import cv2 as cv
import numpy as np
from typing import List, Tuple
import sophus.sophuspy as sp
import os
from vslampy.dataset import TumRgbd

nLevels = 4
max_iterations = 80
min_update = 1e-4
min_reduction = 1.1
w_I = 1.0
w_Z = 1.0 - w_I
w_prior = 0.0
min_dI = 10
min_dZ = np.inf
max_z = 5
max_z_diff = 0.2
fx = 525.0
fy = 525.0
cx = 319.5
cy = 239.5
f_start = 0
n_frames = np.inf

wait_time = 1
np.set_printoptions(precision=4)


class Camera:
    def __init__(self, fx: float, fy: float, cx: float, cy: float, h: int, w: int):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.h = h
        self.w = w
        self.K = np.array([[fx, 0.0, cx], [0, fy, cy], [0, 0, 1]])

        self.Kinv = np.array(
            [[1.0 / fx, 0.0, -cx / fx], [0, 1.0 / fy, -cy / fy], [0, 0, 1]]
        )

    def resize(self, s: float):
        return Camera(
            self.fx * s, self.fy * s, self.cx * s, self.cy * s, self.h * s, self.w * s
        )

    def image_coordinates(self):
        uv = np.dstack(np.meshgrid(np.arange(self.w), np.arange(self.h)))
        return np.reshape(uv, (-1, 2))

    def reconstruct(self, uv: np.array, z: np.array):
        uv1 = np.ones((uv.shape[0], 3))
        uv1[:, :2] = uv
        return z.reshape((-1, 1)) * (self.Kinv @ uv1.T).T

    def project(self, pcl, keep_invalid=False):
        uv = (self.K @ pcl.T).T
        uv /= uv[:, 2, None]
        uv = np.reshape(uv, (-1, 3))

        mask_valid = (
            (pcl[:, 2] > 0)
            & (self.w - 1 > uv[:, 0])
            & (uv[:, 0] > 1)
            & (self.h - 1 > uv[:, 1])
            & (uv[:, 1] > 1)
        )
        if not keep_invalid:
            uv = uv[mask_valid]
        return uv[:, :2], mask_valid


def load_frame(path_img, path_depth) -> Tuple[List[np.array], List[np.array]]:
    if not os.path.exists(path_img):
        raise ValueError(f"Path does not exist: {path_img}")
    if not os.path.exists(path_depth):
        raise ValueError(f"Path does not exist: {path_depth}")

    I = cv.imread(path_img, cv.IMREAD_GRAYSCALE)
    Z = cv.imread(path_depth, cv.IMREAD_ANYDEPTH) / 5000.0
    cv.imshow("I", I)
    cv.imshow("Z", Z)
    cv.waitKey(1)
    I = [I]
    Z = [Z]
    for l in range(1, nLevels):
        I += [cv.pyrDown(I[l - 1])]
        Z += [cv.resize(Z[l - 1], (0, 0), fx=0.5, fy=0.5)]
    return I, Z


def write_result_file(trajectory, filename):
    with open(filename, "w") as f:
        f.writelines([f"{t} {pose.log()}" for t, pose in trajectory])


def compute_jacobian_warp(pcl: np.array, cam: Camera) -> np.array:
    x = pcl[:, 0]
    y = pcl[:, 1]
    z_inv = 1.0 / pcl[:, 2]
    z_inv_2 = z_inv * z_inv

    Jx = np.zeros((pcl.shape[0], 6))
    Jx[:, 0] = z_inv
    Jx[:, 2] = -x * z_inv_2
    Jx[:, 3] = y * Jx[:, 2]
    Jx[:, 4] = 1.0 - x * Jx[:, 2]
    Jx[:, 5] = -y * z_inv
    Jx *= cam.fx

    Jy = np.zeros((pcl.shape[0], 6))
    Jy[:, 1] = z_inv
    Jy[:, 2] = -y * z_inv_2
    Jy[:, 3] = -1.0 + y * Jy[:, 2]
    Jy[:, 4] = -Jy[:, 3]
    Jy[:, 5] = x * z_inv
    Jy *= cam.fy

    return Jx, Jy


def compute_jacobian_se3_z(pcl: np.array) -> np.array:
    J = np.zeros((pcl.shape[0], 6))
    J[:, 2] = 1.0
    J[:, 3] = pcl[:, 1]
    J[:, 4] = -pcl[:, 0]
    return J


def interpolate(I: np.array, Z, uv: np.array, zt: np.array) -> np.array:
    u = uv[:, 0]
    v = uv[:, 1]
    u0 = np.floor(u).astype(int)
    u1 = np.ceil(u).astype(int)
    v0 = np.floor(v).astype(int)
    v1 = np.ceil(v).astype(int)

    w_u1 = u - u0
    w_u0 = 1.0 - w_u1
    w_v1 = v - v0
    w_v0 = 1.0 - w_v1

    w00 = np.reshape(w_u0 * w_v0, (-1,))
    w00[np.abs(Z[v0, u0].reshape((-1,)) - zt) > max_z_diff] = 0
    w00[~np.isfinite(Z[v0, u0].reshape((-1,)))] = 0

    w10 = np.reshape(w_u0 * w_v1, (-1,))
    w10[np.abs(Z[v1, u0].reshape((-1,)) - zt) > max_z_diff] = 0
    w10[~np.isfinite(Z[v1, u0].reshape((-1,)))] = 0

    w01 = np.reshape(w_u1 * w_v0, (-1,))
    w01[np.abs(Z[v0, u1].reshape((-1,)) - zt) > max_z_diff] = 0
    w01[~np.isfinite(Z[v0, u1].reshape((-1,)))] = 0

    w11 = np.reshape(w_u1 * w_v1, (-1,))
    w11[np.abs(Z[v1, u1].reshape((-1,)) - zt) > max_z_diff] = 0
    w11[~np.isfinite(Z[v1, u1].reshape((-1,)))] = 0
    w_sum = np.reshape(w00 + w01 + w10 + w11, (-1,))

    Ivu = w00 * I[v0, u0] + w01 * I[v0, u1] + w10 * I[v1, u0] + w11 * I[v1, u1]
    Ivu /= w_sum

    Zvu = w00 * Z[v0, u0] + w01 * Z[v0, u1] + w10 * Z[v1, u0] + w11 * Z[v1, u1]
    Zvu /= w_sum
    return Ivu.reshape((-1,)), Zvu.reshape((-1,))


def compute_weights(r: np.array, sigma: float) -> np.array:
    v = 5.0
    w = (v + 1) / (v + (r / sigma) ** 2)
    W = np.zeros((r.shape[0], r.shape[0]))
    W[range(r.shape[0]), range(r.shape[0])] = w.reshape((-1,))
    return W


def estimate_weights(r: np.array) -> np.array:
    step_size = np.inf
    sigma = 1.0
    for iter in range(50):
        W = compute_weights(r, sigma)
        sigma_i = np.sqrt(float(r.T @ W @ r) / r.shape[0])
        step_size = np.abs(sigma - sigma_i)
        sigma = sigma_i
        # print(
        #    f"iter = {iter}, sigma = {sigma:4f}, step_size = {step_size:4f} \nW={statsstr(W.diagonal())}"
        # )
        if step_size < 1e-5:
            break

        if sigma <= 1e-9:
            return 1e-9, np.identity(r.shape[0])

    print(
        f"EM: {iter}, precision: {step_size:.4f}, scale: {sigma:.4f}, \nW={statsstr(W.diagonal())}"
    )
    return sigma, compute_weights(r, sigma)


def statsstr(x) -> str:
    return f"{np.linalg.norm(x):.4f}, {x.min():.4f} < {x.mean():.4f} +- {x.std():.4f} < {x.max():.4f} n={x.shape[0]}, d={x.shape}"


def compute_jacobian_image(I, Z):
    dIdx = cv.Sobel(I, cv.CV_64F, dx=1, dy=0)
    dIdy = cv.Sobel(I, cv.CV_64F, dx=0, dy=1)
    # print(f"dIdx: {statsstr(dIdx.reshape((-1,1)))}")
    # print(f"dIdy: {statsstr(dIdy.reshape((-1,1)))}")
    # cv.imshow("dI", np.vstack([dIdx, dIdy]))

    dI = np.dstack([dIdx, dIdy])

    dZdx = cv.Sobel(Z, cv.CV_64F, dx=1, dy=0, scale=1 / 8)
    dZdy = cv.Sobel(Z, cv.CV_64F, dx=0, dy=1, scale=1 / 8)
    # print(f"dZdx: {statsstr(dZdx.reshape((-1,1)))}")
    # print(f"dZdx: {statsstr(dZdx.reshape((-1,1)))}")
    # cv.imshow("dZ", np.vstack([dZdx, dZdy]))

    dZ = np.dstack((dZdx, dZdy))

    return np.reshape(dI, (-1, 2)), np.reshape(dZ, (-1, 2))


def selection_mask(Z, dI, dZ):
    return (
        (np.isfinite(Z[:, 0]))
        & (Z[:, 0] > 0)
        & (Z[:, 0] < max_z)
        & (
            (np.abs(dI[:, 0]) > min_dI)
            | (np.abs(dI[:, 1]) > min_dI)
            | (np.abs(dZ[:, 0]) > min_dZ)
            | (np.abs(dZ[:, 1]) > min_dZ)
        )
    )


def interpolate_pose_between(trajectory, t0, t1):
    trajectory_t0 = [(t, p) for t, p in trajectory.items() if t >= t0]
    tp0 = trajectory_t0[0]
    tp1 = [(t, p) for t, p in dict(trajectory_t0).items() if t >= t1][0]
    dt = t1 - t0
    dt_traj = tp1[0] - tp0[0]
    s = dt / dt_traj if dt_traj != 0 else 1

    dp = sp.SE3.exp(s * (tp1[1] * tp0[1].inverse()).log())

    print(f"Interpolate Pose at t0={tp0[0]} and t1={tp1[0]}, dt={dt} dp={dp.log()}")
    return dp


sequence = TumRgbd("rgbd_dataset_freiburg2_desk")

timestamps, files_I, files_Z = sequence.image_depth_filepaths()

trajectory = {}
trajectory_gt = dict(
    (t, sp.SE3(p).inverse()) for t, p in sequence.gt_trajectory().items()
)

pose = sp.SE3()
f_no0 = f_start
t0 = timestamps[f_start]
I0, Z0 = load_frame(files_I[f_start], files_Z[f_start])
h, w = I0[0].shape[:2]
cam0 = Camera(fx, fy, cx, cy, h, w)
f_end = min([n_frames, len(timestamps)])
motion = sp.SE3()
for f_no in range(f_start + 1, f_end):
    t1 = timestamps[f_no]
    I1, Z1 = load_frame(files_I[f_no], files_Z[f_no])
    prior = motion  # sp.SE3()  # interpolate_pose_between(trajectory_gt, t0, t1)
    motion = prior
    print(
        f"_________Aligning: {f_no0} -> {f_no} / {f_end}, {t0}->{t1}, dt={t1-t0:.3f}___________"
    )
    for l_ in range(nLevels):
        level = nLevels - (l_ + 1)
        print(f"_________Level={level}___________")
        cam = cam0.resize(1 / (2**level))

        dI0, dZ0 = compute_jacobian_image(I0[level], Z0[level])

        mask_selected = selection_mask(Z0[level].reshape(-1, 1), dI0, dZ0)

        pcl0 = cam.reconstruct(
            cam.image_coordinates()[mask_selected],
            Z0[level].reshape(-1, 1)[mask_selected],
        )

        Jwx, Jwy = compute_jacobian_warp(motion * pcl0, cam)

        JIJw = dI0[:, :1][mask_selected] * Jwx + dI0[:, 1:][mask_selected] * Jwy
        JZJw = dZ0[:, :1][mask_selected] * Jwx + dZ0[:, 1:][mask_selected] * Jwy

        chi2 = np.zeros((max_iterations,))
        dx = np.zeros((6,))
        for i in range(max_iterations):
            # print(f"_________Iteration={i}___________")
            pcl0t = motion * pcl0
            uv0t, mask_valid = cam.project(pcl0t)

            i1wxp, z1wxp = interpolate(
                I1[level], Z1[level], uv0t, pcl0t[:, 2].reshape((-1,))[mask_valid]
            )

            mask_occluded = np.logical_not(np.isnan(i1wxp))
            uv0t = uv0t[mask_occluded]
            z1wxp = z1wxp[mask_occluded]
            i1wxp = i1wxp[mask_occluded]
            N = uv0t.shape[0]
            norm = N * 255

            pcl1t = motion.inverse() * (cam.reconstruct(uv0t, z1wxp))

            r_I = (
                i1wxp
                - I0[level].reshape((-1,))[mask_selected][mask_valid][mask_occluded]
            ) / norm
            r_Z = (
                pcl1t[:, 2]
                - Z0[level].reshape((-1,))[mask_selected][mask_valid][mask_occluded]
            )
            # if i == 0:
            #    sigma_I, W_I = estimate_weights(r_I)
            # else:
            #    W_I = compute_weights(r_I, sigma_I)

            # s_Z, W_Z = compute_scale(r_Z)
            chi2[i] = w_I * (r_I.T @ r_I)

            if i > 0:
                # print(
                #    f"r_I: {statsstr(r_I)}\n"
                #    f"r_Z: {statsstr(r_Z)}\n"
                #    f"chi2={chi2[i]:0.6f}, dchi2={(chi2[i]/chi2[i-1])*100:0.2f} %, dchi2={chi2[i]-chi2[i-1]:0.6f}"
                # )
                if chi2[i] / chi2[i - 1] > min_reduction:
                    motion = sp.SE3.exp(dx) * motion
                    print(f"Stop. Error Increased.")
                    break
            else:
                pass
                # print(
                #    f"r_I: {statsstr(r_I)}\n"
                #    f"r_Z: {statsstr(r_Z)}\n"
                #    f"chi2={chi2[i]:0.6f}"
                # )

            if wait_time >= 0:
                I1Wxp = np.zeros_like(I0[level])
                Z1Wxp = np.zeros_like(Z0[level])
                R_I = np.zeros_like(I0[level])
                R_Z = np.zeros_like(Z0[level])
                WI = np.zeros_like(I0[level])
                WZ = np.zeros_like(Z0[level])
                M = np.zeros_like(I0[level])

                uv0 = cam.image_coordinates()[mask_selected][mask_valid][
                    mask_occluded
                ].astype(int)

                I1Wxp[uv0[:, 1], uv0[:, 0]] = i1wxp.reshape((-1,))
                Z1Wxp[uv0[:, 1], uv0[:, 0]] = pcl1t[:, 2].reshape((-1,))
                R_I[uv0[:, 1], uv0[:, 0]] = np.abs(r_I.reshape((-1,))) * norm
                M[uv0[:, 1], uv0[:, 0]] = 255
                # R_Z[uv0[:, 1], uv0[:, 0]] = np.abs(r_Z.reshape((-1,)))
                # WI[uv0[:, 1], uv0[:, 0]] = W_I.diagonal().reshape((-1,))
                # WI = (255.0 * WI / WI.max()).astype(np.uint8)
                # WZ[uv0[:, 1], uv0[:, 0]] = W_Z.diagonal().reshape((-1,))
                intensity_vis = np.hstack([I0[level], I1Wxp, R_I, M])
                intensity_vis = cv.resize(intensity_vis, (640 * 4, 480))
                intensity_vis = cv.putText(
                    intensity_vis,
                    f"#:{f_no}/{f_end} l={level} i={i} chi2/N={255*chi2[i]:.6f} |dx|={np.linalg.norm(dx):0.5f}",
                    (10, 20),
                    cv.FONT_HERSHEY_TRIPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )
                cv.imshow("Intensity", intensity_vis)
                # cv.imshow("Z1Wxp", Z1Wxp)
                # cv.imshow("r_Z", R_Z)
                # cv.imshow("WI", WI)
                # cv.imshow("WZ", WZ)
                cv.waitKey(wait_time)

            JIJw_ = JIJw[mask_valid][mask_occluded] / norm
            JZJpJt_Jtz = JZJw[mask_valid][mask_occluded] - compute_jacobian_se3_z(pcl1t)

            # Az = JZJpJt_Jtz.T @ W_Z @ JZJpJt_Jtz
            # bz = JZJpJt_Jtz.T @ W_Z @ r_Z

            Ai = JIJw_.T @ JIJw_
            bi = JIJw_.T @ r_I

            # A = (w_I * Ai + w_Z * Az) / N
            # b = (np.sqrt(w_I) * bi + np.sqrt(w_Z) * bz) / N

            dx = np.linalg.solve(
                Ai + w_prior * np.identity(6),
                bi + w_prior * (motion.inverse() * prior).log(),
            )
            motion = sp.SE3.exp(-dx) * motion
            # print(f"dx={np.linalg.norm(dx):.6f}")

            # print(
            #    f"t={motion.log()[:3]}m r={np.linalg.norm(motion.log()[3:])*180.0/np.pi:.3f}°"
            # )

            if np.linalg.norm(dx) < min_update:
                print(
                    f"Converged. Min Step Size reached: {np.linalg.norm(dx):.6f}/{min_update:.6f}"
                )
                break

    pose = motion * pose
    trajectory[timestamps[f_no]] = pose.inverse().matrix()
    f_no0 = f_no
    t0 = t1
    I0 = I1
    Z0 = Z1

sequence.evaluate_rpe(trajectory, output_dir="./", upload=False)
