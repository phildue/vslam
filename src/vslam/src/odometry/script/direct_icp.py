import cv2 as cv
import numpy as np
from typing import List, Tuple
from sophus.sophuspy import SE3
import os
from vslampy.dataset import TumRgbd


def statsstr(x) -> str:
    return f"{np.linalg.norm(x):.4f}, {x.min():.4f} < {x.mean():.4f} +- {x.std():.4f} < {x.max():.4f} n={x.shape[0]}, d={x.shape}"


def interpolate_pose_between(trajectory, t0, t1):
    trajectory_t0 = [(t, p) for t, p in trajectory.items() if t >= t0]
    tp0 = trajectory_t0[0]
    tp1 = [(t, p) for t, p in dict(trajectory_t0).items() if t >= t1][0]
    dt = t1 - t0
    dt_traj = tp1[0] - tp0[0]
    s = dt / dt_traj if dt_traj != 0 else 1

    dp = SE3.exp(s * (tp1[1] * tp0[1].inverse()).log())

    print(f"Interpolate Pose at t0={tp0[0]} and t1={tp1[0]}, dt={dt} dp={dp.log()}")
    return dp


def load_frame(path_img, path_depth) -> Tuple[List[np.array], List[np.array]]:
    if not os.path.exists(path_img):
        raise ValueError(f"Path does not exist: {path_img}")
    if not os.path.exists(path_depth):
        raise ValueError(f"Path does not exist: {path_depth}")

    I = cv.imread(path_img, cv.IMREAD_GRAYSCALE)
    Z = cv.imread(path_depth, cv.IMREAD_ANYDEPTH) / 5000.0
    cv.imshow("Frame", np.hstack([I, Z / Z.max() * 255]))
    cv.waitKey(1)
    return I, Z


def write_result_file(trajectory, filename):
    with open(filename, "w") as f:
        f.writelines([f"{t} {pose.log()}" for t, pose in trajectory])


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


class DirectIcp:
    def __init__(
        self,
        cam: Camera,
        nLevels: int,
        weight_intensity=1.0,
        weight_prior=0.0,
        min_gradient_intensity=10 * 8,
        min_gradient_depth=np.inf,
        max_z=5.0,
        max_z_diff=0.2,
        max_iterations=100,
        min_parameter_update=1e-4,
        max_delta_chi2=1.1,
    ):
        self.nLevels = nLevels
        self.cam = [cam.resize(1 / (2**l)) for l in range(nLevels)]
        self.f_no = 0
        self.I0 = None
        self.Z0 = None
        self.t0 = None
        self.weight_intensity = weight_intensity
        self.weight_depth = 1.0 - weight_intensity
        self.weight_prior = weight_prior
        self.min_dI = min_gradient_intensity
        self.min_dZ = min_gradient_depth
        self.max_z = max_z
        self.max_z_diff = max_z_diff
        self.max_iterations = max_iterations
        self.min_parameter_update = min_parameter_update
        self.max_delta_chi2 = max_delta_chi2

    def compute_pyramid(self, I, Z):
        I = [I]
        Z = [Z]
        for l in range(1, self.nLevels):
            I += [cv.pyrDown(I[l - 1])]
            Z += [cv.resize(Z[l - 1], (0, 0), fx=0.5, fy=0.5)]
        return I, Z

    def compute_jacobian_warp_xy(self, pcl: np.array, cam: Camera) -> np.array:
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

    def compute_jacobian_se3_z(self, pcl: np.array) -> np.array:
        J = np.zeros((pcl.shape[0], 6))
        J[:, 2] = 1.0
        J[:, 3] = pcl[:, 1]
        J[:, 4] = -pcl[:, 0]
        return J

    def interpolate(self, I: np.array, Z, uv: np.array, zt: np.array) -> np.array:
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

        w00 = np.reshape(w_u0 * w_v0, (-1, 1))
        w00[np.abs(Z[v0, u0].reshape((-1, 1)) - zt) > self.max_z_diff] = 0
        w00[~np.isfinite(Z[v0, u0].reshape((-1, 1)))] = 0

        w10 = np.reshape(w_u0 * w_v1, (-1, 1))
        w10[np.abs(Z[v1, u0].reshape((-1, 1)) - zt) > self.max_z_diff] = 0
        w10[~np.isfinite(Z[v1, u0].reshape((-1, 1)))] = 0

        w01 = np.reshape(w_u1 * w_v0, (-1, 1))
        w01[np.abs(Z[v0, u1].reshape((-1, 1)) - zt) > self.max_z_diff] = 0
        w01[~np.isfinite(Z[v0, u1].reshape((-1, 1)))] = 0

        w11 = np.reshape(w_u1 * w_v1, (-1, 1))
        w11[np.abs(Z[v1, u1].reshape((-1, 1)) - zt) > self.max_z_diff] = 0
        w11[~np.isfinite(Z[v1, u1].reshape((-1, 1)))] = 0
        w_sum = np.reshape(w00 + w01 + w10 + w11, (-1, 1))

        M = np.dstack([I, Z])
        Mvu = w00 * M[v0, u0] + w01 * M[v0, u1] + w10 * M[v1, u0] + w11 * M[v1, u1]
        Mvu /= w_sum

        mask_occluded = np.logical_not(np.isnan(Mvu[:, 0].reshape((-1,))))

        return (
            Mvu[:, 0].reshape((-1,))[mask_occluded],
            Mvu[:, 1].reshape((-1,))[mask_occluded],
            mask_occluded,
        )

    def compute_weights(self, r: np.array, sigma: float) -> np.array:
        v = 5.0
        w = (v + 1) / (v + (r / sigma) ** 2)
        W = np.zeros((r.shape[0], r.shape[0]))
        W[range(r.shape[0]), range(r.shape[0])] = w.reshape((-1,))
        return W

    def estimate_weights(self, r: np.array) -> np.array:
        step_size = np.inf
        sigma = 1.0
        for iter in range(50):
            W = self.compute_weights(r, sigma)
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
        return sigma, self.compute_weights(r, sigma)

    def compute_jacobian_image(self, I):
        dIdx = cv.Sobel(I, cv.CV_64F, dx=1, dy=0)
        dIdy = cv.Sobel(I, cv.CV_64F, dx=0, dy=1)

        return np.reshape(np.dstack([dIdx, dIdy]), (-1, 2))

    def select_constraints(self, Z, dI, dZ):
        return (
            (np.isfinite(Z[:, 0]))
            & (Z[:, 0] > 0)
            & (Z[:, 0] < self.max_z)
            & (
                (np.abs(dI[:, 0]) > self.min_dI)
                | (np.abs(dI[:, 1]) > self.min_dI)
                | (np.abs(dZ[:, 0]) > self.min_dZ)
                | (np.abs(dZ[:, 1]) > self.min_dZ)
            )
        )

    def error_increased(self, chi2, i):
        if i > 0:
            # print(
            #    f"r_I: {statsstr(r_I)}\n"
            #    f"r_Z: {statsstr(r_Z)}\n"
            #    f"chi2={chi2[i]:0.6f}, dchi2={(chi2[i]/chi2[i-1])*100:0.2f} %, dchi2={chi2[i]-chi2[i-1]:0.6f}"
            # )
            if chi2[i] / chi2[i - 1] > self.max_delta_chi2:
                print(f"Iteration= {i}: Stop. Error Increased.")
                return True
        else:
            pass
            # print(
            #    f"r_I: {statsstr(r_I)}\n"
            #    f"r_Z: {statsstr(r_Z)}\n"
            #    f"chi2={chi2[i]:0.6f}"
            # )
        return False

    def compute_pose_update(self, JIJw, r_I, JZJw, r_Z, prior):
        # Az = JZJpJt_Jtz.T @ W_Z @ JZJpJt_Jtz
        # bz = JZJpJt_Jtz.T @ W_Z @ r_Z

        Ai = JIJw.T @ JIJw
        bi = JIJw.T @ r_I

        # A = (w_I * Ai + w_Z * Az) / N
        # b = (np.sqrt(w_I) * bi + np.sqrt(w_Z) * bz) / N

        return np.linalg.solve(
            Ai + self.weight_prior * np.identity(6),
            bi + self.weight_prior * (motion.inverse() * prior).log(),
        )

    def compute_egomotion(
        self, t1: float, I1: np.array, Z1: np.array, guess=SE3()
    ) -> SE3:
        self.f_no += 1
        I1, Z1 = self.compute_pyramid(I1, Z1)

        if self.I0 is None:
            self.I0 = I1
            self.Z0 = Z1
            self.t0 = t1
            return guess

        prior = guess
        motion = prior

        for l_ in range(self.nLevels):
            level = self.nLevels - (l_ + 1)
            print(f"_________Level={level}___________")

            dI0 = self.compute_jacobian_image(self.I0[level])
            dZ0 = self.compute_jacobian_image(self.Z0[level])

            mask_selected = self.select_constraints(
                self.Z0[level].reshape(-1, 1), dI0, dZ0
            )

            pcl0 = self.cam[level].reconstruct(
                self.cam[level].image_coordinates()[mask_selected],
                self.Z0[level].reshape(-1, 1)[mask_selected],
            )

            Jwx, Jwy = self.compute_jacobian_warp_xy(motion * pcl0, self.cam[level])

            JIJw = dI0[:, :1][mask_selected] * Jwx + dI0[:, 1:][mask_selected] * Jwy
            JZJw = dZ0[:, :1][mask_selected] * Jwx + dZ0[:, 1:][mask_selected] * Jwy

            chi2 = np.zeros((self.max_iterations,))
            dx = np.zeros((6,))
            for i in range(self.max_iterations):
                # print(f"_________Iteration={i}___________")
                pcl0t = motion * pcl0
                uv0t, mask_valid = self.cam[level].project(pcl0t)

                i1wxp, z1wxp, mask_occluded = self.interpolate(
                    I1[level],
                    Z1[level],
                    uv0t,
                    pcl0t[:, 2].reshape((-1, 1))[mask_valid],
                )
                N = i1wxp.shape[0]
                norm_I = N * 255

                i0x = self.I0[level].reshape((-1,))[mask_selected][mask_valid][
                    mask_occluded
                ]

                pcl1t = motion.inverse() * (
                    self.cam[level].reconstruct(uv0t[mask_occluded], z1wxp)
                )
                z0x = self.Z0[level].reshape((-1,))[mask_selected][mask_valid][
                    mask_occluded
                ]
                r_I = (i1wxp - i0x) / norm_I
                r_Z = pcl1t[:, 2] - z0x
                # if i == 0:
                #    sigma_I, W_I = estimate_weights(r_I)
                # else:
                #    W_I = compute_weights(r_I, sigma_I)

                # s_Z, W_Z = compute_scale(r_Z)
                chi2[i] = self.weight_intensity * (r_I.T @ r_I) + self.weight_depth * (
                    r_Z.T @ r_Z
                )

                if self.error_increased(chi2, i):
                    motion = SE3.exp(dx) * motion
                    break

                if wait_time >= 0:
                    I1Wxp = np.zeros_like(self.I0[level])
                    Z1Wxp = np.zeros_like(self.Z0[level])
                    R_I = np.zeros_like(self.I0[level])
                    R_Z = np.zeros_like(self.Z0[level])
                    WI = np.zeros_like(self.I0[level])
                    WZ = np.zeros_like(self.Z0[level])
                    M = np.zeros_like(self.I0[level])

                    uv0 = (
                        self.cam[level]
                        .image_coordinates()[mask_selected][mask_valid][mask_occluded]
                        .astype(int)
                    )

                    I1Wxp[uv0[:, 1], uv0[:, 0]] = i1wxp.reshape((-1,))
                    Z1Wxp[uv0[:, 1], uv0[:, 0]] = pcl1t[:, 2].reshape((-1,))
                    R_I[uv0[:, 1], uv0[:, 0]] = np.abs(r_I.reshape((-1,))) * norm_I
                    M[uv0[:, 1], uv0[:, 0]] = 255
                    # R_Z[uv0[:, 1], uv0[:, 0]] = np.abs(r_Z.reshape((-1,)))
                    # WI[uv0[:, 1], uv0[:, 0]] = W_I.diagonal().reshape((-1,))
                    # WI = (255.0 * WI / WI.max()).astype(np.uint8)
                    # WZ[uv0[:, 1], uv0[:, 0]] = W_Z.diagonal().reshape((-1,))
                    intensity_vis = np.hstack([self.I0[level], I1Wxp, R_I, M])
                    intensity_vis = cv.resize(intensity_vis, (640 * 4, 480))
                    intensity_vis = cv.putText(
                        intensity_vis,
                        f"#:{self.f_no} l={level} i={i} chi2/N={255*chi2[i]:.6f} |dx|={np.linalg.norm(dx):0.5f}",
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

                dx = self.compute_pose_update(
                    JIJw[mask_valid][mask_occluded] / norm_I,
                    r_I,
                    JZJw[mask_valid][mask_occluded]
                    - self.compute_jacobian_se3_z(pcl1t),
                    r_Z,
                    prior,
                )

                motion = SE3.exp(-dx) * motion
                # print(f"dx={np.linalg.norm(dx):.6f}")

                # print(
                #    f"t={motion.log()[:3]}m r={np.linalg.norm(motion.log()[3:])*180.0/np.pi:.3f}°"
                # )

                if np.linalg.norm(dx) < self.min_parameter_update:
                    print(
                        f"Iteration= {i}: Converged. Min Step Size reached: {np.linalg.norm(dx):.6f}/{self.min_parameter_update:.6f}"
                    )
                    break

        self.I0 = I1
        self.Z0 = Z1
        self.t0 = t1
        return motion


if __name__ == "__main__":
    f_start = 0
    n_frames = 250  # np.inf

    wait_time = 1
    np.set_printoptions(precision=4)

    sequence = TumRgbd("rgbd_dataset_freiburg2_desk")
    direct_icp = DirectIcp(
        Camera(fx=525.0, fy=525.0, cx=319.5, cy=239.5, h=480, w=640),
        nLevels=4,
        weight_intensity=1.0,
        weight_prior=0.0,
        min_gradient_intensity=1 * 8,
        min_gradient_depth=np.inf,
        max_z=5.0,
        max_z_diff=0.2,
        max_iterations=100,
        min_parameter_update=1e-4,
        max_delta_chi2=1.1,
    )
    timestamps, files_I, files_Z = sequence.image_depth_filepaths()

    trajectory = {}
    trajectory_gt = dict(
        (t, SE3(p).inverse()) for t, p in sequence.gt_trajectory().items()
    )

    pose = SE3()
    f_no0 = f_start
    t0 = timestamps[f_start]
    f_end = min([n_frames, len(timestamps)])
    motion = SE3()
    for f_no in range(f_start, f_end):
        t1 = timestamps[f_no]
        I1, Z1 = load_frame(files_I[f_no], files_Z[f_no])
        print(
            f"_________Aligning: {f_no0} -> {f_no} / {f_end}, {t0}->{t1}, dt={t1-t0:.3f}___________"
        )
        motion = direct_icp.compute_egomotion(t1, I1, Z1, motion)

        pose = motion * pose
        trajectory[timestamps[f_no]] = pose.inverse().matrix()
        f_no0 = f_no
        t0 = t1
        I0 = I1
        Z0 = Z1

    sequence.evaluate_rpe(trajectory, output_dir="./", upload=False)
