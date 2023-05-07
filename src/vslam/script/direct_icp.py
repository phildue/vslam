import cv2 as cv
import numpy as np
from typing import List, Tuple
import sophus.sophuspy as sp
import os

nLevels = 4
max_iterations = 100
min_update = 1e-6
min_reduction = 1.1
w_I = 1.0
w_Z = 1.0 - w_I
min_gradient_I = 20.0 / 255.0
min_gradient_Z = 10
fx = 525.0
fy = 525.0
cx = 319.5
cy = 239.5

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

    def image_coordinates(self):
        uv = np.dstack(np.meshgrid(np.arange(self.w), np.arange(self.h)))
        return np.reshape(uv, (-1, 2))

    def backproject(self, uv: np.array, z: np.array):
        uv1 = np.ones((uv.shape[0], 3))
        uv1[:, :2] = uv
        return z * (self.Kinv @ uv1.T).T

    def project(self, pcl, keep_invalid=False):
        uv = (cam.K @ pcl.T).T
        uv /= uv[:, 2, None]
        uv = np.reshape(uv, (-1, 3))

        mask_valid = (
            (pcl[:, 2] > 0)
            & (w - 1 > uv[:, 0])
            & (uv[:, 0] > 1)
            & (h - 1 > uv[:, 1])
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
    I = [I]
    Z = [Z]
    for l in range(1, nLevels):
        I += [cv.pyrDown(I[l - 1])]
        Z += [cv.resize(Z[l - 1], (0, 0), fx=0.5, fy=0.5)]
    return I, Z


def load_association_file(folder: str, filename: str):
    timestamps = []
    filenames_depth = []
    filenames_intensity = []

    for line in open(folder + "/" + filename, "r"):
        elements = line.split(" ")
        timestamps += [float(elements[0])]
        filenames_depth += [folder + "/" + elements[1]]
        filenames_intensity += [folder + "/" + elements[3][:-1]]

    print(f"Found {len(timestamps)} frames")
    return timestamps, filenames_intensity, filenames_depth


def computeJw(pcl: np.array, dI: np.array, cam: Camera) -> np.array:
    J = np.zeros((pcl.shape[0], 6, 2))
    x = pcl[:, 0]
    y = pcl[:, 1]
    z_inv = 1.0 / pcl[:, 2]
    z_inv_2 = z_inv * z_inv

    J[:, 0, 0] = z_inv
    J[:, 2, 0] = -x * z_inv_2
    J[:, 3, 0] = y * J[:, 2, 0]
    J[:, 4, 0] = 1.0 - x * J[:, 2, 0]
    J[:, 5, 0] = -y * z_inv
    J[:, :, 0] *= cam.fx

    J[:, 1, 1] = z_inv
    J[:, 2, 1] = -y * z_inv_2
    J[:, 3, 1] = -1.0 + y * J[:, 2, 1]
    J[:, 4, 1] = -J[:, 3, 1]
    J[:, 5, 1] = x * z_inv
    J[:, :, 1] *= cam.fy

    return dI[:, :1] * J[:, :, 0] + dI[:, 1:] * J[:, :, 1]


def computeJtz(pcl: np.array) -> np.array:
    J = np.zeros((pcl.shape[0], 6))
    J[:, 2] = 1.0
    J[:, 3] = pcl[:, 1]
    J[:, 4] = -pcl[:, 0]
    return J


def interpolate(mat: np.array, uv: np.array) -> np.array:
    u = uv[:, 0]
    v = uv[:, 1]

    u1 = np.floor(u).astype(int)
    u2 = np.ceil(u).astype(int)
    v1 = np.floor(v).astype(int)
    v2 = np.ceil(v).astype(int)
    Q11 = mat[v1, u1]
    Q12 = mat[v1, u2]
    Q21 = mat[v2, u1]
    Q22 = mat[v2, u2]

    R1 = np.zeros((uv.shape[0], mat.shape[-1]))
    R2 = np.zeros((uv.shape[0], mat.shape[-1]))
    R1[u2 == u1] = Q11[u2 == u1]
    R2[u2 == u1] = Q12[u2 == u1]
    m1 = (u2[u2 != u1] - u[u2 != u1]) / (u2[u2 != u1] - u1[u2 != u1])
    m2 = (u[u2 != u1] - u1[u2 != u1]) / (u2[u2 != u1] - u1[u2 != u1])
    R1[u2 != u1] = m1[:, None] * Q11[u2 != u1] + m2[:, None] * Q21[u2 != u1]

    m1 = (u2[u2 != u1] - u[u2 != u1]) / (u2[u2 != u1] - u1[u2 != u1])
    m2 = (u[u2 != u1] - u1[u2 != u1]) / (u2[u2 != u1] - u1[u2 != u1])

    R2[u2 != u1] = m1[:, None] * Q12[u2 != u1] + m1[:, None] * Q22[u2 != u1]

    P = np.zeros((uv.shape[0], mat.shape[-1]))
    P[v2 == v1] = R1[v2 == v1]

    m1 = (v2[v2 != v1] - v[v2 != v1]) / (v2[v2 != v1] - v1[v2 != v1])
    m2 = (v[v2 != v1] - v1[v2 != v1]) / (v2[v2 != v1] - v1[v2 != v1])

    P[v2 != v1] = m1[:, None] * R1[v2 != v1] + m2[:, None] * R2[v2 != v1]

    return P


def compute_weights(r: np.array, sigma: float) -> np.array:
    v = 5
    w = (v + 1) / (v + (r / sigma) ** 2)
    W = np.zeros((r.shape[0], r.shape[0]))
    W[range(r.shape[0]), range(r.shape[0])] = w.reshape((-1,))
    return W


def compute_scale(r: np.array) -> np.array:
    step_size = np.inf
    sigma = 1.0
    for iter in range(50):
        W = compute_weights(r, sigma)
        sigma_i = np.sqrt(float(r.T @ W @ r) / r.shape[0])
        step_size = np.abs(sigma - sigma_i)
        sigma = sigma_i
        # print(f"iter = {iter}, sigma = {sigma:4f}, step_size = {step_size:4f}")
        if step_size < 1e-5:
            break

    print(
        f"EM[scale]: {iter}, precision: {step_size:.4f}, scale: {sigma:.4f}, \nW={statsstr(W.diagonal())}"
    )
    return sigma, W


def statsstr(x) -> str:
    return f"{np.linalg.norm(x):.4f}, {x.min():.4f} < {x.mean():.4f} +- {x.std():.4f} < {x.max():.4f}"


def compute_gradients(I, Z):
    dIdx = cv.Sobel(I, cv.CV_64F, dx=1, dy=0)
    dIdy = cv.Sobel(I, cv.CV_64F, dx=0, dy=1)

    cv.imshow("dIdx", dIdx)
    cv.imshow("dIdy", dIdy)

    dI = np.dstack([dIdx, dIdy]) / 255.0

    dZdx = cv.Sobel(Z, cv.CV_64F, dx=1, dy=0)
    dZdy = cv.Sobel(Z, cv.CV_64F, dx=0, dy=1)

    cv.imshow("dZdx", dZdx)
    cv.imshow("dZdy", dZdy)

    dZ = np.dstack((dZdx, dZdy))

    return np.reshape(dI, (-1, 2)), np.reshape(dZ, (-1, 2))


def selection_mask(pcl, dI, dZ):
    z = pcl[:, 2]
    return (z > 0) & (
        (np.abs(dI[:, 0]) > min_gradient_I)
        | (np.abs(dI[:, 1]) > min_gradient_I)
        | (np.abs(dZ[:, 0]) > min_gradient_Z)
        | (np.abs(dZ[:, 1]) > min_gradient_Z)
    )


# load assoc file
timestamps, files_I, files_Z = load_association_file(
    "/mnt/dataset/tum_rgbd/rgbd_dataset_freiburg2_desk/rgbd_dataset_freiburg2_desk",
    "assoc.txt",
)

pose = sp.SE3()
I0, Z0 = load_frame(files_I[0], files_Z[0])
for f_no in range(1, len(timestamps)):
    I1, Z1 = load_frame(files_I[f_no], files_Z[f_no])
    for l_ in range(nLevels):
        level = nLevels - (l_ + 1)
        h, w = I0[level].shape[:2]
        cam = Camera(
            fx / (2**level),
            fy / (2**level),
            cx / (2**level),
            cy / (2**level),
            h,
            w,
        )

        dI0, dZ0 = compute_gradients(I0[level], Z0[level])

        pcl0 = cam.backproject(cam.image_coordinates(), Z0[level].reshape(-1, 1))

        mask_selected = selection_mask(pcl0, dI0, dZ0)

        dI0 = dI0[mask_selected]
        dZ0 = dZ0[mask_selected]

        JwI = computeJw(pcl0[mask_selected], dI0, cam)
        JwZ = computeJw(pcl0[mask_selected], dZ0, cam)

        chi2 = np.zeros((max_iterations,))
        for i in range(max_iterations):
            uv0t, mask_valid = cam.project(pose * pcl0[mask_selected])
            N = uv0t.shape[0]

            i1wxp, z1wxp = np.split(
                interpolate(np.dstack([I1[level], Z1[level]]), uv0t), 2, axis=-1
            )

            pcl1t = pose.inverse() * (cam.backproject(uv0t, z1wxp))

            uv0 = cam.image_coordinates()[mask_selected][mask_valid]
            I0x = I0[level][uv0[:, 1], uv0[:, 0]].reshape((-1, 1))
            Z0x = Z0[level][uv0[:, 1], uv0[:, 0]]
            r_I = (i1wxp - I0x) / 255.0
            r_Z = pcl1t[:, 2] - Z0x
            r_Z = np.reshape(r_Z, (-1, 1))
            print(f"r_I: {statsstr(r_I)}")

            s_I, W_I = compute_scale(r_I)

            s_Z, W_Z = compute_scale(r_Z)

            chi2[i] = float(
                w_I * (r_I.T @ W_I @ r_I) / N + w_Z + (r_Z.T @ W_Z @ r_Z) / N
            )

            print(
                f"_________fNo={f_no}/{len(timestamps)}, t={timestamps[f_no]}, Level={level}, Iteration={i}___________"
            )
            print(f"chi2={chi2[i]:0.6f}")
            # cv.imshow("I1Wxp", I1Wxp)
            # cv.imshow("Z1Wxp", Z1Wxp)
            I1Wxp = np.zeros_like(I0[level])
            Z1Wxp = np.zeros_like(Z0[level])
            R_I = np.zeros_like(I0[level])
            R_Z = np.zeros_like(Z0[level])
            WI = np.zeros_like(I0[level])
            WZ = np.zeros_like(Z0[level])

            I1Wxp[uv0[:, 1], uv0[:, 0]] = i1wxp.reshape((-1,)) * 255
            Z1Wxp[uv0[:, 1], uv0[:, 0]] = z1wxp.reshape((-1,))
            R_I[uv0[:, 1], uv0[:, 0]] = (np.abs(r_I.reshape((-1,)))) * 255
            R_Z[uv0[:, 1], uv0[:, 0]] = np.abs(r_Z.reshape((-1,)))
            WI[uv0[:, 1], uv0[:, 0]] = W_I.diagonal().reshape((-1,))
            WI = (255.0 * WI / WI.max()).astype(np.uint8)
            WZ[uv0[:, 1], uv0[:, 0]] = W_Z.diagonal().reshape((-1,))

            cv.imshow("r_I", R_I)
            cv.imshow("r_Z", R_Z)
            cv.imshow("WI", WI)
            cv.imshow("WZ", WZ)

            cv.waitKey(0)

            if i > 0 and chi2[i] / chi2[i - 1] > min_reduction:
                pose = sp.SE3.exp(dx) * pose
                print(f"Stop. Error Increased.")
                break

            JwI_ = JwI[mask_valid]
            JZJpJt_Jtz = JwZ[mask_valid] - computeJtz(pcl1t)

            Az = JZJpJt_Jtz.T @ W_Z @ JZJpJt_Jtz
            bz = JZJpJt_Jtz.T @ W_Z @ r_Z

            Ai = JwI_.T @ W_I @ JwI_
            bi = JwI_.T @ W_I @ r_I

            A = (w_I * Ai + w_Z * Az) / N
            b = (np.sqrt(w_I) * bi + np.sqrt(w_Z) * bz) / N

            dx = np.linalg.solve(A, b)
            pose = sp.SE3.exp(-dx) * pose

            print(f"dx={np.linalg.norm(dx):.6f}")

            print(
                f"t={np.linalg.norm(pose.log()[:2]):.3f}m r={np.linalg.norm(pose.log()[2:])*180.0/np.pi:.3f}°"
            )

            if np.linalg.norm(dx) < min_update:
                print(
                    f"Converged. Min Step Size reached: {np.linalg.norm(dx):.6f}/{min_update:.6f}"
                )
                break
