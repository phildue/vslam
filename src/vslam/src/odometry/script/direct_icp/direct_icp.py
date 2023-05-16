import cv2 as cv
import numpy as np
from sophus.sophuspy import SE3
import logging

from overlay import OverlayNull
from weights import TDistributionWeights
from utils import statsstr
from camera import Camera


class DirectIcp:
    def __init__(
        self,
        cam: Camera,
        nLevels: int,
        weight_intensity=1.0,
        weight_prior=0.0,
        min_gradient_intensity=10 * 8,
        min_gradient_depth=np.inf,
        max_gradient_depth=0.5,
        max_z=5.0,
        max_z_diff=0.2,
        max_iterations=100,
        min_parameter_update=1e-4,
        max_delta_chi2=1.1,
        weight_function=TDistributionWeights(5, 1),
        image_log=OverlayNull(),
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
        self.max_dZ = max_gradient_depth

        self.max_z = max_z
        self.max_z_diff = max_z_diff
        self.max_iterations = max_iterations
        self.min_parameter_update = min_parameter_update
        self.max_delta_chi2 = max_delta_chi2
        self.weight_function = weight_function
        self.image_log = image_log
        self.border_dist = 0.01
        self.log = logging.getLogger("DirectIcp")

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
        sigma_I = 1.0
        sigma_Z = 1.0
        for l_ in range(self.nLevels):
            level = self.nLevels - (l_ + 1)
            self.log.info(f"_________Level={level}___________")

            dI0 = self.compute_jacobian_image(self.I0[level])
            dZ0 = self.compute_jacobian_depth(self.Z0[level])

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
            reason = "Max iterations exceeded"
            for i in range(self.max_iterations):
                pcl0t = motion * pcl0
                uv0t = self.cam[level].project(pcl0t)
                mask_valid = self.select_visible(pcl0t[:, 2], uv0t, self.cam[level])
                uv0t = uv0t[mask_valid]
                if uv0t.shape[0] < 6:
                    reason = "Not enough constraints"
                    motion = SE3()
                    l_ = 0
                    break

                i1wxp, z1wxp, mask_occluded = self.interpolate(
                    I1[level],
                    Z1[level],
                    uv0t,
                    pcl0t[:, 2].reshape((-1, 1))[mask_valid],
                )
                norm_I = i1wxp.shape[0] * 255
                norm_Z = z1wxp.shape[0]

                i0x = self.I0[level].reshape((-1,))[mask_selected][mask_valid][
                    mask_occluded
                ]

                pcl1t = motion.inverse() * (
                    self.cam[level].reconstruct(uv0t[mask_occluded], z1wxp)
                )
                z0x = self.Z0[level].reshape((-1,))[mask_selected][mask_valid][
                    mask_occluded
                ]
                r_I = i1wxp - i0x
                r_Z = pcl1t[:, 2] - z0x

                sigma_I, w_I = self.weight_function.fit(r_I, sigma_I)
                sigma_Z, w_Z = self.weight_function.fit(r_Z, sigma_Z)

                # s_Z, W_Z = compute_scale(r_Z)
                chi2[i] = (
                    self.weight_intensity * ((r_I.T * w_I) @ r_I) / norm_I
                    + self.weight_depth * ((r_Z.T * w_Z) @ r_Z) / norm_Z
                )

                self.log_errors(chi2, i, r_I, r_Z)

                if i > 0 and chi2[i] / chi2[i - 1] > self.max_delta_chi2:
                    reason = f"Error Increased. dchi2={(chi2[i]/chi2[i-1])*100:0.2f} %, dchi2={chi2[i]-chi2[i-1]:0.6f}"
                    motion = SE3.exp(dx) * motion
                    break

                uv0 = (
                    self.cam[level]
                    .image_coordinates()[mask_selected][mask_valid][mask_occluded]
                    .astype(int)
                )
                self.image_log.log(
                    level,
                    i,
                    uv0,
                    self.I0,
                    self.Z0,
                    i1wxp,
                    z1wxp,
                    r_I,
                    r_Z,
                    w_I,
                    w_Z,
                    chi2,
                    dx,
                    sigma_I,
                    sigma_Z,
                )

                dx = self.compute_pose_update(
                    JIJw[mask_valid][mask_occluded],
                    r_I,
                    w_I,
                    JZJw[mask_valid][mask_occluded]
                    - self.compute_jacobian_se3_z(pcl1t),
                    r_Z,
                    w_Z,
                    prior,
                    motion,
                )

                motion = SE3.exp(-dx) * motion

                if np.linalg.norm(dx) < self.min_parameter_update:
                    reason = f"Min Step Size reached: {np.linalg.norm(dx):.6f}/{self.min_parameter_update:.6f}"
                    break
            self.log.info(
                f"Aligned after {i} iterations because [{reason}]\nt={motion.log()[:3]}m r={np.linalg.norm(motion.log()[3:])*180.0/np.pi:.3f}°"
            )
        self.I0 = I1
        self.Z0 = Z1
        self.t0 = t1
        return motion

    def compute_pyramid(self, I, Z):
        I = [I]
        Z = [Z]
        for l in range(1, self.nLevels):
            I += [cv.pyrDown(I[l - 1])]
            Z += [
                cv.resize(
                    Z[l - 1], (0, 0), fx=0.5, fy=0.5, interpolation=cv.INTER_NEAREST
                )
            ]
        return I, Z

    def compute_jacobian_image(self, I):
        dIdx = cv.Sobel(I, cv.CV_64F, dx=1, dy=0)
        dIdy = cv.Sobel(I, cv.CV_64F, dx=0, dy=1)

        return np.reshape(np.dstack([dIdx, dIdy]), (-1, 2))

    def compute_jacobian_depth(self, Z):
        dZ = np.gradient(Z)

        return np.reshape(np.dstack([dZ[1], dZ[0]]), (-1, 2))

    def select_constraints(self, Z, dI, dZ):
        return (
            (np.isfinite(Z[:, 0]))
            & (Z[:, 0] > 0)
            & (Z[:, 0] < self.max_z)
            & (np.abs(dZ[:, 0]) < self.max_dZ)
            & (np.abs(dZ[:, 1]) < self.max_dZ)
            & (
                (np.abs(dI[:, 0]) > self.min_dI)
                | (np.abs(dI[:, 1]) > self.min_dI)
                | (np.abs(dZ[:, 0]) > self.min_dZ)
                | (np.abs(dZ[:, 1]) > self.min_dZ)
            )
        )

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

    def select_visible(self, z: np.array, uv: np.array, cam: Camera) -> np.array:
        border = max((1, int(self.border_dist * cam.w)))
        return (
            (z > 0)
            & (cam.w - border > uv[:, 0])
            & (uv[:, 0] > border)
            & (cam.h - border > uv[:, 1])
            & (uv[:, 1] > border)
        )

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

    def log_errors(self, chi2, i, r_I, r_Z):
        if i > 0:
            self.log.debug(
                f"r_I: {statsstr(r_I)}\n"
                f"r_Z: {statsstr(r_Z)}\n"
                f"chi2={chi2[i]:0.6f}, dchi2={(chi2[i]/chi2[i-1])*100:0.2f} %, dchi2={chi2[i]-chi2[i-1]:0.6f}"
            )
        else:
            self.log.debug(
                f"r_I: {statsstr(r_I)}\n"
                f"r_Z: {statsstr(r_Z)}\n"
                f"chi2={chi2[i]:0.6f}"
            )

    def compute_pose_update(self, JIJw, r_I, w_I, JZJw_Jtz, r_Z, w_Z, prior, motion):
        norm_I = r_I.shape[0] * 255
        norm_Z = r_Z.shape[0]
        JZJw_Jtzw = self.weight_depth * w_Z.reshape((-1, 1)) / norm_Z * JZJw_Jtz
        Az = JZJw_Jtzw.T @ JZJw_Jtz
        bz = JZJw_Jtzw.T @ r_Z

        JIJww = self.weight_intensity * w_I.reshape((-1, 1)) / norm_I * JIJw
        Ai = JIJww.T @ JIJw
        bi = JIJww.T @ r_I

        Ap = self.weight_prior * np.identity(6)
        bp = self.weight_prior * (motion * prior.inverse()).log()

        return np.linalg.solve(Ai + Az, bi + bz + bp)
