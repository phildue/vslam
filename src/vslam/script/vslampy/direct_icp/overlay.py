import numpy as np
import cv2 as cv


class Overlay:
    def __init__(self):
        self.rmse_t = np.inf
        self.rmse_r = np.inf

    def log(
        self,
        level: int,
        i,
        uv0: np.array,
        I0,
        Z0,
        i1wxp: np.array,
        z1wxp: np.array,
        r_I,
        r_Z,
        w_I,
        w_Z,
        chi2,
        dx,
        sigma_I,
        sigma_Z,
    ):
        pass


class OverlayShow(Overlay):
    def __init__(self, nFrames: int, wait_time=1):
        self.nFrames = nFrames
        self.f_no = 0
        self.wait_time = wait_time
        self.max_z = 5.0
        self.rmse_t = np.inf
        self.rmse_r = np.inf

    def create_overlay(self, uv0, I0, i1wxp, residuals, weights):
        Warped = np.zeros_like(I0)
        Residual = np.zeros_like(I0)
        Weights = np.zeros_like(I0)
        R_W = np.zeros_like(I0)
        Valid = np.zeros_like(I0, dtype=np.uint8)

        Warped[uv0[:, 1], uv0[:, 0]] = i1wxp.reshape((-1,)).astype(np.uint8)
        Residual[uv0[:, 1], uv0[:, 0]] = np.abs(residuals.reshape((-1,))).astype(
            np.uint8
        )
        Valid[uv0[:, 1], uv0[:, 0]] = 255

        Weights[uv0[:, 1], uv0[:, 0]] = weights.reshape((-1,))
        Weights = (255.0 * Weights / Weights.max()).astype(np.uint8)

        R_W[uv0[:, 1], uv0[:, 0]] = np.abs(
            residuals.reshape((-1,)) * weights.reshape((-1,))
        )
        R_W = (255 * R_W / R_W.max()).astype(np.uint8)
        imgs = [I0, Warped, Valid, Residual, Weights, R_W]
        stack = np.hstack(imgs)
        stack = cv.resize(stack, (int(640 * len(imgs) / 2), int(480 / 2)))

        return stack

    def create_overlay_depth(self, uv0, Z0, z1wxp, residuals, weights):
        Warped = np.zeros_like(Z0)
        Residual = np.zeros_like(Z0)
        Weights = np.zeros_like(Z0)
        R_W = np.zeros_like(Z0)
        Valid = np.zeros_like(Z0, dtype=np.uint8)

        Warped[uv0[:, 1], uv0[:, 0]] = z1wxp.reshape((-1,)) / self.max_z * 255
        Residual[uv0[:, 1], uv0[:, 0]] = (
            255 / self.max_z * np.abs(residuals.reshape((-1,)))
        )
        Valid[uv0[:, 1], uv0[:, 0]] = 255

        Weights[uv0[:, 1], uv0[:, 0]] = weights.reshape((-1,))
        Weights = 255.0 * Weights / Weights.max()

        R_W[uv0[:, 1], uv0[:, 0]] = np.abs(
            residuals.reshape((-1,)) * weights.reshape((-1,))
        )
        R_W = 255 * R_W / R_W.max()
        Z0 = Z0 / self.max_z * 255
        imgs = [Z0, Warped, Valid, Residual, Weights, R_W]
        imgs = [img.astype(np.uint8) for img in imgs]
        stack = np.hstack(imgs)
        stack = cv.resize(stack, (int(640 * len(imgs) / 2), int(480 / 2)))

        return stack

    def log(
        self,
        level: int,
        i,
        uv0: np.array,
        I0,
        Z0,
        i1wxp: np.array,
        z1wxp: np.array,
        r_I,
        r_Z,
        w_I,
        w_Z,
        chi2,
        dx,
        sigma_I,
        sigma_Z,
    ):
        overlay_I = self.create_overlay(uv0, I0[level], i1wxp, r_I, w_I)
        overlay_Z = self.create_overlay_depth(uv0, Z0[level], z1wxp, r_Z, w_Z)
        info = np.zeros((80, overlay_I.shape[1]), dtype=np.uint8)
        font = cv.FONT_HERSHEY_TRIPLEX
        color = (255, 255, 255)
        text = f"#:{self.f_no}/{self.nFrames} l={level} i={i} chi2={chi2[i]:.6f} |dx|={np.linalg.norm(dx):0.5f}"
        text2 = f"rmse_t = {self.rmse_t:.3f} m rmse_r = {self.rmse_r:.3f} deg"
        info = cv.putText(info, text, (15, 15), font, 0.5, color, 1)
        info = cv.putText(info, text2, (15, 40), font, 0.5, color, 1)
        info = cv.putText(info, "Mask", (int(640 * 2 / 2), 15), font, 0.5, color, 1)
        info = cv.putText(info, "Residual", (int(640 * 3 / 2), 15), font, 0.5, color, 1)
        info = cv.putText(
            info,
            f"|r_Z| = {np.linalg.norm(r_I):.2f}",
            (int(640 * 3 / 2), 40),
            font,
            0.5,
            color,
            1,
        )
        info = cv.putText(
            info,
            f"|r_Z| = {np.linalg.norm(r_Z):.2f}",
            (int(640 * 3 / 2), 65),
            font,
            0.5,
            color,
            1,
        )
        info = cv.putText(info, f"Weights", (int(640 * 4 / 2), 15), font, 0.5, color, 1)
        info = cv.putText(
            info, f"s_I = {sigma_I:.3f}", (int(640 * 4 / 2), 40), font, 0.5, color, 1
        )
        info = cv.putText(
            info, f"s_Z = {sigma_Z:.3f}", (int(640 * 4 / 2), 65), font, 0.5, color, 1
        )
        info = cv.putText(
            info,
            f"Weighted Residual Normalized.",
            (int(640 * 5 / 2), 15),
            font,
            0.5,
            color,
            1,
        )
        info = cv.putText(
            info,
            f"rw_max_I = {(w_I * r_I).max():.2f}",
            (int(640 * 5 / 2), 40),
            font,
            0.5,
            color,
            1,
        )
        info = cv.putText(
            info,
            f"rw_max_Z = {(w_Z * r_Z).max():.2f}",
            (int(640 * 5 / 2), 65),
            font,
            0.5,
            color,
            1,
        )
        cv.imshow("DirectIcp", np.vstack([overlay_I, info, overlay_Z]))
        cv.waitKey(self.wait_time)
