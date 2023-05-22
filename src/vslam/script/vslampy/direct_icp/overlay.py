import numpy as np
import cv2 as cv
import logging
from vslampy.utils.utils import statsstr
from vslampy.direct_icp.weights import TDistributionMultivariateWeights
import matplotlib.pyplot as plt


class Log:
    def __init__(self):
        self.rmse_t = np.inf
        self.rmse_r = np.inf

    def log_level(self, level):
        pass

    def log_iteration(
        self,
        level: int,
        i,
        uv0: np.array,
        IZ0,
        iz1wxp,
        r,
        scale,
        chi2,
        dx,
    ):
        pass

    def log_converged(self, reason, motion):
        pass


class LogShow(Log):
    def __init__(
        self,
        nFrames: int,
        wait_time,
        weight_function,
    ):
        self.nFrames = nFrames
        self.f_no = 0
        self.wait_time = wait_time
        self.max_z = 1
        self.rmse_t = np.inf
        self.rmse_r = np.inf
        self.weight_function = weight_function
        self.level = np.inf
        self.iteration = 0
        self.text_log = logging.getLogger("DirectIcp")

    def log_level(self, level):
        self.text_log.info(f"_________Level={level}___________")

        self.level = level
        self.iteration = 0

        self.plot = Plot(500)

    def log_iteration(
        self,
        i,
        uv0: np.array,
        IZ0,
        iz1wxp,
        r,
        weights,
        error,
        dx,
    ):
        self.iteration = i
        level = self.level
        w_I = weights[:, 0, 0] / self.weight_function.scale[0, 0]
        w_Z = weights[:, 1, 1] / self.weight_function.scale[1, 1]

        r_I = r[:, 0]
        r_Z = r[:, 1]
        I0, Z0 = IZ0
        i1wxp, z1wxp = iz1wxp

        if self.wait_time >= 0:
            self.plot.scale[i] = self.weight_function.scale
            self.plot.error[i] = error[i]
            self.plot.step_size[i] = np.linalg.norm(dx)
            self.plot.iteration = i

            overlay_Z = self.create_overlay_depth(uv0, Z0[level], z1wxp, r_Z, w_Z)
            overlay_I = self.create_overlay(uv0, I0[level], i1wxp, r_I * 255, w_I)
            info = self.create_info_bar(overlay_I.shape[1], i, r, weights, error, dx)

            cv.imshow("DirectIcp", np.vstack([overlay_I, info, overlay_Z]))
            cv.waitKey(self.wait_time)

    def log_converged(self, reason, motion):
        self.text_log.info(
            f"Aligned after {self.iteration} iterations because [{reason}]\nt={motion.log()[:3]}m r={np.linalg.norm(motion.log()[3:])*180.0/np.pi:.3f}°"
        )
        # self.plot.plot()

        pass

    def create_overlay(self, uv0, I0, i1wxp, residuals, weights):
        h, w = I0.shape[:2]
        Warped = np.zeros((h, w))
        Residual = np.zeros((h, w))
        Weights = np.zeros((h, w))
        R_W = np.zeros((h, w))
        Valid = np.zeros((h, w))

        Warped[uv0[:, 1], uv0[:, 0]] = i1wxp.reshape((-1,))
        Residual[uv0[:, 1], uv0[:, 0]] = np.abs(residuals.reshape((-1,)))
        Residual = 255.0 * Residual / Residual.max()
        Valid[uv0[:, 1], uv0[:, 0]] = 255

        Weights[uv0[:, 1], uv0[:, 0]] = weights.reshape((-1,))
        Weights = 255.0 * Weights / Weights.max()

        R_W[uv0[:, 1], uv0[:, 0]] = np.abs(
            residuals.reshape((-1,)) * weights.reshape((-1,))
        )
        R_W = 255.0 * R_W / R_W.max()
        imgs = [I0, Warped, Valid, Residual, Weights, R_W]
        imgs = [img.astype(np.uint8) for img in imgs]
        stack = np.hstack(imgs)
        stack = cv.resize(
            stack.astype(np.uint8), (int(640 * len(imgs) / 2), int(480 / 2))
        )

        return stack

    def create_overlay_depth(self, uv0, Z0, z1wxp, residuals, weights):
        h, w = Z0.shape[:2]
        Warped = np.zeros((h, w))
        Residual = np.zeros((h, w))
        Weights = np.zeros((h, w))
        R_W = np.zeros((h, w))
        Valid = np.zeros((h, w))
        self.max_z = Z0.max()

        Warped[uv0[:, 1], uv0[:, 0]] = z1wxp.reshape((-1,)) / self.max_z * 255
        Residual[uv0[:, 1], uv0[:, 0]] = np.abs(residuals.reshape((-1,)))
        Residual = 255.0 * Residual / Residual.max()

        Valid[uv0[:, 1], uv0[:, 0]] = 255

        Weights[uv0[:, 1], uv0[:, 0]] = weights.reshape((-1,))
        Weights = 255.0 * Weights / Weights.max()

        R_W[uv0[:, 1], uv0[:, 0]] = np.abs(
            residuals.reshape((-1,)) * weights.reshape((-1,))
        )
        R_W = 255.0 * R_W / R_W.max()
        Z0 = Z0 / self.max_z * 255.0
        imgs = [Z0, Warped, Valid, Residual, Weights, R_W]
        imgs = [img.astype(np.uint8) for img in imgs]
        stack = np.hstack(imgs)
        stack = cv.resize(stack, (int(640 * len(imgs) / 2), int(480 / 2)))

        return stack

    def create_info_bar(self, width, i, r, weights, chi2, dx):
        level = self.level
        w_I = weights[:, 0, 0] / self.weight_function.scale[0, 0]
        w_Z = weights[:, 1, 1] / self.weight_function.scale[1, 1]

        r_I = r[:, 0]
        r_Z = r[:, 1]

        info = np.zeros((80, width), dtype=np.uint8)
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
            f"|r_I| = {np.linalg.norm(r_I):.2f}",
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
            info,
            f"s_I = {self.weight_function.scale[0,0]:.3f}, {self.weight_function.scale[0,1]:.3f}",
            (int(640 * 4 / 2), 40),
            font,
            0.5,
            color,
            1,
        )
        info = cv.putText(
            info,
            f"s_Z = {self.weight_function.scale[1,0]:.3f}, {self.weight_function.scale[1,1]:.3f}",
            (int(640 * 4 / 2), 65),
            font,
            0.5,
            color,
            1,
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
        """
        
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
        """
        info = cv.putText(
            info,
            f"|wr_I| = {np.linalg.norm(r_I*w_I):.2f} mean(wr_I) = {np.sum(r_I*w_I):.2f}",
            (int(640 * 5 / 2), 40),
            font,
            0.5,
            color,
            1,
        )
        info = cv.putText(
            info,
            f"|wr_Z| = {np.linalg.norm(r_Z*w_Z):.2f} mean(wr_Z) = {np.sum(r_Z*w_Z):.2f}",
            (int(640 * 5 / 2), 65),
            font,
            0.5,
            color,
            1,
        )
        return info


class Plot:
    def __init__(self, n_iterations=100):
        self.scale = np.zeros((n_iterations, 2, 2))
        self.error = np.zeros((n_iterations, 1))
        self.step_size = np.zeros((n_iterations, 1))
        self.iteration = 0

    def plot(self):
        plt.figure(num=1, figsize=(8, 6))
        plt.subplot(1, 3, 1)
        plt.title("$error^2$")
        plt.xlabel("Iteration")
        plt.ylabel("$error^2$")
        plt.plot(self.error[: self.iteration])

        plt.subplot(1, 3, 2)
        plt.title("Scale")
        plt.xlabel("Iteration")
        plt.ylabel("$\\Sigma$")
        plt.plot(self.scale[: self.iteration, 0, 0])
        plt.plot(self.scale[: self.iteration, 1, 1])
        plt.plot(self.scale[: self.iteration, 0, 1])
        plt.legend(["$\\sigma_I^2$", "$\\sigma_Z^2$", "$\\sigma_IZ^2$"])

        plt.subplot(1, 3, 3)
        plt.title("Step Size")
        plt.xlabel("Iteration")
        plt.ylabel("$|dx|$")
        plt.plot(self.step_size[: self.iteration])
        plt.tight_layout()
        plt.show(block=False)
