import logging
import numpy as np
from vslampy.utils.utils import statsstr


class LinearCombination:
    def __init__(
        self, weight_function_intensity, weight_function_depth, weight_intensity=0.5
    ):
        self.weight_intensity = weight_intensity
        self.weight_depth = 1.0 - weight_intensity
        self.weight_function_intensity = weight_function_intensity
        self.weight_function_depth = weight_function_depth
        self.scale = np.identity(2)

    def compute_weight_matrices(self, r: np.array):
        scale_I, w_I = self.weight_function_intensity.fit(r[:, 0])
        scale_Z, w_Z = self.weight_function_depth.fit(r[:, 1])
        self.scale[0, 0] = scale_I
        self.scale[1, 1] = scale_Z
        # scale_I = 1 / r.shape[0]
        # scale_Z = 1 / r.shape[0]
        weights = np.zeros((r.shape[0], 2, 2))
        weights[:, 0, 0] = w_I * scale_I
        weights[:, 1, 1] = w_Z * scale_Z
        return weights


class TDistributionWeights:
    def __init__(self, dof=5.0, sigma=1.0):
        self.dof = dof
        self.sigma = sigma
        self.log = logging.getLogger("WeightEstimation")

    def compute_weights(self, r: np.array) -> np.array:
        w = (self.dof + 1.0) / (self.dof + r * r * self.sigma)
        return w

    def fit(self, r: np.array, precision=1e-3, max_iterations=50) -> np.array:
        step_size = np.inf
        for iter in range(max_iterations):
            w = self.compute_weights(r)
            sigma_i = 1.0 / (float((w * r).T @ r) / r.shape[0])
            step_size = np.abs(self.sigma - sigma_i)
            self.sigma = sigma_i
            self.log.debug(
                f"\titer = {iter}, sigma = {self.sigma:4f}, step_size = {step_size:4f} \n\tW={statsstr(w)})"
            )
            if step_size < precision:
                break

        self.log.info(
            f"\tEM: {iter}, precision: {step_size:.4f}, scale: {self.sigma:.4f}, \nW={statsstr(w)}"
        )
        return self.sigma, w


class TDistributionMultivariateWeights:
    def __init__(self, dof=5.0, scale=None, dim=2):
        if scale is None:
            scale = np.identity(2)
        self.dof = dof
        self.scale = scale
        self.dim = dim
        self.log = logging.getLogger("WeightEstimation")

    def compute_weight_matrices(self, r: np.array, w: np.array):
        scale, w = self.fit(r, w)
        weights = w.reshape(-1, 1, 1) * scale
        return weights

    def compute_weights(self, r: np.array) -> np.array:
        self._check_shape(r)
        rT = np.transpose(r, (0, 2, 1))
        return (self.dof + self.dim) / (self.dof + rT @ self.scale @ r)

    def fit(
        self, r: np.array, w: np.array, precision=1e-3, max_iterations=50
    ) -> np.array:
        self._check_shape(r)
        step_size = np.inf
        r = r[:, :, np.newaxis]
        rT = np.transpose(r, (0, 2, 1))
        for iter in range(max_iterations):
            scale_i = np.sum(w.reshape((-1, 1, 1)) * r @ rT, axis=0) / (r.shape[0])
            scale_i = np.linalg.inv(scale_i)
            step_size = np.linalg.norm(self.scale - scale_i)
            self.scale = scale_i
            w = self.compute_weights(r)
            self.log.debug(
                f"\titer = {iter}, sigma = {self.scale}, step_size = {step_size:4f} \n\tW={statsstr(w)})"
            )
            if step_size < precision:
                break

        self.log.info(
            f"\tEM: {iter}, precision: {step_size:.4f}, scale: {self.scale}, \nW={statsstr(w)}"
        )
        return self.scale, w

    def _check_shape(self, r: np.array):
        if r.shape[1] != self.dim:
            raise ValueError(
                f"Residual has wrong dimension of {r.shape[1]} should be {self.dim}"
            )
