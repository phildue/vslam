import logging
import numpy as np
from utils import statsstr


class TDistributionWeights:
    def __init__(self, dof=5.0, sigma=1.0):
        self.dof = dof
        self.sigma = sigma
        self.log = logging.getLogger("WeightEstimation")

    def compute_weights(self, r: np.array) -> np.array:
        w = (self.dof + 1.0) / (self.dof + (r / self.sigma) ** 2)
        return w

    def fit(self, r: np.array, sigma0, precision=1e-3, max_iterations=50) -> np.array:
        self.sigma = sigma0
        step_size = np.inf
        for iter in range(max_iterations):
            w = self.compute_weights(r)
            sigma_i = np.sqrt(float((w * r).T @ r) / r.shape[0])
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
    def __init__(self, dof=5.0, sigma=1.0, dim=2):
        self.dof = dof
        self.sigma = sigma
        self.dim = dim
        self.log = logging.getLogger("WeightEstimation")

    def compute_weights(self, r: np.array) -> np.array:
        self._check_shape(r)
        return (self.dof + self.dim) / (
            self.dof + np.sum(np.dot(r, self.sigma) * r, axis=1)
        )

    def fit(
        self, r: np.array, sigma0: np.array, precision=1e-3, max_iterations=50
    ) -> np.array:
        self._check_shape(r)
        self.sigma = sigma0
        step_size = np.inf
        for iter in range(max_iterations):
            w = self.compute_weights(r)
            sigma_i = (
                np.sum((w * r[:, :, np.newaxis]) @ r[:, np.newaxis, :], axis=0)
                / r.shape[0]
            )

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

    def _check_shape(self, r: np.array):
        if r.shape[1] != self.dim:
            raise ValueError(
                f"Residual has wrong dimension of {r.shape[1]} should be {self.dim}"
            )
