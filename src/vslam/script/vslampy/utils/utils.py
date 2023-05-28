import os
from typing import Tuple, List
import numpy as np
import cv2 as cv
from sophus.sophuspy import SE3


def statsstr(x) -> str:
    return f"{np.linalg.norm(x):.4f}, {x.min():.4f} < {x.mean():.4f} +- {x.std():.4f} < {x.max():.4f} n={x.shape[0]}, d={x.shape}"


def ensure_dims(x: np.array, shape: np.shape):
    if x.shape != shape:
        raise ValueError(f"Invalid shape: {x.shape} should be {shape}")


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







def create_intensity_depth_overlay(I, Z):
    Z = 255.0 * Z / Z.max()
    I = cv.cvtColor(I.astype(np.uint8), cv.COLOR_GRAY2BGR)
    Z = cv.cvtColor(Z.astype(np.uint8), cv.COLOR_GRAY2BGR)
    Z = cv.applyColorMap(Z, cv.COLORMAP_JET)
    return (0.7 * I + 0.3 * Z).astype(np.uint8)


import time


class Timer:
    stack = []
    timers = {}

    def tick(name: str):
        Timer.timers[name] = time.perf_counter()
        Timer.stack.append(name)

    def tock(name: str = "", verbose=True):
        if not name:
            name = Timer.stack.pop()
        dt = time.perf_counter() - Timer.timers[name]
        Timer.timers.pop(name)
        if verbose:
            print(f"[{name}] ran for [{dt}:.4f]s")
        return dt
