import cv2 as cv
import numpy as np
from typing import List, Tuple
from sophus.sophuspy import SE3
import os
from vslampy.dataset import TumRgbd
from direct_icp import DirectIcp, TDistributionWeights, Camera, ImageLog
import logging
import logging.config
import yaml


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


if __name__ == "__main__":
    f_start = 0
    n_frames = np.inf

    wait_time = 1
    np.set_printoptions(precision=4)
    logging.config.dictConfig(
        {
            "version": 1,
            "root": {"level": "INFO"},
            "loggers": {
                "DirectIcp": {"level": "INFO"},
                "WeightEstimation": {"level": "WARNING"},
            },
        }
    )
    sequence = TumRgbd("rgbd_dataset_freiburg2_desk")
    direct_icp = DirectIcp(
        Camera(fx=525.0, fy=525.0, cx=319.5, cy=239.5, h=480, w=640),
        nLevels=4,
        weight_intensity=1.0,
        weight_prior=0.0,
        min_gradient_intensity=5 * 8,
        min_gradient_depth=np.inf,
        max_z=5.0,
        max_z_diff=0.2,
        max_iterations=100,
        min_parameter_update=1e-4,
        max_delta_chi2=1.1,
        weight_function=TDistributionWeights(5.0, 1),
        image_log=ImageLog(n_frames, wait_time),
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
        logging.info(
            f"_________Aligning: {f_no0} -> {f_no} / {f_end}, {t0}->{t1}, dt={t1-t0:.3f}___________"
        )
        motion = direct_icp.compute_egomotion(t1, I1, Z1, motion)

        pose = motion * pose
        trajectory[timestamps[f_no]] = pose.inverse().matrix()
        f_no0 = f_no
        t0 = t1
        I0 = I1
        Z0 = Z1

        if f_no > 0 and f_no % 100 == 0:
            sequence.evaluate_rpe(trajectory, output_dir="./", upload=False)

    sequence.evaluate_rpe(trajectory, output_dir="./", upload=False)
