import cv2 as cv
import numpy as np
from typing import List, Tuple
from sophus.sophuspy import SE3
import logging
import logging.config
from vslampy.evaluation.tum import TumRgbd
from vslampy.direct_icp.direct_icp import DirectIcp, Camera
from vslampy.direct_icp.overlay import LogShow, Log
from vslampy.direct_icp.weights import (
    TDistributionWeights,
    TDistributionMultivariateWeights,
    LinearCombination,
)
from vslampy.utils.utils import (
    load_frame,
    write_result_file,
    Timer,
    statsstr,
    interpolate_pose_between,
)
import wandb
import os
import argparse
import matplotlib.pyplot as plt


def create_intensity_depth_overlay(I, Z):
    Z = 255.0 * Z / Z.max()
    I = cv.cvtColor(I.astype(np.uint8), cv.COLOR_GRAY2BGR)
    Z = cv.cvtColor(Z.astype(np.uint8), cv.COLOR_GRAY2BGR)
    Z = cv.applyColorMap(Z, cv.COLORMAP_JET)
    return (0.7 * I + 0.3 * Z).astype(np.uint8)


if __name__ == "__main__":
    f_start = 240
    n_frames = np.inf

    wait_time = 1
    upload = False
    parser = argparse.ArgumentParser(
        description="""
    Run evaluation of algorithm"""
    )
    parser.add_argument(
        "--experiment_name", help="Name for the experiment", default="test"
    )
    parser.add_argument(
        "--sequence_id",
        help="Id of the sequence to run on)",
        default="rgbd_dataset_freiburg1_360",
    )
    args = parser.parse_args()

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

    sequence = TumRgbd(args.sequence_id)

    timestamps_Z, files_Z, timestamps_I, files_I = sequence.image_depth_filepaths()
    timestamps = timestamps_I
    f_end = min([f_start + n_frames, len(timestamps)])

    cam = Camera(fx=525.0, fy=525.0, cx=319.5, cy=239.5, h=480, w=640)

    trajectory = {}
    trajectory_gt = dict(
        (t, SE3(p).inverse()) for t, p in sequence.gt_trajectory().items()
    )

    pose = SE3()
    f_no0 = f_start
    t0 = timestamps[f_start]
    I0, Z0 = load_frame(files_I[f_start], files_Z[f_start])
    speed = np.zeros((6,))
    dt_IZ = np.zeros((f_end - f_start,))
    for f_no in range(f_start, f_end):
        t1 = timestamps[f_no]
        dt = t1 - t0
        I1, Z1 = load_frame(files_I[f_no], files_Z[f_no])
        dt_IZ[f_no - f_start] = timestamps_I[f_no] - timestamps_Z[f_no]

        motion = interpolate_pose_between(
            trajectory_gt, timestamps_Z[f_no], timestamps_I[f_no]
        )
        pclt = motion * cam.reconstruct(
            cam.image_coordinates(),
            Z1.reshape((-1,)),
        )
        uv = cam.project(pclt)
        mask_visible = cam.select_visible(uv, pclt[:, 2])
        uv = uv[mask_visible].astype(int)
        pcl1t = pclt[mask_visible]
        Z1_comp = np.zeros_like(Z1)
        Z1_comp[uv[:, 1], uv[:, 0]] = pcl1t[:, 2]

        print(f"dt_IZ: {timestamps_I[f_no] - timestamps_Z[f_no]:.3f}")
        o = np.hstack(
            [
                create_intensity_depth_overlay(I1, Z1),
                create_intensity_depth_overlay(I1, Z1_comp),
            ]
        )
        cv.imshow("Frame", o)
        cv.waitKey(0 if np.abs(dt_IZ[f_no - f_start]) > 0.01 else 1)

        speed = motion.log() / dt if dt > 0 else np.zeros((6,))
        pose = motion * pose
        f_no0 = f_no
        t0 = t1
        I0 = I1
        Z0 = Z1

    print(f"dt_IZ: {statsstr(dt_IZ)}")
    plt.figure()
    plt.plot(dt_IZ)
    plt.show()
