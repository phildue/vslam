import cv2 as cv
import numpy as np
from typing import List, Tuple
from sophus.sophuspy import SE3
import os
from vslampy.dataset import TumRgbd
from direct_icp import DirectIcp, Camera
from overlay import OverlayShow, Overlay
from weights import TDistributionWeights
from utils import load_frame, write_result_file
import logging
import logging.config


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
    timestamps, files_I, files_Z = sequence.image_depth_filepaths()
    f_end = min([n_frames, len(timestamps)])
    image_log = OverlayShow(f_end, wait_time) if wait_time > 0 else Overlay()
    direct_icp = DirectIcp(
        Camera(fx=525.0, fy=525.0, cx=319.5, cy=239.5, h=480, w=640),
        nLevels=4,
        weight_intensity=0.7,
        weight_prior=0.0,
        min_gradient_intensity=5 * 8,  #
        min_gradient_depth=np.inf,
        max_gradient_depth=np.inf,
        max_z=5.0,
        max_z_diff=0.2,
        max_iterations=100,
        min_parameter_update=1e-4,
        max_delta_chi2=1.1,
        weight_function=TDistributionWeights(5.0, 1),
        image_log=image_log,
    )

    trajectory = {}
    trajectory_gt = dict(
        (t, SE3(p).inverse()) for t, p in sequence.gt_trajectory().items()
    )

    pose = SE3()
    f_no0 = f_start
    t0 = timestamps[f_start]
    motion = SE3()
    for f_no in range(f_start, f_end):
        t1 = timestamps[f_no]
        I1, Z1 = load_frame(files_I[f_no], files_Z[f_no])
        logging.info(
            f"_________Aligning: {f_no0} -> {f_no} / {f_end}, {t0}->{t1}, dt={t1-t0:.3f}___________"
        )
        image_log.f_no = f_no
        motion = direct_icp.compute_egomotion(t1, I1, Z1, motion)

        pose = motion * pose
        trajectory[timestamps[f_no]] = pose.inverse().matrix()
        f_no0 = f_no
        t0 = t1
        I0 = I1
        Z0 = Z1

        if f_no > 0 and f_no % 100 == 0:
            image_log.rmse_t, image_log.rmse_r = sequence.evaluate_rpe(
                trajectory, output_dir="./", upload=False
            )

    sequence.evaluate_rpe(trajectory, output_dir="./", upload=False)
    write_result_file(trajectory, f"{sequence._sequence_id}-algo.txt")
