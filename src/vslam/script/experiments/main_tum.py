import cv2 as cv
import numpy as np
from typing import List, Tuple
from sophus.sophuspy import SE3
import logging
import logging.config
from vslampy.dataset.tum import TumRgbd
from vslampy.direct_icp.direct_icp import DirectIcp, Camera
from vslampy.direct_icp.overlay import OverlayShow, Overlay
from vslampy.direct_icp.weights import (
    TDistributionWeights,
    TDistributionMultivariateWeights,
    LinearCombination,
)
from vslampy.utils.utils import load_frame, write_result_file, Timer
import wandb
import os
import argparse

if __name__ == "__main__":
    f_start = 0
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
        default="rgbd_dataset_freiburg2_desk",
    )
    args = parser.parse_args()

    np.set_printoptions(precision=4)
    logging.config.dictConfig(
        {
            "version": 1,
            "root": {"level": "INFO"},
            "loggers": {
                "DirectIcp": {"level": "WARNING"},
                "WeightEstimation": {"level": "WARNING"},
            },
        }
    )

    params = {
        "nLevels": 4,
        "weight_prior": 0.0,
        "min_gradient_intensity": 10 * 8,  #
        "min_gradient_depth": np.inf,
        "max_gradient_depth": np.inf,
        "max_z": 5.0,
        "max_z_diff": 0.2,
        "max_iterations": 100,
        "min_parameter_update": 1e-4,
    }
    sequence = TumRgbd(args.sequence_id)

    if upload:
        os.environ["WANDB_BASE_URL"] = "http://localhost:8080"
        os.environ["WANDB_API_KEY"] = "local-837a2a9d75b14cf1ae7886da28a78394a9a7b053"
        wandb.init(
            project="vslam",
            entity="phild",
            config=params,
        )
        wandb.run.name = f"{args.sequence_id}.{args.experiment_name}"

    timestamps, files_I, files_Z = sequence.image_depth_filepaths()
    f_end = min([n_frames, len(timestamps)])
    image_log = OverlayShow(f_end, wait_time) if wait_time >= 0 else Overlay()
    t_multi = TDistributionMultivariateWeights(5.0, np.identity(2))
    t_combi = LinearCombination(
        TDistributionWeights(5, 1),
        TDistributionWeights(5, 1),
    )

    direct_icp = DirectIcp(
        cam=Camera(fx=525.0, fy=525.0, cx=319.5, cy=239.5, h=480, w=640),
        weight_function=t_combi,
        image_log=image_log,
        **params,
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
        Timer.tick("compute_egomotion")
        motion = direct_icp.compute_egomotion(t1, I1, Z1, motion)
        Timer.tock()
        pose = motion * pose
        trajectory[timestamps[f_no]] = pose.inverse().matrix()
        f_no0 = f_no
        t0 = t1
        I0 = I1
        Z0 = Z1

        if f_no > 0 and f_no % 100 == 0:
            image_log.rmse_t, image_log.rmse_r = sequence.evaluate_rpe(
                trajectory, output_dir="./", upload=upload
            )
            write_result_file(trajectory, f"{sequence._sequence_id}-algo.txt")

    sequence.evaluate_rpe(trajectory, output_dir="./", upload=upload)

    if upload:
        wandb.finish()
