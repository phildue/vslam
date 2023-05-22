import cv2 as cv
import numpy as np
from typing import List, Tuple
from sophus.sophuspy import SE3
import logging
import logging.config
from vslampy.dataset.tum import TumRgbd
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
    create_intensity_depth_overlay,
)
import wandb
import os
import argparse
import matplotlib.pyplot as plt


if __name__ == "__main__":
    f_start = 0
    n_frames = np.inf
    rate_eval = 25

    wait_time = 1
    upload = True
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
        default="rgbd_dataset_freiburg1_room",
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

    params = {
        "nLevels": 4,
        "weight_prior": 0.0,
        "min_gradient_intensity": 5,
        "min_gradient_depth": 0.05,
        "max_gradient_depth": 0.3,
        "max_z": 5.0,
        "max_iterations": 100,
        "min_parameter_update": 1e-4,
        "max_error_increase": 1.1,
        "weight_function": "Multivariate",
    }
    sequence = TumRgbd(args.sequence_id)

    os.environ["WANDB_BASE_URL"] = "http://localhost:8080"
    os.environ["WANDB_API_KEY"] = "local-837a2a9d75b14cf1ae7886da28a78394a9a7b053"
    wandb.init(
        project="vslam",
        entity="phild",
        config=params,
    )
    wandb.run.name = f"{args.sequence_id}.{args.experiment_name}"

    timestamps_Z, files_Z, timestamps_I, files_I = sequence.image_depth_filepaths()
    timestamps = timestamps_I
    f_end = min([f_start + n_frames, len(timestamps)])

    cam = Camera(fx=525.0, fy=525.0, cx=319.5, cy=239.5, h=480, w=640)
    weight_function = (
        LinearCombination(
            TDistributionWeights(5, 1),
            TDistributionWeights(5, 1),
        )
        if params["weight_function"] == "LinearCombination"
        else TDistributionMultivariateWeights(5.0, np.identity(2))
    )
    params.pop("weight_function")
    log = LogShow(f_end, wait_time, weight_function) if wait_time >= 0 else Log()

    direct_icp = DirectIcp(
        cam=cam,
        weight_function=weight_function,
        log=log,
        **params,
    )

    trajectory = {}
    pose = SE3()
    f_no0 = f_start
    t0 = timestamps[f_start]
    I0, Z0 = load_frame(files_I[f_start], files_Z[f_start])
    motion = SE3()
    speed = np.zeros((6,))
    for f_no in range(f_start, f_end):
        t1 = timestamps[f_no]
        dt = t1 - t0
        I1, Z1 = load_frame(files_I[f_no], files_Z[f_no])

        o = np.hstack(
            [
                create_intensity_depth_overlay(I0, Z0),
                create_intensity_depth_overlay(I1, Z1),
            ]
        )
        cv.imshow("Frame", o)
        cv.waitKey(1)

        logging.info(
            f"_________Aligning: {f_no0} -> {f_no} / {f_end}, {t0}->{t1}, dt={dt:.3f}___________"
        )
        log.f_no = f_no
        Timer.tick("compute_egomotion")
        motion = direct_icp.compute_egomotion(t1, I1, Z1, SE3.exp(speed * dt))
        Timer.tock()
        speed = motion.log() / dt if dt > 0 else np.zeros((6,))
        pose = motion * pose
        trajectory[timestamps[f_no]] = pose.inverse().matrix()
        f_no0 = f_no
        t0 = t1
        I0 = I1
        Z0 = Z1

        if f_no - f_start > 25 and f_no % rate_eval == 0:
            try:
                log.rmse_t, log.rmse_r = sequence.evaluate_rpe(
                    trajectory, output_dir="./", upload=upload
                )
                write_result_file(trajectory, f"{sequence._sequence_id}-algo.txt")
            except Exception as e:
                print(e)

    sequence.evaluate_rpe(trajectory, output_dir="./", upload=upload)
    if upload:
        wandb.finish()
