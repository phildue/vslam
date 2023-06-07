#!/usr/bin/python3
import os
import sys
import argparse
import git
import yaml
import shutil
from datetime import datetime
from vslampy.evaluation.tum import TumRgbd
from vslampy.evaluation.kitty import Kitti

from vslampy.evaluation.evaluation import Evaluation
import wandb
from pathlib import Path
from vslampy.plot.plot_logs import plot_logs
from vslampy.plot.parse_performance_log import parse_performance_log
from threading import Thread
from time import sleep

script_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(script_dir)
parser = argparse.ArgumentParser(
    description="""
Run evaluation of algorithm"""
)
parser.add_argument("--experiment_name", help="Name for the experiment", default="test")
parser.add_argument(
    "--sequence_id",
    help="Id of the sequence to run on)",
    default="rgbd_dataset_freiburg2_desk",
)
parser.add_argument(
    "--sequence_root", help="Root folder for sequences", default="/mnt/dataset/tum_rgbd"
)
parser.add_argument(
    "--out_root",
    help="Root folder for generating output, defaults to subfolder of sequence",
    default="",
)
parser.add_argument(
    "--launch_without_algo",
    help="Start everything without algo for debugging",
    default="False",
)
parser.add_argument(
    "--upload", help="Upload results to experiment tracking tool", action="store_true"
)

parser.add_argument(
    "--commit_hash", help="Id to identify algorithm version", default=""
)
parser.add_argument(
    "--workspace_dir",
    help="Directory of repository (only applicable if commit_hash not given)",
    default="/home/ros/vslam_ros",
)
args = parser.parse_args()


dataset = (
    TumRgbd(args.sequence_id)
    if args.sequence_id in TumRgbd.sequences()
    else Kitti(args.sequence_id)
)
config_file = os.path.join(args.workspace_dir, "config", f"node_config_{dataset.name()}.yaml")
params = yaml.safe_load(
                Path(os.path.join(config_file)).read_text()
            )
evaluation = Evaluation(sequence=dataset, experiment_name=args.experiment_name)

evaluation.prepare_run(parameters=params, sha=args.commit_hash, workspace_dir=args.workspace_dir, override=True)

if args.upload:
    evaluation.prepare_upload()

running = True

def intermediate_evaluation(t):
    sleep(20)
    while running:
        sleep(t)
        try:
            evaluation.evaluate(final=False)
        except Exception as e:
            print(e)

thread=Thread(target=intermediate_evaluation, args=(30, ))
thread.start()

print("---------Running Algorithm-----------------")
try:
    os.system(
        f"{args.workspace_dir}/install/vslam_ros/lib/composition_evaluation_{dataset.name()} --ros-args --params-file {config_file} \
        -p bag_file:={dataset.filepath()} \
        -p gtTrajectoryFile:={dataset.gt_filepath()} \
        -p outputDirectory:={evaluation.folder_results} \
        -p algoOutputFile:={evaluation.filepath_trajectory_algo} \
        -p replayMode:=True \
        -p sync_topic:={dataset.sync_topic()} \
        -p log.root_dir:={evaluation.folder_logs} \
        {dataset.remappings()} \
        2>&1 | tee {evaluation.filepath_console_log}"
    )
except Exception as e:
    print(e)
running = False
thread.join()
evaluation.evaluate(final=True)

