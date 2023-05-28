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
from vslampy.plot.plot_traj import plot_trajectory
from vslampy.plot.parse_performance_log import parse_performance_log

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
    "--run_algo", help="Set to create algorithm results", action="store_true"
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
parser.add_argument(
    "--dataset",
    help="Name of dataset. Available options: [tum, kitti]",
    default="kitti",
)
args = parser.parse_args()


dataset = (
    Kitti(args.sequence_id)
    if args.sequence_id in Kitti.sequences()
    else TumRgbd(args.sequence_id)
)
config_file = os.path.join(args.workspace_dir, "config", "node_config.yaml")
params = yaml.safe_load(
                Path(os.path.join(config_file)).read_text()
            )

evaluation = Evaluation(sequence=dataset, parameters=params, experiment_name=args.experiment_name, upload=args.upload,out_root=args.out_root,run_algo=args.run_algo)

if args.run_algo:
    print("---------Running Algorithm-----------------")

    os.system(
        f"{args.workspace_dir}/install/vslam_ros/lib/composition_evaluation_{args.dataset} --ros-args --params-file {config_file} \
        -p bag_file:={dataset.filepath()} \
        -p gtTrajectoryFile:={dataset.gt_filepath()} \
        -p algoOutputFile:={evaluation.output_dir}/{args.sequence_id}-algo.txt \
        -p replayMode:=True \
        -p sync_topic:={dataset.sync_topic()} \
        -p log.root_dir:={os.path.join(evaluation.output_dir, 'log')} \
        {dataset.remappings()} \
        2>&1 | tee {os.path.join(evaluation.output_dir,'log','log.txt')}"
    )
# TODO plot, fix paths
print("---------Creating Plots-----------------")
plot_trajectory(evaluation.filepath_trajectory_algo, dataset.gt_filepath(), evaluation.filepath_trajectory_plot, None,upload=args.upload)

plot_logs(args.experiment_name, args.sequence_id, args.sequence_root)

dataset.run_evaluation_scripts(dataset.gt_filepath(), evaluation.filepath_trajectory_algo, evaluation.output_dir, script_dir)

print("Parsing performance log..")
parse_performance_log(os.path.join(evaluation.output_dir, "log", "vslam.log"))

if args.upload:
    wandb.finish()
