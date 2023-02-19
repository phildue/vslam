#!/usr/bin/python3
import os
import sys
import argparse
import git
import yaml
import shutil
from datetime import datetime
from vslam_evaluation.dataset import Kitti, TumRgbd
import wandb
from pathlib import Path
from vslam_evaluation.plot.plot_logs import plot_logs
from vslam_evaluation.plot.plot_traj import plot_trajectory


script_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(script_dir)
parser = argparse.ArgumentParser(description='''
Run evaluation of algorithm''')
parser.add_argument('--experiment_name', help='Name for the experiment',
                    default="test")
parser.add_argument('--sequence_id', help='Id of the sequence to run on)',
                    default='rgbd_dataset_freiburg2_desk')
parser.add_argument('--sequence_root',
                    help='Root folder for sequences',
                    default='/mnt/dataset/tum_rgbd')
parser.add_argument('--out_root',
                    help='Root folder for generating output, defaults to subfolder of sequence',
                    default='')
parser.add_argument('--launch_without_algo', help='Start everything without algo for debugging',
                    default='False')
parser.add_argument('--run_algo', help='Set to create algorithm results', action="store_true")
parser.add_argument('--upload', help='Upload results to experiment tracking tool', action="store_false")

parser.add_argument('--commit_hash',
                    help='Id to identify algorithm version',
                    default='')
parser.add_argument('--workspace_dir',
                    help='Directory of repository (only applicable if commit_hash not given)',
                    default='/home/ros/vslam_ros')
parser.add_argument('--dataset',
                    help='Name of dataset. Available options: [tum, kitti]',
                    default='kitti')
args = parser.parse_args()


dataset = Kitti(args.sequence_id) if args.sequence_id in Kitti.sequences() else TumRgbd(args.sequence_id)

if not args.out_root:
    args.out_root = f'{dataset.bag_filepath()}/algorithm_results/'
output_dir = f'{args.out_root}/{args.experiment_name}'
algo_traj = os.path.join(output_dir, args.sequence_id+"-algo.txt")
traj_plot = os.path.join(output_dir, "trajectory.png")


gt_traj = os.path.join(args.sequence_root, args.sequence_id, args.sequence_id + "-groundtruth.txt")
if not os.path.exists(output_dir):
    if not args.run_algo:
        raise ValueError(f"There is no algorithm output for: {args.experiment_name}. \
            Create it by setting --run_algo")
    os.makedirs(output_dir)


if args.upload:
    # os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_BASE_URL"] = 'http://localhost:8080'
    os.environ["WANDB_API_KEY"] = 'local-837a2a9d75b14cf1ae7886da28a78394a9a7b053'
    wandb.init(project="vslam", entity="phild")
    wandb.run.name = f'{args.sequence_id}.{args.experiment_name}'

if args.run_algo:
    print("---------Running Algorithm-----------------")
    sha = args.commit_hash if args.commit_hash else git.Repo(args.workspace_dir).head.object.hexsha
    with open(os.path.join(output_dir, 'meta.yaml'), 'w') as f:
        yaml.dump([
                {'date': datetime.now()},
                {'name': args.experiment_name},
                {'code_sha': sha}
                ], f)
    
    config_file = os.path.join(args.workspace_dir, 'config', 'node_config.yaml')
    shutil.copy(config_file, os.path.join(output_dir, 'node_config.yaml'))
    
    os.system(f"{args.workspace_dir}/install/vslam_ros/lib/composition_evaluation_{args.dataset} --ros-args --params-file {config_file} \
        -p bag_file:={dataset.bag_filepath()} \
        -p gtTrajectoryFile:={dataset.gt_filepath()} \
        -p algoOutputFile:={output_dir}/{args.sequence_id}-algo.txt \
        -p replayMode:=True \
        -p sync_topic:={dataset.sync_topic()} \
        -p log.root_dir:={os.path.join(output_dir, 'log')} \
        {dataset.remappings()}")

# TODO plot, fix paths
print("---------Creating Plots-----------------")
plot_trajectory(algo_traj, gt_traj, traj_plot, None, True)

plot_logs(args.experiment_name, args.sequence_id, args.sequence_root)

dataset.run_evaluation_scripts(gt_traj, algo_traj, output_dir, script_dir)

if args.upload:
    wandb.config = yaml.safe_load(Path(os.path.join(output_dir, 'node_config.yaml')).read_text())
    wandb.finish()

