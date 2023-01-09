#!/usr/bin/python3
import os
import sys
import argparse
import git
import yaml
import shutil
from datetime import datetime
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
                    help='Root folder for generating output, defaults to <sequence_root>',
                    default='')
parser.add_argument('--launch_without_algo', help='Start everything without algo for debugging',
                    default='False')
parser.add_argument('--run_algo', help='Set to create algorithm results', action="store_true")
parser.add_argument('--commit_hash',
                    help='Id to identify algorithm version',
                    default='')
parser.add_argument('--workspace_dir',
                    help='Directory of repository (only applicable if commit_hash not given)',
                    default='/home/ros/vslam_ros')
args = parser.parse_args()

if not args.out_root:
    args.out_root = args.sequence_root
output_dir = os.path.join(args.out_root, args.sequence_id, 'algorithm_results', args.experiment_name)
algo_traj = os.path.join(output_dir, args.sequence_id+"-algo.txt")
rpe_plot = os.path.join(output_dir, "rpe.png")
ate_plot = os.path.join(output_dir, "ate.png")
traj_plot = os.path.join(output_dir, "trajectory.png")
rpe_txt = os.path.join(output_dir, 'rpe.txt')
ate_txt = os.path.join(output_dir, 'ate.txt')

gt_traj = os.path.join(args.sequence_root, args.sequence_id, args.sequence_id + "-groundtruth.txt")
if not os.path.exists(output_dir):
    if not args.run_algo:
        raise ValueError(f"There is no algorithm output for: {args.experiment_name}. \
            Create it by setting --run_algo")
    os.makedirs(output_dir)

if args.run_algo:
    print("---------Running Algorithm-----------------")
    sha = args.commit_hash if args.commit_hash else git.Repo(args.workspace_dir).head.object.hexsha
    with open(os.path.join(output_dir, 'meta.yaml'), 'w') as f:
        yaml.dump([
                {'date': datetime.now()},
                {'name': args.experiment_name},
                {'code_sha': sha}
                ], f)
    
    # shutil.copy(os.path.join(args.workspace_dir, 'launch', 'evaluation.launch.py'), os.path.join(output_dir, 'evaluation.launch.py'))
    # os.system(f"ros2 launch vslam_ros evaluation.launch.py \
    #    sequence_root:={args.sequence_root} sequence_id:={args.sequence_id} \
    #    experiment_name:={args.experiment_name} launch_without_algo:={args.launch_without_algo}")
    
    config_file = os.path.join(args.workspace_dir, 'config', 'nodeMapping.yaml')
    shutil.copy(config_file, os.path.join(output_dir, 'nodeMapping.yaml'))
    
    os.system(f"{args.workspace_dir}/install/vslam_ros/bin/composition_evaluation --ros-args --params-file {config_file} \
        -p bag_file:=/mnt/dataset/tum_rgbd/{args.sequence_id} \
        -p gtTrajectoryFile:=/mnt/dataset/tum_rgbd/{args.sequence_id}/{args.sequence_id}-groundtruth.txt \
        -p algoOutputFile:=/mnt/dataset/tum_rgbd/{args.sequence_id}/algorithm_results/{args.experiment_name}/{args.sequence_id}-algo.txt \
        -p replayMode:=True \
        -p log.root_dir:={os.path.join(output_dir, 'log')}")

# TODO plot, fix paths
print("---------Creating Plots-----------------")
os.system(f"python3 {script_dir}/vslam_evaluation/plot/plot_traj.py \
    {algo_traj} --gt_file {gt_traj} --out {traj_plot}")

os.system(f"python3 {script_dir}/vslam_evaluation/plot/plot_logs.py \
    --experiment_name {args.experiment_name} --sequence_id {args.sequence_id}")

print("---------Evaluating Relative Pose Error-----------------")
os.system(f"python3 {script_dir}/vslam_evaluation/tum/evaluate_rpe.py \
    {gt_traj} {algo_traj} \
    --verbose --plot {rpe_plot} --fixed_delta --delta_unit s --save {rpe_plot} \
        > {output_dir}/rpe_summary.txt && cat {output_dir}/rpe_summary.txt")

print("---------Evaluating Average Trajectory Error------------")
os.system(f"python3 {script_dir}/vslam_evaluation/tum/evaluate_ate.py \
    {gt_traj} {algo_traj} \
    --verbose --plot {ate_plot} --save {ate_txt} \
        > {output_dir}/ate_summary.txt && cat {output_dir}/ate_summary.txt")


# TODO upload to WandB
