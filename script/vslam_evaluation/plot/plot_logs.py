import matplotlib.pyplot as plt
import numpy as np
import argparse
import pandas as pd
import os

from plot.plot_log_alignment import plot_alignment
from plot.plot_log_kalman import plot_kalman
from plot.plot_log_residual import plot_residual

plots = {
    "Alignment": plot_alignment,
    "Kalman": plot_kalman,
    "ResidualFinal": plot_residual
}


def plot_logs(experiment_name, sequence_id, sequence_root):
    result_dir = os.path.join(sequence_root, sequence_id, 'algorithm_results', experiment_name)
    log_dir = os.path.join(result_dir, 'log')
    for log in os.listdir(log_dir):
        if log in plots.keys():
            try:
                print(f"Processing: [{log}] in [{os.path.join(log_dir, log)}]")
                plots[log](os.path.join(log_dir, log), result_dir)
            except Exception as e:
                print(e)
        else:
            print(f"No Display for: {log}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='''
    Run evaluation of algorithm''')
    parser.add_argument('--experiment_name', help='Name for the experiment',
                        default="no_motion_model")
    parser.add_argument('--sequence_id', help='Id of the sequence to run on)',
                        default='rgbd_dataset_freiburg1_floor')
    parser.add_argument('--sequence_root',
                        help='Root folder for sequences',
                        default='/mnt/dataset/tum_rgbd')
    args = parser.parse_args()
    plot_logs(args.experiment_name, args.sequence_id, args.sequence_root)
   
