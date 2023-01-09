import matplotlib.pyplot as plt
import numpy as np
import argparse
import pandas as pd
import os

from plot_log_alignment import plot_alignment
from plot_log_kalman import plot_kalman

     
plots = {
    "Alignment": plot_alignment,
    "Kalman": plot_kalman
}

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
    result_dir = os.path.join(args.sequence_root, args.sequence_id, 'algorithm_results', args.experiment_name)
    log_dir = os.path.join(result_dir, 'log')
    for log in os.listdir(log_dir):
        if log in plots.keys():
            plots[log](os.path.join(log_dir, log))
        else:
            print(f"No Display for: {log}")
