import matplotlib.pyplot as plt
import numpy as np
import argparse
from tum.dataset_analysis import read_trajectory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    description='''Plot two trajectories for comparison ( xy, z )''')
    parser.add_argument('first_file', help='first text file (format: timestamp data)')
    parser.add_argument('second_file', help='second text file (format: timestamp data)')
    parser.add_argument('--show', help='if true show live', action='store_true')
    parser.add_argument('--xy_out', help='where to save plot png xy', default="")
    parser.add_argument('--z_out', help='where to save plot png z', default="")

    args = parser.parse_args()

    files = [args.first_file, args.second_file]
    trajs = [read_trajectory(f) for f in files]

    # TODO alignment

    transs = []
    for traj in trajs:
        transs.append(np.array([pose[:3, 3] for pose in traj.values()]))
    plt.figure()
    for trans in transs:
        plt.scatter(trans[:, 0], trans[:, 1], marker=".")
    plt.xlabel("tx")
    plt.xlabel("ty")
    plt.legend(files)
    if args.xy_out:
        plt.savefig(args.xy_out)
    plt.figure()
    for trans in transs:
        plt.scatter(np.arange(trans.shape[0]), trans[:, 2], marker=".")
    plt.xlabel("t")
    plt.ylabel("tz")
    plt.legend(files)
    if args.z_out:
        plt.savefig(args.z_out)
    if args.show:
        plt.show()
