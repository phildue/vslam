import matplotlib.pyplot as plt
import numpy as np
import argparse
from vslampy._tum.dataset_analysis import read_trajectory, ominus
import wandb
import os


def compute_relative_poses(traj):
    t = list(traj.keys())
    relative_poses = {t[0]: np.identity(4)}

    for i in range(1, len(t)):
        relative_poses[t[i]] = ominus(traj[t[i - 1]], traj[t[i]])

    return relative_poses


def plot_trajectory(algo_file, gt_file, out, show, upload):
    files = [algo_file]
    if gt_file and os.path.exists(gt_file):
        files.append(gt_file)
    trajs = [read_trajectory(f, matrix=True) for f in files]
    motions = [compute_relative_poses(t) for t in trajs]
    plt.figure(figsize=(20, 10))

    plt.subplot(2, 2, 1)
    plt.ylabel("$t_y   [m]$")
    plt.xlabel("$t_x   [m]$")
    plt.grid("both")
    t0 = list(trajs[0].keys())[0]
    t_to_s = 1

    for traj in trajs:
        x = np.array([pose[0, 3] for _, pose in traj.items()])
        y = np.array([pose[1, 3] for _, pose in traj.items()])
        plt.plot(x, y, ".--")
        plt.axis("equal")
    plt.legend([f[-10:-3] for f in files])

    plt.subplot(2, 2, 2)
    plt.ylabel("$t_z   [m]$")
    plt.xlabel("$t-t_0 [s]$")
    plt.grid("both")

    for traj in trajs:
        z = [pose[2, 3] for _, pose in traj.items()]
        t = [(t - t0) * t_to_s for t, _ in traj.items()]
        plt.plot(t, z, ".--")
    plt.legend([f for f in files])
    plt.subplot(2, 2, 3)

    plt.ylabel("$v [m/s]$")
    plt.xlabel("$t-t_0 [s]$")
    plt.grid("both")
    for motion in motions:
        t = [(t - t0) * t_to_s for t, _ in motion.items()]
        v = [np.linalg.norm(dpose[:3, 3]) for _, dpose in motion.items()]
        plt.plot(t, v, ".--")

    try:
        cov = read_trajectory(algo_file, covariance=True)[1]

        plt.subplot(2, 2, 4)
        plt.ylabel("$|\Sigma| $")
        plt.xlabel("$t-t_0 [s]$")
        plt.grid("both")
        t = [(t - t0) * t_to_s for t, _ in cov.items()]
        y = [np.linalg.norm(np.diag(c)) for _, c in cov.items()]
        plt.plot(t, y, ".--")
    except Exception as e:
        print(e)
    if out:
        plt.savefig(out)
    if show:
        plt.show()
    if upload:
        wandb.log({"trajectory": plt})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Plot two trajectories for comparison ( xy, z )"""
    )
    parser.add_argument(
        "algo_file",
        help="algo text file (format: timestamp data)",
        default="/mnt/dataset/tum_rgbd/freiburg1_floor/",
    )
    parser.add_argument("--gt_file", help="gt text file (format: timestamp data)")
    parser.add_argument("--show", help="if true show live", action="store_true")
    parser.add_argument("--upload", help="if true show live", action="store_true")
    parser.add_argument("--out", help="where to save plot png", default="")

    args = parser.parse_args()

    plot_trajectory(args.algo_file, args.gt_file, args.out, args.show, args.upload)
