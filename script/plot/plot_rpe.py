import matplotlib.pyplot as plt
import numpy as np
import argparse
from tum.dataset_analysis import ominus, read_trajectory
from tum.evaluate_rpe import find_closest_index


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='''
    Plot RPE and covariance over frame.''')
    parser.add_argument('trajectory_file_gt', help='trajectory text file (format: timestamp data)')
    parser.add_argument('trajectory_file_algo',
                        help='trajectory text file (format: timestamp data)')
    parser.add_argument('--show', help='if true show live', action='store_true')
    parser.add_argument('--out_file', help='where to save plot png xy', default="")
    parser.add_argument('--max_diff', help='max difference to gt timestamp', default=1000)

    args = parser.parse_args()

    traj_algo, cov = read_trajectory(args.trajectory_file_algo, True, True)
    traj_gt = read_trajectory(args.trajectory_file_gt, True, False)

    stamps_algo = list(traj_algo.keys())
    stamps_algo.sort()
    stamps_gt = list(traj_gt.keys())
    stamps_gt.sort()
    error = {}
    relative_poses_algo = {}
    relative_poses_gt = {}
    fId_max = 100
    for fId, t0 in enumerate(stamps_algo[:fId_max]):
        t1 = stamps_algo[fId+1]
        relative_pose_algo = ominus(traj_algo[t0], traj_algo[t1])
        t0_gt = stamps_gt[find_closest_index(stamps_gt, t0)]
        t1_gt = stamps_gt[find_closest_index(stamps_gt, t1)]
        if abs(t0_gt - t1_gt) > args.max_diff or abs(t1_gt - t1) > args.max_diff:
            raise IndexError("Did not find matching GT timestamp within range.")
        relative_pose_gt = ominus(traj_gt[t0_gt], traj_gt[t1_gt])
        error[t1] = ominus(relative_pose_algo, relative_pose_gt)
        relative_poses_algo[t1] = relative_pose_algo
        relative_poses_gt[t1] = relative_pose_gt

    t_mag = np.zeros((fId_max))
    t_mag_gt = np.zeros((fId_max))
    t_err = np.zeros((fId_max))
    h_det = np.zeros((fId_max))
    for i, t in enumerate(stamps_algo[1:fId_max]):
        t_mag[i] = np.sum(relative_poses_algo[t][:3, 3])
        t_mag_gt[i] = np.sum(relative_poses_gt[t][:3, 3])
        t_err[i] = np.linalg.norm(error[t][:3, 3])
        h_det[i] = np.linalg.det(cov[t])

    print('RMSE: {} < {} +- {} < {} '.format(t_err.min(), t_err.mean(), t_err.std(), t_err.max()))
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(t_err)
    plt.xlabel("fNo")
    plt.ylabel("Translational Error")
    plt.subplot(2, 2, 2)
    plt.plot(t_mag)
    plt.plot(t_mag_gt)
    plt.legend(['Algo', 'Gt'])
    plt.xlabel("fNo")
    plt.ylabel("Relative Translation")
    plt.subplot(2, 2, 3)
    plt.xlabel("fNo")
    plt.ylabel("|H|")
    plt.plot(h_det)

    if args.out_file:
        plt.savefig(args.out_file)
    if args.show:
        plt.show()
