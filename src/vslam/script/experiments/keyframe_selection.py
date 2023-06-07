from vslampy.evaluation.tum import TumRgbd
from vslampy.evaluation.evaluation import Evaluation
from vslampy.evaluation._tum.evaluate_rpe import read_trajectory
from vslampy.evaluation._tum.dataset_analysis import read_trajectory
from vslampy.utils.utils import interpolate_pose_between,interpolate_pose_at
import numpy as np
from sophus.sophuspy import SE3
import matplotlib.pyplot as plt
sequence = TumRgbd("rgbd_dataset_freiburg1_desk")

evaluation = Evaluation(sequence, "keyframe_selection")
trajectory_gt = dict([(t, SE3(pose).inverse()) for t, pose in sequence.gt_trajectory().items()])

trajectory, covariances = read_trajectory(evaluation.filepath_trajectory_algo, covariance=True)
trajectory = dict([(t, SE3(pose).inverse()) for t, pose in trajectory.items()])

t_50 = list(trajectory.keys())[0]

pose_pairs = [(trajectory[t], interpolate_pose_between(trajectory_gt, t_50, t)) for t in trajectory.keys()]
errors = [pose_algo * pose_gt.inverse() for pose_algo, pose_gt in pose_pairs]
errors_translation = np.array([np.linalg.norm(e.translation()) for e in errors])
entropies = np.array([np.log(np.linalg.det(cov)) for cov in covariances.values()])
entropy_ratio = entropies/entropies[1]
plt.plot(entropy_ratio, "x--")
plt.plot(errors_translation, "x--")
plt.legend(['Entropy', 'Error Translation'])
plt.grid('both')
plt.show()    