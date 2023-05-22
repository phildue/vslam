import os
from typing import List, Tuple
from vslampy.evaluation._tum.evaluate_rpe import read_trajectory, evaluate_trajectory
from vslampy.evaluation.dataset import Dataset
from vslampy.camera import Camera
import numpy as np
import cv2 as cv
import logging


class TumRgbd(Dataset):
    def __init__(self, sequence_id):
        if sequence_id not in TumRgbd.sequences():
            raise ValueError(f"This is not a tum_rgbd sequence: {sequence_id}")

        super(TumRgbd, self).__init__(sequence_id)

        try:
            self._t_Z, self._files_Z, self._t_I, self._files_I = self.parse_data()
        except Exception as e:
            logging.warning(
                f"Extracted data is not available for: {sequence_id} because: \n{e}"
            )

    def directory(self) -> str:
        return f"/mnt/dataset/tum_rgbd/{self._sequence_id}"

    def bag_filepath(self) -> str:
        return f"/mnt/dataset/tum_rgbd/{self._sequence_id}"

    def gt_filepath(self) -> str:
        return f"/mnt/dataset/tum_rgbd/{self._sequence_id}/{self._sequence_id}-groundtruth.txt"

    def gt_trajectory(self):
        return read_trajectory(self.gt_filepath())

    def sync_topic(self) -> str:
        return "/camera/depth/image"

    def camera(self):
        return Camera(fx=525.0, fy=525.0, cx=319.5, cy=239.5, h=480, w=640)

    def image_depth_filepaths(self):
        return self._t_Z, self._files_Z, self._t_I, self._files_I

    def timestamps(self, sensor_name="image"):
        if sensor_name == "image":
            return self._t_I
        if sensor_name == "depth":
            return self._t_Z
        raise ValueError(f"{sensor_name} not found.")

    def parse_data(self):
        timestamps_depth = []
        timestamps_intensity = []
        filenames_depth = []
        filenames_intensity = []
        folder = f"/mnt/dataset/tum_rgbd/{self._sequence_id}/{self._sequence_id}"
        for line in open(f"{folder}/assoc.txt", "r"):
            elements = line.split(" ")
            timestamps_depth += [float(elements[0])]
            timestamps_intensity += [float(elements[2])]
            filenames_depth += [folder + "/" + elements[1]]
            filenames_intensity += [folder + "/" + elements[3][:-1]]

        print(f"Found {len(timestamps_depth)} frames")
        return (
            timestamps_depth,
            filenames_depth,
            timestamps_intensity,
            filenames_intensity,
        )

    def load_frame(self, f_no) -> Tuple[np.array, np.array]:
        path_img = self._files_I[f_no]
        path_depth = self._files_Z[f_no]

        if not os.path.exists(path_img):
            raise ValueError(f"Path does not exist: {path_img}")
        if not os.path.exists(path_depth):
            raise ValueError(f"Path does not exist: {path_depth}")

        I = cv.imread(path_img, cv.IMREAD_GRAYSCALE)
        Z = cv.imread(path_depth, cv.IMREAD_ANYDEPTH) / 5000.0
        return I, Z

    def evaluate_rpe(self, traj_est, output_dir="./", upload=True):
        rpe_plot = os.path.join(output_dir, "rpe.png")
        rpe_txt = os.path.join(output_dir, "rpe.txt")
        max_pairs = 10000
        fixed_delta = True
        delta = 1.0
        delta_unit = "s"
        save = rpe_txt
        plot = rpe_plot
        verbose = True
        offset = 0
        scale = 1.0
        print("---------Evaluating Relative Pose Error-----------------")
        traj_gt = read_trajectory(self.gt_filepath())

        result = evaluate_trajectory(
            traj_gt,
            traj_est,
            int(max_pairs),
            fixed_delta,
            float(delta),
            delta_unit,
            float(offset),
            float(scale),
        )

        stamps = np.array(result)[:, 0]
        trans_error = np.array(result)[:, 4]
        rot_error = np.array(result)[:, 5]
        rmse_t = np.sqrt(np.dot(trans_error, trans_error) / len(trans_error))
        rmse_r = np.sqrt(np.dot(rot_error, rot_error) / len(rot_error)) * 180.0 / np.pi
        if save:
            f = open(save, "w")
            f.write("\n".join([" ".join(["%f" % v for v in line]) for line in result]))
            f.close()

        if verbose:
            print("compared_pose_pairs %d pairs" % (len(trans_error)))

            print("translational_error.rmse %f m" % rmse_t)
            print("translational_error.mean %f m" % np.mean(trans_error))
            print("translational_error.median %f m" % np.median(trans_error))
            print("translational_error.std %f m" % np.std(trans_error))
            print("translational_error.min %f m" % np.min(trans_error))
            print("translational_error.max %f m" % np.max(trans_error))

            print("rotational_error.rmse %f deg" % (rmse_r))
            print("rotational_error.mean %f deg" % (np.mean(rot_error) * 180.0 / np.pi))
            print(
                "rotational_error.median %f deg"
                % (np.median(rot_error) * 180.0 / np.pi)
            )
            print("rotational_error.std %f deg" % (np.std(rot_error) * 180.0 / np.pi))
            print("rotational_error.min %f deg" % (np.min(rot_error) * 180.0 / np.pi))
            print("rotational_error.max %f deg" % (np.max(rot_error) * 180.0 / np.pi))
        else:
            print(np.mean(trans_error))

        if plot:
            print("---Plotting---")

            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.pylab as pylab

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(stamps - stamps[0], trans_error, "-", color="blue")
            # ax.plot([t for t,e in err_rot],[e for t,e in err_rot],'-',color="red")
            ax.set_xlabel("time [s]")
            ax.set_ylabel("translational error [m]")
            plt.savefig(plot, dpi=300)

        if upload:
            print("---Uploading results---")
            import wandb

            metric_t = wandb.define_metric("Timestamp", summary=None, hidden=True)
            wandb.define_metric(
                "translational_error",
                summary="mean",
                goal="minimize",
                step_metric=metric_t,
            )
            wandb.define_metric(
                "rotational_error",
                summary="mean",
                goal="minimize",
                step_metric=metric_t,
            )

            for i in range(trans_error.shape[0]):
                wandb.log(
                    {
                        "Timestamp": stamps[i] - stamps[0],
                        "translational_error": trans_error[i],
                        "rotational_error": rot_error[i],
                    }
                )
            wandb.run.summary["translational_error.RMSE"] = np.sqrt(
                np.dot(trans_error, trans_error) / len(trans_error)
            )
            wandb.run.summary["rotational_error.RMSE"] = np.sqrt(
                np.dot(rot_error, rot_error) / len(rot_error)
            )
        return rmse_t, rmse_r

    def run_evaluation_scripts(self, gt_traj, algo_traj, output_dir, script_dir):
        ate_plot = os.path.join(output_dir, "ate.png")
        ate_txt = os.path.join(output_dir, "ate.txt")
        traj_est = read_trajectory(algo_traj)

        self.evaluate_rpe(traj_est, output_dir)

        print("---------Evaluating Average Trajectory Error------------")
        os.system(
            f"python3 {script_dir}/vslam_evaluation/tum/evaluate_ate.py \
            {gt_traj} {algo_traj} \
            --verbose --plot {ate_plot} --save {ate_txt} \
                > {output_dir}/ate_summary.txt && cat {output_dir}/ate_summary.txt"
        )

    def remappings(self) -> str:
        return ""  # TODO

    @staticmethod
    def sequences() -> List[str]:
        return [
            "rgbd_dataset_freiburg1_desk",
            "rgbd_dataset_freiburg1_desk_validation",
            "rgbd_dataset_freiburg1_desk2",
            "rgbd_dataset_freiburg1_desk2_validation",
            "rgbd_dataset_freiburg1_floor",
            "rgbd_dataset_freiburg1_room",
            "rgbd_dataset_freiburg1_rpy",
            "rgbd_dataset_freiburg1_teddy",
            "rgbd_dataset_freiburg1_xyz",
            "rgbd_dataset_freiburg1_360",
            "rgbd_dataset_freiburg2_desk",
            "rgbd_dataset_freiburg2_desk_validation",
            "rgbd_dataset_freiburg2_pioneer_360",
            "rgbd_dataset_freiburg2_pioneer_slam",
            "rgbd_dataset_freiburg3_long_office_household",
        ]
