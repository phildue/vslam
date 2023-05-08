import os
from typing import List
from vslampy.tum.evaluate_rpe import read_trajectory, evaluate_trajectory
import numpy


class Dataset:
    def __init__(self, sequence_id):
        self._sequence_id = sequence_id

    def bag_filepath(self) -> str:
        pass

    def gt_filepath(self) -> str:
        pass

    def sync_topic(self) -> str:
        pass

    def run_evaluation_scripts(self, gt_traj, algo_traj, output_dir, script_dir):
        pass

    def remappings(self) -> str:
        pass


class TumRgbd(Dataset):
    def __init__(self, sequence_id):
        if sequence_id not in TumRgbd.sequences():
            raise ValueError(f"This is not a tum_rgbd sequence: {sequence_id}")

        super(TumRgbd, self).__init__(sequence_id)

    def bag_filepath(self) -> str:
        return f"/mnt/dataset/tum_rgbd/{self._sequence_id}"

    def gt_filepath(self) -> str:
        return f"/mnt/dataset/tum_rgbd/{self._sequence_id}/{self._sequence_id}-groundtruth.txt"

    def sync_topic(self) -> str:
        return "/camera/depth/image"

    def image_depth_filepaths(self):
        timestamps = []
        filenames_depth = []
        filenames_intensity = []
        folder = f"/mnt/dataset/tum_rgbd/{self._sequence_id}/{self._sequence_id}"
        for line in open(f"{folder}/assoc.txt", "r"):
            elements = line.split(" ")
            timestamps += [float(elements[0])]
            filenames_depth += [folder + "/" + elements[1]]
            filenames_intensity += [folder + "/" + elements[3][:-1]]

        print(f"Found {len(timestamps)} frames")
        return timestamps, filenames_intensity, filenames_depth

    def evaluate_rpe(self, algo_traj, output_dir="./", upload=True):
        gt_traj = read_trajectory(self.gt_filepath)
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
        # os.system(f"python3 {script_dir}/vslam_evaluation/tum/evaluate_rpe.py \
        #    {gt_traj} {algo_traj} \
        #    --verbose --plot {rpe_plot} --fixed_delta --delta_unit s --save {rpe_txt} \
        #        > {output_dir}/rpe_summary.txt && cat {output_dir}/rpe_summary.txt")

        traj_gt = read_trajectory(gt_traj)
        traj_est = read_trajectory(algo_traj)

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

        stamps = numpy.array(result)[:, 0]
        trans_error = numpy.array(result)[:, 4]
        rot_error = numpy.array(result)[:, 5]

        if save:
            f = open(save, "w")
            f.write("\n".join([" ".join(["%f" % v for v in line]) for line in result]))
            f.close()

        if verbose:
            print("compared_pose_pairs %d pairs" % (len(trans_error)))

            print(
                "translational_error.rmse %f m"
                % numpy.sqrt(numpy.dot(trans_error, trans_error) / len(trans_error))
            )
            print("translational_error.mean %f m" % numpy.mean(trans_error))
            print("translational_error.median %f m" % numpy.median(trans_error))
            print("translational_error.std %f m" % numpy.std(trans_error))
            print("translational_error.min %f m" % numpy.min(trans_error))
            print("translational_error.max %f m" % numpy.max(trans_error))

            print(
                "rotational_error.rmse %f deg"
                % (
                    numpy.sqrt(numpy.dot(rot_error, rot_error) / len(rot_error))
                    * 180.0
                    / numpy.pi
                )
            )
            print(
                "rotational_error.mean %f deg"
                % (numpy.mean(rot_error) * 180.0 / numpy.pi)
            )
            print(
                "rotational_error.median %f deg"
                % (numpy.median(rot_error) * 180.0 / numpy.pi)
            )
            print(
                "rotational_error.std %f deg"
                % (numpy.std(rot_error) * 180.0 / numpy.pi)
            )
            print(
                "rotational_error.min %f deg"
                % (numpy.min(rot_error) * 180.0 / numpy.pi)
            )
            print(
                "rotational_error.max %f deg"
                % (numpy.max(rot_error) * 180.0 / numpy.pi)
            )
        else:
            print(numpy.mean(trans_error))

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
            wandb.run.summary["translational_error.RMSE"] = numpy.sqrt(
                numpy.dot(trans_error, trans_error) / len(trans_error)
            )
            wandb.run.summary["rotational_error.RMSE"] = numpy.sqrt(
                numpy.dot(rot_error, rot_error) / len(rot_error)
            )

    def run_evaluation_scripts(self, gt_traj, algo_traj, output_dir, script_dir):
        ate_plot = os.path.join(output_dir, "ate.png")
        ate_txt = os.path.join(output_dir, "ate.txt")

        self.evaluate_rpe(algo_traj, output_dir)

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
            "rgbd_dataset_freiburg1_rpy",
            "rgbd_dataset_freiburg1_xyz",
            "rgbd_dataset_freiburg1_360",
            "rgbd_dataset_freiburg2_desk",
            "rgbd_dataset_freiburg2_desk_validation",
            "rgbd_dataset_freiburg2_pioneer_360",
            "rgbd_dataset_freiburg2_pioneer_slam",
            "rgbd_dataset_freiburg3_long_office_household",
        ]


class Kitti(Dataset):
    def __init__(self, sequence_id):
        if sequence_id not in Kitti.sequences():
            raise ValueError(f"This is not a kitti sequence: {sequence_id}")
        super(Kitti, self).__init__(sequence_id)

    def bag_filepath(self) -> str:
        return f"/mnt/dataset/kitti/data_odometry_gray/dataset/sequences/{self._sequence_id}"

    def gt_filepath(self) -> str:
        return f"/mnt/dataset/kitti/data_odometry_gray/dataset/poses/{self._sequence_id}.txt"

    def sync_topic(self) -> str:
        return "/kitti/camera_gray_left/image_rect"

    def run_evaluation_scripts(self, gt_traj, algo_traj, output_dir, script_dir):
        print("No evaluation scripts implemented")

    @staticmethod
    def sequences() -> List[str]:
        return ["00"]

    def remappings(self) -> str:
        return "-r /left/camera_info:=/kitti/camera_gray_left/camera_info \
        -r /left/image_rect:=/kitti/camera_gray_left/image_rect \
        -r /right/camera_info:=/kitti/camera_gray_right/camera_info \
        -r /right/image_rect:=/kitti/camera_gray_right/image_rect \
        -r /camera/depth/image:=/depth \
        -r /camera/rgb/image_color:=/kitti/camera_gray_left/image_rect \
        -r /camera/rgb/camera_info:=/kitti/camera_gray_left/camera_info"
