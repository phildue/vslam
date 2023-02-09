import os
from typing import List


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
        if sequence_id not in TumRgbd.sequences(): raise ValueError(f"This is not a tum_rgbd sequence: {sequence_id}")

        super(TumRgbd, self).__init__(sequence_id)

    def bag_filepath(self) -> str:
        return f'/mnt/dataset/tum_rgbd/{self._sequence_id}'

    def gt_filepath(self) -> str:
        return f'/mnt/dataset/tum_rgbd/{self._sequence_id}/{self._sequence_id}-groundtruth.txt'

    def sync_topic(self) -> str:
        return '/camera/depth/image'

    def run_evaluation_scripts(self, gt_traj, algo_traj, output_dir, script_dir):
        rpe_plot = os.path.join(output_dir, "rpe.png")
        ate_plot = os.path.join(output_dir, "ate.png")
        rpe_txt = os.path.join(output_dir, 'rpe.txt')
        ate_txt = os.path.join(output_dir, 'ate.txt')
        print("---------Evaluating Relative Pose Error-----------------")
        os.system(f"python3 {script_dir}/vslam_evaluation/tum/evaluate_rpe.py \
            {gt_traj} {algo_traj} \
            --verbose --plot {rpe_plot} --fixed_delta --delta_unit s --save {rpe_txt} \
                > {output_dir}/rpe_summary.txt && cat {output_dir}/rpe_summary.txt")

        print("---------Evaluating Average Trajectory Error------------")
        os.system(f"python3 {script_dir}/vslam_evaluation/tum/evaluate_ate.py \
            {gt_traj} {algo_traj} \
            --verbose --plot {ate_plot} --save {ate_txt} \
                > {output_dir}/ate_summary.txt && cat {output_dir}/ate_summary.txt")    

    def remappings(self) -> str:
        return "" # TODO

    @staticmethod
    def sequences() -> List[str]:
        return["rgbd_dataset_freiburg1_desk",
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
                "rgbd_dataset_freiburg3_long_office_household"]

class Kitti(Dataset):

    def __init__(self, sequence_id):
        if sequence_id not in Kitti.sequences(): raise ValueError(f"This is not a kitti sequence: {sequence_id}")
        super(Kitti, self).__init__(sequence_id)

    def bag_filepath(self) -> str:
        return f'/mnt/dataset/kitti/data_odometry_gray/dataset/sequences/{self._sequence_id}'

    def gt_filepath(self) -> str:
        return f'/mnt/dataset/kitti/data_odometry_gray/dataset/poses/{self._sequence_id}.txt'
    
    def sync_topic(self) -> str:
        return '/kitti/camera_gray_left/image_rect'
    
    def run_evaluation_scripts(self, gt_traj, algo_traj, output_dir, script_dir):
        print("No evaluation scripts implemented")

    @staticmethod
    def sequences() -> List[str]:
        return[
        "00"]
    
    def remappings(self) -> str:
        return "-r /left/camera_info:=/kitti/camera_gray_left/camera_info \
        -r /left/image_rect:=/kitti/camera_gray_left/image_rect \
        -r /right/camera_info:=/kitti/camera_gray_right/camera_info \
        -r /right/image_rect:=/kitti/camera_gray_right/image_rect \
        -r /camera/depth/image:=/depth \
        -r /camera/rgb/image_color:=/kitti/camera_gray_left/image_rect \
        -r /camera/rgb/camera_info:=/kitti/camera_gray_left/camera_info"