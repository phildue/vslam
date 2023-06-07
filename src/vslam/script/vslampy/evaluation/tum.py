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

    def filepath(self) -> str:
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
    @staticmethod
    def name() -> str:
        return "tum"