from vslampy.evaluation.dataset import Dataset
from typing import List
class Kitti(Dataset):
    def __init__(self, sequence_id):
        if sequence_id not in Kitti.sequences():
            raise ValueError(f"This is not a kitti sequence: {sequence_id}")
        super(Kitti, self).__init__(sequence_id)

    def filepath(self) -> str:
        return f"/mnt/dataset/kitti/data_odometry_gray/dataset/sequences/{self._sequence_id}"

    def bag_filepath(self) -> str:
        return f"/mnt/dataset/kitti/data_odometry_gray/dataset/sequences/{self._sequence_id}"

    def gt_filepath(self) -> str:
        return f"/mnt/dataset/kitti/data_odometry_gray/dataset/poses/{self._sequence_id}.txt"

    def sync_topic(self) -> str:
        return "/kitti/camera_gray_left/image_rect"

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
