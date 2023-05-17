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
