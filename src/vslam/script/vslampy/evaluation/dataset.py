from typing import List


class Dataset:
    def __init__(self, sequence_id):
        self._sequence_id = sequence_id

    def id(self):
        return self._sequence_id

    def filepath(self) -> str:
        pass

    def gt_filepath(self) -> str:
        pass

    def sync_topic(self) -> str:
        pass

    def remappings(self) -> str:
        pass

    @staticmethod
    def name() -> str:
        pass
