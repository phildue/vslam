from vslampy.evaluation.dataset import Dataset
import wandb
import os


class Evaluation:
    def __init__(
        self, sequence: Dataset, parameters, experiment_name, upload=False, store=True
    ):
        self.sequence = sequence
        self.upload = upload
        self.store = store
        if upload:
            os.environ["WANDB_BASE_URL"] = "http://localhost:8080"
            os.environ[
                "WANDB_API_KEY"
            ] = "local-837a2a9d75b14cf1ae7886da28a78394a9a7b053"
            wandb.init(
                project="vslam",
                entity="phild",
                config=parameters,
            )
            wandb.run.name = f"{sequence.id()}.{experiment_name}"

    def finalize(self, trajectory=None):
        if trajectory:
            self.sequence.evaluate_rpe(
                trajectory,
                output_dir=self.sequence.directory() if self.store else None,
                upload=self.upload,
            )
        if self.upload:
            wandb.finish()
