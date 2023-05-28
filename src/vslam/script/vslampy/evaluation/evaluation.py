from vslampy.evaluation.dataset import Dataset
import wandb
import os
import git
import yaml
import shutil
from datetime import datetime

class Evaluation:
    
    def __init__(
        self, sequence: Dataset, parameters, experiment_name, upload=False, store=True, out_root=None, run_algo=True, sha=None,workspace_dir="/home/ros/vslam_ros/"):
        self.sequence = sequence
        self.upload = upload
        self.store = store
        self.experiment_name = experiment_name
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

        if not out_root:
            self.out_root = f"{sequence.filepath()}/algorithm_results/"
        else:
            self.out_root = out_root
        self.output_dir = f"{self.out_root}/{self.experiment_name}"
        self.filepath_trajectory_algo = os.path.join(self.output_dir, sequence.id() + "-algo.txt")
        self.filepath_trajectory_plot = os.path.join(self.output_dir, "trajectory.png")
        if store:
            #if os.path.exists(self.output_dir):
                #os.removedirs(self.output_dir)
            os.makedirs(self.output_dir,exist_ok=True)

            sha = (sha if sha else git.Repo(workspace_dir).head.object.hexsha)
            with open(os.path.join(self.output_dir, "meta.yaml"), "w") as f:
                yaml.dump(
                    [
                        {"date": datetime.now()},
                        {"name": self.experiment_name},
                        {"code_sha": sha},
                    ],
                    f,
                )
            with open(os.path.join(self.output_dir, "params.yaml"), "w") as f:
                yaml.dump(
                    parameters,
                    f,
                )
            
            




    def finalize(self, trajectory=None):
        if trajectory:
            self.sequence.evaluate_rpe(
                trajectory,
                output_dir=self.sequence.directory() if self.store else None,
                upload=self.upload,
            )
        if self.upload:
            wandb.finish()
