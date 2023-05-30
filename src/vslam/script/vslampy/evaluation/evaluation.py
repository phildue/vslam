from vslampy.evaluation.dataset import Dataset
from vslampy.evaluation._tum.evaluate_rpe import read_trajectory, evaluate_trajectory
from vslampy.evaluation._tum import associate
from vslampy.evaluation._tum.evaluate_ate import align,plot_traj
from vslampy.plot.parse_performance_log import parse_performance_log
from vslampy.plot.plot_traj import plot_trajectory
from vslampy.plot.plot_logs import plot_logs
from vslampy.evaluation.tum import TumRgbd
from vslampy.evaluation.kitty import Kitti
from scipy.spatial.transform import Rotation as R

import wandb
import os
import git
import yaml
import shutil
from datetime import datetime
import numpy as np
import sys
from sophus.sophuspy import SE3
class Evaluation:
    
    def __init__(
        self, sequence: Dataset, experiment_name, out_root=None):
        self.sequence = sequence
        self.experiment_name = experiment_name
        self.upload = False
        if not out_root:
            self.out_root = f"{sequence.filepath()}/algorithm_results/"
        else:
            self.out_root = out_root
        self.output_dir = f"{self.out_root}/{self.experiment_name}"
        self.filepath_trajectory_algo = os.path.join(self.output_dir, sequence.id() + "-algo.txt")
        self.filepath_trajectory_plot = os.path.join(self.output_dir, "trajectory.png")            
            
    def prepare_run(self, parameters, upload=False, sha=None, workspace_dir="/home/ros/vslam_ros/"):
        self.upload = upload
        if upload:
            os.environ["WANDB_BASE_URL"] = "http://localhost:8080"
            os.environ[
                "WANDB_API_KEY"
            ] = "local-837a2a9d75b14cf1ae7886da28a78394a9a7b053"
            wandb.init(
                project="vslam",
                entity="phild",
                config=parameters,
                id = f"{self.sequence.id()}.{self.experiment_name}"
            )
            wandb.run.name = f"{self.sequence.id()}.{self.experiment_name}"

        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

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

    def evaluate(self, trajectory=None, final = True):
        if trajectory:
            lines = []
            for ts, pose in trajectory.items():
                q = R.from_matrix(pose[:3,:3]).as_quat()
                t = pose[:3,3]
                lines += [f"{ts} {t[0]:.4f} {t[1]:.4f} {t[2]:.4f} {q[0]:.4f} {q[1]:.4f} {q[2]:.4f} {q[3]:.4f}\n"]
            with open(self.filepath_trajectory_algo, "w") as f:
                f.writelines([
                    f"# Algorithm Trajectory\n",
                    f"# file: {self.filepath_trajectory_algo}\n",
                    f"# timestamp tx ty tz qx qy qz qw\n"
                ])
                f.writelines(lines)
                
            
        self.evaluate_rpe()

        self.evaluate_ate()
        logfile = os.path.join(self.output_dir, "log", "vslam.log")
        if os.path.exists(logfile):
            parse_performance_log(logfile,upload=self.upload)

        if os.path.exists(os.path.join(self.output_dir,"log")):
            plot_logs(self.output_dir)

        plot_trajectory(self.filepath_trajectory_algo, self.sequence.gt_filepath(), self.filepath_trajectory_plot, None, upload=self.upload)

        if final and self.upload:
            wandb.finish()

    def evaluate_rpe(self):
        rpe_plot = os.path.join(self.output_dir, "rpe.png")
        rpe_txt = os.path.join(self.output_dir, "rpe.txt")
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
        traj_gt = read_trajectory(self.sequence.gt_filepath())
        traj_est = read_trajectory(self.filepath_trajectory_algo)

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

        if self.upload:
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


    def evaluate_ate(self):
        print("---------Evaluating Average Trajectory Error-----------------")
        offset = 0
        max_difference = 0.02
        scale = 1.0
        first_list = associate.read_file_list(self.filepath_trajectory_algo)
        second_list = associate.read_file_list(self.sequence.gt_filepath())
        save = self.output_dir + "/" + "associations_ate.txt"
        plot = self.output_dir + "/" + "ate.png"
        verbose = True

        matches = associate.associate(first_list, second_list,float(offset),float(max_difference))    
        if len(matches)<2:
            sys.exit("Couldn't find matching timestamp pairs between groundtruth and estimated trajectory! Did you choose the correct sequence?")


        first_xyz = np.matrix([[float(value) for value in first_list[a][0:3]] for a,b in matches]).transpose()
        second_xyz = np.matrix([[float(value)*float(scale) for value in second_list[b][0:3]] for a,b in matches]).transpose()
        rot,trans,trans_error = align(second_xyz,first_xyz)
        
        second_xyz_aligned = rot * second_xyz + trans
        
        first_stamps = list(first_list)
        first_stamps.sort()
        first_xyz_full = np.matrix([[float(value) for value in first_list[b][0:3]] for b in first_stamps]).transpose()
        
        second_stamps = list(second_list)
        second_stamps.sort()
        second_xyz_full = np.matrix([[float(value)*float(scale) for value in second_list[b][0:3]] for b in second_stamps]).transpose()
        second_xyz_full_aligned = rot * second_xyz_full + trans
        rmse_t = np.sqrt(np.dot(trans_error,trans_error) / len(trans_error))
        if verbose:
            print ("compared_pose_pairs %d pairs"%(len(trans_error)))

            print ("absolute_translational_error.rmse %f m"%rmse_t)
            print ("absolute_translational_error.mean %f m"%np.mean(trans_error))
            print ("absolute_translational_error.median %f m"%np.median(trans_error))
            print ("absolute_translational_error.std %f m"%np.std(trans_error))
            print ("absolute_translational_error.min %f m"%np.min(trans_error))
            print ("absolute_translational_error.max %f m"%np.max(trans_error))
        else:
            print ("%f"%np.sqrt(numpy.dot(trans_error,trans_error) / len(trans_error)))
            
        if save:
            file = open(save,"w")
            file.write("\n".join(["%f "%stamp+" ".join(["%f"%d for d in line]) for stamp,line in zip(second_stamps,second_xyz_full_aligned.transpose().A)]))
            file.close()

        if plot:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import matplotlib.pylab as pylab
            from matplotlib.patches import Ellipse
            fig = plt.figure()
            ax = fig.add_subplot(111)
            plot_traj(ax,first_stamps,first_xyz_full.transpose().A,'-',"black","ground truth")
            plot_traj(ax,second_stamps,second_xyz_full_aligned.transpose().A,'-',"blue","estimated")

            label="difference"
            for (a,b),(x1,y1,z1),(x2,y2,z2) in zip(matches,first_xyz.transpose().A,second_xyz_aligned.transpose().A):
                ax.plot([x1,x2],[y1,y2],'-',color="red",label=label)
                label=""
                
            ax.legend()
                
            ax.set_xlabel('x [m]')
            ax.set_ylabel('y [m]')
            plt.savefig(plot,dpi=90)
        if self.upload:
            print("---Uploading results---")
            import wandb

            metric_t = wandb.define_metric("Timestamp", summary=None, hidden=True)
            wandb.define_metric(
                "ate",
                summary="mean",
                goal="minimize",
                step_metric=metric_t,
            )

            for i in range(trans_error.shape[0]):
                wandb.log(
                    {
                        "Timestamp": first_stamps[i] - first_stamps[0],
                        "translational_error": trans_error[i],
                    }
                )
            wandb.run.summary["ate.RMSE"] = rmse_t

            return rmse_t
            


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="""
    Run evaluation of algorithm"""
    )
    parser.add_argument(
        "--experiment_name", help="Name for the experiment", default="test-python"
    )
    parser.add_argument(
        "--sequence_id",
        help="Id of the sequence to run on)",
        default="rgbd_dataset_freiburg1_floor",
    )
    args = parser.parse_args()

    dataset = (Kitti(args.sequence_id) if args.sequence_id in Kitti.sequences() else TumRgbd(args.sequence_id))
    evaluation = Evaluation(sequence=dataset, experiment_name=args.experiment_name)
    evaluation.evaluate()
