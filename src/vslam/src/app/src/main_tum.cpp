

#include <opencv2/highgui.hpp>

#include "vslam/vslam.h"
using namespace vslam;

int main(int UNUSED(argc), char ** UNUSED(argv))
{
  const std::string experimentId = "test_c++";
  auto dl =
    std::make_unique<tum::DataLoader>("/mnt/dataset/tum_rgbd/", "rgbd_dataset_freiburg1_teddy");

  const std::string trajectoryAlgoPath = format(
    "{}/algorithm_results/{}/{}-algo.txt", dl->datasetPath(), experimentId, dl->sequenceId());
  const int tRmse = 25;

  const std::map<std::string, double> params = {
    {"nLevels", 4.0},           {"weightPrior", 0.0},         {"minGradientIntensity", 5},
    {"minGradientDepth", 0.01}, {"maxGradientDepth", 0.3},    {"maxDepth", 5.0},
    {"maxIterations", 100},     {"minParameterUpdate", 1e-4}, {"maxErrorIncrease", 5.0}};

  auto weightFunction = std::make_shared<TDistributionBivariate>(5.0, Mat2d::Identity());

  auto directIcp = std::make_shared<DirectIcp>(dl->cam(), weightFunction, params);

  Trajectory::ShPtr traj = std::make_shared<Trajectory>();
  const size_t fEnd = dl->timestamps().size();
  SE3d motion;
  SE3d pose;
  for (size_t fId = 0; fId < fEnd; fId++) {
    try {
      print(
        "{}/{}: {} m, {:.3f}°\n", fId, fEnd, pose.translation().transpose(),
        pose.log().block(3, 0, 3, 1).norm() * 180.0 / M_PI);

      const cv::Mat img = dl->loadIntensity(fId);
      const cv::Mat depth = dl->loadDepth(fId);

      cv::imshow("Frame", colorizedRgbd(img, depth));
      cv::waitKey(1);

      motion = directIcp->computeEgomotion(img, depth, motion);
      pose = motion * pose;
      traj->append(dl->timestamps()[fId], Pose(pose.inverse(), Mat6d::Identity()));

    } catch (const std::runtime_error & e) {
      std::cerr << e.what() << std::endl;
    }
    if (fId > tRmse && fId % tRmse == 0) {
      tum::writeTrajectory(*traj, trajectoryAlgoPath);
      tum::runEvaluateRPEpy(trajectoryAlgoPath, dl->pathGt());
    };
  }
  tum::writeTrajectory(*traj, trajectoryAlgoPath);
  tum::runEvaluateRPEpy(trajectoryAlgoPath, dl->pathGt());
}
