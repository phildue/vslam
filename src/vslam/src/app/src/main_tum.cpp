

#include <opencv2/highgui.hpp>

#include "vslam/vslam.h"
using namespace vslam;

int main(int UNUSED(argc), char ** UNUSED(argv))
{
  const std::string experimentId = "test_c++";
  auto dl = std::make_unique<evaluation::tum::DataLoader>(
    "/mnt/dataset/tum_rgbd/", "rgbd_dataset_freiburg2_desk");

  const std::string outPath = format("{}/algorithm_results/{}", dl->datasetPath(), experimentId);
  const std::string trajectoryAlgoPath = format("{}-algo.txt", outPath, dl->sequenceId());
  const int tRmse = 25;
  log::initialize(outPath);

  const std::map<std::string, double> params = {
    {"nLevels", 4.0},           {"weightPrior", 0.0},         {"minGradientIntensity", 5},
    {"minGradientDepth", 0.01}, {"maxGradientDepth", 0.3},    {"maxDepth", 5.0},
    {"maxIterations", 100},     {"minParameterUpdate", 1e-4}, {"maxErrorIncrease", 5.0}};

  auto directIcp = std::make_shared<DirectIcp>(dl->cam(), params);

  Trajectory::ShPtr traj = std::make_shared<Trajectory>();
  const size_t fEnd = 200;  //dl->timestamps().size();
  Pose motion;
  Pose pose;
  for (size_t fId = 0; fId < fEnd; fId++) {
    try {
      print(
        "{}/{}: {} m, {:.3f}°\n", fId, fEnd, pose.translation().transpose(),
        pose.totalRotationDegrees());

      const cv::Mat img = dl->loadIntensity(fId);
      const cv::Mat depth = dl->loadDepth(fId);

      cv::imshow("Frame", colorizedRgbd(img, depth));
      cv::waitKey(1);

      motion = directIcp->computeEgomotion(img, depth, motion);
      pose = motion * pose;
      traj->append(dl->timestamps()[fId], pose.inverse());

    } catch (const std::runtime_error & e) {
      std::cerr << e.what() << std::endl;
    }
    if (fId > tRmse && fId % tRmse == 0) {
      evaluation::tum::writeTrajectory(*traj, trajectoryAlgoPath);
      evaluation::tum::runEvaluateRPEpy(trajectoryAlgoPath, dl->pathGt());
    };
  }
  evaluation::tum::writeTrajectory(*traj, trajectoryAlgoPath);
  evaluation::tum::runEvaluateRPEpy(trajectoryAlgoPath, dl->pathGt());
  evaluation::runPerformanceLogParserpy(format("{}/vslam.log", outPath));
}
