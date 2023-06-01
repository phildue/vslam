

#include <opencv2/highgui.hpp>

#include "vslam/core.h"
#include "vslam/direct_icp.h"
#include "vslam/evaluation.h"
#include "vslam/utils.h"

using namespace vslam;

int main(int UNUSED(argc), char ** UNUSED(argv))
{
  const std::string experimentId = "test_c++";
  auto dl = std::make_unique<evaluation::tum::DataLoader>(
    "/mnt/dataset/tum_rgbd/", "rgbd_dataset_freiburg2_desk");

  const std::string outPath = format("{}/algorithm_results/{}", dl->datasetPath(), experimentId);
  const std::string trajectoryAlgoPath = format("{}-algo.txt", outPath, dl->sequenceId());
  const int tRmse = 200;
  log::initialize(outPath);

  auto directIcp = std::make_shared<DirectIcp>(DirectIcp::defaultParameters());

  Trajectory::ShPtr traj = std::make_shared<Trajectory>();
  const size_t fEnd = 100;  //dl->timestamps().size();
  Pose motion;
  Pose pose;
  cv::Mat img0 = dl->loadIntensity(0), depth0 = dl->loadDepth(0);
  traj->append(dl->timestamps()[0], pose);
  for (size_t fId = 0; fId < fEnd; fId++) {
    try {
      print(
        "{}/{}: {} m, {:.3f}°\n", fId, fEnd, pose.translation().transpose(),
        pose.totalRotationDegrees());

      const cv::Mat img = dl->loadIntensity(fId);
      const cv::Mat depth = dl->loadDepth(fId);

      cv::imshow("Frame", colorizedRgbd(img, depth));
      cv::waitKey(1);

      motion = directIcp->computeEgomotion(dl->cam(), img0, depth0, img, depth, motion);
      pose = motion * pose;
      traj->append(dl->timestamps()[fId], pose.inverse());
      img0 = img;
      depth0 = depth;

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
  evaluation::runPerformanceLogParserpy(format("{}/runtime.log", outPath));
}
