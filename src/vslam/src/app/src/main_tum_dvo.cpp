
#include <fmt/chrono.h>
#include <fmt/core.h>
using fmt::format;
using fmt::print;
#include <utils/utils.h>

#include <filesystem>
namespace fs = std::filesystem;

#include <opencv2/rgbd.hpp>
#include <opencv4/opencv2/core/eigen.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/opencv.hpp>

#include "dvo/dense_tracking.h"

using namespace pd;
using namespace pd::vslam;
using namespace dvo;
using namespace dvo::core;

class Main
{
public:
  Main(int UNUSED(argc), char ** UNUSED(argv))
  {
    _dl =
      std::make_unique<tum::DataLoader>("/mnt/dataset/tum_rgbd/", "rgbd_dataset_freiburg2_desk");
    IntrinsicMatrix intrinsics = IntrinsicMatrix::create(
      _dl->cam()->fx(), _dl->cam()->fy(), _dl->cam()->principalPoint().x(),
      _dl->cam()->principalPoint().y());

    DenseTracker::Config tracker_cfg = DenseTracker::getDefaultConfig();
    std::cout << tracker_cfg << std::endl;
    _tracker = std::make_unique<DenseTracker>(intrinsics, tracker_cfg);
  }

  std::pair<Timestamp, std::unique_ptr<RgbdImagePyramid>> loadFrame(size_t fNo)
  {
    cv::Mat intensity =
      cv::imread(_dl->datasetPath() + "/" + _dl->pathsImage()[fNo], cv::IMREAD_GRAYSCALE);
    cv::Mat depth =
      cv::imread(_dl->datasetPath() + "/" + _dl->pathsDepth()[fNo], cv::IMREAD_ANYDEPTH) / 5000.0;
    intensity.convertTo(intensity, CV_32F);
    depth.convertTo(depth, CV_32F);

    return {_dl->timestamps()[fNo], std::make_unique<RgbdImagePyramid>(intensity, depth)};
  }

  void run()
  {
    Trajectory::ShPtr traj = std::make_shared<Trajectory>();
    std::unique_ptr<RgbdImagePyramid> frameRef;
    Eigen::Affine3d global;
    global.setIdentity();
    const std::string pathOut = format(
      "{}/algorithm_results/{}/{}-algo.txt", _dl->datasetPath(), "app_out", _dl->sequenceId());
    fs::create_directories(pathOut);
    print("Writing to: {}\n", pathOut);
    for (size_t fId = 50; fId < _dl->nFrames(); fId++) {
      std::cout << fId << "/" << _dl->nFrames() << std::endl;
      try {
        Eigen::Affine3d local;
        local.setIdentity();
        auto frame = loadFrame(fId);
        if (frameRef) {
          _tracker->match(*frameRef, *frame.second, local);
        }
        global = local.cast<double>() * global;
        traj->append(
          frame.first, Pose(SE3d(global.rotation(), global.translation()), Mat6d::Identity()));
        frameRef = std::move(frame.second);
      } catch (const std::runtime_error & e) {
        std::cerr << e.what() << std::endl;
      }
      utils::writeTrajectory(*traj, pathOut);
    }
  }

protected:
  tum::DataLoader::ConstUnPtr _dl;
  std::unique_ptr<DenseTracker> _tracker;
};

int main(int argc, char ** argv)
{
  Main m(argc, argv);
  m.run();
  return 0;
}