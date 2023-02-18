
#include <fmt/chrono.h>
#include <fmt/core.h>
using fmt::format;
using fmt::print;
#include <vslam/vslam.h>

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
    tracker_cfg.InfluenceFuntionType = InfluenceFunctions::Unit;
    tracker_cfg.UseWeighting = false;
    std::cout << tracker_cfg << std::endl;
    _tracker = std::make_unique<DenseTracker>(intrinsics, tracker_cfg);

    auto solver = std::make_shared<least_squares::GaussNewton>(1e-7, 100, 1e-7, 1e-7, 1e9);
    auto loss =
      std::make_shared<least_squares::QuadraticLoss>(std::make_shared<least_squares::Scaler>());
    _rgbdAlignment = std::make_unique<RgbdAlignmentRgb>(
      solver, loss, false, false, 4, std::vector<double>({0, 0, 0, 0}), 0.1, 0.1, 0.1, 100.0,
      std::vector<double>({1.0, 1.0, 1.0, 1.0}));
  }

  void loadFramePair(size_t fNo)
  {
    {
      cv::Mat intensity =
        cv::imread(_dl->datasetPath() + "/" + _dl->pathsImage()[fNo], cv::IMREAD_GRAYSCALE);
      cv::Mat depth =
        cv::imread(_dl->datasetPath() + "/" + _dl->pathsDepth()[fNo], cv::IMREAD_ANYDEPTH) / 5000.0;
      intensity.convertTo(intensity, CV_32F);
      depth.convertTo(depth, CV_32F);
      _reference.reset(new RgbdImagePyramid(intensity, depth));
    }
    {
      cv::Mat intensity =
        cv::imread(_dl->datasetPath() + "/" + _dl->pathsImage()[fNo + 1], cv::IMREAD_GRAYSCALE);
      cv::Mat depth =
        cv::imread(_dl->datasetPath() + "/" + _dl->pathsDepth()[fNo + 1], cv::IMREAD_ANYDEPTH) /
        5000.0;
      intensity.convertTo(intensity, CV_32F);
      depth.convertTo(depth, CV_32F);
      _current.reset(new RgbdImagePyramid(intensity, depth));
    }
    _t = _dl->timestamps()[fNo + 1];
  }

  void run()
  {
    int level = 1;
    least_squares::NormalEquations::UnPtr ne;
    {
      Frame::ShPtr ref = _dl->loadFrame(0);
      Frame::ShPtr cur = _dl->loadFrame(1);
      ref->computePyramid(3);
      ref->computeIntensityDerivatives();
      ref->computePcl();
      cur->computePyramid(3);
      cur->computeIntensityDerivatives();
      cur->computePcl();

      auto lsp = _rgbdAlignment->setupProblem(SE3d().log(), ref, cur, level);
      ne = lsp->computeNormalEquations();

      std::cout << "VSLAM:" << ne->toString() << std::endl;
    }
    NormalEquationsLeastSquares ls;
    {
      loadFramePair(0);
      _reference->compute(3);
      _current->compute(3);

      _tracker->computeLeastSquaresEquationsInverseCompositional(
        _reference->level(level), _current->level(level), _tracker->intrinsics_[level],
        AffineTransform::Identity(), ls);

      std::cout << "DVO err=" << ls.error << "#:" << ls.num_constraints << " A=\n"
                << ls.A << " b=\n"
                << ls.b << std::endl;
    }
    std::cout << "Err: e=\n" << ls.error - ne->chi2() << std::endl;

    std::cout << "Diff: A=\n"
              << ls.A - ne->A().cast<float>() << "\n"
              << "|A-A| = " << (ls.A - ne->A().cast<float>()).norm() << std::endl;
    std::cout << "Diff: b=\n"
              << ls.b - ne->b().cast<float>() << "\n"
              << "|b-b| = " << (ls.b - ne->b().cast<float>()).norm() << std::endl;
  }

protected:
  tum::DataLoader::ConstUnPtr _dl;
  std::unique_ptr<DenseTracker> _tracker;
  RgbdAlignmentRgb::UnPtr _rgbdAlignment;
  boost::shared_ptr<dvo::core::RgbdImagePyramid> _current, _reference;
  Timestamp _t;
};

int main(int argc, char ** argv)
{
  Main m(argc, argv);
  m.run();
  return 0;
}