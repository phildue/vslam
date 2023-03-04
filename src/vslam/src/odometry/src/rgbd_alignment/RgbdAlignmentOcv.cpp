// Copyright 2022 Philipp.Duernay
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

#include <fmt/core.h>
#include <fmt/ostream.h>

#include <Eigen/Dense>
#include <opencv4/opencv2/calib3d.hpp>
#include <opencv4/opencv2/core/eigen.hpp>
#include <opencv4/opencv2/core/utility.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/rgbd.hpp>

#include "RgbdAlignmentOcv.h"
#include "utils/utils.h"
using fmt::format;
using fmt::print;
template <typename T>
struct fmt::formatter<T, std::enable_if_t<std::is_base_of_v<Eigen::DenseBase<T>, T>, char>>
: ostream_formatter
{
};
#define LOG_ODOM(level) CLOG(level, "odometry")
using namespace pd::vslam::least_squares;
namespace pd::vslam
{
RgbdAlignmentOcv::RgbdAlignmentOcv(
  bool initializeOnPrediction, int nLevels, const std::vector<double> & minGradient,
  double minDepth, double maxDepth, double maxDepthDiff, const std::vector<double> & maxPoints,
  const std::string & odometryType)
: RgbdAlignment(
    nullptr, nullptr, false, initializeOnPrediction, nLevels, minGradient, minDepth, maxDepth,
    maxDepthDiff, maxDepthDiff, maxPoints, 0.0),
  _odometryType(odometryType)
{
}

void RgbdAlignmentOcv::preprocessReference(Frame::ShPtr f) const {}
void RgbdAlignmentOcv::preprocessTarget(Frame::ShPtr f) const {}

Pose RgbdAlignmentOcv::align(Frame::ConstShPtr from, Frame::ConstShPtr to) const
{
  TIMED_SCOPE(timerF, "align");

  cv::Mat intensityFrom, depthFrom, K, intensityTo, depthTo;
  cv::eigen2cv(from->intensity(), intensityFrom);
  cv::eigen2cv(from->depth(), depthFrom);
  cv::eigen2cv(from->camera()->K(), K);
  cv::eigen2cv(to->intensity(), intensityTo);
  cv::eigen2cv(to->depth(), depthTo);
  intensityFrom.convertTo(intensityFrom, CV_8UC1);
  intensityTo.convertTo(intensityTo, CV_8UC1);
  depthFrom.convertTo(depthFrom, CV_32FC1);
  depthTo.convertTo(depthTo, CV_32FC1);

  cv::Mat Rt;
  cv::Mat Rtinit;
  if (_initializeOnPrediction) {
    SE3d relativePose = algorithm::computeRelativeTransform(from->pose().SE3(), to->pose().SE3());
    cv::eigen2cv(relativePose.matrix(), Rtinit);
  }
  auto odom = cv::rgbd::Odometry::create(_odometryType);
  odom->setCameraMatrix(K);
  //odom->setMaxPointsPart(1.0);
  //odom->setMinGradientMagnitudes({10, 10, 10, 10});
  //odom->setIterationCounts({30, 30, 30, 30});
  odom->compute(intensityFrom, depthFrom, cv::Mat(), intensityTo, depthTo, cv::Mat(), Rt, Rtinit);
  Mat<double, 4, 4> Rtout;
  cv::cv2eigen(Rt, Rtout);
  return Pose((SE3d(Rtout) * from->pose().SE3()).log(), Mat6d::Identity());
}

Pose RgbdAlignmentOcv::align(const Frame::VecConstShPtr & from, Frame::ConstShPtr to) const
{
  throw pd::Exception(
    "Not implemented: RgbdAlignmentOcv::align(Frame::VecConstShPtr, Frame::ConstShPtr)");
}

}  // namespace pd::vslam
