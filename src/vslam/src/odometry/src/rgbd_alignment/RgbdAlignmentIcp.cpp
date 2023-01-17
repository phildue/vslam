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

#include "IterativeClosestPoint.h"
#include "PlotAlignment.h"
#include "RgbdAlignmentIcp.h"
#include "lukas_kanade/lukas_kanade.h"
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
RgbdAlignmentIcp::RgbdAlignmentIcp(
  Solver::ShPtr solver, Loss::ShPtr loss, bool includePrior, bool initializeOnPrediction,
  const std::vector<double> & minGradient, double minDepth, double maxDepth, double maxDepthDiff,
  const std::vector<double> & maxPointsPart)
: RgbdAlignment(
    solver, loss, includePrior, initializeOnPrediction, minGradient, minDepth, maxDepth,
    maxPointsPart),
  _maxDepthDiff(maxDepthDiff)
{
}

least_squares::Problem::UnPtr RgbdAlignmentIcp::setupProblem(
  const Vec6d & twistInit, Frame::ConstShPtr from, Frame::ConstShPtr to, int level) const
{
  auto interestPoints = selectInterestPoints(from, level);

  if (_includePrior) {
    throw pd::Exception("Not Implemented!");

  } else {
    return std::make_unique<IterativeClosestPoint>(
      SE3d::exp(twistInit), from, to, interestPoints, level, _maxDepthDiff, _loss);
  }
}
std::vector<Vec2i> RgbdAlignmentIcp::selectInterestPoints(Frame::ConstShPtr frame, int level) const
{
  std::vector<Eigen::Vector2i> interestPoints;
  interestPoints.reserve(frame->width(level) * frame->height(level));
  forEachPixel(frame->depth(level), [&](int u, int v, double d) {
    if (
      frame->withinImage({u, v}, _distanceToBorder, level) && d >= _minDepth && d <= _maxDepth &&
      std::isfinite(frame->normal(v, u, level).x())) {
      interestPoints.emplace_back(u, v);
    }
  });

  //TODO is this faster/better than grid based subsampling?
  const size_t needCount = std::max<size_t>(
    100, size_t(frame->width(level) * frame->height(level) * _maxPointsPart[level]));
  std::vector<bool> mask(frame->width(level) * frame->height(level), false);
  std::vector<Eigen::Vector2i> subset;
  subset.reserve(interestPoints.size());
  if (needCount < interestPoints.size()) {
    while (subset.size() < needCount) {
      auto ip = interestPoints[random::U(0, interestPoints.size() - 1)];
      const size_t idx = ip.y() * frame->width(level) + ip.x();
      if (!mask[idx]) {
        subset.push_back(ip);
        mask[idx] = true;
      }
    }
    interestPoints = std::move(subset);
  }

  LOG_ODOM(INFO) << format(
    "Selected [{}] features at level [{}] with min gradient magnitude [{}]", interestPoints.size(),
    level, _minGradient2[level]);
  return interestPoints;
}

}  // namespace pd::vslam
