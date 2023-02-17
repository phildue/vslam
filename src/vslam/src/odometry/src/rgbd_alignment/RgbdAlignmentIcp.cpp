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
  const std::vector<double> & minGradient, double minDepth, double maxDepth, double minDepthDiff,
  double maxDepthDiff, const std::vector<double> & maxPointsPart)
: RgbdAlignment(
    solver, loss, includePrior, initializeOnPrediction, minGradient, minDepth, maxDepth,
    minDepthDiff, maxDepthDiff, maxPointsPart)
{
}

least_squares::Problem::UnPtr RgbdAlignmentIcp::setupProblem(
  const Vec6d & twistInit, Frame::ConstShPtr from, Frame::ConstShPtr to, int level) const
{
  auto interestPoints = selectInterestPoints(from, level);

  return std::make_unique<IterativeClosestPoint>(
    SE3d::exp(twistInit), from, to, interestPoints, level, _maxDepthDiff, _loss);
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

  interestPoints = uniformSubselection(frame, interestPoints, level);

  LOG_ODOM(DEBUG) << format(
    "Selected [{}] features at level [{}] in frame[{}]", interestPoints.size(), level, frame->id());
  return interestPoints;
}

}  // namespace pd::vslam
