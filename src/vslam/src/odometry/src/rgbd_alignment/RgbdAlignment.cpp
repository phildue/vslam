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

#include "LukasKanadeWithDepth.h"
#include "PlotAlignment.h"
#include "RgbdAlignment.h"
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
RgbdAlignment::RgbdAlignment(
  Solver::ShPtr solver, Loss::ShPtr loss, bool includePrior, bool initializeOnPrediction,
  const std::vector<double> & minGradient, double minDepth, double maxDepth, double minDepthDiff,
  double maxDepthDiff, const std::vector<double> & maxPointsPart)
: _loss(loss),
  _solver(solver),
  _includePrior(includePrior),
  _initializeOnPrediction(initializeOnPrediction),
  _minDepth(minDepth),
  _maxDepth(maxDepth),
  _minDepthDiff(minDepthDiff),
  _maxDepthDiff(maxDepthDiff),
  _maxPointsPart(maxPointsPart),
  _distanceToBorder(0)
{
  std::transform(
    minGradient.begin(), minGradient.end(), std::back_inserter(_minGradient2),
    [&](auto g) { return std::pow(g, 2); });
  Log::get("odometry");
  LOG_IMG("ImageWarped");
  LOG_IMG("Residual");
  LOG_IMG("Weights");
  LOG_IMG("Image");
  LOG_IMG("Template");
  LOG_IMG("Depth");
  LOG_IMG("Alignment");
}

Pose RgbdAlignment::align(Frame::ConstShPtr from, Frame::ConstShPtr to) const
{
  Vec6d twist = _initializeOnPrediction
                  ? algorithm::computeRelativeTransform(from->pose().SE3(), to->pose().SE3()).log()
                  : Vec6d::Zero();

  Mat<double, 6, 6> covariance = Mat<double, 6, 6>::Identity();
  PlotAlignment::ShPtr plot = PlotAlignment::make(to->t());
  for (int level = from->nLevels() - 1; level >= 0; level--) {
    TIMED_SCOPE(timerI, "align at level ( " + std::to_string(level) + " )");
    LOG_ODOM(INFO) << "Aligning at level: [" << level << "] image size: [" << from->width(level)
                   << "," << from->height(level) << "].";

    LOG_IMG("Image") << to->intensity(level);
    LOG_IMG("Template") << from->intensity(level);
    LOG_IMG("Depth") << from->depth(level);

    least_squares::Problem::ShPtr p = setupProblem(twist, from, to, level);
    least_squares::Solver::Results::ConstShPtr r = _solver->solve(p);

    if (r->hasSolution()) {
      twist = r->solution();
      covariance = r->covariance();
    }

    LOG_ODOM(INFO) << format(
      "Aligned with {} iterations: {} +- {}", r->iteration, twist.transpose(),
      covariance.diagonal().cwiseSqrt().transpose());

    plot << PlotAlignment::Entry({level, r});
  }
  LOG_IMG("Alignment") << plot;

  return Pose(twist, covariance);
}

std::vector<Vec2i> RgbdAlignment::selectInterestPoints(Frame::ConstShPtr frame, int level) const
{
  std::vector<Eigen::Vector2i> interestPoints;
  interestPoints.reserve(frame->width(level) * frame->height(level));
  const MatXd gradientMagnitude =
    frame->dIdx(level).array().pow(2) + frame->dIdy(level).array().pow(2);
  const auto & depth = frame->depth(level);
  forEachPixel(gradientMagnitude, [&](int u, int v, double p) {
    if (
      frame->withinImage({u, v}, _distanceToBorder, level) && p >= _minGradient2[level] &&
      _minDepth < depth(v, u) && depth(v, u) < _maxDepth && depth(v - 1, u) > _minDepth &&
      std::isfinite(frame->normal(v, u, level).x()) && depth(v - 1, u - 1) > _minDepth &&
      depth(v + 1, u) > _minDepth && depth(v + 1, u + 1) > _minDepth &&
      depth(v, u + 1) > _minDepth) {
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

least_squares::Problem::UnPtr RgbdAlignment::setupProblem(
  const Vec6d & twistInit, Frame::ConstShPtr from, Frame::ConstShPtr to, int level) const
{
  auto interestPoints = selectInterestPoints(from, level);

  if (_includePrior) {
    throw pd::Exception("Not Implemented!");

  } else {
    return std::make_unique<LukasKanadeWithDepth>(
      SE3d::exp(twistInit), from, to, interestPoints, level, 0.0, _maxDepthDiff, _loss);
  }
}

}  // namespace pd::vslam
