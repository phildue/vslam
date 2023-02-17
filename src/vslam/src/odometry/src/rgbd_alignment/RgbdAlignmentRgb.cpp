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

#include "MotionPrior.h"
#include "PlotAlignment.h"
#include "RgbdAlignmentRgb.h"
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
RgbdAlignmentRgb::RgbdAlignmentRgb(
  Solver::ShPtr solver, Loss::ShPtr loss, bool includePrior, bool initializeOnPrediction,
  const std::vector<double> & minGradient, double minDepth, double maxDepth, double minDepthDiff,
  double maxDepthDiff, const std::vector<double> & maxPointsPart)
: RgbdAlignment(
    solver, loss, includePrior, initializeOnPrediction, minGradient, minDepth, maxDepth,
    minDepthDiff, maxDepthDiff, maxPointsPart)
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

Pose RgbdAlignmentRgb::align(Frame::ConstShPtr from, Frame::ConstShPtr to) const
{
  TIMED_SCOPE(timerF, "align");

  Vec6d twist = _initializeOnPrediction ? to->pose().SE3().log() : from->pose().SE3().log();

  Mat<double, 6, 6> covariance = Mat<double, 6, 6>::Identity();
  PlotAlignment::ShPtr plot = PlotAlignment::make(to->t());
  for (int level = from->nLevels() - 1; level >= 0; level--) {
    //TIMED_SCOPE(timerI, "align at level ( " + std::to_string(level) + " )");

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
      "Aligned at level [{}] with [{}] iterations: {} +- {}, Reason: [{}]. Solution is {}.", level,
      r->iteration, twist.transpose(), covariance.diagonal().cwiseSqrt().transpose(),
      to_string(r->convergenceCriteria), r->hasSolution() ? "valid" : "not valid");
    plot << PlotAlignment::Entry({level, r});
  }
  LOG_IMG("Alignment") << plot;

  return Pose(twist, covariance);
}

std::vector<Vec2i> RgbdAlignmentRgb::selectInterestPoints(Frame::ConstShPtr frame, int level) const
{
  std::vector<Eigen::Vector2i> interestPoints;
  interestPoints.reserve(frame->width(level) * frame->height(level));
  const MatXd gradientMagnitude =
    frame->dIdx(level).array().pow(2) + frame->dIdy(level).array().pow(2);
  const auto & depth = frame->depth(level);
  forEachPixel(gradientMagnitude, [&](int u, int v, double p) {
    if (
      frame->withinImage({u, v}, _distanceToBorder, level) && p >= _minGradient2[level] &&
      _minDepth < depth(v, u) && depth(v, u) < _maxDepth) {
      interestPoints.emplace_back(u, v);
    }
  });

  interestPoints = uniformSubselection(frame, interestPoints, level);

  LOG_ODOM(DEBUG) << format(
    "Selected [{}] features at level [{}] in frame[{}]", interestPoints.size(), level, frame->id());
  return interestPoints;
}

least_squares::Problem::UnPtr RgbdAlignmentRgb::setupProblem(
  const Vec6d & twist, Frame::ConstShPtr from, Frame::ConstShPtr to, int level) const
{
  auto interestPoints = selectInterestPoints(from, level);

  auto warp = std::make_shared<lukas_kanade::WarpSE3>(
    SE3d::exp(twist), to->depth(level), from->pcl(level, false), from->width(level),
    from->camera(level), to->camera(level), from->pose().SE3(), _minDepthDiff, _maxDepthDiff);
  least_squares::Problem::UnPtr lk = std::make_unique<lukas_kanade::InverseCompositional>(
    from->intensity(level), from->dIdx(level), from->dIdy(level), to->intensity(level), warp,
    interestPoints, _loss);

  return std::make_unique<PriorRegularizedLeastSquares>(
    SE3d::exp(twist), to->pose(), std::move(lk));
}

}  // namespace pd::vslam
