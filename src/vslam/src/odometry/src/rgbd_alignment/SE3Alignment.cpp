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

#include "SE3Alignment.h"
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
/*
We expect the new pose not be too far away from a prediction.
Namely we expect it to be normally distributed around the prediction ( mean ) with some uncertainty ( covariance ).
*/
class MotionPrior : public Prior
{
public:
  MotionPrior(const PoseWithCovariance & predictedPose, const PoseWithCovariance & referencePose)
  : Prior(),
    _xPred((predictedPose.pose() * referencePose.pose().inverse()).log()),
    _information(predictedPose.cov().inverse())
  {
    LOG_ODOM(DEBUG) << "Prior: " << _xPred.transpose() << " \nInformation:\n " << _information;
  }

  void apply(NormalEquations::ShPtr ne, const Eigen::VectorXd & x) const override
  {
    //normalize by maximal photometric error
    constexpr double normalizer = 1.0 / (255.0 * 255.0);
    ne->A().noalias() = ne->A() * normalizer;
    ne->b().noalias() = ne->b() * normalizer;

    ne->A().noalias() += _information;
    ne->b().noalias() += _information * (_xPred - x);
    LOG_ODOM(DEBUG) << format(
      "Prior deviation: {}\n Weighted: {}\n", (_xPred - x).transpose(),
      (_information * (_xPred - x)).transpose());
  }

private:
  Eigen::VectorXd _xPred;
  Eigen::MatrixXd _information;
};

SE3Alignment::SE3Alignment(
  double minGradient, Solver::ShPtr solver, Loss::ShPtr loss, bool includePrior,
  bool initializeOnPrediction)
: _minGradient2(minGradient * minGradient),
  _loss(loss),
  _solver(solver),
  _includePrior(includePrior),
  _initializeOnPrediction(initializeOnPrediction)
{
  Log::get("odometry");
}

PoseWithCovariance::UnPtr SE3Alignment::align(Frame::ConstShPtr from, Frame::ConstShPtr to) const
{
  auto prior =
    _includePrior
      ? least_squares::Prior::ShPtr(std::make_shared<MotionPrior>(to->pose(), from->pose()))
      : least_squares::Prior::ShPtr(std::make_shared<least_squares::NoPrior>());
  PoseWithCovariance::UnPtr pose = _initializeOnPrediction
                                     ? std::make_unique<PoseWithCovariance>(to->pose())
                                     : std::make_unique<PoseWithCovariance>(from->pose());
  for (int level = from->nLevels() - 1; level >= 0; level--) {
    TIMED_SCOPE(timerI, "align at level ( " + std::to_string(level) + " )");
    LOG_ODOM(INFO) << "Aligning " << level << " image size: [" << from->width(level) << ","
                   << from->height(level) << "].";

    LOG_IMG("Image") << to->intensity(level);
    LOG_IMG("Template") << from->intensity(level);
    LOG_IMG("Depth") << from->depth(level);

    auto w = std::make_shared<lukas_kanade::WarpSE3>(
      pose->pose(), from->pcl(level, false), from->width(level), from->camera(level),
      to->camera(level), from->pose().pose());

    std::vector<Eigen::Vector2i> interestPoints;
    interestPoints.reserve(from->width(level) * from->height(level));
    const MatXd gradientMagnitude =
      from->dIx(level).array().pow(2) + from->dIy(level).array().pow(2);
    forEachPixel(gradientMagnitude, [&](int u, int v, double p) {
      double s = 1.0 / std::pow(0.5, level);
      if (
        p >= _minGradient2 &&
        from->depth()(static_cast<int>(v * s), static_cast<int>(u * s)) > 0.0) {
        interestPoints.emplace_back(u, v);
      }
    });

    auto lk = std::make_shared<lukas_kanade::InverseCompositional>(
      from->intensity(level), from->dIx(level), from->dIy(level), to->intensity(level), w,
      interestPoints, _loss, prior);

    auto results = _solver->solve(lk);
    LOG_ODOM(INFO) << format(
      "Aligned with {} iterations: {} +- {}", results->iteration, pose->twist().transpose(),
      pose->twistCov().diagonal().transpose());
    if (results->iteration >= 1) {
      pose = std::make_unique<PoseWithCovariance>(
        w->poseCur(), results->normalEquations[results->iteration]->A().inverse());
    }
  }
  return pose;
}
PoseWithCovariance::UnPtr SE3Alignment::align(
  const Frame::VecConstShPtr & from, Frame::ConstShPtr to) const
{
  PoseWithCovariance::UnPtr pose =
    _initializeOnPrediction ? std::make_unique<PoseWithCovariance>(to->pose())
                            : std::make_unique<PoseWithCovariance>(from[from.size() - 1]->pose());
  for (int level = from[0]->nLevels() - 1; level >= 0; level--) {
    TIMED_SCOPE(timerI, "align at level ( " + std::to_string(level) + " )");

    std::vector<std::shared_ptr<lukas_kanade::InverseCompositional>> frames;
    std::vector<std::shared_ptr<lukas_kanade::WarpSE3>> warps;

    for (const auto & f : from) {
      auto prior =
        _includePrior
          ? least_squares::Prior::ShPtr(std::make_shared<MotionPrior>(to->pose(), f->pose()))
          : least_squares::Prior::ShPtr(std::make_shared<least_squares::NoPrior>());

      auto w = std::make_shared<lukas_kanade::WarpSE3>(
        pose->pose(), f->pcl(level), f->width(level), f->camera(level), to->camera(level),
        f->pose().pose());

      std::vector<Eigen::Vector2i> interestPoints;
      interestPoints.reserve(f->width(level) * f->height(level));
      const MatXd gradientMagnitude = f->dIx(level).array().pow(2) + f->dIy(level).array().pow(2);
      forEachPixel(gradientMagnitude, [&](int u, int v, double p) {
        if (p >= _minGradient2 && f->depth(level)(v, u) > 0.0) {
          interestPoints.emplace_back(u, v);
        }
      });

      auto lk = std::make_shared<lukas_kanade::InverseCompositional>(
        f->intensity(level), f->dIx(level), f->dIy(level), to->intensity(level), w, interestPoints,
        _loss, prior);

      frames.push_back(lk);
      warps.push_back(w);
    }
    auto lk = std::make_shared<lukas_kanade::InverseCompositionalStacked>(frames);

    auto results = _solver->solve(lk);
    LOG_ODOM(INFO) << format(
      "Aligned with {} iterations: {} +- {}", results->iteration, pose->twist().transpose(),
      pose->twistCov().norm());
    if (results->iteration >= 1) {
      pose = std::make_unique<PoseWithCovariance>(
        warps[0]->poseCur(), results->normalEquations[results->iteration]->A().inverse());
    }
  }
  return pose;
}

}  // namespace pd::vslam
