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
class GaussianPrior
{
  /*
      We want to estimate p(€|R) * p(€) with R being the residuals and € the pose.
      Assuming the samples are i.i.d and taking the log this becomes:
      log p(€) + sum_N log(€|R_n)
      The latter becomes the standard MLE and with NLS can be formulated to J^TWJ d€ + JWr(0)
      A log gaussian prior derived with respect to x becomes:
      S^-1(x - €_0) with S being the covariance and €_0 the mean.
      x is simply the current € + d€ so the optimal solution becomes:
      S^-1(€ + d€ - €_0) + J^TWJ d€ + JWr(0) = 0
      rearranged:
      (J^TWJ + S^-1)d€ = JWr(0) + S^-1(€ - €_0)
      so to apply the prior we simply add S^-1 to lhs of the normal equations
      and S^-1(€ - €_0) to rhs
    */
public:
  typedef std::shared_ptr<GaussianPrior> ShPtr;
  typedef std::unique_ptr<GaussianPrior> UnPtr;
  typedef std::shared_ptr<const GaussianPrior> ConstShPtr;
  typedef std::unique_ptr<const GaussianPrior> ConstUnPtr;

  GaussianPrior(const SE3d & se3Init, const Pose & prior)
  : _b(Vec6d::Zero()), _priorTwist(prior.twist()), _priorInformation(prior.twistCov().inverse())
  {
    setX(se3Init.log());
  }
  void setX(const Eigen::VectorXd & twist) { _b = _priorInformation * (_priorTwist - twist); }
  Mat<double, 6, 6> A() { return _priorInformation; }
  Vec6d b() { return _b; }

private:
  Vec6d _b;
  const Vec6d _priorTwist;
  const Mat<double, 6, 6> _priorInformation;
};

class LukasKanadeWithGaussianPrior : public least_squares::Problem
{
public:
  typedef std::shared_ptr<LukasKanadeWithGaussianPrior> ShPtr;
  typedef std::unique_ptr<LukasKanadeWithGaussianPrior> UnPtr;
  typedef std::shared_ptr<const LukasKanadeWithGaussianPrior> ConstShPtr;
  typedef std::unique_ptr<const LukasKanadeWithGaussianPrior> ConstUnPtr;

  LukasKanadeWithGaussianPrior(
    const SE3d & se3Init, const Pose & prior, Frame::ConstShPtr f0, Frame::ConstShPtr f1, int level,
    const std::vector<Eigen::Vector2i> & keyPoints, least_squares::Loss::ShPtr loss)
  : Problem(6),
    _f0(f0),
    _f1(f1),
    _level(level),
    _keyPoints(keyPoints),
    _prior(std::make_unique<GaussianPrior>(se3Init, prior)),
    _warp(std::make_shared<lukas_kanade::WarpSE3>(
      se3Init, f0->pcl(level, false), f0->width(level), f0->camera(level), f1->camera(level))),
    _lk(std::make_unique<lukas_kanade::InverseCompositional>(
      f0->intensity(level), f0->dIx(level), f0->dIy(level), f1->intensity(level), _warp, keyPoints,
      loss))
  {
  }
  void setX(const Eigen::VectorXd & x)
  {
    _lk->setX(x);
    _prior->setX(_lk->x());
  }
  void updateX(const Eigen::VectorXd & dx)
  {
    _lk->updateX(dx);
    _prior->setX(_lk->x());
  }
  Eigen::VectorXd x() const { return _lk->x(); }
  NormalEquations::ConstShPtr computeNormalEquations()
  {
    auto nePh = _lk->computeNormalEquations();
    //normalize by maximal photometric error to get error in similar state space
    constexpr double normalizer = 1.0 / (255.0 * 255.0);
    const MatXd A = nePh->A() * normalizer + _prior->A();
    const MatXd b = nePh->b() * normalizer + _prior->b();
    return std::make_shared<NormalEquations>(A, b, nePh->chi2(), nePh->nConstraints());
  }

private:
  const Frame::ConstShPtr _f0, _f1;
  const int _level;
  const std::vector<Eigen::Vector2i> _keyPoints;
  const GaussianPrior::UnPtr _prior;
  const lukas_kanade::WarpSE3::ShPtr _warp;
  const lukas_kanade::InverseCompositional::UnPtr _lk;
};

RgbdAlignment::RgbdAlignment(
  Solver::ShPtr solver, Loss::ShPtr loss, bool includePrior, bool initializeOnPrediction,
  const std::vector<double> & minGradient, double minDepth, double maxDepth,
  const std::vector<double> & maxPointsPart)
: _loss(loss),
  _solver(solver),
  _includePrior(includePrior),
  _initializeOnPrediction(initializeOnPrediction),
  _minDepth(minDepth),
  _maxDepth(maxDepth),
  _maxPointsPart(maxPointsPart)
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
}

PoseWithCovariance::UnPtr RgbdAlignment::align(Frame::ConstShPtr from, Frame::ConstShPtr to) const
{
  Vec6d twist = _initializeOnPrediction
                  ? algorithm::computeRelativeTransform(from->pose().SE3(), to->pose().SE3()).log()
                  : Vec6d::Zero();

  Mat<double, 6, 6> covariance = Mat<double, 6, 6>::Identity();

  for (int level = from->nLevels() - 1; level >= 0; level--) {
    TIMED_SCOPE(timerI, "align at level ( " + std::to_string(level) + " )");
    LOG_ODOM(INFO) << "Aligning at level: [" << level << "] image size: [" << from->width(level)
                   << "," << from->height(level) << "].";

    LOG_IMG("Image") << to->intensity(level);
    LOG_IMG("Template") << from->intensity(level);
    LOG_IMG("Depth") << from->depth(level);

    least_squares::Problem::ShPtr p = setupProblem(twist, from, to, level);

    least_squares::Solver::Results::ConstUnPtr r = _solver->solve(p);

    if (r->hasSolution()) {
      twist = r->solution();
      covariance = r->covariance();
      //TODO threshold on maximum speed ?
    }
    LOG_ODOM(INFO) << format(
      "Aligned with {} iterations: {} +- {}", r->iteration, twist.transpose(),
      covariance.diagonal().transpose());
  }
  //TODO how to convert the covariance to the covariance of the absolute pose?
  return std::make_unique<PoseWithCovariance>(SE3d::exp(twist) * from->pose().SE3(), covariance);
}

std::vector<Vec2i> RgbdAlignment::selectInterestPoints(Frame::ConstShPtr frame, int level) const
{
  std::vector<Eigen::Vector2i> interestPoints;
  interestPoints.reserve(frame->width(level) * frame->height(level));
  const MatXd gradientMagnitude =
    frame->dIx(level).array().pow(2) + frame->dIy(level).array().pow(2);
  forEachPixel(gradientMagnitude, [&](int u, int v, double p) {
    if (
      p >= _minGradient2[level] && _minDepth < frame->depth(level)(v, u) &&
      frame->depth(level)(v, u) < _maxDepth && frame->depth(level)(v - 1, u) > _minDepth &&
      frame->depth(level)(v - 1, u - 1) > _minDepth && frame->depth(level)(v + 1, u) > _minDepth &&
      frame->depth(level)(v + 1, u + 1) > _minDepth && frame->depth(level)(v, u + 1) > _minDepth) {
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
    return std::make_unique<LukasKanadeWithGaussianPrior>(
      SE3d::exp(twistInit), to->pose(), from, to, level, interestPoints, _loss);

  } else {
    return std::make_unique<lukas_kanade::InverseCompositional>(
      from->intensity(level), from->dIx(level), from->dIy(level), to->intensity(level),
      std::make_shared<lukas_kanade::WarpSE3>(
        SE3d::exp(twistInit), from->pcl(level, false), from->width(level), from->camera(level),
        to->camera(level)),
      interestPoints, _loss);
  }
}

}  // namespace pd::vslam