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
    LOG_ODOM(INFO) << "Prior: " << _xPred.transpose() << " \nInformation:\n " << _information;
  }

  void apply(NormalEquations::ShPtr ne, const Eigen::VectorXd & x) const override
  {
    //normalize by maximal photometric error
    constexpr double normalizer = 1.0 / (255.0 * 255.0);
    ne->A().noalias() = ne->A() * normalizer;
    ne->b().noalias() = ne->b() * normalizer;

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

SE3Alignment::InterestPointSelection::InterestPointSelection(
  const std::vector<double> & minGradient, double minDepth, double maxDepth, double maxPointsPart)
: _minDepth(minDepth), _maxDepth(maxDepth), _maxPointsPart(maxPointsPart)
{
  double sobelScale = 1. / 8.;
  const float sobelScale2_inv = 1.f / (float)(sobelScale * sobelScale);

  std::transform(
    minGradient.begin(), minGradient.end(), std::back_inserter(_minGradient2),
    [&](auto g) { return std::pow(g, 2) * sobelScale2_inv; });
}
std::vector<Eigen::Vector2i> SE3Alignment::InterestPointSelection::select(
  Frame::ConstShPtr frameRef, Frame::ConstShPtr frameCur, int level) const
{
  std::vector<Eigen::Vector2i> interestPoints;
  interestPoints.reserve(frameRef->width(level) * frameRef->height(level));
  const MatXd gradientMagnitude =
    frameRef->dIx(level).array().pow(2) + frameRef->dIy(level).array().pow(2);
  forEachPixel(gradientMagnitude, [&](int u, int v, double p) {
    if (
      p >= _minGradient2[level] && _minDepth < frameRef->depth(level)(v, u) &&
      frameRef->depth(level)(v, u) < _maxDepth) {
      interestPoints.emplace_back(u, v);
    }
  });
  LOG_ODOM(INFO) << "Selected interest points : " << interestPoints.size();

  interestPoints = selectRandomSubset(
    interestPoints, frameRef->height(level), frameCur->width(level), _maxPointsPart);
  LOG_ODOM(INFO) << "Sub-Selected interest points: " << interestPoints.size();
  return interestPoints;
}

std::vector<Eigen::Vector2i> SE3Alignment::InterestPointSelection::selectRandomSubset(
  const std::vector<Eigen::Vector2i> & interestPoints, int height, int width, double part) const
{
  const size_t minPointsCount = 1000;  // minimum point count (we can process them fast)
  const size_t needCount = std::max<size_t>(minPointsCount, size_t(width * height * part));
  std::vector<bool> mask(width * height, false);
  std::vector<Eigen::Vector2i> subset;
  subset.reserve(interestPoints.size());
  if (needCount < interestPoints.size()) {
    while (subset.size() < needCount) {
      auto ip = interestPoints[random::U(0, interestPoints.size() - 1)];
      if (!mask[ip.y() * width + ip.x()]) {
        subset.push_back(ip);
        mask[ip.y() * width + ip.x()] = true;
      }
    }
    return subset;
  }
  return interestPoints;
}

SE3Alignment::SE3Alignment(
  double minGradient, Solver::ShPtr solver, Loss::ShPtr loss, bool includePrior,
  bool initializeOnPrediction)
: _minGradient(minGradient),
  _loss(loss),
  _solver(solver),
  _includePrior(includePrior),
  _initializeOnPrediction(initializeOnPrediction)
{
  std::vector<double> minGradients;
  for (int i = 0; i < 4; i++) {
    minGradients.push_back(minGradient);
  }

  _interestPointSelection = std::make_unique<InterestPointSelection>(minGradients);

  Log::get("odometry");
  LOG_IMG("ImageWarped");
  LOG_IMG("Residual");
  LOG_IMG("Weights");
  LOG_IMG("Image");
  LOG_IMG("Template");
  LOG_IMG("Depth");
}

PoseWithCovariance::UnPtr SE3Alignment::align(Frame::ConstShPtr from, Frame::ConstShPtr to) const
{
  Vec6d twist = Vec6d::Zero();
  Mat<double, 6, 6> covariance = Mat<double, 6, 6>::Identity();

  for (int level = from->nLevels() - 1; level >= 0; level--) {
    TIMED_SCOPE(timerI, "align at level ( " + std::to_string(level) + " )");
    LOG_ODOM(INFO) << "Aligning " << level << " image size: [" << from->width(level) << ","
                   << from->height(level) << "].";

    LOG_IMG("Image") << to->intensity(level);
    LOG_IMG("Template") << from->intensity(level);
    LOG_IMG("Depth") << from->depth(level);

    std::vector<Eigen::Vector2i> interestPoints = _interestPointSelection->select(from, to, level);
    //auto p =
    //  std::make_shared<LeastSquaresProblem>(SE3d::exp(twist), from, to, interestPoints, level);

    auto w = std::make_shared<lukas_kanade::WarpSE3>(
      SE3d::exp(twist), from->pcl(level, false), from->width(level), from->camera(level),
      to->camera(level));

    auto p = std::make_shared<lukas_kanade::InverseCompositional>(
      from->intensity(level), from->dIx(level), from->dIy(level), to->intensity(level), w,
      interestPoints, _loss);
    auto r = _solver->solve(p);
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
PoseWithCovariance::UnPtr SE3Alignment::align(
  const Frame::VecConstShPtr & from, Frame::ConstShPtr to) const
{
  PoseWithCovariance::UnPtr pose =
    _initializeOnPrediction ? std::make_unique<PoseWithCovariance>(to->pose())
                            : std::make_unique<PoseWithCovariance>(from[from.size() - 1]->pose());
  std::vector<double> minGradients = {_minGradient};

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

      std::vector<Eigen::Vector2i> interestPoints = _interestPointSelection->select(f, to, level);

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
        warps[0]->poseCur(), results->normalEquations[results->iteration - 1]->A().inverse());
    }
  }
  return pose;
}

}  // namespace pd::vslam
