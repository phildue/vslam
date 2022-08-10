#include "SE3Alignment.h"
#include "utils/utils.h"

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
    _information(MatXd::Identity(6, 6)) {}

  void apply(NormalEquations::ShPtr ne, const Eigen::VectorXd & x) const override
  {
    const double normalizer = 1.0 / (255.0 * 255.0);                    //otherwise prior has no influence ?
    ne->A().noalias() = ne->A() * normalizer;
    ne->b().noalias() = ne->b() * normalizer;

    ne->A().noalias() += _information;
    ne->b().noalias() += _information * (_xPred - x);

    LOG_ODOM(DEBUG) << "Prior: " << _xPred.transpose() << " \nInformation:\n " << _information;
  }

private:
  Eigen::VectorXd _xPred;
  Eigen::MatrixXd _information;
};

SE3Alignment::SE3Alignment(
  double minGradient, Solver::ShPtr solver, Loss::ShPtr loss,
  bool includePrior)
: _minGradient2(minGradient * minGradient),
  _loss(loss),
  _solver(solver),
  _includePrior(includePrior)
{
  Log::get("odometry", ODOMETRY_CFG_DIR "/log/odometry.conf");
}


PoseWithCovariance::UnPtr SE3Alignment::align(
  FrameRgbd::ConstShPtr from,
  FrameRgbd::ConstShPtr to) const
{
  auto prior = _includePrior ? std::make_shared<MotionPrior>(to->pose(), from->pose()) : nullptr;
  PoseWithCovariance::UnPtr pose = std::make_unique<PoseWithCovariance>(to->pose());
  for (int level = from->nLevels() - 1; level >= 0; level--) {
    TIMED_SCOPE(timerI, "align at level ( " + std::to_string(level) + " )");
    LOG_ODOM(INFO) << "Aligning from: \n" << from->pose().pose().log().transpose() << " to " <<
      pose->pose().log().transpose() <<
      "\nat " << level << " image size: [" << from->width(level) << "," << from->height(level) <<
      "].";

    LOG_IMG("Image") << to->intensity(level);
    LOG_IMG("Template") << from->intensity(level);
    LOG_IMG("Depth") << from->depth(level);

    auto w = std::make_shared<lukas_kanade::WarpSE3>(
      pose->pose(), from->pcl(level, false), from->width(level),
      from->camera(level), to->camera(level), from->pose().pose());

    std::vector<Eigen::Vector2i> interestPoints;
    interestPoints.reserve(from->width(level) * from->height(level));
    const MatXd gradientMagnitude = from->dIx(level).array().pow(2) +
      from->dIy(level).array().pow(2);
    forEachPixel(
      gradientMagnitude, [&](int u, int v, double p)
      {
        double s = 1.0 / std::pow(0.5, level);
        if (p >= _minGradient2 && from->depth()((int)(v * s), (int)(u * s)) > 0.0) {
          interestPoints.emplace_back(u, v);
        }
      });

    auto lk = std::make_shared<lukas_kanade::InverseCompositional>(
      from->intensity(level), from->dIx(level), from->dIy(level),
      to->intensity(level), w, interestPoints,
      _loss, prior);

    auto results = _solver->solve(lk);
    auto covariance = results->normalEquations.at(results->iteration)->A().inverse();
    pose = std::make_unique<PoseWithCovariance>(w->poseCur(), covariance);

  }
  return pose;
}
PoseWithCovariance::UnPtr SE3Alignment::align(
  const std::vector<FrameRgbd::ConstShPtr> & from,
  FrameRgbd::ConstShPtr to) const
{
  PoseWithCovariance::UnPtr pose = std::make_unique<PoseWithCovariance>(to->pose());
  for (int level = from[0]->nLevels() - 1; level >= 0; level--) {
    TIMED_SCOPE(timerI, "align at level ( " + std::to_string(level) + " )");

    std::vector<std::shared_ptr<lukas_kanade::InverseCompositional>> frames;
    std::vector<std::shared_ptr<lukas_kanade::WarpSE3>> warps;

    for (const auto & f : from) {
      auto prior = _includePrior ? std::make_shared<MotionPrior>(to->pose(), f->pose()) : nullptr;

      auto w = std::make_shared<lukas_kanade::WarpSE3>(
        pose->pose(), f->pcl(level), f->width(level),
        f->camera(level), to->camera(level), f->pose().pose());

      std::vector<Eigen::Vector2i> interestPoints;
      interestPoints.reserve(f->width(level) * f->height(level));
      const MatXd gradientMagnitude = f->dIx(level).array().pow(2) + f->dIy(level).array().pow(2);
      forEachPixel(
        gradientMagnitude, [&](int u, int v, double p)
        {
          if (p >= _minGradient2 && f->depth(level)(v, u) > 0.0) {
            interestPoints.emplace_back(u, v);
          }
        });

      auto lk = std::make_shared<lukas_kanade::InverseCompositional>(
        f->intensity(level), f->dIx(level), f->dIy(level),
        to->intensity(level), w, interestPoints, _loss, prior);

      frames.push_back(lk);
      warps.push_back(w);

    }
    auto lk = std::make_shared<lukas_kanade::InverseCompositionalStacked>(frames);

    auto results = _solver->solve(lk);
    auto covariance = results->normalEquations.at(results->iteration)->A().inverse();
    pose = std::make_unique<PoseWithCovariance>(warps[0]->poseCur(), covariance);

  }
  return pose;
}


}
