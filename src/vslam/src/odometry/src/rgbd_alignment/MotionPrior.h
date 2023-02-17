
#ifndef VSLAM_MOTION_PRIOR_H__
#define VSLAM_MOTION_PRIOR_H__
#include <core/core.h>
#include <least_squares/least_squares.h>
#include <lukas_kanade/lukas_kanade.h>
namespace pd::vslam
{
class MotionPrior
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
  typedef std::shared_ptr<MotionPrior> ShPtr;
  typedef std::unique_ptr<MotionPrior> UnPtr;
  typedef std::shared_ptr<const MotionPrior> ConstShPtr;
  typedef std::unique_ptr<const MotionPrior> ConstUnPtr;

  MotionPrior(const SE3d & se3Init, const Pose & prior);
  void setX(const Eigen::VectorXd & twist);
  Mat<double, 6, 6> A() { return _priorInformation; }
  Vec6d b() { return _b; }

private:
  Vec6d _b;
  Vec6d _priorTwist;
  SE3d _priorSE3;
  const Mat<double, 6, 6> _priorInformation;
};

class PriorRegularizedLeastSquares : public least_squares::Problem
{
public:
  typedef std::shared_ptr<PriorRegularizedLeastSquares> ShPtr;
  typedef std::unique_ptr<PriorRegularizedLeastSquares> UnPtr;
  typedef std::shared_ptr<const PriorRegularizedLeastSquares> ConstShPtr;
  typedef std::unique_ptr<const PriorRegularizedLeastSquares> ConstUnPtr;

  virtual ~PriorRegularizedLeastSquares() = default;
  PriorRegularizedLeastSquares(
    const SE3d & se3Init, const Pose & prior, least_squares::Problem::UnPtr p);
  void setX(const Eigen::VectorXd & x);
  void updateX(const Eigen::VectorXd & dx);
  Eigen::VectorXd x() const;
  least_squares::NormalEquations::UnPtr computeNormalEquations();

private:
  const MotionPrior::UnPtr _prior;
  const least_squares::Problem::UnPtr _p;
};

}  // namespace pd::vslam

#endif  //VSLAM_MOTION_PRIOR_H__