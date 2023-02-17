
#include "MotionPrior.h"

namespace pd::vslam
{
MotionPrior::MotionPrior(const SE3d & se3Init, const Pose & prior)
: _b(Vec6d::Zero()),
  _priorTwist(prior.twist()),
  _priorSE3(prior.SE3()),
  _priorInformation(prior.twistCov().inverse())
{
  setX(se3Init.log());
}
void MotionPrior::setX(const Eigen::VectorXd & twist)
{
  _b = _priorInformation * (SE3d::exp(twist) * _priorSE3.inverse()).log();
  //_priorTwist = twist;
  //_priorSE3 = SE3d::exp(twist);
}
PriorRegularizedLeastSquares::PriorRegularizedLeastSquares(
  const SE3d & se3Init, const Pose & prior, least_squares::Problem::UnPtr p)
: Problem(6), _prior(std::make_unique<MotionPrior>(se3Init, prior)), _p(std::move(p))
{
  if (_p->nParameters() != 6) {
    throw pd::Exception("This class expects optimization on the 6 parameter egomotion twist!.");
  }
}
void PriorRegularizedLeastSquares::setX(const Eigen::VectorXd & x)
{
  _p->setX(x);
  _prior->setX(_p->x());
}
void PriorRegularizedLeastSquares::updateX(const Eigen::VectorXd & dx)
{
  _p->updateX(dx);
  _prior->setX(_p->x());
}
Eigen::VectorXd PriorRegularizedLeastSquares::x() const { return _p->x(); }
least_squares::NormalEquations::UnPtr PriorRegularizedLeastSquares::computeNormalEquations()
{
  auto ne = _p->computeNormalEquations();
  const MatXd A = ne->A() + _prior->A();
  const MatXd b = ne->b() + _prior->b();
  return std::make_unique<least_squares::NormalEquations>(
    A, b, ne->chi2() + _prior->b().squaredNorm(), ne->nConstraints());
}

}  // namespace pd::vslam