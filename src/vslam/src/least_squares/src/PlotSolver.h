#ifndef VSLAM_PLOT_SOLVER_H__
#define VSLAM_PLOT_SOLVER_H__

#include "core/core.h"
#include "utils/utils.h"
namespace pd::vslam::least_squares
{
class PlotGaussNewton : public vis::Plot
{
public:
  PlotGaussNewton(int nIterations, const Eigen::VectorXd & chi2, const Eigen::VectorXd & stepSize)
  : _chi2(chi2), _stepSize(stepSize), _nIterations(nIterations)
  {
  }

  void plot() const override;
  std::string csv() const override;

private:
  const Eigen::VectorXd _chi2;
  const Eigen::VectorXd _stepSize;
  const int _nIterations;
};

class PlotLevenbergMarquardt : public vis::Plot
{
public:
  PlotLevenbergMarquardt(
    int nIterations, const Eigen::VectorXd & chi2, const Eigen::VectorXd & dChi2,
    const Eigen::VectorXd & chi2pred, const Eigen::VectorXd & lambda,
    const Eigen::VectorXd & stepSize, const Eigen::VectorXd & rho)
  : _chi2(chi2),
    _chi2pred(chi2pred),
    _lambda(lambda),
    _stepSize(stepSize),
    _nIterations(nIterations),
    _dChi2(dChi2),
    _rho(rho)
  {
  }

  void plot() const override;
  std::string csv() const override;

private:
  const Eigen::VectorXd _chi2;
  const Eigen::VectorXd _chi2pred;
  const Eigen::VectorXd _lambda;
  const Eigen::VectorXd _stepSize;
  const int _nIterations;
  const Eigen::VectorXd _dChi2;
  const Eigen::VectorXd _rho;
};

}  // namespace pd::vslam::least_squares
#endif