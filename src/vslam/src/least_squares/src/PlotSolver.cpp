#include "PlotSolver.h"
using namespace pd::vslam::vis;
namespace pd::vslam::least_squares
{
void PlotLevenbergMarquardt::plot() const
{
  plt::figure();
  plt::subplot(1, 5, 1);
  plt::title("Squared Error $\\chi^2$");
  std::vector<double> chi2v(_chi2.data(), _chi2.data() + _nIterations);
  plt::named_plot("$\\chi^2$", chi2v);
  plt::xlabel("Iteration");
  plt::legend();

  plt::subplot(1, 5, 2);
  plt::title("Error Reduction $\\Delta \\chi^2$");
  std::vector<double> chi2predv(_chi2pred.data(), _chi2pred.data() + _nIterations);
  plt::named_plot("$\\Delta \\chi^2*$", chi2predv);
  std::vector<double> dChi2v(_dChi2.data(), _dChi2.data() + _nIterations);
  plt::named_plot("$\\Delta \\chi^2$", dChi2v);
  plt::xlabel("Iteration");
  plt::legend();

  plt::subplot(1, 5, 3);
  plt::title("Improvement Ratio $\\rho$");
  std::vector<double> rhov(_rho.data(), _rho.data() + _nIterations);
  plt::named_plot("$\\rho$", rhov);
  //plt::ylim(0.0,1.0);
  plt::xlabel("Iteration");
  plt::legend();

  plt::subplot(1, 5, 4);
  plt::title("Damping Factor $\\lambda$");
  std::vector<double> lambdav(_lambda.data(), _lambda.data() + _nIterations);
  plt::named_plot("$\\lambda$", lambdav);
  plt::xlabel("Iteration");

  plt::legend();
  plt::subplot(1, 5, 5);
  plt::title("Step Size $||\\Delta x||_2$");
  std::vector<double> stepsizev(_stepSize.data(), _stepSize.data() + _nIterations);
  plt::named_plot("$||\\Delta x||_2$", stepsizev);
  plt::xlabel("Iteration");
  plt::legend();
}
std::string PlotLevenbergMarquardt::csv() const { return ""; }

void PlotGaussNewton::plot() const
{
  plt::figure();
  plt::subplot(1, 3, 1);
  plt::title("Squared Error $\\chi^2$");
  std::vector<double> chi2v(_chi2.data(), _chi2.data() + _nIterations);
  plt::named_plot("$\\chi^2$", chi2v);
  plt::xlabel("Iteration");
  plt::legend();

  plt::subplot(1, 3, 2);
  plt::title("Error Reduction $\\Delta \\chi^2$");

  std::vector<double> dChi2v(_nIterations);
  dChi2v[0] = 0;
  for (int i = 1; i < _nIterations; i++) {
    dChi2v[i] = chi2v[i] - chi2v[i - 1];
  }
  plt::named_plot("$\\Delta \\chi^2$", dChi2v);
  plt::xlabel("Iteration");
  plt::legend();

  plt::legend();
  plt::subplot(1, 3, 3);
  plt::title("Step Size $||\\Delta x||_2$");
  std::vector<double> stepsizev(_stepSize.data(), _stepSize.data() + _nIterations);
  plt::named_plot("$||\\Delta x||_2$", stepsizev);
  plt::xlabel("Iteration");
  plt::legend();
}
std::string PlotGaussNewton::csv() const
{
  std::vector<double> dChi2(_nIterations);
  dChi2[0] = 0;
  for (int i = 1; i < _nIterations; i++) {
    dChi2[i] = _chi2[i] - _chi2[i - 1];
  }
  std::stringstream ss;
  ss << "Iteration,Squared Error,Error Reduction,Step Size\r\n";
  for (int i = 0; i < _nIterations; i++) {
    ss << _chi2(i) << "," << dChi2[i] << "," << _stepSize(i) << "\r\n";
  }
  return ss.str();
}
}  // namespace pd::vslam::least_squares