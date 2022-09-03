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

//
// Created by phil on 07.08.21.
//
#include <Eigen/Dense>
#include <experimental/filesystem>
#include <filesystem>
#include <opencv2/imgproc.hpp>
#include <opencv4/opencv2/core/eigen.hpp>
#include <opencv4/opencv2/highgui.hpp>
namespace fs = std::experimental::filesystem;
#include "visuals.h"
namespace pd::vslam::vis
{
cv::Mat drawAsImage(const Eigen::MatrixXd & mat)
{
  return drawMat((algorithm::normalize(mat) * 255).cast<uint8_t>());
}
cv::Mat drawMat(const Image & matEigen)
{
  cv::Mat mat;
  cv::eigen2cv(matEigen, mat);

  return mat;
}

void Histogram::plot() const
{
  const double minH = _h.minCoeff();
  const double maxH = _h.maxCoeff();
  const double range = maxH - minH;
  const double binSize = range / static_cast<double>(_nBins);
  std::vector<int> bins(_nBins, 0);
  std::vector<std::string> ticksS(_nBins);
  std::vector<int> ticks(_nBins);
  for (int i = 0; i < _nBins; i++) {
    ticksS[i] = std::to_string(i * binSize + minH);
    ticks[i] = i;
  }
  for (int i = 0; i < _h.rows(); i++) {
    if (std::isfinite(_h(i))) {
      auto idx = static_cast<int>(std::floor(((_h(i) - minH) / binSize)));
      if (idx < _nBins) {
        bins[idx]++;
      }
    }
  }
  for (int i = 0; i < _nBins; i++) {
    std::cout << minH + i * binSize << " :" << bins[i] << std::endl;
  }
  //plt::figure();
  //plt::title(_title.c_str());
  //std::vector<double> hv(_h.data(),_h.data()+_h.rows());
  //plt::hist(hv);
  //plt::bar(bins);
  //plt::xticks(ticks,ticksS);
}

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
std::string PlotGaussNewton::csv() const { return ""; }

}  // namespace pd::vslam::vis
