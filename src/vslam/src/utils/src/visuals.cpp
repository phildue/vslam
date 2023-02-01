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
  return drawMat(
    (algorithm::normalize(mat) * std::numeric_limits<image_value_t>::max()).cast<image_value_t>());
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

}  // namespace pd::vslam::vis
