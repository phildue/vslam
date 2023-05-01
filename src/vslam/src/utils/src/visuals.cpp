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

void imshow(const std::string & name, const Image & mat, int waitKey)
{
  cv::Mat matcv = drawMat(mat);
  matcv.convertTo(matcv, CV_8UC3);
  cv::imshow(name, matcv);
  if (waitKey > -1) {
    cv::waitKey(waitKey);
  }
}

cv::Mat colorMap(const cv::Mat & in /*CV_32FC1*/, int color_map)
{
  //thx to: https://stackoverflow.com/questions/28825520/is-there-something-like-matlabs-colorbar-for-opencv
  int num_bar_w = 30;
  int color_bar_w = 10;
  int vline = 10;
  cv::Mat input = in;

  cv::Mat win_mat(
    cv::Size(input.cols + num_bar_w + num_bar_w + vline, input.rows), CV_8UC3,
    cv::Scalar(255, 255, 255));

  //Input image to
  double Min, Max;
  cv::minMaxLoc(input, &Min, &Max);
  int max_int = ceil(Max);

  std::cout << " Min " << Min << " Max " << Max << std::endl;
  input.convertTo(input, CV_8UC3, 255.0 / (Max - Min), -255.0 * Min / (Max - Min));
  input.convertTo(input, CV_8UC3);

  cv::Mat M;
  cv::applyColorMap(input, M, color_map);

  M.copyTo(win_mat(cv::Rect(0, 0, input.cols, input.rows)));

  //Scale
  cv::Mat num_window(cv::Size(num_bar_w, input.rows), CV_8UC3, cv::Scalar(255, 255, 255));
  for (int i = 0; i <= max_int; i++) {
    int j = i * input.rows / max_int;
    cv::putText(
      num_window, std::to_string(i), cv::Point(5, num_window.rows - j - 5),
      cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1, 2, false);
  }

  //color bar
  cv::Mat color_bar(cv::Size(color_bar_w, input.rows), CV_8UC3, cv::Scalar(255, 255, 255));
  cv::Mat cb;
  for (int i = 0; i < color_bar.rows; i++) {
    for (int j = 0; j < color_bar_w; j++) {
      int v = 255 - 255 * i / color_bar.rows;
      color_bar.at<cv::Vec3b>(i, j) = cv::Vec3b(v, v, v);
    }
  }

  color_bar.convertTo(color_bar, CV_8UC3);
  cv::applyColorMap(color_bar, cb, color_map);
  num_window.copyTo(win_mat(cv::Rect(input.cols + vline + color_bar_w, 0, num_bar_w, input.rows)));
  cb.copyTo(win_mat(cv::Rect(input.cols + vline, 0, color_bar_w, input.rows)));
  return win_mat;
}
void Histogram::plot(matplot::figure_handle f)
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
