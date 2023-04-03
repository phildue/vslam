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

#ifndef VSLAM_CORE_IMAGE_TRANSFORM_H__
#define VSLAM_CORE_IMAGE_TRANSFORM_H__
#include <Eigen/Dense>
#include <map>
#include <opencv4/opencv2/calib3d.hpp>
#include <opencv4/opencv2/core/eigen.hpp>
#include <opencv4/opencv2/core/utility.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/rgbd.hpp>

#include "types.h"
namespace pd::vslam
{
template <typename Derived, typename Operation>
void forEachPixel(const Eigen::Matrix<Derived, -1, -1> & image, Operation op)
{
  //give option to parallelize?
  for (int v = 0; v < image.rows(); v++) {
    for (int u = 0; u < image.cols(); u++) {
      op(u, v, image(v, u));
    }
  }
}

template <typename Derived, typename Operation>
void forEach(const Eigen::Matrix<Derived, -1, -1> & mat, Operation op)
{
  //give option to parallelize?
  for (int v = 0; v < mat.rows(); v++) {
    for (int u = 0; u < mat.cols(); u++) {
      op(u, v);
    }
  }
}

template <typename Derived>
Mat<Derived, -1, -1> sobel(
  const Mat<Derived, -1, -1> & mat, int dx, int dy, int kernelSize, double scale = 1.0)
{
  cv::Mat mat_(mat.rows(), mat.cols(), CV_32F);
  cv::eigen2cv(mat, mat_);
  cv::Mat dId_;
  cv::Sobel(mat_, dId_, CV_32F, dx, dy, kernelSize, scale);
  Mat<Derived, -1, -1> dId;
  cv::cv2eigen(dId_, dId);
  return dId;
}

template <typename Derived>
Mat<Derived, -1, -1> scharr(
  const Mat<Derived, -1, -1> & mat, int dx, int dy, int kernelSize, double scale = 1.0)
{
  cv::Mat mat_(mat.rows(), mat.cols(), CV_32F);
  cv::eigen2cv(mat, mat_);
  cv::Mat dId_;
  cv::Scharr(mat_, dId_, CV_32F, dx, dy, kernelSize, scale);
  Mat<Derived, -1, -1> dId;
  cv::cv2eigen(dId_, dId);
  return dId;
}

}  // namespace pd::vslam

#endif
