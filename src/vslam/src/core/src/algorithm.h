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
// Created by phil on 02.07.21.
//

#ifndef ALGORITHM_H__
#define ALGORITHM_H__
#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <functional>
#include <random>
#include <vector>

#include "Frame.h"
#include "Kernel2d.h"
#include "types.h"

namespace pd::vslam
{
namespace algorithm
{
template <typename Derived>
double bilinearInterpolation(const Eigen::MatrixBase<Derived> & mat, double x, double y)
{
  /*We want to interpolate P
        * http://supercomputingblog.com/graphics/coding-bilinear-interpolation/
        *
        * y2 |Q12    R2          Q22
        *    |
        * y  |       P
        *    |
        * y1 |Q11    R1          Q21
        *    _______________________
        *    x1      x            x2
        * */
  const double x1 = std::floor(x);
  const double x2 = std::ceil(x);
  const double y1 = std::floor(y);
  const double y2 = std::ceil(y);
  const double Q11 = mat(static_cast<int>(y1), static_cast<int>(x1));
  const double Q12 = mat(static_cast<int>(y1), static_cast<int>(x2));
  const double Q21 = mat(static_cast<int>(y2), static_cast<int>(x1));
  const double Q22 = mat(static_cast<int>(y2), static_cast<int>(x2));
  double R1 = 0, R2 = 0;

  if (x2 == x1) {
    R1 = Q11;
    R2 = Q12;
  } else {
    R1 = ((x2 - x) / (x2 - x1)) * Q11 + ((x - x1) / (x2 - x1)) * Q21;
    R2 = ((x2 - x) / (x2 - x1)) * Q12 + ((x - x1) / (x2 - x1)) * Q22;
  }

  double P = 0;
  if (y2 == y1) {
    P = R1;
  } else {
    // After the two R values are calculated, the value of P can finally be calculated by a weighted average of R1 and R2.
    P = ((y2 - y) / (y2 - y1)) * R1 + ((y - y1) / (y2 - y1)) * R2;
  }

  return P;
}
template <typename Derived>
double bilinearInterpolation(const Eigen::MatrixBase<Derived> & mat, const Eigen::Vector2d & xy)
{
  return bilinearInterpolation<Derived>(mat, xy.x(), xy.y());
}

template <class T, typename Derived>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> resize(
  const Eigen::MatrixBase<Derived> & mat, double scale)
{
  // TODO(phil) interpolation function as parameter
  if (scale == 1.0) {
    return mat;
  }

  const double scaleInv = 1.0 / scale;
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> res(
    static_cast<int>(mat.rows() * scale), static_cast<int>(mat.cols() * scale));
  for (int i = 0; i < res.rows(); i++) {
    for (int j = 0; j < res.cols(); j++) {
      res(i, j) = bilinearInterpolation(mat, j * scaleInv, i * scaleInv);
    }
  }
  return res;
}

template <typename Derived, typename Derived1>
void warpAffine(
  const Eigen::MatrixBase<Derived> & img, const Eigen::Matrix<Derived1, 3, 3> & warp,
  Eigen::MatrixBase<Derived> & imgWarped)
{
  const double cx = img.cols() / 2;
  const double cy = img.rows() / 2;
  for (int v = 0; v < imgWarped.rows(); v++) {
    for (int u = 0; u < imgWarped.cols(); u++) {
      const Eigen::Vector3d xy1(u - cx, v - cy, 1);
      const Eigen::Vector3d xy1Ref = warp.inverse() * xy1;
      Eigen::Vector2d uvRef;
      uvRef << xy1Ref.x() + cx, xy1Ref.y() + cy;
      if (
        1 < uvRef.x() && uvRef.x() < img.cols() - 1 && 1 < uvRef.y() &&
        uvRef.y() < img.rows() - 1) {
        imgWarped(v, u) = algorithm::bilinearInterpolation(img, uvRef.x(), uvRef.y());
      }
    }
  }
}

template <typename Derived, typename Derived1>
void shift(
  const Eigen::MatrixBase<Derived> & img, const Eigen::Matrix<Derived1, 2, 1> & shift,
  Eigen::MatrixBase<Derived> & imgWarped)
{
  for (int v = 0; v < imgWarped.rows(); v++) {
    for (int u = 0; u < imgWarped.cols(); u++) {
      Eigen::Vector2d uvRef;
      uvRef << u - shift.x(), v - shift.y();
      if (
        1 < uvRef.x() && uvRef.x() < img.cols() - 1 && 1 < uvRef.y() &&
        uvRef.y() < img.rows() - 1) {
        imgWarped(v, u) = algorithm::bilinearInterpolation(img, uvRef.x(), uvRef.y());
      }
    }
  }
}

template <typename Derived>
void insertionSort(std::vector<Derived> & v, Derived e)
{
  auto it = std::find_if(v.begin(), v.end(), [&](Derived i) { return e < i; });
  v.insert(it, e);
}

template <typename Derived>
Derived median(const std::vector<Derived> & v, bool isSorted = false)
{
  const int n = v.size();
  std::vector<Derived> vv;
  std::copy_if(
    v.begin(), v.end(), std::back_inserter(vv), [&](auto & v_) { return std::isfinite(v_); });
  //TODO use partial sort?
  if (!isSorted) {
    std::sort(vv.begin(), vv.end());
  }
  if (n % 2 == 0) {
    return (vv[n / 2 - 1] + vv[n / 2 + 1]) / 2;
  } else {
    return vv[n / 2];
  }
}
template <typename Derived>
Derived median(
  const Eigen::Matrix<Derived, Eigen::Dynamic, Eigen::Dynamic> & d, bool isSorted = false)
{
  return median<Derived>(std::vector<Derived>(&d(0), d.data() + d.cols() * d.rows()), isSorted);
}
template <typename Derived>
Derived median(const Eigen::Matrix<Derived, 1, Eigen::Dynamic> & d, bool isSorted = false)
{
  return median<Derived>(std::vector<Derived>(&d(0), d.data() + d.cols() * d.rows()), isSorted);
}
template <typename Derived>
Derived median(const Eigen::Matrix<Derived, Eigen::Dynamic, 1> & d, bool isSorted = false)
{
  return median<Derived>(std::vector<Derived>(&d(0), d.data() + d.cols() * d.rows()), isSorted);
}
template <typename Derived>
Eigen::Matrix<Derived, Eigen::Dynamic, Eigen::Dynamic> medianBlur(
  const Eigen::Matrix<Derived, -1, -1> & mat, int sizeX, int sizeY,
  std::function<bool(Derived)> mask)
{
  typedef int Idx;
  // TODO(phil) is this the most efficient way? add padding
  Eigen::Matrix<Derived, Eigen::Dynamic, Eigen::Dynamic> res(mat.rows(), mat.cols());
  res.setZero();
  const Idx kX_2 = static_cast<Idx>(sizeX / 2);
  const Idx kY_2 = static_cast<Idx>(sizeY / 2);

  for (Idx i = kY_2; i < res.rows() - kY_2; i++) {
    for (Idx j = kX_2; j < res.cols() - kX_2; j++) {
      std::vector<double> values;
      values.reserve(sizeX * sizeY);
      for (Idx ki = i - kY_2; ki < i - kY_2 + sizeY; ki++) {
        for (Idx kj = j - kX_2; kj < j - kX_2 + sizeX; kj++) {
          auto v = static_cast<double>(mat(ki, kj));
          if (!mask(v)) {
            values.push_back(v);
          }
        }
      }
      res(i, j) = values.size() > 0 ? (Derived)median(values) : 0.0;
    }
  }
  return res;
}

double rmse(const Eigen::MatrixXi & patch1, const Eigen::MatrixXi & patch2);
double sad(const Eigen::MatrixXi & patch1, const Eigen::MatrixXi & patch2);
Image resize(const Image & mat, double scale);
Eigen::MatrixXd resize(const Eigen::MatrixXd & mat, double scale);

Eigen::MatrixXd normalize(const Eigen::MatrixXd & mat);
Eigen::MatrixXd normalize(const Eigen::MatrixXd & mat, double min, double max);

/// Computes T01 from T0 and T1
Sophus::SE3d computeRelativeTransform(const Sophus::SE3d & t0, const Sophus::SE3d & t1);
Pose computeRelativeTransform(const Pose & t0, const Pose & t1);

MatXd computeF(const Mat3d & Kref, const Sophus::SE3d & Rtref2cur, const Mat3d & Kcur);
MatXd computeF(Frame::ConstShPtr frameRef, Frame::ConstShPtr frameCur);

}  // namespace algorithm

namespace transforms
{
Eigen::MatrixXd createdTransformMatrix2D(double x, double y, double angle);
double deg2rad(double deg);
double rad2deg(double rad);
Eigen::Quaterniond euler2quaternion(double rx, double ry, double rz);

}  // namespace transforms

namespace random
{
template <int dim>
struct Gaussian
{
  Matd<dim, dim> cov;
  Matd<dim, 1> mean;
};

double U(double min, double max);
uint64_t U(uint64_t min, uint64_t max);

int sign();
Eigen::VectorXd N(const Eigen::MatrixXd & cov);
}  // namespace random

namespace time
{
std::chrono::time_point<std::chrono::high_resolution_clock> to_time_point(Timestamp t);
}
}  // namespace pd::vslam
#endif  // ALGORITHM_H__
