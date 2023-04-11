
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

#ifndef VSLAM_LINALG_H__
#define VSLAM_LINALG_H__
#include <Eigen/Dense>

#include "image_transform.h"
#include "macros.h"
#include "types.h"
namespace pd::vslam::linalg
{
template <typename Derived>
double variance(const Eigen::Matrix<Derived, -1, -1> & mat, double mean)
{
  double sum = 0;
  forEach(mat, [&](int UNUSED(x), int UNUSED(y), double v) { sum += std::pow(v - mean, 2); });
  return sum / (mat.rows() * mat.cols());
}
template <typename Derived>
double stddev(const Eigen::Matrix<Derived, -1, -1> & mat, double mean)
{
  return std::sqrt(variance(mat, mean));
}
template <typename Derived>
double variance(const Eigen::Matrix<Derived, -1, -1> & mat)
{
  return variance(mat, mat.mean());
}
template <typename Derived>
double stddev(const Eigen::Matrix<Derived, -1, -1> & mat)
{
  return stddev(mat, mat.mean());
}
double stddev(const std::vector<double> & vs);
double stddev(const std::vector<double> & vs, double mean);

}  // namespace pd::vslam::linalg
#endif