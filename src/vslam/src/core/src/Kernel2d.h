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

#ifndef VSLAM_IMAGE_KERNEL_H__
#define VSLAM_IMAGE_KERNEL_H__
#include <Eigen/Dense>

#include "types.h"
namespace pd::vslam
{
//TODO give argument for size
template <typename T>
class Kernel2d
{
public:
  static Eigen::Matrix<T, -1, -1> dY()
  {
    Eigen::Matrix<T, 3, 3> g;
    g << 0, -1, 0, 0, 0, 0, 0, +1, 0;
    return g;
  }

  static Eigen::Matrix<T, -1, -1> dX()
  {
    Eigen::Matrix<T, 3, 3> g;
    g << 0, 0, 0, -1, 0, 1, 0, 0, 0;
    return g;
  }

  static Eigen::Matrix<T, -1, -1> sobelY()
  {
    Eigen::Matrix<T, 3, 3> sobel;
    sobel << -1, -2, -1, 0, 0, 0, +1, +2, +1;
    return sobel;
  }
  static Eigen::Matrix<T, -1, -1> sobelX()
  {
    Eigen::Matrix<T, 3, 3> sobel;
    sobel << -1, 0, 1, -2, 0, 2, -1, 0, 1;
    return sobel;
  }
  static Eigen::Matrix<T, -1, -1> scharrY()
  {
    Eigen::Matrix<T, 3, 3> scharr;
    scharr << -3, -10, -3, 0, 0, 0, +3, +10, +3;
    return scharr;
  }
  static Eigen::Matrix<T, -1, -1> scharrX()
  {
    Eigen::Matrix<T, 3, 3> scharr;
    scharr << -3, 0, 3, -10, 0, 10, -3, 0, 3;
    return scharr;
  }
  static Eigen::Matrix<T, -1, -1> gaussian()
  {
    Eigen::Matrix<T, 3, 3> g;
    g << 1, 2, 1, 2, 4, 2, 1, 2, 1;
    return g;
  }
};
}  // namespace pd::vslam
#endif
