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
      op(u, v, mat(v, u));
    }
  }
}
}  // namespace pd::vslam

#endif
