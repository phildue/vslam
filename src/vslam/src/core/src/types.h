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

#ifndef DIRECT_IMAGE_ALIGNMENT_TYPES_H
#define DIRECT_IMAGE_ALIGNMENT_TYPES_H

#include <Eigen/Dense>
#include <limits>
#include <sophus/se3.hpp>
namespace Eigen
{
typedef Eigen::Matrix<double, 6, 1> Vector6d;
}

namespace pd::vslam
{
typedef std::uint8_t image_value_t;
typedef Eigen::Matrix<image_value_t, Eigen::Dynamic, Eigen::Dynamic> Image;
typedef std::vector<Image> ImageVec;

typedef double depth_value_t;
typedef Eigen::Matrix<depth_value_t, Eigen::Dynamic, Eigen::Dynamic> DepthMap;
typedef std::vector<DepthMap> DepthMapVec;

typedef std::uint64_t Timestamp;
typedef Sophus::SE3d SE3d;
typedef Sophus::SE3d SE3;

typedef Eigen::Matrix<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic> MatXui8;
typedef Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> MatXi;
typedef std::vector<MatXi> MatXiVec;

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatXd;
typedef std::vector<MatXd> MatXdVec;

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> MatXf;
typedef Eigen::Matrix<double, 3, 3> Mat3d;
typedef Eigen::Matrix<double, 2, 2> Mat2d;

template <typename Derived, int nRows, int nCols>
using Mat = Eigen::Matrix<Derived, nRows, nCols>;

template <int nRows, int nCols>
using Matd = Eigen::Matrix<double, nRows, nCols>;

template <int nRows, int nCols>
using Matf = Eigen::Matrix<double, nRows, nCols>;

typedef Eigen::VectorXd VecXd;
typedef Eigen::Vector2d Vec2d;
typedef Eigen::Vector2i Vec2i;
typedef Eigen::Vector3d Vec3d;
typedef Eigen::Vector4d Vec4d;
typedef Eigen::Vector6d Vec6d;
typedef Eigen::Matrix<double, 6, 6> Mat6d;
typedef Eigen::Matrix<double, 12, 1> Vec12d;
typedef Eigen::Matrix<double, 12, 12> Mat12d;

typedef double num_t;
typedef Eigen::Matrix<num_t, Eigen::Dynamic, Eigen::Dynamic> MatX;
typedef Mat<num_t, 3, 3> Mat3;
typedef Mat<num_t, 2, 2> Mat2;
typedef Mat<num_t, -1, 1> VecX;
typedef Mat<num_t, 2, 1> Vec2;
typedef Mat<num_t, 3, 1> Vec3;
typedef Mat<num_t, 4, 1> Vec4;
typedef Mat<num_t, 6, 1> Vec6;
typedef Mat<num_t, 6, 6> Mat6;
typedef Mat<num_t, 12, 1> Vec12;
typedef Mat<num_t, 12, 12> Mat12;
typedef std::vector<MatX> MatXVec;

typedef std::uint64_t Timestamp;

}  // namespace pd::vslam

#endif  //DIRECT_IMAGE_ALIGNMENT_TYPES_H
