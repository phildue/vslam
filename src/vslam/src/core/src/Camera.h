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
// Created by phil on 30.06.21.
//

#ifndef VSLAM_CAMERA_H
#define VSLAM_CAMERA_H

#include <Eigen/Dense>
#include <memory>

#include "types.h"
namespace pd::vslam
{
class Camera
{
public:
  using ConstShPtr = std::shared_ptr<const Camera>;
  using ShPtr = std::shared_ptr<Camera>;
  using Ptr = std::unique_ptr<Camera>;
  typedef std::vector<ConstShPtr> ConstShPtrVec;

  Camera(double f, double cx, double cy);
  Camera(double fx, double fy, double cx, double cy);

  Eigen::Vector2d camera2image(const Eigen::Vector3d & pCamera) const;
  Eigen::Vector3d image2camera(const Eigen::Vector2d & pImage, double depth = 1.0) const;
  Eigen::Vector3d image2ray(const Eigen::Vector2d & pImage) const;
  void resize(double s);

  const double & focalLength() const { return _K(0, 0); }
  const double & fx() const { return _K(0, 0); }
  const double & fy() const { return _K(1, 1); }

  Eigen::Vector2d principalPoint() const { return {_K(0, 2), _K(1, 2)}; }
  const Eigen::Matrix3d & K() const { return _K; }
  const Eigen::Matrix3d & Kinv() const { return _Kinv; }
  ShPtr static resize(ConstShPtr cam, double s);

private:
  Eigen::Matrix<double, 3, 3> _K;     //< Intrinsic camera matrix
  Eigen::Matrix<double, 3, 3> _Kinv;  //< Intrinsic camera matrix inverted
};
}  // namespace pd::vslam

#endif  //VSLAM_CAMERA_H
