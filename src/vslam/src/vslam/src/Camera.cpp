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
#include "Camera.h"
namespace vslam
{
Eigen::Vector2d Camera::camera2image(const Eigen::Vector3d & p) const
{
  if (p.z() <= 0) {
    return {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()};
  }

  return {(fx() * p(0) / p(2)) + cx(), (fy() * p(1) / p(2)) + cy()};
}

Eigen::Vector3d Camera::image2camera(const Eigen::Vector2d & uv, double z) const
{
  return Eigen::Vector3d((uv(0) - cx()) / fx() * z, (uv(1) - cy()) / fy() * z, z);
}

Camera::Camera(double f, double cx, double cy) : Camera(f, f, cx, cy) {}

Camera::Camera(double fx, double fy, double cx, double cy)
: Camera(fx, fy, cx, cy, std::ceil(2 * cx), std::ceil(2 * cy))
{
}
Camera::Camera(double fx, double fy, double cx, double cy, int width, int height)
: _width(width), _height(height)
{
  _K << fx, 0, cx, 0, fy, cy, 0, 0, 1;
  _Kinv = _K.inverse();
}

Camera::ShPtr Camera::resize(Camera::ConstShPtr cam, double s)
{
  return std::make_shared<Camera>(
    cam->fx() * s, cam->fy() * s, cam->principalPoint().x() * s, cam->principalPoint().y() * s);
}
std::string Camera::toString() const
{
  return format(
    "Focal Length: {},{} | Principal Point: {},{} | Resolution: {},{}", fx(), fy(),
    principalPoint().x(), principalPoint().y(), _width, _height);
}

}  // namespace vslam
