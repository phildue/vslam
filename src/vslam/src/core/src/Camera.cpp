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
namespace pd::vslam
{
Eigen::Vector2d Camera::camera2image(const Eigen::Vector3d & pWorld) const
{
  if (pWorld.z() <= 0) {
    return {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()};
  }
  Eigen::Vector3d pProj = _K * pWorld;
  return {pProj.x() / pProj.z(), pProj.y() / pProj.z()};
}

Eigen::Vector3d Camera::image2camera(const Eigen::Vector2d & pImage, double depth) const
{
  return _Kinv * (Eigen::Vector3d({pImage.x(), pImage.y(), 1}) * depth);
}
Eigen::Vector3d Camera::image2ray(const Eigen::Vector2d & pImage) const
{
  return _Kinv * Eigen::Vector3d({pImage.x(), pImage.y(), 1});
}

Camera::Camera(double f, double cx, double cy) : Camera(f, f, cx, cy) {}

Camera::Camera(double fx, double fy, double cx, double cy)
{
  _K << fx, 0, cx, 0, fy, cy, 0, 0, 1;
  _Kinv = _K.inverse();
}
void Camera::resize(double s)
{
  _K *= s;
  _Kinv = _K.inverse();
}
Camera::ShPtr Camera::resize(Camera::ConstShPtr cam, double s)
{
  return std::make_shared<Camera>(
    cam->fx() * s, cam->fy() * s, cam->principalPoint().x() * s, cam->principalPoint().y() * s);
}

}  // namespace pd::vslam
