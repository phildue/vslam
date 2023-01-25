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

#include <iostream>
#include <memory>

#include "Warp.h"
#include "core/core.h"
namespace pd::vslam::lukas_kanade
{
double Warp::apply(const Image & img, int u, int v) const
{
  Eigen::Vector2d uvI = apply(u, v);
  if (1 < uvI.x() && uvI.x() < img.cols() - 1 && 1 < uvI.y() && uvI.y() < img.rows() - 1) {
    return algorithm::bilinearInterpolation(img, uvI.x(), uvI.y());
  } else {
    return std::numeric_limits<double>::quiet_NaN();
  }
}
Image Warp::apply(const Image & img) const
{
  Image warped = Image::Zero(img.rows(), img.cols());
  for (int v = 0; v < warped.rows(); v++) {
    for (int u = 0; u < warped.cols(); u++) {
      warped(v, u) = apply(img, u, v);
    }
  }
  return warped;
}
WarpAffine::WarpAffine(const Eigen::VectorXd & x, double cx, double cy)
: Warp(6, x), _cx(cx), _cy(cy)
{
  _w = toMat(_x);
}
void WarpAffine::updateAdditive(const Eigen::VectorXd & dx)
{
  _x.noalias() += dx;
  _w = toMat(x());
}
void WarpAffine::updateCompositional(const Eigen::VectorXd & dx)
{
  _x(0) = _x(0) + dx(0) + _x(0) * dx(0) + _x(2) * dx(1);
  _x(1) = _x(1) + dx(1) + _x(1) * dx(0) + _x(3) * dx(1);
  _x(2) = _x(2) + dx(2) + _x(0) * dx(2) + _x(2) * dx(3);
  _x(3) = _x(3) + dx(3) + _x(1) * dx(2) + _x(3) * dx(3);
  _x(4) = _x(4) + dx(4) + _x(0) * dx(4) + _x(2) * dx(5);
  _x(5) = _x(5) + dx(5) + _x(1) * dx(4) + _x(3) * dx(5);

  _w = toMat(_x);
}
Eigen::Vector2d WarpAffine::apply(int u, int v) const
{
  Eigen::Vector3d uv1;
  uv1 << u, v, 1;
  auto wuv1 = _w * uv1;
  return {wuv1.x(), wuv1.y()};
}
Eigen::MatrixXd WarpAffine::J(int u, int v) const
{
  Eigen::Matrix<double, 2, 6> J;
  J << u - _cx, 0, v - _cy, 0, 1, 0, 0, u - _cx, 0, v - _cy, 0, 1;
  return J;
}
void WarpAffine::setX(const Eigen::VectorXd & x)
{
  _x = x;
  _w = toMat(x);
}
Eigen::Matrix3d WarpAffine::toMat(const Eigen::VectorXd & x) const
{
  Eigen::Matrix3d w;
  w << 1 + x(0), x(2), x(4), x(1), 1 + x(3), x(5), 0, 0, 1;
  return w;
}

WarpOpticalFlow::WarpOpticalFlow(const Eigen::VectorXd & x) : Warp(2, x) { _w = toMat(_x); }
void WarpOpticalFlow::updateAdditive(const Eigen::VectorXd & dx)
{
  _x.noalias() += dx;
  _w = toMat(_x);
}
void WarpOpticalFlow::updateCompositional(const Eigen::VectorXd & dx)
{
  // TODO(unknown):
  _w = _w * toMat(dx);
  _x(0) = _w(0, 2);
  _x(1) = _w(1, 2);
}
Eigen::Vector2d WarpOpticalFlow::apply(int u, int v) const
{
  Eigen::Vector2d uv;
  uv << u, v;
  Eigen::Vector2d wuv = uv + _x;
  return wuv;
}
Eigen::MatrixXd WarpOpticalFlow::J(int UNUSED(u), int UNUSED(v)) const
{
  return Eigen::Matrix<double, 2, 2>::Identity();
}
void WarpOpticalFlow::setX(const Eigen::VectorXd & x)
{
  _x = x;
  _w = toMat(x);
}
Eigen::Matrix3d WarpOpticalFlow::toMat(const Eigen::VectorXd & x) const
{
  Eigen::Matrix3d w;
  w << 1, 0, x(0), 0, 1, x(1), 0, 0, 1;
  return w;
}

WarpSE3::WarpSE3(
  const SE3d & poseCur, const Eigen::MatrixXd & depth, Camera::ConstShPtr camCur,
  Camera::ConstShPtr camRef, const SE3d & poseRef)
: Warp(6),
  _se3(poseCur * poseRef.inverse()),
  _poseRef(poseRef),
  _width(depth.cols()),
  _camCur(camCur),
  _camRef(camRef),
  _pcl(depth.rows() * depth.cols())
{
  _x = _se3.log();
  // TODO(unknown): move pcl to frame so its only computed once
  for (int v = 0; v < depth.rows(); v++) {
    for (int u = 0; u < depth.cols(); u++) {
      /* Exclude pixels that are close to not having depth since we do bilinear interpolation later*/
      if (
        std::isfinite(depth(v, u)) && depth(v, u) > 0 && std::isfinite(depth(v + 1, u + 1)) &&
        depth(v + 1, u + 1) > 0 && std::isfinite(depth(v + 1, u - 1)) && depth(v + 1, u - 1) > 0 &&
        std::isfinite(depth(v - 1, u + 1)) && depth(v - 1, u + 1) > 0 &&
        std::isfinite(depth(v - 1, u - 1)) &&
        depth(v - 1, u - 1) > 0)  // TODO(unknown): move to actual interpolation?
      {
        _pcl[v * _width + u] = _camRef->image2camera({u, v}, depth(v, u));
      } else {
        _pcl[v * _width + u] = Eigen::Vector3d::Zero();
      }
    }
  }
}
WarpSE3::WarpSE3(
  const SE3d & poseCur, const std::vector<Vec3d> & pcl, size_t width, Camera::ConstShPtr camCur,
  Camera::ConstShPtr camRef, const SE3d & poseRef)
: Warp(6),
  _se3(poseCur * poseRef.inverse()),
  _poseRef(poseRef),
  _width(width),
  _camCur(camCur),
  _camRef(camRef),
  _pcl(pcl)
{
  _x = _se3.log();
}

void WarpSE3::updateAdditive(const Eigen::VectorXd & dx)
{
  _se3 = Sophus::SE3d::exp(dx) * _se3;
  _x = _se3.log();
}
void WarpSE3::updateCompositional(const Eigen::VectorXd & dx)
{
  _se3 = Sophus::SE3d::exp(dx) * _se3;
  _x = _se3.log();
}
Eigen::Vector2d WarpSE3::apply(int u, int v) const
{
  auto & p = _pcl[v * _width + u];
  return p.z() > 0.0
           ? _camCur->camera2image(_se3 * p)
           : Eigen::Vector2d(
               std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN());
}
Eigen::MatrixXd WarpSE3::J(int u, int v) const
{
  /*A tutorial on SE(3) transformation parameterizations and on-manifold optimization
  A.2. Projection of a point p.43
  Jacobian of uv = K * T_SE3 * p3d
  Jw = J_K * J_T
  with respect to tx,ty,tz,rx,ry,rz the parameters of the lie algebra element of T_SE3
  */
  Eigen::Matrix<double, 2, 6> jac;
  jac.setConstant(std::numeric_limits<double>::quiet_NaN());
  const Eigen::Vector3d & p = _pcl[v * _width + u];
  if (p.z() <= 0.0) {
    return jac;
  }

  const double & x = p.x();
  const double & y = p.y();
  const double z_inv = 1. / p.z();
  const double z_inv_2 = z_inv * z_inv;

  jac(0, 0) = z_inv;
  jac(0, 1) = 0.0;
  jac(0, 2) = -x * z_inv_2;
  jac(0, 3) = y * jac(0, 2);
  jac(0, 4) = 1.0 - x * jac(0, 2);
  jac(0, 5) = -y * z_inv;

  jac(1, 0) = 0.0;
  jac(1, 1) = z_inv;
  jac(1, 2) = -y * z_inv_2;
  jac(1, 3) = -1.0 + y * jac(1, 2);
  jac(1, 4) = -jac(0, 3);
  jac(1, 5) = x * z_inv;
  jac.row(0) *= _camRef->fx();
  jac.row(1) *= _camRef->fy();

  return jac;
}
double WarpSE3::apply(const Image & img, int u, int v) const
{
  const Vec3d & p = _pcl[v * _width + u];
  if (p.z() > 0.0) {
    const Vec3d pt = _se3 * p;
    const Vec2d uv = _camCur->camera2image(pt);
    if (1 < uv(0) && uv(0) < img.cols() - 1 && 1 < uv(1) && uv(1) < img.rows() - 1) {
      return interpolate(img, uv, pt.z());
    } else {
      return std::numeric_limits<double>::quiet_NaN();
    }
  } else {
    return std::numeric_limits<double>::quiet_NaN();
  }
}

double WarpSE3::interpolate(const Image & img, const Eigen::Vector2d & uv, double z) const
{
  const double x = uv(0);
  const double y = uv(1);

  const int x0 = (int)std::floor(x);
  const int y0 = (int)std::floor(y);
  const int x1 = x0 + 1;
  const int y1 = y0 + 1;

  const float x1_weight = x - x0;
  const float x0_weight = 1.0f - x1_weight;
  const float y1_weight = y - y0;
  const float y0_weight = 1.0f - y1_weight;
  const float z_eps = z - 0.05f;

  float val = 0.0f;
  float sum = 0.0f;

  auto depth = [&](int v, int u) { return _pcl[v * _width + u].z(); };

  if (std::isfinite(depth(y0, x0)) && depth(y0, x0) > z_eps) {
    val += x0_weight * y0_weight * img(y0, x0);
    sum += x0_weight * y0_weight;
  }

  if (std::isfinite(depth(y0, x1)) && depth(y0, x1) > z_eps) {
    val += x1_weight * y0_weight * img(y0, x1);
    sum += x1_weight * y0_weight;
  }

  if (std::isfinite(depth(y1, x0)) && depth(y1, x0) > z_eps) {
    val += x0_weight * y1_weight * img(y1, x0);
    sum += x0_weight * y1_weight;
  }

  if (std::isfinite(depth(y1, x1)) && depth(y1, x1) > z_eps) {
    val += x1_weight * y1_weight * img(y1, x1);
    sum += x1_weight * y1_weight;
  }

  return sum > 0.0f ? val / sum : std::numeric_limits<double>::quiet_NaN();
}

DepthMap WarpSE3::apply(const DepthMap & img) const
{
  DepthMap warped = DepthMap::Zero(img.rows(), img.cols());
  for (int i = 0; i < warped.rows(); i++) {
    for (int j = 0; j < warped.cols(); j++) {
      Eigen::Vector2d uvI = apply(j, i);
      if (
        1 < uvI.x() && uvI.x() < img.cols() - 1 && 1 < uvI.y() &&
        uvI.y() < img.rows() - 1) {  // TODO(unknown): check for invalid
        warped(i, j) = algorithm::bilinearInterpolation(img, uvI.x(), uvI.y());
      }
    }
  }
  return warped;
}

void WarpSE3::setX(const Eigen::VectorXd & x)
{
  _x = x;
  _se3 = Sophus::SE3d::exp(x);
}

SE3d WarpSE3::poseCur() const { return _se3 * _poseRef; }

}  // namespace pd::vslam::lukas_kanade
