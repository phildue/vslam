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
  const SE3d & poseCur, const DepthMap & depth1, const std::vector<Vec3d> & pcl, size_t width,
  Camera::ConstShPtr camCur, Camera::ConstShPtr camRef, const SE3d & poseRef, double minDepthDiff,
  double maxDepthDiff)
: Warp(6),
  _se3(poseCur),
  _pose0(poseRef),
  _depth1(depth1),
  _width(width),
  _cam1(camCur),
  _cam0(camRef),
  _pcl0(pcl),
  _minDepthDiff(minDepthDiff),
  _maxDepthDiff(maxDepthDiff)
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
  auto & p = _pcl0[v * _width + u];
  return p.z() > 0.0
           ? _cam1->camera2image(_se3 * _pose0.inverse() * p)
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
  const Eigen::Vector3d p = _pcl0[v * _width + u];
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
  jac.row(0) *= _cam0->fx();
  jac.row(1) *= _cam0->fy();

  return jac;
}
double WarpSE3::apply(const Image & img, int u, int v) const
{
  const Vec3d & p0 = _pcl0[v * _width + u];
  if (p0.z() > 0.0) {
    const Vec3d pt0 = _se3 * _pose0.inverse() * p0;
    return interpolate(img, _cam1->camera2image(pt0), pt0.z());
  }
  return std::numeric_limits<double>::quiet_NaN();
}

double WarpSE3::interpolate(const Image & img1, const Eigen::Vector2d & uv1, double z0t) const
{
  const double x = uv1(0);
  const double y = uv1(1);

  const int x0 = (int)std::max(0.0, std::floor(x));
  const int y0 = (int)std::max(0.0, std::floor(y));
  const int x1 = std::min<int>(img1.cols(), x0 + 1);
  const int y1 = std::min<int>(img1.rows(), y0 + 1);

  const float x1_weight = x - x0;
  const float x0_weight = 1.0f - x1_weight;
  const float y1_weight = y - y0;
  const float y0_weight = 1.0f - y1_weight;
  const float zmax = std::max(0.1, z0t + _maxDepthDiff);
  const double zmin = std::max(0.0, std::min(zmax - 0.01, z0t - _minDepthDiff));
  float val = 0.0f;
  float sum = 0.0f;
  auto validZ = [&zmin, &zmax](double z) -> bool { return zmin < z && z < zmax; };
  if (validZ(_depth1(y0, x0))) {
    val += x0_weight * y0_weight * (double)img1(y0, x0);
    sum += x0_weight * y0_weight;
  }

  if (validZ(_depth1(y0, x1))) {
    val += x1_weight * y0_weight * (double)img1(y0, x1);
    sum += x1_weight * y0_weight;
  }

  if (validZ(_depth1(y1, x0))) {
    val += x0_weight * y1_weight * (double)img1(y1, x0);
    sum += x0_weight * y1_weight;
  }

  if (validZ(_depth1(y1, x1))) {
    val += x1_weight * y1_weight * (double)img1(y1, x1);
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

SE3d WarpSE3::poseCur() const { return _se3 * _pose0; }

}  // namespace pd::vslam::lukas_kanade
