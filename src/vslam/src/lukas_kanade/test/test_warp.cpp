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

#include <core/core.h>
#include <gtest/gtest.h>
#include <utils/utils.h>

#include "lukas_kanade/lukas_kanade.h"

using namespace testing;
using namespace pd;
using namespace pd::vslam;
using namespace pd::vslam::least_squares;
using namespace pd::vslam::lukas_kanade;

//OpenCV implementation
void calcRgbdEquationCoeffs(
  double * C, double dIdx, double dIdy, const Vec3d & p3d, double fx, double fy)
{
  double invz = 1. / p3d.z(), v0 = dIdx * fx * invz, v1 = dIdy * fy * invz,
         v2 = -(v0 * p3d.x() + v1 * p3d.y()) * invz;

  C[3] = -p3d.z() * v1 + p3d.y() * v2;
  C[4] = p3d.z() * v0 - p3d.x() * v2;
  C[5] = -p3d.y() * v0 + p3d.x() * v1;
  C[0] = v0;
  C[1] = v1;
  C[2] = v2;
}

Eigen::Vector6d ourJ(double dIdx, double dIdy, const Vec3d & p, double fx, double fy)
{
  Eigen::Matrix<double, 2, 6> jac;
  jac.setConstant(std::numeric_limits<double>::quiet_NaN());

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
  jac.row(0) *= fx;
  jac.row(1) *= fy;
  return jac.row(0) * dIdx + jac.row(1) * dIdy;
}

TEST(WarpSE3Test, DISABLED_Jacobian)
{
  SE3d pose;
  DepthMap depth = utils::loadDepth(TEST_RESOURCE "/depth.jpg") / 5000.0;
  Image img = utils::loadImage(TEST_RESOURCE "/rgb.jpg");
  auto cam = std::make_shared<Camera>(525.0, 525.0, 319.5, 239.5);
  auto frame = std::make_shared<Frame>(
    img, depth, cam, 1, 0, PoseWithCovariance(pose, Eigen::MatrixXd::Identity(6, 6)));
  auto w = std::make_shared<WarpSE3>(pose, frame->pcl(), frame->width(), cam, cam);
  const int u = 240;
  const int v = 300;
  std::vector<double> A_buf(6);
  double * A_ptr = &A_buf[0];
  calcRgbdEquationCoeffs(
    A_ptr, frame->dIx()(v, u), frame->dIy()(v, u), frame->p3d(v, u), cam->fx(), cam->fy());
  Eigen::Map<Eigen::Vector6d> jacobianOpencv(A_buf.data(), A_buf.size());
  Eigen::Vector6d jacobian =
    w->J(u, v).row(0) * frame->dIx()(v, u) + w->J(u, v).row(1) * frame->dIy()(v, u);

  std::cout << "OpenCV: " << jacobianOpencv.transpose() << std::endl;
  std::cout << "Ours: " << jacobian.transpose() << std::endl;
  std::cout << "Ours: "
            << ourJ(frame->dIx()(v, u), frame->dIy()(v, u), frame->p3d(v, u), cam->fx(), cam->fy())
                 .transpose()
            << std::endl;
}
