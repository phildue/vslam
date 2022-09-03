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

#include "EKFConstantVelocitySE3.h"

#include <memory>

namespace pd::vslam::kalman
{
EKFConstantVelocitySE3::State::UnPtr EKFConstantVelocitySE3::predict(Timestamp t) const
{
  auto state = std::make_unique<State>();
  Matd<12, 12> P;
  Matd<12, 12> Jfx;
  predict(t, state->pose, state->velocity, P, Jfx);
  state->covPose = P.block(0, 0, 6, 6);
  state->covVel = P.block(0, 0, 6, 6);
  return state;
}

void EKFConstantVelocitySE3::update(const Vec6d & motion, const Matd<6, 6> & covMotion, Timestamp t)
{
  Timestamp dt = t - _t;
  Matd<12, 12> Jfx;
  predict(t, _pose, _velocity, _covState, Jfx);

  auto e = _velocity * dt;
  auto Jhx = computeJacobianMeasurement(dt);
  auto E = Jhx * _covState * Jhx.transpose();

  auto y = motion - e;
  auto Z = E + covMotion;

  MatXd K = _covState * Jhx.transpose() * Z.inverse();

  MatXd dx = K * y;
  // there is no position update anyway
  _velocity += dx.block(6, 0, 12, 1);

  _covState -= K * Z * K.transpose();
  _t = t;
}

void EKFConstantVelocitySE3::predict(
  Timestamp t, Vec6d & pose, Vec6d & vel, Matd<12, 12> & P, Matd<12, 12> & Jfx) const
{
  Timestamp dt = t - _t;
  pose = (SE3d::exp(_pose) * SE3d::exp(_velocity * dt)).log();
  vel = _velocity;
  Jfx = computeJacobianProcess(SE3d::exp(pose));
  P = Jfx * (P * Jfx.transpose()) + _covProcess;
}

Matd<12, 12> EKFConstantVelocitySE3::computeJacobianProcess(const SE3d & pose) const
{
  MatXd J_f_x = MatXd::Zero(12, 12);
  Vec3d t = pose.translation();
  Eigen::Matrix3d t_hat;
  t_hat << 0, -t(2), t(1), t(2), 0, -t(0), -t(1), t(0), 0;
  auto R = pose.rotationMatrix();
  MatXd adj = MatXd::Zero(6, 6);
  adj.block(0, 0, 3, 3) = R;
  adj.block(0, 3, 3, 6) = t_hat;
  adj.block(3, 3, 6, 6) = R;
  J_f_x.block(6, 6, 12, 12) = adj.inverse();
  return J_f_x;
}

Matd<6, 12> EKFConstantVelocitySE3::computeJacobianMeasurement(Timestamp dt) const
{
  Matd<6, 12> mat;
  mat << MatXd::Zero(6, 6), MatXd::Identity(6, 6) * dt;
  return mat;
}

}  // namespace pd::vslam::kalman
