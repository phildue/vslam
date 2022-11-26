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

#include <memory>

#include "EKFConstantVelocitySE3.h"

namespace pd::vslam::odometry
{
EKFConstantVelocitySE3::EKFConstantVelocitySE3(const Matd<12, 12> & covarianceProcess, Timestamp t0)
: _t(t0), _covProcess(covarianceProcess)
{
}
EKFConstantVelocitySE3::State::UnPtr EKFConstantVelocitySE3::predict(Timestamp t) const
{
  auto state = std::make_unique<State>();
  Timestamp dt = t - _t;
  state->pose = (SE3d::exp(_pose) * SE3d::exp(_velocity * dt)).log();
  state->velocity = _velocity;
  Mat<double, 12, 12> Jfx = computeJacobianProcess(SE3d::exp(state->pose));
  Mat<double, 12, 12> P = Jfx * (_covState * Jfx.transpose()) + _covProcess;
  state->covPose = P.block(0, 0, 6, 6);
  state->covVel = P.block(6, 6, 6, 6);
  return state;
}

void EKFConstantVelocitySE3::update(const Vec6d & motion, const Matd<6, 6> & covMotion, Timestamp t)
{
  Timestamp dt = t - _t;
  //Prediction
  _pose = (SE3d::exp(_pose) * SE3d::exp(_velocity * dt)).log();
  _velocity = _velocity;
  Mat<double, 12, 12> Jfx = computeJacobianProcess(SE3d::exp(_pose));
  _covState = Jfx * (_covState * Jfx.transpose()) + _covProcess;

  //Correction
  Vec6d e = _velocity * dt;
  Matd<6, 12> Jhx = computeJacobianMeasurement(dt);
  Matd<6, 6> E = Jhx * _covState * Jhx.transpose();

  Vec6d y = motion - e;
  Matd<6, 6> Z = E + covMotion;

  //State update
  MatXd K = _covState * Jhx.transpose() * Z.inverse();

  MatXd dx = K * y;
  // there is no position update anyway
  _velocity += dx.block(6, 0, 6, 1);

  _covState -= K * Z * K.transpose();
  _t = t;
}

Matd<12, 12> EKFConstantVelocitySE3::computeJacobianProcess(const SE3d & pose) const
{
  Matd<12, 12> J_f_x = MatXd::Zero(12, 12);
  Vec3d t = pose.translation();
  Eigen::Matrix3d t_hat;
  t_hat << 0, -t(2), t(1), t(2), 0, -t(0), -t(1), t(0), 0;
  auto R = pose.rotationMatrix();
  MatXd adj = MatXd::Zero(6, 6);
  adj.block(0, 0, 3, 3) = R;
  adj.block(0, 3, 3, 3) = t_hat;
  adj.block(3, 3, 3, 3) = R;
  J_f_x.block(6, 6, 6, 6) = adj.inverse();
  return J_f_x;
}

Matd<6, 12> EKFConstantVelocitySE3::computeJacobianMeasurement(Timestamp dt) const
{
  Matd<6, 12> mat;
  mat << MatXd::Zero(6, 6), MatXd::Identity(6, 6) * dt;
  return mat;
}

}  // namespace pd::vslam::kalman
