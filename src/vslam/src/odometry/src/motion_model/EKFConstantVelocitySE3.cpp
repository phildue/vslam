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
#include "utils/utils.h"

#define LOG_ODOM(level) CLOG(level, "odometry")
namespace pd::vslam::odometry
{
EKFConstantVelocitySE3::EKFConstantVelocitySE3(
  const Matd<12, 12> & covarianceProcess, Timestamp t0, const Matd<12, 12> & covState)
: _t(t0),
  _pose(Vec6d::Zero()),
  _velocity(Vec6d::Zero()),
  _covState(covState),
  _covProcess(covarianceProcess),
  _plot(PlotKalman::get())
{
  Log::get("odometry");
}
EKFConstantVelocitySE3::State::UnPtr EKFConstantVelocitySE3::predict(Timestamp t) const
{
  auto state = std::make_unique<State>();
  const double dt = t - _t;
  if (dt <= 0) {
    state->pose = _pose;
    state->velocity = _velocity;
    state->covPose = _covState.block(0, 0, 6, 6);
    state->covVel = _covState.block(6, 6, 6, 6);
  } else {
    auto motion = SE3d::exp(_velocity * dt);
    state->pose = (SE3d::exp(_pose) * motion).log();
    state->velocity = _velocity;
    Mat<double, 12, 12> Jfx = computeJacobianProcess(motion);
    Mat<double, 12, 12> P = Jfx * (_covState * Jfx.transpose()) + _covProcess;
    state->covPose = P.block(0, 0, 6, 6);
    state->covVel = P.block(6, 6, 6, 6);
  }

  return state;
}

void EKFConstantVelocitySE3::update(
  const Vec6d & measurement, const Matd<6, 6> & covMeasurement, Timestamp t)
{
  const double dt = t - _t;
  if (dt <= 0) {
    return;
  }
  // Prediction
  auto motion = SE3d::exp(_velocity * dt);
  _pose = (SE3d::exp(_pose) * motion).log();
  _velocity = _velocity;
  Mat<double, 12, 12> Jfx = computeJacobianProcess(motion);
  _covState = Jfx * (_covState * Jfx.transpose()) + _covProcess;
  LOG_ODOM(DEBUG) << "Prediction. Pose: " << _pose.transpose()
                  << " Twist: " << _velocity.transpose();
  // Expectation
  Vec6d e = _velocity * dt;
  Matd<6, 12> Jhx = computeJacobianMeasurement(dt);
  Matd<6, 6> E = Jhx * _covState * Jhx.transpose();
  LOG_ODOM(DEBUG) << "Expectation. Twist: " << e.transpose();
  LOG_ODOM(DEBUG) << "Measurement. Twist: " << measurement.transpose();

  // Correction
  Vec6d y = measurement - e;
  Matd<6, 6> Z = E + covMeasurement;
  LOG_ODOM(DEBUG) << "Correction. Twist: " << y.transpose();

  // State update
  MatXd K = _covState * Jhx.transpose() * Z.inverse();
  LOG_ODOM(DEBUG) << "Gain. |K| = " << K;

  MatXd dx = K * y;
  _pose += dx.block(0, 0, 6, 1);
  _velocity += dx.block(6, 0, 6, 1);
  LOG_ODOM(DEBUG) << "State Update. Dx: = " << dx.transpose();

  _covState -= K * Z * K.transpose();
  _t = t;
  LOG_ODOM(DEBUG) << "State. Pose = " << _pose.transpose() << " Twist = " << _velocity.transpose()
                  << " |XX|: " << _covState.determinant();

  _plot->append(t, {state(), e, measurement, y, dx, _covState, E, covMeasurement, K});
}
Vec12d EKFConstantVelocitySE3::state() const
{
  Vec12d x;
  x << _pose, _velocity;
  return x;
}

Matd<12, 12> EKFConstantVelocitySE3::computeJacobianProcess(const SE3d & vdt) const
{
  //Compute Jacobian of f(x) = pose * exp(twist * dt).
  // J_F_x:         Ad_Exp(v*dt)^(-1)
  Matd<12, 12> J_f_x = MatXd::Zero(12, 12);
  Vec3d t = vdt.translation();
  Eigen::Matrix3d t_hat;
  t_hat << 0, -t(2), t(1), t(2), 0, -t(0), -t(1), t(0), 0;
  auto R = vdt.rotationMatrix();
  MatXd adj = MatXd::Zero(6, 6);
  adj.block(0, 0, 3, 3) = R;
  adj.block(0, 3, 3, 3) = t_hat;
  adj.block(3, 3, 3, 3) = R;
  J_f_x.block(0, 0, 6, 6) = adj.inverse();
  return J_f_x;
}
Matd<6, 12> EKFConstantVelocitySE3::computeJacobianMeasurement(double dt) const
{
  //Compute Jacobian of h(x) = twist * dt.
  Matd<6, 12> J_h_x = MatXd::Zero(6, 12);
  J_h_x.block(0, 6, 6, 6) = MatXd::Identity(6, 6) * dt;
  return J_h_x;
}

}  // namespace pd::vslam::odometry
