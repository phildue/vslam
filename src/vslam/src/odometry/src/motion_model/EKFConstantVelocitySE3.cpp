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

#include <manif/SE3.h>

#include <memory>

#include "EKFConstantVelocitySE3.h"
#include "utils/utils.h"
#define LOG_ODOM(level) CLOG(level, "odometry")

namespace pd::vslam
{
EKFConstantVelocitySE3::EKFConstantVelocitySE3(
  const Matd<12, 12> & covarianceProcess, Timestamp t0, const Matd<12, 12> & covState)
: _t(t0),
  _state(std::make_unique<State>(Vec6d::Zero(), Vec6d::Zero(), covState)),
  _covProcess(covarianceProcess),
  _K(Matd<12, 6>::Zero()),
  _q(_covProcess(0, 0)),
  _windowSize(100),
  _stateTransitionMats(_windowSize, Matd<12, 12>::Identity()),
  _innovMats(_windowSize, Matd<6, 6>::Identity()),
  _nSamples(0U),
  _log(PlotKalman::make())
{
  Log::get("odometry");
}
EKFConstantVelocitySE3::State::UnPtr EKFConstantVelocitySE3::predict(
  Timestamp t, Mat12d & J_f_x) const
{
  const double dt = t - _t;
  // Prediction
  manif::SE3d Xp = manif::SE3Tangentd(_state->pose()).exp();
  manif::SE3d Xv = manif::SE3Tangentd(_state->velocity()).exp();

  Vec6d motion = _state->velocity() * dt;
  manif::SE3d::Jacobian J_xp, J_xv;
  Xp = Xp.lplus(manif::SE3Tangentd(motion), J_xp, J_xv);  // exp(u) * X with Jacobians
  Xv = Xv;
  J_f_x = Matd<12, 12>::Zero();
  J_f_x.block(0, 0, 6, 6) = J_xp;
  J_f_x.block(6, 0, 6, 6) = J_xv;
  const Matd<12, 12> P = J_f_x * _state->covariance() * J_f_x.transpose() + _covProcess;

  return std::make_unique<State>(Xp.log().coeffs(), Xv.log().coeffs(), P);
}
EKFConstantVelocitySE3::State::State(
  const Vec6d & pose, const Vec6d & velocity, const Mat12d & covariance)
: _covariance(covariance)
{
  _state << pose, velocity;
}

EKFConstantVelocitySE3::State::UnPtr EKFConstantVelocitySE3::predict(Timestamp t) const
{
  Mat12d J;
  return predict(t, J);
}

void EKFConstantVelocitySE3::update(
  const Vec6d & measurement, const Matd<6, 6> & covMeasurement, Timestamp t)
{
  const double dt = t - _t;
  auto statePred = predict(t);

  LOG_ODOM(DEBUG) << "Prediction. Pose: " << statePred->pose().transpose()
                  << " Velocity: " << statePred->velocity().transpose();

  // Expectation
  Vec6d motion = statePred->velocity() * dt;
  Vec6d e = motion;
  Matd<6, 12> Jhx = computeJ_h_x(dt);
  Matd<6, 6> E = Jhx * statePred->covariance() * Jhx.transpose();
  LOG_ODOM(DEBUG) << "Expectation: " << e.transpose();
  LOG_ODOM(DEBUG) << "Measurement: " << measurement.transpose();

  // Innovation
  Vec6d y = measurement - e;
  Matd<6, 6> Z = E + covMeasurement;
  _innovMats[_nSamples++ % _windowSize] = y * y.transpose();

  LOG_ODOM(DEBUG) << "Innovation: " << y.transpose();

  // State update
  _K = statePred->covariance() * Jhx.transpose() * Z.inverse();
  LOG_ODOM(DEBUG) << "Gain. |K| = " << _K;
  MatXd dx = _K * y;
  LOG_ODOM(DEBUG) << "State Update. Dx: = " << dx.transpose();

  const manif::SE3d pose =
    manif::SE3Tangentd(statePred->pose()).exp() + manif::SE3Tangentd(dx.block(0, 0, 6, 1));
  const manif::SE3d velocity =
    manif::SE3Tangentd(statePred->velocity()).exp() + manif::SE3Tangentd(dx.block(6, 0, 6, 1));
  const Mat12d P = statePred->covariance() - _K * Z * _K.transpose();
  _state = std::make_unique<State>(pose.log().coeffs(), velocity.log().coeffs(), P);
  _covProcess = computeProcessNoise(dt);
  _t = t;
  LOG_ODOM(DEBUG) << "State. Pose = " << _state->pose().transpose()
                  << " Twist = " << _state->velocity().transpose()
                  << " |XX|: " << _state->covariance().determinant();

  //TODO avoid singleton? nicer api?
  _log << PlotKalman::Entry(
    {t, _state->x(), e, measurement, y, dx, _state->covariance(), E, covMeasurement, _K});
}

void EKFConstantVelocitySE3::reset(Timestamp t, const EKFConstantVelocitySE3::State & state)
{
  _t = t;
  _state = std::make_unique<State>(state);
}
Matd<12, 12> EKFConstantVelocitySE3::computeJ_f_x(const SE3d & pose, const SE3d & vdt) const
{
  /*
  * Compute Jacobian of f(x):   [f1(x);f2(x)]' = [pose,twist]' = [pose * exp(twist * dt),twist]'
  * J_f_x =  |J_f1_pose,  J_f2_pose |
  *          |J_f2_twist, J_f2_twist|
  *
  * J_f1_pose:         Ad_Exp(v*dt)^(-1)
  * J_f1_twist:        
  * J_f2_pose:         0
  * J_f2_twist:        0
  *
  * 
  * 
  */

  Matd<12, 12> J_f_x = MatXd::Zero(12, 12);
#if 0
  // J_f1_pose:         Ad_Exp(v*dt)^(-1)
  Vec3d t = vdt.translation();
  Eigen::Matrix3d t_hat;
  t_hat << 0, -t(2), t(1), t(2), 0, -t(0), -t(1), t(0), 0;
  auto R = vdt.rotationMatrix();
  MatXd adj = MatXd::Zero(6, 6);
  adj.block(0, 0, 3, 3) = R;
  adj.block(0, 3, 3, 3) = t_hat;
  adj.block(3, 3, 3, 3) = R;
  J_f_x.block(0, 0, 6, 6) = adj.inverse();

  // J_f1_twist:         Ad_Exp(v*dt)^(-1)
#elif 1
  manif::SE3d pose_(pose.translation(), manif::SO3d(pose.unit_quaternion()));
  manif::SE3Tangentd twist_(vdt.log());
  manif::SE3d::Jacobian J_f1_pose, J_f1_twist;
  pose_.lplus(twist_, J_f1_pose, J_f1_twist);
  J_f_x.block(0, 0, 6, 6) = J_f1_pose;
  J_f_x.block(6, 0, 6, 6) = J_f1_twist;

#endif
  LOG_ODOM(DEBUG) << "J_f1_pose = \n" << J_f1_pose;
  LOG_ODOM(DEBUG) << "J_f1_twist = \n" << J_f1_twist;
  LOG_ODOM(DEBUG) << "J_f_x = \n" << J_f_x;
  return J_f_x;
}
Matd<6, 12> EKFConstantVelocitySE3::computeJ_h_x(double dt) const
{
  //Compute Jacobian of h(x) = twist * dt.
  Matd<6, 12> J_h_x;
  J_h_x << MatXd::Zero(6, 6), MatXd::Identity(6, 6) * dt;
  return J_h_x;
}
Matd<12, 12> EKFConstantVelocitySE3::computeProcessNoise(double dt) const
{
  Mat<double, 12, 12> Q = Mat<double, 12, 12>::Identity();

//
#if 1
  /*
1d
x  = [x vx]
C = [c(x) c(v,x)
     c(v,x) c(v)]

F = [1 dt dt^2/2
     0  1  dt
     0  0   1]
6d
x = [px py pz rx ry rz vx vy vz rvx rvy rvz]
C = [c(px) c(px,py) c(px,pz) c(px,rz) c(px, ry) c()]
     ...
C = [1 0]
*/
  Q = _covProcess;
#elif 0
  Mat<double, 12, 12> Qa = Mat<double, 12, 12>::Zero();
  Qa(11, 11) = 1;
  Mat<double, 12, 12> Jfx = computeJacobianProcess(SE3d::exp(pose()), motion);

  Q = Jfx * Qa * Jfx.transpose() * _q;
#elif 0
  // Approach 2 - Innovation based adaptive method
  Matd<6, 6> sum = Matd<6, 6>::Zero();
  for (const auto & i : _innovMats) {
    sum += i;
  }
  Q = K * (sum / _innovMats.size()) * K.transpose();
#elif 0
  // Approach 3- Generative
  Matd<12, 12> sum = Matd<12, 12>::Zero();
  for (const auto & xx : _stateTransitionMats) {
    sum += xx;
  }
  Q = sum / _stateTransitionMats.size();
#endif

  LOG_ODOM(DEBUG) << "Process Noise. Q = \n" << Q.transpose();
  return Q;
}

}  // namespace pd::vslam
