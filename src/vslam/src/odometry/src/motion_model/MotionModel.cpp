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

#include "MotionModel.h"
#include "utils/utils.h"
#define LOG_MOTION_PREDICTION(level) CLOG(level, "motion_prediction")
namespace pd::vslam
{
MotionModelNoMotion::MotionModelNoMotion()
: MotionModel(),
  _speed(Vec6d::Zero()),
  _lastPose(std::make_shared<PoseWithCovariance>(SE3d(), MatXd::Identity(6, 6))),
  _lastT(0U)
{
}
void MotionModelNoMotion::update(PoseWithCovariance::ConstShPtr pose, Timestamp timestamp)
{
  if (timestamp < _lastT) {
    throw pd::Exception("New timestamp is older than last one!");
  }
  const double dT = (static_cast<double>(timestamp) - static_cast<double>(_lastT)) / 1e9;

  _speed = algorithm::computeRelativeTransform(_lastPose->pose(), pose->pose()).log() / dT;
  _lastPose = pose;
  _lastT = timestamp;
}
void MotionModelNoMotion::update(Frame::ConstShPtr frameRef, Frame::ConstShPtr frameCur)
{
  if (frameCur->t() < frameRef->t()) {
    throw pd::Exception("New timestamp is older than last one!");
  }
  const double dT = (static_cast<double>(frameCur->t()) - static_cast<double>(frameRef->t())) / 1e9;

  _speed =
    algorithm::computeRelativeTransform(frameRef->pose().pose(), frameCur->pose().pose()).log() /
    dT;
  _lastPose = std::make_shared<PoseWithCovariance>(frameCur->pose());
  _lastT = frameCur->t();
}

PoseWithCovariance::UnPtr MotionModelNoMotion::predictPose(Timestamp UNUSED(timestamp)) const
{
  return pose();
}
PoseWithCovariance::UnPtr MotionModelNoMotion::pose() const
{
  return std::make_unique<PoseWithCovariance>(_lastPose->pose(), _lastPose->cov());
}
PoseWithCovariance::UnPtr MotionModelNoMotion::speed() const
{
  return std::make_unique<PoseWithCovariance>(SE3d::exp(_speed), _lastPose->cov());
}

MotionModelConstantSpeed::MotionModelConstantSpeed() : MotionModelNoMotion() {}

PoseWithCovariance::UnPtr MotionModelConstantSpeed::predictPose(Timestamp timestamp) const
{
  const double dT = (static_cast<double>(timestamp) - static_cast<double>(_lastT)) / 1e9;
  const SE3d predictedRelativePose = SE3d::exp(_speed * dT);
  return std::make_unique<PoseWithCovariance>(
    predictedRelativePose * _lastPose->pose(), MatXd::Identity(6, 6));
}

MotionModelConstantSpeedKalman::MotionModelConstantSpeedKalman()
: MotionModel(),
  _kalman(std::make_unique<odometry::EKFConstantVelocitySE3>(Matd<12, 12>::Identity())),
  _lastPose(std::make_shared<PoseWithCovariance>(SE3d(), MatXd::Identity(6, 6))),
  _lastT(0U)
{
}
PoseWithCovariance::UnPtr MotionModelConstantSpeedKalman::predictPose(Timestamp timestamp) const
{
  odometry::EKFConstantVelocitySE3::State::ConstPtr state = _kalman->predict(timestamp);
  return std::make_unique<PoseWithCovariance>(SE3d::exp(state->pose), state->covPose);
}
void MotionModelConstantSpeedKalman::update(
  PoseWithCovariance::ConstShPtr pose, Timestamp timestamp)
{
  if (timestamp < _lastT) {
    throw pd::Exception("New timestamp is older than last one!");
  }
  const double dT = (static_cast<double>(timestamp) - static_cast<double>(_lastT)) / 1e9;

  auto speed = algorithm::computeRelativeTransform(_lastPose->pose(), pose->pose()).log() / dT;
  _lastPose = pose;
  _lastT = timestamp;
  _kalman->update(speed, Matd<6, 6>::Identity(), timestamp);
}

void MotionModelConstantSpeedKalman::update(Frame::ConstShPtr frameRef, Frame::ConstShPtr frameCur)
{
  if (frameCur->t() < frameRef->t()) {
    throw pd::Exception("New timestamp is older than last one!");
  }
  const double dT = (static_cast<double>(frameCur->t()) - static_cast<double>(frameRef->t())) / 1e9;

  auto speed =
    algorithm::computeRelativeTransform(frameRef->pose().pose(), frameCur->pose().pose()).log() /
    dT;
  _lastPose = std::make_shared<PoseWithCovariance>(frameCur->pose());
  _lastT = frameCur->t();
  _kalman->update(speed, Matd<6, 6>::Identity(), frameCur->t());
}

PoseWithCovariance::UnPtr MotionModelConstantSpeedKalman::speed() const
{
  odometry::EKFConstantVelocitySE3::State::ConstPtr state = _kalman->predict(0);
  return std::make_unique<PoseWithCovariance>(SE3d::exp(state->velocity), state->covVel);
}

PoseWithCovariance::UnPtr MotionModelConstantSpeedKalman::pose() const
{
  return predictPose(_lastT);
}

}  // namespace pd::vslam
