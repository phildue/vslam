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

#include <fmt/chrono.h>
#include <fmt/core.h>

#include "MotionModel.h"
#include "utils/utils.h"
using fmt::format;
using fmt::print;
template <typename T>
struct fmt::formatter<T, std::enable_if_t<std::is_base_of_v<Eigen::DenseBase<T>, T>, char>>
: ostream_formatter
{
};
#define LOG_ODOM(level) CLOG(level, "odometry")
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
  if (dT > 0) {
    _speed = algorithm::computeRelativeTransform(_lastPose->pose(), pose->pose()).log() / dT;
  }
  _lastPose = pose;
  _lastT = timestamp;
}
void MotionModelNoMotion::update(Frame::ConstShPtr frameRef, Frame::ConstShPtr frameCur)
{
  if (frameCur->t() < frameRef->t()) {
    throw pd::Exception("New timestamp is older than last one!");
  }
  const double dT = (static_cast<double>(frameCur->t()) - static_cast<double>(frameRef->t())) / 1e9;
  if (dT > 0) {
    _speed =
      algorithm::computeRelativeTransform(frameRef->pose().pose(), frameCur->pose().pose()).log() /
      dT;
  }

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

MotionModelMovingAverage::MotionModelMovingAverage(Timestamp timeFrame)
: MotionModel(),
  _timeFrame(timeFrame),
  _traj(std::make_unique<Trajectory>()),
  _speed(Vec6d::Zero()),
  _lastPose(std::make_shared<PoseWithCovariance>(SE3d(), MatXd::Identity(6, 6))),
  _lastT(0U)
{
}
void MotionModelMovingAverage::update(PoseWithCovariance::ConstShPtr pose, Timestamp timestamp)
{
  if (timestamp < _lastT) {
    throw pd::Exception("New timestamp is older than last one!");
  }
  _traj->append(timestamp, pose);
  if (_traj->tEnd() - _traj->tStart() <= _timeFrame) {
    const double dT = (static_cast<double>(timestamp) - static_cast<double>(_lastT)) / 1e9;
    _speed = algorithm::computeRelativeTransform(_lastPose->pose(), pose->pose()).log() / dT;

  } else {
    _speed =
      _traj->meanMotion(timestamp - _timeFrame, timestamp, timestamp - _lastT)->pose().log() * 1e9;
  }
  _lastPose = pose;
  _lastT = timestamp;
}
void MotionModelMovingAverage::update(Frame::ConstShPtr frameRef, Frame::ConstShPtr frameCur) {}

PoseWithCovariance::UnPtr MotionModelMovingAverage::predictPose(Timestamp timestamp) const
{
  const double dT = (static_cast<double>(timestamp) - static_cast<double>(_lastT)) / 1e9;
  const SE3d predictedRelativePose = SE3d::exp(_speed * dT);
  return std::make_unique<PoseWithCovariance>(
    predictedRelativePose * _lastPose->pose(), MatXd::Identity(6, 6));
}
PoseWithCovariance::UnPtr MotionModelMovingAverage::pose() const
{
  return std::make_unique<PoseWithCovariance>(_lastPose->pose(), _lastPose->cov());
}
PoseWithCovariance::UnPtr MotionModelMovingAverage::speed() const
{
  return std::make_unique<PoseWithCovariance>(SE3d::exp(_speed), _lastPose->cov());
}

MotionModelConstantSpeedKalman::MotionModelConstantSpeedKalman(
  const Matd<12, 12> & covProcess, const Matd<12, 12> & covState)
: MotionModel(),
  _kalman(std::make_unique<EKFConstantVelocitySE3>(
    covProcess, std::numeric_limits<Timestamp>::max(), covState)),
  _lastPose(std::make_shared<PoseWithCovariance>(SE3d(), MatXd::Identity(6, 6))),
  _lastT(0U)
{
}
PoseWithCovariance::UnPtr MotionModelConstantSpeedKalman::predictPose(Timestamp timestamp) const
{
  Pose::UnPtr pose;
  if (_kalman->t() == std::numeric_limits<uint64_t>::max()) {
    pose = std::make_unique<PoseWithCovariance>(*_lastPose);
  } else {
    auto state = _kalman->predict(timestamp);
    pose = std::make_unique<PoseWithCovariance>(state->pose(), state->covPose());
  }
  LOG_ODOM(DEBUG) << format(
    "Prediction: {} +- {}", pose->mean().transpose(), pose->twistCov().diagonal().transpose());
  return pose;
}
void MotionModelConstantSpeedKalman::update(
  PoseWithCovariance::ConstShPtr pose, Timestamp timestamp)
{
  //Since the system model of kalman is x' = x * exp(v * dt) we have to compute v*dT = log(x.inv() * x')
  auto motion = (_lastPose->pose().inverse() * pose->pose()).log();
  _lastPose = pose;
  _lastT = timestamp;
  if (_kalman->t() == std::numeric_limits<Timestamp>::max()) {
    _kalman->reset(timestamp, {pose->twist(), Vec6d::Zero(), _kalman->state()->covariance()});
  } else {
    _kalman->update(motion, Matd<6, 6>::Identity(), timestamp);
  }
}

void MotionModelConstantSpeedKalman::update(Frame::ConstShPtr frameRef, Frame::ConstShPtr frameCur)
{
  if (frameCur->t() < frameRef->t()) {
    throw pd::Exception("New timestamp is older than last one!");
  }
  auto speed =
    algorithm::computeRelativeTransform(frameRef->pose().pose(), frameCur->pose().pose()).log();
  _lastPose = std::make_shared<PoseWithCovariance>(frameCur->pose());
  _lastT = frameCur->t();
  _kalman->update(speed, Matd<6, 6>::Identity(), frameCur->t());
}

PoseWithCovariance::UnPtr MotionModelConstantSpeedKalman::speed() const
{
  auto state = _kalman->state();
  return std::make_unique<PoseWithCovariance>(SE3d::exp(state->velocity()), state->covVelocity());
}

PoseWithCovariance::UnPtr MotionModelConstantSpeedKalman::pose() const
{
  auto state = _kalman->state();
  return std::make_unique<PoseWithCovariance>(SE3d::exp(state->pose()), state->covPose());
}

}  // namespace pd::vslam
