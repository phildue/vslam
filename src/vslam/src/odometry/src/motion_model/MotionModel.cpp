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
: MotionModel(), _speed(Vec6d::Zero()), _lastPose(SE3d(), MatXd::Identity(6, 6)), _lastT(0U)
{
}
void MotionModelNoMotion::update(const Pose & relativePose, Timestamp timestamp)
{
  if (timestamp < _lastT) {
    throw pd::Exception("New timestamp is older than last one!");
  }
  const double dT = (static_cast<double>(timestamp) - static_cast<double>(_lastT)) / 1e9;
  if (dT > 0) {
    _speed = relativePose.twist() / dT;
    _speedCov = relativePose.twistCov() / dT;
  }
  _lastPose = relativePose * _lastPose;
  _lastT = timestamp;
}

Pose MotionModelNoMotion::predictPose(Timestamp UNUSED(timestamp)) const { return pose(); }
Pose MotionModelNoMotion::pose() const { return Pose(_lastPose.pose(), _lastPose.cov()); }
Pose MotionModelNoMotion::speed() const { return Pose(SE3d::exp(_speed), _speedCov); }

MotionModelConstantSpeed::MotionModelConstantSpeed(const Mat6d & covariance)
: MotionModelNoMotion(), _covariance(covariance)
{
}

Pose MotionModelConstantSpeed::predictPose(Timestamp timestamp) const
{
  const double dT = (static_cast<double>(timestamp) - static_cast<double>(_lastT)) / 1e9;
  const Pose predictedRelativePose(SE3d::exp(_speed * dT), dT * _speedCov);
  return Pose(predictedRelativePose * _lastPose);
}

MotionModelMovingAverage::MotionModelMovingAverage(Timestamp timeFrame)
: MotionModel(),
  _timeFrame(timeFrame),
  _traj(std::make_unique<Trajectory>()),
  _speed(Vec6d::Zero()),
  _lastPose(SE3d(), MatXd::Identity(6, 6)),
  _lastT(0U)
{
}
void MotionModelMovingAverage::update(const Pose & relativePose, Timestamp timestamp)
{
  if (timestamp < _lastT) {
    throw pd::Exception("New timestamp is older than last one!");
  }
  auto pose = relativePose * _lastPose;
  _traj->append(timestamp, pose);
  if (_traj->tEnd() - _traj->tStart() <= _timeFrame) {
    const double dT = (static_cast<double>(timestamp) - static_cast<double>(_lastT)) / 1e9;
    _speed = relativePose.twist() / dT;
    _speedCov = relativePose.twistCov() / dT;

  } else {
    _speed =
      _traj->meanMotion(timestamp - _timeFrame, timestamp, timestamp - _lastT)->pose().log() * 1e9;
  }
  _lastPose = pose;
  _lastT = timestamp;
}

Pose MotionModelMovingAverage::predictPose(Timestamp timestamp) const
{
  const double dT = (static_cast<double>(timestamp) - static_cast<double>(_lastT)) / 1e9;
  const Pose predictedRelativePose(SE3d::exp(_speed * dT), dT * _speedCov);
  return Pose(predictedRelativePose * _lastPose);
}
Pose MotionModelMovingAverage::pose() const { return Pose(_lastPose.pose(), _lastPose.cov()); }
Pose MotionModelMovingAverage::speed() const { return Pose(SE3d::exp(_speed), _lastPose.cov()); }

MotionModelConstantSpeedKalman::MotionModelConstantSpeedKalman(
  const Matd<12, 12> & covProcess, const Matd<12, 12> & covState)
: MotionModel(),
  _kalman(std::make_unique<EKFConstantVelocitySE3>(
    covProcess, std::numeric_limits<Timestamp>::max(), covState)),
  _lastPose(SE3d(), MatXd::Identity(6, 6)),
  _lastT(0U)
{
}
Pose MotionModelConstantSpeedKalman::predictPose(Timestamp timestamp) const
{
  Pose pose = _lastPose;
  if (_kalman->t() != std::numeric_limits<uint64_t>::max()) {
    auto state = _kalman->predict(timestamp);
    pose = Pose(state->pose(), state->covPose());
  }
  LOG_ODOM(DEBUG) << format(
    "Prediction: {} +- {}", pose.mean().transpose(), pose.twistCov().diagonal().transpose());
  return pose;
}
void MotionModelConstantSpeedKalman::update(const Pose & relativePose, Timestamp timestamp)
{
  if (_kalman->t() == std::numeric_limits<Timestamp>::max()) {
    _kalman->reset(timestamp, {Vec6d::Zero(), Vec6d::Zero(), _kalman->state()->covariance()});
  } else {
    _kalman->update(relativePose.twist(), relativePose.twistCov(), timestamp);
  }
  _lastPose = pose();
  _lastT = timestamp;
}

Pose MotionModelConstantSpeedKalman::speed() const
{
  auto state = _kalman->state();
  return Pose(SE3d::exp(state->velocity() / 1e9), state->covVelocity() / 1e9);
}

Pose MotionModelConstantSpeedKalman::pose() const
{
  auto state = _kalman->state();
  return Pose(SE3d::exp(state->pose()), state->covPose());
}

}  // namespace pd::vslam
