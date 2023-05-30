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
namespace pd::vslam::motion_model
{
NoMotion::NoMotion(double maxTranslationalVelocity, double maxAngularVelocity)
: MotionModel(),
  _maxTranslationalVelocity(maxTranslationalVelocity),
  _maxAngularVelocity(maxAngularVelocity),
  _speed(Vec6d::Zero()),
  _lastPose(SE3d(), MatXd::Identity(6, 6)),
  _lastT(0U)
{
}
bool NoMotion::exceedsThresholds(const Vec6d & speed) const
{
  using time::to_time_point;

  const double translationalVelocity = speed.block(0, 0, 3, 1).norm();
  const double angularVelocity = speed.block(3, 0, 3, 1).norm();
  const bool exceeds =
    translationalVelocity > _maxTranslationalVelocity || angularVelocity > _maxAngularVelocity;

  if (exceeds) {
    LOG_ODOM(WARNING) << format(
      "Speed exceeded, ignoring t={:%Y-%m-%d %H:%M:%S} with vt = {}/{} m/s, va = {}/{} deg/s",
      to_time_point(_lastT), translationalVelocity, _maxTranslationalVelocity,
      angularVelocity * 180.0 / M_PI, _maxAngularVelocity * 180.0 / M_PI);
  }
  return exceeds;
}

void NoMotion::update(const Pose & relativePose, Timestamp timestamp)
{
  if (timestamp < _lastT) {
    throw pd::Exception("New timestamp is older than last one!");
  }
  const double dT = (static_cast<double>(timestamp) - static_cast<double>(_lastT)) / 1e9;
  if (_lastT > 0 && dT > 0) {
    const Vec6d speed = relativePose.twist() / dT;
    const Mat6d speedCov = relativePose.twistCov() / dT;

    if (exceedsThresholds(speed)) return;

    _speed = speed;
    _speedCov = speedCov;
  }
  _lastPose = relativePose * _lastPose;
  _lastT = timestamp;
}

Pose NoMotion::predictPose(Timestamp UNUSED(timestamp)) const { return pose(); }
Pose NoMotion::pose() const { return Pose(_lastPose.pose(), _lastPose.cov()); }
Pose NoMotion::speed() const { return Pose(SE3d::exp(_speed), _speedCov); }

ConstantMotion::ConstantMotion(
  const Mat6d & covariance, double maxTranslationalVelocity, double maxAngularVelocity)
: NoMotion(maxTranslationalVelocity, maxAngularVelocity), _covariance(covariance)
{
}

Pose ConstantMotion::predictPose(Timestamp timestamp) const
{
  const double dT = (static_cast<double>(timestamp) - static_cast<double>(_lastT)) / 1e9;
  const Pose predictedRelativePose(SE3d::exp(_speed * dT), dT * _speedCov);
  Pose predictedPose(predictedRelativePose.SE3() * _lastPose.SE3(), _covariance);
  LOG_ODOM(DEBUG) << format(
    "Prediction: {} +- {}", predictedPose.mean().transpose(),
    predictedPose.twistCov().diagonal().transpose());
  return predictedPose;
}

ConstantMotionWindow::ConstantMotionWindow(Timestamp timeFrame)
: MotionModel(),
  _timeFrame(timeFrame),
  _traj(std::make_unique<Trajectory>()),
  _speed(Vec6d::Zero()),
  _lastPose(SE3d(), MatXd::Identity(6, 6)),
  _lastT(0U)
{
}
void ConstantMotionWindow::update(const Pose & relativePose, Timestamp timestamp)
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

Pose ConstantMotionWindow::predictPose(Timestamp timestamp) const
{
  const double dT = (static_cast<double>(timestamp) - static_cast<double>(_lastT)) / 1e9;
  const Pose predictedRelativePose(SE3d::exp(_speed * dT), dT * _speedCov);
  return Pose(predictedRelativePose * _lastPose);
}
Pose ConstantMotionWindow::pose() const { return Pose(_lastPose.pose(), _lastPose.cov()); }
Pose ConstantMotionWindow::speed() const { return Pose(SE3d::exp(_speed), _lastPose.cov()); }

ConstantMotionKalman::ConstantMotionKalman(
  const Matd<12, 12> & covProcess, const Matd<12, 12> & covState)
: MotionModel(),
  _kalman(std::make_unique<EKFConstantVelocitySE3>(
    covProcess, std::numeric_limits<Timestamp>::max(), covState)),
  _lastPose(SE3d(), MatXd::Identity(6, 6)),
  _lastT(0U)
{
}
Pose ConstantMotionKalman::predictPose(Timestamp timestamp) const
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
void ConstantMotionKalman::update(const Pose & relativePose, Timestamp timestamp)
{
  if (_kalman->t() == std::numeric_limits<Timestamp>::max()) {
    _kalman->reset(timestamp, {Vec6d::Zero(), Vec6d::Zero(), _kalman->state()->covariance()});
  } else {
    _kalman->update(relativePose.twist(), relativePose.twistCov(), timestamp);
  }
  _lastPose = pose();
  _lastT = timestamp;
}

Pose ConstantMotionKalman::speed() const
{
  auto state = _kalman->state();
  return Pose(SE3d::exp(state->velocity() / 1e9), state->covVelocity() / 1e9);
}

Pose ConstantMotionKalman::pose() const
{
  auto state = _kalman->state();
  return Pose(SE3d::exp(state->pose()), state->covPose());
}

}  // namespace pd::vslam::motion_model
