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

#include "MotionPrediction.h"

#include "utils/utils.h"
#define LOG_MOTION_PREDICTION(level) CLOG(level, "motion_prediction")
namespace pd::vslam
{
MotionPrediction::ShPtr MotionPrediction::make(const std::string & model)
{
  if (model == "NoMotion") {
    return std::make_shared<MotionPredictionNoMotion>();
  } else if (model == "ConstantMotion") {
    return std::make_shared<MotionPredictionConstant>();

  } else if (model == "Kalman") {
    return std::make_shared<MotionPredictionKalman>();
  } else {
    LOG_MOTION_PREDICTION(WARNING)
      << "Unknown motion model! Falling back to constant motion model.";
    return std::make_shared<MotionPredictionConstant>();
  }
}

void MotionPredictionConstant::update(PoseWithCovariance::ConstShPtr pose, Timestamp timestamp)
{
  if (timestamp < _lastT) {
    throw pd::Exception("New timestamp is older than last one!");
  }
  const double dT = (static_cast<double>(timestamp) - static_cast<double>(_lastT)) / 1e9;

  _speed = algorithm::computeRelativeTransform(_lastPose->pose(), pose->pose()).log() / dT;
  _lastPose = pose;
  _lastT = timestamp;
}
PoseWithCovariance::UnPtr MotionPredictionConstant::predict(Timestamp timestamp) const
{
  const double dT = (static_cast<double>(timestamp) - static_cast<double>(_lastT)) / 1e9;
  const SE3d predictedRelativePose = SE3d::exp(_speed * dT);
  return std::make_unique<PoseWithCovariance>(
    predictedRelativePose * _lastPose->pose(), MatXd::Identity(6, 6));
}

MotionPredictionKalman::MotionPredictionKalman()
: MotionPrediction(),
  _kalman(std::make_unique<kalman::EKFConstantVelocitySE3>(Matd<12, 12>::Identity())),
  _speed(Vec6d::Zero()),
  _lastPose(std::make_shared<PoseWithCovariance>(SE3d(), MatXd::Identity(6, 6))),
  _lastT(0U)
{
}
PoseWithCovariance::UnPtr MotionPredictionKalman::predict(Timestamp timestamp) const
{
  kalman::EKFConstantVelocitySE3::State::ConstPtr state = _kalman->predict(timestamp);
  return std::make_unique<PoseWithCovariance>(SE3d::exp(state->pose), state->covPose);
}
void MotionPredictionKalman::update(PoseWithCovariance::ConstShPtr pose, Timestamp timestamp)
{
  if (timestamp < _lastT) {
    throw pd::Exception("New timestamp is older than last one!");
  }
  const double dT = (static_cast<double>(timestamp) - static_cast<double>(_lastT)) / 1e9;

  _speed = algorithm::computeRelativeTransform(_lastPose->pose(), pose->pose()).log() / dT;
  _lastPose = pose;
  _lastT = timestamp;
  _kalman->update(_speed, Matd<6, 6>::Identity(), timestamp);
}

}  // namespace pd::vslam
