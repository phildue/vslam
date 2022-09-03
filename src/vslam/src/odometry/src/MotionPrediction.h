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

#ifndef VSLAM_MOTION_PREDICTION
#define VSLAM_MOTION_PREDICTION

#include "core/core.h"
#include "kalman/kalman.h"
namespace pd::vslam
{
class MotionPrediction
{
public:
  typedef std::shared_ptr<MotionPrediction> ShPtr;
  typedef std::unique_ptr<MotionPrediction> UnPtr;
  typedef std::shared_ptr<const MotionPrediction> ConstShPtr;
  typedef std::unique_ptr<const MotionPrediction> ConstUnPtr;

  virtual void update(PoseWithCovariance::ConstShPtr pose, Timestamp timestamp) = 0;
  virtual PoseWithCovariance::UnPtr predict(uint64_t timestamp) const = 0;

  static ShPtr make(const std::string & model);
};
class MotionPredictionNoMotion : public MotionPrediction
{
public:
  typedef std::shared_ptr<MotionPredictionNoMotion> ShPtr;
  typedef std::unique_ptr<MotionPredictionNoMotion> UnPtr;
  typedef std::shared_ptr<const MotionPredictionNoMotion> ConstShPtr;
  typedef std::unique_ptr<const MotionPredictionNoMotion> ConstUnPtr;
  MotionPredictionNoMotion()
  : MotionPrediction(),
    _lastPose(std::make_shared<PoseWithCovariance>(SE3d(), MatXd::Identity(6, 6)))
  {
  }

  void update(PoseWithCovariance::ConstShPtr pose, Timestamp UNUSED(timestamp)) override
  {
    _lastPose = pose;
  }
  PoseWithCovariance::UnPtr predict(Timestamp UNUSED(timestamp)) const override
  {
    return std::make_unique<PoseWithCovariance>(_lastPose->pose(), _lastPose->cov());
  }

private:
  PoseWithCovariance::ConstShPtr _lastPose;
};
class MotionPredictionConstant : public MotionPrediction
{
public:
  typedef std::shared_ptr<MotionPredictionConstant> ShPtr;
  typedef std::unique_ptr<MotionPredictionConstant> UnPtr;
  typedef std::shared_ptr<const MotionPredictionConstant> ConstShPtr;
  typedef std::unique_ptr<const MotionPredictionConstant> ConstUnPtr;
  MotionPredictionConstant()
  : MotionPrediction(),
    _speed(Vec6d::Zero()),
    _lastPose(std::make_shared<PoseWithCovariance>(SE3d(), MatXd::Identity(6, 6))),
    _lastT(0U)
  {
  }

  void update(PoseWithCovariance::ConstShPtr pose, Timestamp timestamp) override;
  PoseWithCovariance::UnPtr predict(Timestamp timestamp) const override;

private:
  Vec6d _speed = Vec6d::Zero();
  PoseWithCovariance::ConstShPtr _lastPose;
  Timestamp _lastT;
};
class MotionPredictionKalman : public MotionPrediction
{
public:
  typedef std::shared_ptr<MotionPredictionKalman> ShPtr;
  typedef std::unique_ptr<MotionPredictionKalman> UnPtr;
  typedef std::shared_ptr<const MotionPredictionKalman> ConstShPtr;
  typedef std::unique_ptr<const MotionPredictionKalman> ConstUnPtr;
  MotionPredictionKalman();
  void update(PoseWithCovariance::ConstShPtr pose, Timestamp timestamp) override;
  PoseWithCovariance::UnPtr predict(Timestamp timestamp) const override;

private:
  const kalman::EKFConstantVelocitySE3::UnPtr _kalman;
  Vec6d _speed = Vec6d::Zero();
  PoseWithCovariance::ConstShPtr _lastPose;
  Timestamp _lastT;
};
}  // namespace pd::vslam
#endif  // VSLAM_MOTION_PREDICTION
