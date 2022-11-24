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
class MotionModel
{
public:
  typedef std::shared_ptr<MotionModel> ShPtr;
  typedef std::unique_ptr<MotionModel> UnPtr;
  typedef std::shared_ptr<const MotionModel> ConstShPtr;
  typedef std::unique_ptr<const MotionModel> ConstUnPtr;

  virtual void update(PoseWithCovariance::ConstShPtr pose, Timestamp timestamp) = 0;
  virtual void update(Frame::ConstShPtr frameRef, Frame::ConstShPtr frameCur) = 0;

  virtual PoseWithCovariance::UnPtr predictPose(uint64_t timestamp) const = 0;
  virtual PoseWithCovariance::UnPtr pose() const = 0;
  virtual PoseWithCovariance::UnPtr speed() const = 0;
};
class MotionModelNoMotion : public MotionModel
{
public:
  typedef std::shared_ptr<MotionModelNoMotion> ShPtr;
  typedef std::unique_ptr<MotionModelNoMotion> UnPtr;
  typedef std::shared_ptr<const MotionModelNoMotion> ConstShPtr;
  typedef std::unique_ptr<const MotionModelNoMotion> ConstUnPtr;
  MotionModelNoMotion();
  void update(PoseWithCovariance::ConstShPtr pose, Timestamp timestamp) override;
  void update(Frame::ConstShPtr frameRef, Frame::ConstShPtr frameCur) override;
  PoseWithCovariance::UnPtr predictPose(Timestamp UNUSED(timestamp)) const override;
  PoseWithCovariance::UnPtr pose() const override;
  PoseWithCovariance::UnPtr speed() const override;

protected:
  Vec6d _speed = Vec6d::Zero();
  PoseWithCovariance::ConstShPtr _lastPose;
  Timestamp _lastT;
};
class MotionModelConstantSpeed : public MotionModelNoMotion
{
public:
  typedef std::shared_ptr<MotionModelConstantSpeed> ShPtr;
  typedef std::unique_ptr<MotionModelConstantSpeed> UnPtr;
  typedef std::shared_ptr<const MotionModelConstantSpeed> ConstShPtr;
  typedef std::unique_ptr<const MotionModelConstantSpeed> ConstUnPtr;
  MotionModelConstantSpeed();
  PoseWithCovariance::UnPtr predictPose(Timestamp timestamp) const override;
};
class MotionModelConstantSpeedKalman : public MotionModel
{
public:
  typedef std::shared_ptr<MotionModelConstantSpeedKalman> ShPtr;
  typedef std::unique_ptr<MotionModelConstantSpeedKalman> UnPtr;
  typedef std::shared_ptr<const MotionModelConstantSpeedKalman> ConstShPtr;
  typedef std::unique_ptr<const MotionModelConstantSpeedKalman> ConstUnPtr;
  MotionModelConstantSpeedKalman();
  void update(PoseWithCovariance::ConstShPtr pose, Timestamp timestamp) override;
  void update(Frame::ConstShPtr frameRef, Frame::ConstShPtr frameCur) override;

  PoseWithCovariance::UnPtr predictPose(Timestamp timestamp) const override;
  PoseWithCovariance::UnPtr pose() const override;
  PoseWithCovariance::UnPtr speed() const override;

private:
  const kalman::EKFConstantVelocitySE3::UnPtr _kalman;
  PoseWithCovariance::ConstShPtr _lastPose;
  Timestamp _lastT;
};
}  // namespace pd::vslam
#endif  // VSLAM_MOTION_PREDICTION
