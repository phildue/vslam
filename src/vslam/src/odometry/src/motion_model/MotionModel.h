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

#include "EKFConstantVelocitySE3.h"
#include "core/core.h"
namespace pd::vslam
{
class MotionModel
{
public:
  typedef std::shared_ptr<MotionModel> ShPtr;
  typedef std::unique_ptr<MotionModel> UnPtr;
  typedef std::shared_ptr<const MotionModel> ConstShPtr;
  typedef std::unique_ptr<const MotionModel> ConstUnPtr;

  virtual ~MotionModel() = default;
  virtual void update(const Pose & relativePose, Timestamp timestamp) = 0;

  virtual Pose predictPose(uint64_t timestamp) const = 0;
  virtual Pose pose() const = 0;
  virtual Pose speed() const = 0;
};
class MotionModelNoMotion : public MotionModel
{
public:
  typedef std::shared_ptr<MotionModelNoMotion> ShPtr;
  typedef std::unique_ptr<MotionModelNoMotion> UnPtr;
  typedef std::shared_ptr<const MotionModelNoMotion> ConstShPtr;
  typedef std::unique_ptr<const MotionModelNoMotion> ConstUnPtr;
  MotionModelNoMotion();
  void update(const Pose & relativePose, Timestamp timestamp) override;
  Pose predictPose(Timestamp UNUSED(timestamp)) const override;
  Pose pose() const override;
  Pose speed() const override;

protected:
  Vec6d _speed = Vec6d::Zero();
  Mat6d _speedCov = Mat6d::Identity();
  Pose _lastPose;
  Timestamp _lastT;
};
class MotionModelConstantSpeed : public MotionModelNoMotion
{
public:
  typedef std::shared_ptr<MotionModelConstantSpeed> ShPtr;
  typedef std::unique_ptr<MotionModelConstantSpeed> UnPtr;
  typedef std::shared_ptr<const MotionModelConstantSpeed> ConstShPtr;
  typedef std::unique_ptr<const MotionModelConstantSpeed> ConstUnPtr;
  MotionModelConstantSpeed(const Mat6d & covariance = Mat6d::Identity());
  Pose predictPose(Timestamp timestamp) const override;

private:
  Mat6d _covariance;
};

class MotionModelMovingAverage : public MotionModel
{
public:
  typedef std::shared_ptr<MotionModelMovingAverage> ShPtr;
  typedef std::unique_ptr<MotionModelMovingAverage> UnPtr;
  typedef std::shared_ptr<const MotionModelMovingAverage> ConstShPtr;
  typedef std::unique_ptr<const MotionModelMovingAverage> ConstUnPtr;
  MotionModelMovingAverage(Timestamp timeFrame);
  void update(const Pose & relativePose, Timestamp timestamp) override;
  Pose predictPose(Timestamp timestamp) const override;
  Pose pose() const override;
  Pose speed() const override;

protected:
  const Timestamp _timeFrame;
  Trajectory::UnPtr _traj;
  Vec6d _speed = Vec6d::Zero();
  Mat6d _speedCov = Mat6d::Identity();
  Pose _lastPose;
  Timestamp _lastT;
};
class MotionModelConstantSpeedKalman : public MotionModel
{
public:
  typedef std::shared_ptr<MotionModelConstantSpeedKalman> ShPtr;
  typedef std::unique_ptr<MotionModelConstantSpeedKalman> UnPtr;
  typedef std::shared_ptr<const MotionModelConstantSpeedKalman> ConstShPtr;
  typedef std::unique_ptr<const MotionModelConstantSpeedKalman> ConstUnPtr;
  MotionModelConstantSpeedKalman(const Matd<12, 12> & covProcess, const Matd<12, 12> & covState);
  void update(const Pose & pose, Timestamp timestamp) override;

  Pose predictPose(Timestamp timestamp) const override;
  Pose pose() const override;
  Pose speed() const override;

private:
  const EKFConstantVelocitySE3::UnPtr _kalman;
  Pose _lastPose;
  Timestamp _lastT;
};
}  // namespace pd::vslam
#endif  // VSLAM_MOTION_PREDICTION
