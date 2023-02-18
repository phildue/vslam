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
namespace motion_model
{
class NoMotion : public MotionModel
{
public:
  typedef std::shared_ptr<NoMotion> ShPtr;
  typedef std::unique_ptr<NoMotion> UnPtr;
  typedef std::shared_ptr<const NoMotion> ConstShPtr;
  typedef std::unique_ptr<const NoMotion> ConstUnPtr;
  NoMotion(double maxTranslationalVelocity, double maxAngularVelocity);
  void update(const Pose & relativePose, Timestamp timestamp) override;
  Pose predictPose(Timestamp UNUSED(timestamp)) const override;
  Pose pose() const override;
  Pose speed() const override;
  bool exceedsThresholds(const Vec6d & speed) const;

protected:
  const double _maxTranslationalVelocity, _maxAngularVelocity;
  Vec6d _speed = Vec6d::Zero();
  Mat6d _speedCov = Mat6d::Identity();
  Pose _lastPose;
  Timestamp _lastT;
};
class ConstantMotion : public NoMotion
{
public:
  typedef std::shared_ptr<ConstantMotion> ShPtr;
  typedef std::unique_ptr<ConstantMotion> UnPtr;
  typedef std::shared_ptr<const ConstantMotion> ConstShPtr;
  typedef std::unique_ptr<const ConstantMotion> ConstUnPtr;
  ConstantMotion(
    const Mat6d & covariance, double maxTranslationalVelocity, double maxAngularVelocity);
  Pose predictPose(Timestamp timestamp) const override;

private:
  Mat6d _covariance;
};

class ConstantMotionWindow : public MotionModel
{
public:
  typedef std::shared_ptr<ConstantMotionWindow> ShPtr;
  typedef std::unique_ptr<ConstantMotionWindow> UnPtr;
  typedef std::shared_ptr<const ConstantMotionWindow> ConstShPtr;
  typedef std::unique_ptr<const ConstantMotionWindow> ConstUnPtr;
  ConstantMotionWindow(Timestamp timeFrame);
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
class ConstantMotionKalman : public MotionModel
{
public:
  typedef std::shared_ptr<ConstantMotionKalman> ShPtr;
  typedef std::unique_ptr<ConstantMotionKalman> UnPtr;
  typedef std::shared_ptr<const ConstantMotionKalman> ConstShPtr;
  typedef std::unique_ptr<const ConstantMotionKalman> ConstUnPtr;
  ConstantMotionKalman(const Matd<12, 12> & covProcess, const Matd<12, 12> & covState);
  void update(const Pose & pose, Timestamp timestamp) override;

  Pose predictPose(Timestamp timestamp) const override;
  Pose pose() const override;
  Pose speed() const override;

private:
  const EKFConstantVelocitySE3::UnPtr _kalman;
  Pose _lastPose;
  Timestamp _lastT;
};
}  // namespace motion_model
}  // namespace pd::vslam
#endif  // VSLAM_MOTION_PREDICTION
