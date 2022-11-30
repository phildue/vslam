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

#ifndef VSLAM_KALMAN_FILTER_SE3_H__
#define VSLAM_KALMAN_FILTER_SE3_H__

#include "core/core.h"
namespace pd::vslam::odometry
{
class EKFConstantVelocitySE3
{
public:
  typedef std::shared_ptr<EKFConstantVelocitySE3> ShPtr;
  typedef std::unique_ptr<EKFConstantVelocitySE3> UnPtr;
  typedef std::shared_ptr<const EKFConstantVelocitySE3> ConstPtr;

  struct State
  {
    typedef std::shared_ptr<State> ShPtr;
    typedef std::unique_ptr<State> UnPtr;
    typedef std::shared_ptr<const State> ConstPtr;

    Vec6d pose;
    Vec6d velocity;
    Matd<6, 6> covPose;
    Matd<6, 6> covVel;
  };

  EKFConstantVelocitySE3(
    const Matd<12, 12> & covProcess, Timestamp t0 = std::numeric_limits<uint64_t>::max(),
    const Matd<12, 12> & covState = Matd<12, 12>::Identity());

  State::UnPtr predict(Timestamp t) const;

  void update(const Vec6d & motion, const Matd<6, 6> & covMotion, Timestamp t);
  const Timestamp & t() const { return _t; }
  const Vec6d & pose() const { return _pose; }
  const Vec6d & velocity() const { return _velocity; }
  const Matd<12, 12> & covState() const { return _covState; }
  const Matd<12, 12> & covProcess() const { return _covProcess; }

private:
  Matd<12, 12> computeJacobianProcess(const SE3d & pose) const;
  Matd<6, 12> computeJacobianMeasurement(Timestamp t) const;

  Timestamp _t;
  Vec6d _pose;
  Vec6d _velocity;
  Matd<12, 12> _covState;
  Matd<12, 12> _covProcess;
};
}  // namespace pd::vslam::odometry
#endif  //VSLAM_KALMAN_H__
