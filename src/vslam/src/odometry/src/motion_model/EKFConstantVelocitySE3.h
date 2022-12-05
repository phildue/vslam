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

#include "PlotKalman.h"
#include "core/core.h"
namespace pd::vslam::odometry
{
/**
     * Extended Kalman Filter with Constant Velocity Model in SE3
     * 
     * State Vector x:                [pose, twist]
     * System Function F(x):          pose = log(exp(pose) * exp(twist*dt)) | p (+) v*dt
     *                                twist = twist 
     *                 J_F_x:         Ad_Exp(v*dt)^(-1)
     * 
     * Measurement Function H(x):     motion = 0*p + vx * dT
     *                 J_H_x:         
     *        
     * 
    */
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

    Vec6d pose = Vec6d::Zero();
    Vec6d velocity = Vec6d::Zero();
    Matd<6, 6> covPose = Matd<6, 6>::Identity();
    Matd<6, 6> covVel = Matd<6, 6>::Identity();
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
  Timestamp & t() { return _t; }
  Matd<12, 12> & covProcess() { return _covProcess; }
  Vec6d & pose() { return _pose; }
  Vec12d state() const;

  PlotKalman::ConstShPtr plot() const { return _plot; }

private:
  Matd<12, 12> computeJacobianProcess(const SE3d & vdt) const;
  Matd<6, 12> computeJacobianMeasurement(double dt) const;

  Timestamp _t;
  Vec6d _pose;
  Vec6d _velocity;
  Matd<12, 12> _covState;
  Matd<12, 12> _covProcess;

  PlotKalman::ShPtr _plot;
};
}  // namespace pd::vslam::odometry
#endif  //VSLAM_KALMAN_H__
