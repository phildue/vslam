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

namespace pd::vslam
{
/**
     * Extended Kalman Filter with Constant Velocity Model in SE3
     * 
     * State Vector x:                [pose, twist]
     * System Function F(x):          pose = log(exp(pose) * exp(twist*dt)) | p (+) v*dt
     *                                twist = twist 
     *                 J_F_x:         Ad_Exp(v*dt)^(-1)
     * 
     * Measurement Function H(x):     motion* = vel*dt
     *                     J_H_x:     dt    
     *                                
     *                                    
     * 
    */
class EKFConstantVelocitySE3
{
public:
  typedef std::shared_ptr<EKFConstantVelocitySE3> ShPtr;
  typedef std::unique_ptr<EKFConstantVelocitySE3> UnPtr;
  typedef std::shared_ptr<const EKFConstantVelocitySE3> ConstShPtr;
  using Mat12d = Matd<12, 12>;
  using Mat6d = Matd<6, 6>;

  class State
  {
  public:
    typedef std::shared_ptr<State> ShPtr;
    typedef std::unique_ptr<State> UnPtr;
    typedef std::shared_ptr<const State> ConstShPtr;
    typedef std::unique_ptr<const State> ConstUnPtr;
    State(const Vec6d & pose, const Vec6d & velocity, const Mat12d & covariance);

    Vec6d pose() const { return _state.block(0, 0, 6, 1); }
    Mat6d covPose() const { return _covariance.block(0, 0, 6, 6); }
    Vec6d velocity() const { return _state.block(6, 0, 6, 1); }
    Mat6d covVelocity() const { return _covariance.block(0, 0, 6, 6); }
    const Mat12d & covariance() const { return _covariance; }
    const Vec12d & state() const { return _state; }
    const Mat12d & P() const { return _covariance; }
    const Vec12d & x() const { return _state; }

  private:
    Vec12d _state;
    Mat12d _covariance;
  };

  EKFConstantVelocitySE3(
    const Matd<12, 12> & covProcess, Timestamp t0 = std::numeric_limits<uint64_t>::max(),
    const Matd<12, 12> & covState = Matd<12, 12>::Identity() * 200);

  State::UnPtr predict(Timestamp t) const;

  void update(const Vec6d & motion, const Matd<6, 6> & covMotion, Timestamp t);

  void reset(Timestamp t, const State & state);

  const Timestamp & t() const { return _t; }

  const Mat12d & covarianceProcess() const { return _covProcess; }
  const Matd<12, 12> & Q() const { return covarianceProcess(); }
  Matd<12, 12> & covarianceProcess() { return _covProcess; }
  Matd<12, 12> & Q() { return covarianceProcess(); }

  State::UnPtr state() const { return std::make_unique<State>(*_state); }

private:
  Matd<12, 12> computeJ_f_x(const SE3d & pose, const SE3d & vdt) const;
  Matd<6, 12> computeJ_h_x(double dt) const;
  Matd<12, 12> computeProcessNoise(double dt) const;
  State::UnPtr predict(Timestamp t, Mat12d & J_f_x) const;

  Timestamp _t;
  State::UnPtr _state;
  Matd<12, 12> _covProcess;
  Matd<12, 6> _K;

  double _q;
  size_t _windowSize;
  std::vector<Matd<12, 12>> _stateTransitionMats;
  std::vector<Matd<6, 6>> _innovMats;
  size_t _nSamples;

  PlotKalman::ShPtr _log;
};
}  // namespace pd::vslam
#endif  //VSLAM_KALMAN_H__
