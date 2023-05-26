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

#ifndef VSLAM_POSE_H__
#define VSLAM_POSE_H__
#include <memory>

#include "types.h"
namespace vslam
{
class Pose
{
public:
  typedef std::shared_ptr<Pose> ShPtr;
  typedef std::unique_ptr<Pose> UnPtr;
  typedef std::shared_ptr<const Pose> ConstShPtr;
  typedef std::unique_ptr<const Pose> ConstUnPtr;

  Pose(const Vec6d & x = Vec6d::Zero(), const Matd<6, 6> & cov = Matd<6, 6>::Identity())
  : _pose(SE3d::exp(x)), _cov(cov)
  {
  }
  //PoseWithCovariance( const Vec3d& t, const Vec4d& q, const Matd<6,6>& cov):_x(SE3d(q,t).log()),_cov(cov){}
  Pose(const SE3d & pose, const Matd<6, 6> & cov) : _pose(pose), _cov(cov) {}

  const SE3d & pose() const { return _pose; }
  SE3d & pose() { return _pose; }
  const SE3d & SE3() const { return _pose; }
  SE3d & SE3() { return _pose; }

  Vec6d twist() const { return _pose.log(); }
  Matd<6, 6> cov() const { return _cov; }
  Matd<6, 6> twistCov() const { return _cov; }
  Vec6d mean() const { return _pose.log(); }
  Pose inverse() const;

private:
  SE3d _pose;
  Matd<6, 6> _cov;
};
Pose operator*(const Pose & p1, const Pose & p0);

}  // namespace vslam
#endif
