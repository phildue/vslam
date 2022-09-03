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

#ifndef VSLAM_TRAJECTORY_H__
#define VSLAM_TRAJECTORY_H__
#include <core/core.h>

#include <map>

namespace pd::vslam
{
class Trajectory
{
public:
  typedef std::shared_ptr<Trajectory> ShPtr;
  typedef std::unique_ptr<Trajectory> UnPtr;
  typedef std::shared_ptr<const Trajectory> ConstShPtr;
  typedef std::unique_ptr<const Trajectory> ConstUnPtr;

  Trajectory();
  Trajectory(const std::map<Timestamp, PoseWithCovariance::ConstShPtr> & poses);
  Trajectory(const std::map<Timestamp, SE3d> & poses);
  PoseWithCovariance::ConstShPtr poseAt(Timestamp t, bool interpolate = true) const;
  PoseWithCovariance::ConstShPtr motionBetween(
    Timestamp t0, Timestamp t1, bool interpolate = true) const;
  void append(Timestamp t, PoseWithCovariance::ConstShPtr pose);
  const std::map<Timestamp, PoseWithCovariance::ConstShPtr> & poses() const { return _poses; }

private:
  PoseWithCovariance::ConstShPtr interpolateAt(Timestamp t) const;

  std::map<Timestamp, PoseWithCovariance::ConstShPtr> _poses;
};
}  // namespace pd::vslam

#endif
