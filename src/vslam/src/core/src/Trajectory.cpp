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

#include "Trajectory.h"
namespace pd::vslam
{
Trajectory::Trajectory() {}
Trajectory::Trajectory(const std::map<Timestamp, PoseWithCovariance::ConstShPtr> & poses)
: _poses(poses)
{
}
Trajectory::Trajectory(const std::map<Timestamp, SE3d> & poses)
{
  for (const auto & p : poses) {
    _poses[p.first] = std::make_shared<PoseWithCovariance>(p.second, MatXd::Identity(6, 6));
  }
}

PoseWithCovariance::ConstShPtr Trajectory::poseAt(Timestamp t, bool interpolate) const
{
  auto it = _poses.find(t);
  if (it == _poses.end()) {
    return interpolate ? interpolateAt(t)
                       : throw std::runtime_error("No pose at: " + std::to_string(t));
  } else {
    return it->second;
  }
}
PoseWithCovariance::ConstShPtr Trajectory::motionBetween(
  Timestamp t0, Timestamp t1, bool interpolate) const
{
  auto p0 = poseAt(t0, interpolate);
  return std::make_shared<PoseWithCovariance>(
    algorithm::computeRelativeTransform(p0->pose(), poseAt(t1, interpolate)->pose()), p0->cov());
}

PoseWithCovariance::ConstShPtr Trajectory::interpolateAt(Timestamp t) const
{
  Timestamp t0 = 0U, t1 = 0U;
  for (const auto & t_pose : _poses) {
    if (t_pose.first <= t) {
      t0 = t_pose.first;
    }
    if (t_pose.first > t) {
      t1 = t_pose.first;
    }
    if (t0 != 0 && t1 != 0) {
      break;
    }
  }
  if (_poses.find(t0) == _poses.end() || _poses.find(t1) == _poses.end()) {
    throw pd::Exception("Cannot interpolate to: [" + std::to_string(t) + "].");
  }
  // TODO(unknown): handle corner cases at boundaries
  const int64_t dT = static_cast<int64_t>(t1) - static_cast<int64_t>(t0);
  auto p0 = _poses.find(t0)->second;
  auto p1 = _poses.find(t1)->second;

  const Vec6d speed =
    algorithm::computeRelativeTransform(p0->pose(), p1->pose()).log() / static_cast<double>(dT);
  const SE3d dPose = SE3d::exp((static_cast<double>(t) - static_cast<double>(t0)) * speed);
  return std::make_shared<PoseWithCovariance>(dPose * p0->pose(), p0->cov());
}
void Trajectory::append(Timestamp t, PoseWithCovariance::ConstShPtr pose) { _poses[t] = pose; }

}  // namespace pd::vslam
