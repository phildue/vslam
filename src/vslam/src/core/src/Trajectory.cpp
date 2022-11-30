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

#include <fmt/chrono.h>
#include <fmt/core.h>
using fmt::format;

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
  using time::to_time_point;
  auto it = std::find_if(_poses.begin(), _poses.end(), [&](auto p) { return t < p.first; });
  if (it == _poses.begin() || it == _poses.end()) {
    auto ref = (it == _poses.begin()) ? _poses.begin()->first : _poses.rbegin()->first;
    throw pd::Exception(format(
      "Cannot interpolate to: [{:%Y-%m-%d %H:%M:%S}] it is outside the time range of [{:%Y-%m-%d "
      "%H:%M:%S}] to [{:%Y-%m-%d %H:%M:%S}] by [{:%S}] seconds.",
      to_time_point(t), to_time_point(_poses.begin()->first), to_time_point(_poses.rbegin()->first),
      to_time_point(t - ref)));
  }
  Timestamp t1 = it->first;
  auto p1 = it->second;
  --it;
  Timestamp t0 = it->first;
  auto p0 = it->second;

  const int64_t dT = static_cast<int64_t>(t1) - static_cast<int64_t>(t0);

  const Vec6d speed =
    algorithm::computeRelativeTransform(p0->pose(), p1->pose()).log() / static_cast<double>(dT);
  const SE3d dPose = SE3d::exp((static_cast<double>(t) - static_cast<double>(t0)) * speed);
  return std::make_shared<PoseWithCovariance>(dPose * p0->pose(), p0->cov());
}
void Trajectory::append(Timestamp t, PoseWithCovariance::ConstShPtr pose) { _poses[t] = pose; }
Timestamp Trajectory::tStart() const { return _poses.begin()->first; }
Timestamp Trajectory::tEnd() const { return _poses.rbegin()->first; }

}  // namespace pd::vslam
