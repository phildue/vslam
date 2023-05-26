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
#include "utils.h"
namespace vslam
{
Trajectory::Trajectory() {}
Trajectory::Trajectory(const std::map<Timestamp, Pose::ConstShPtr> & poses) : _poses(poses) {}
Trajectory::Trajectory(const std::map<Timestamp, SE3d> & poses)
{
  for (const auto & p : poses) {
    _poses[p.first] = std::make_shared<Pose>(p.second, MatXd::Identity(6, 6));
  }
}

Pose::ConstShPtr Trajectory::poseAt(Timestamp t, bool interpolate) const
{
  auto it = _poses.find(t);
  if (it == _poses.end()) {
    return interpolate ? interpolateAt(t)
                       : throw std::runtime_error("No pose at: " + std::to_string(t));
  } else {
    return it->second;
  }
}
std::pair<Timestamp, Pose::ConstShPtr> Trajectory::nearestPoseAt(Timestamp t) const
{
  std::pair<Timestamp, Pose::ConstShPtr> min;
  double minDiff = std::numeric_limits<double>::max();
  double lastDiff = std::numeric_limits<double>::max();
  for (auto t_p : _poses) {
    double diff = std::abs(static_cast<double>(min.first) - static_cast<double>(t));
    if (diff < minDiff) {
      min = t_p;
      minDiff = diff;
    }
    if (diff > lastDiff) {
      break;
    }
    lastDiff = diff;
  }
  return min;
}

Pose::ConstShPtr Trajectory::motionBetween(Timestamp t0, Timestamp t1, bool interpolate) const
{
  auto p0 = poseAt(t0, interpolate);
  return std::make_shared<Pose>(poseAt(t1, interpolate)->pose() * p0->pose().inverse(), p0->cov());
}

Pose::ConstShPtr Trajectory::interpolateAt(Timestamp t) const
{
  using time::to_time_point;
  auto it = std::find_if(_poses.begin(), _poses.end(), [&](auto p) { return t < p.first; });
  if (it == _poses.begin() || it == _poses.end()) {
    auto ref = (it == _poses.begin()) ? _poses.begin()->first : _poses.rbegin()->first;
    throw std::runtime_error(format(
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

  const Vec6d speed = (p1->pose() * p0->pose().inverse()).log() / static_cast<double>(dT);
  const SE3d dPose = SE3d::exp((static_cast<double>(t) - static_cast<double>(t0)) * speed);
  return std::make_shared<Pose>(dPose * p0->pose(), p0->cov());
}
void Trajectory::append(Timestamp t, Pose::ConstShPtr pose) { _poses[t] = pose; }
void Trajectory::append(Timestamp t, const Pose & pose)
{
  _poses[t] = std::make_shared<Pose>(pose);
}
Timestamp Trajectory::tStart() const { return _poses.begin()->first; }
Timestamp Trajectory::tEnd() const { return _poses.rbegin()->first; }
Trajectory Trajectory::inverse() const
{
  std::map<Timestamp, Pose::ConstShPtr> posesInverted;
  for (auto t_p : _poses) {
    posesInverted[t_p.first] = std::make_shared<Pose>(t_p.second->inverse());
  }
  return Trajectory(posesInverted);
}

}  // namespace vslam
