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
std::pair<Timestamp, PoseWithCovariance::ConstShPtr> Trajectory::nearestPoseAt(Timestamp t) const
{
  std::pair<Timestamp, PoseWithCovariance::ConstShPtr> min;
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
PoseWithCovariance::ConstShPtr Trajectory::meanMotion(Timestamp t0, Timestamp t1) const
{
  auto itRef = std::find_if(_poses.begin(), _poses.end(), [&](auto p) { return t0 >= p.first; });
  auto end = std::find_if(_poses.begin(), _poses.end(), [&](auto p) { return t1 <= p.first; });

  std::vector<Vec6d> relativePoses;
  relativePoses.reserve(_poses.size());
  auto itCur = itRef;
  ++itCur;

  while (itCur != end) {
    const double dT = itCur->first - itRef->first;
    relativePoses.push_back(
      algorithm::computeRelativeTransform(itRef->second->pose(), itCur->second->pose()).log() / dT);
    ++itCur;
    ++itRef;
  }
  return computeMean(relativePoses);
}
Trajectory Trajectory::inverse() const
{
  std::map<Timestamp, Pose::ConstShPtr> posesInverted;
  for (auto t_p : _poses) {
    posesInverted[t_p.first] = std::make_shared<Pose>(t_p.second->inverse());
  }
  return Trajectory(posesInverted);
}

PoseWithCovariance::ConstShPtr Trajectory::meanMotion() const
{
  std::vector<Vec6d> relativePoses;
  relativePoses.reserve(_poses.size());
  auto itRef = _poses.begin();
  auto itCur = _poses.begin();
  ++itCur;

  for (; itCur != _poses.end(); ++itCur) {
    const double dT = itCur->first - itRef->first;
    relativePoses.push_back(
      algorithm::computeRelativeTransform(itRef->second->pose(), itCur->second->pose()).log() / dT);
    ++itRef;
  }
  return computeMean(relativePoses);
}

PoseWithCovariance::ConstShPtr Trajectory::meanMotion(
  Timestamp t0, Timestamp t1, Timestamp dT) const
{
  std::vector<Vec6d> relativePoses;
  relativePoses.reserve(_poses.size());
  for (Timestamp t = t0 + dT; t < t1; t += dT) {
    relativePoses.push_back(
      algorithm::computeRelativeTransform(poseAt(t - dT)->pose(), poseAt(t)->pose()).log() /
      static_cast<double>(dT));
  }
  return computeMean(relativePoses);
}
PoseWithCovariance::ConstShPtr Trajectory::meanMotion(Timestamp dT) const
{
  return meanMotion(tStart(), tEnd(), dT);
}
PoseWithCovariance::ConstShPtr Trajectory::meanAcceleration(Timestamp t0, Timestamp t1) const
{
  auto itRef = std::find_if(_poses.begin(), _poses.end(), [&](auto p) { return t0 >= p.first; });
  auto end = std::find_if(_poses.begin(), _poses.end(), [&](auto p) { return t1 <= p.first; });

  std::vector<Vec6d> relativePoses;
  std::vector<double> dTs;
  relativePoses.reserve(_poses.size());
  auto itCur = itRef;
  ++itCur;

  while (itCur != end) {
    const double dT = itCur->first - itRef->first;
    relativePoses.push_back(
      algorithm::computeRelativeTransform(itRef->second->pose(), itCur->second->pose()).log() / dT);
    ++itCur;
    ++itRef;
    dTs.push_back(dT);
  }

  std::vector<Vec6d> accelerations(relativePoses.size() - 1);
  for (size_t i = 1; i < relativePoses.size(); i++) {
    accelerations[i - 1] = algorithm::computeRelativeTransform(
                             SE3d::exp(relativePoses[i - 1]), SE3d::exp(relativePoses[i]))
                             .log() /
                           dTs[i];
  }
  return computeMean(relativePoses);
}
PoseWithCovariance::ConstShPtr Trajectory::meanAcceleration() const
{
  std::vector<Vec6d> relativePoses;
  std::vector<double> dTs;
  relativePoses.reserve(_poses.size());
  auto itRef = _poses.begin();
  auto itCur = _poses.begin();
  ++itCur;

  for (; itCur != _poses.end(); ++itCur) {
    const double dT = itCur->first - itRef->first;
    relativePoses.push_back(
      algorithm::computeRelativeTransform(itRef->second->pose(), itCur->second->pose()).log() / dT);
    dTs.push_back(dT);
    ++itRef;
  }
  std::vector<Vec6d> accelerations(relativePoses.size() - 1);
  for (size_t i = 1; i < relativePoses.size(); i++) {
    accelerations[i - 1] = algorithm::computeRelativeTransform(
                             SE3d::exp(relativePoses[i - 1]), SE3d::exp(relativePoses[i]))
                             .log() /
                           dTs[i];
  }
  return computeMean(accelerations);
}

PoseWithCovariance::ConstShPtr Trajectory::meanAcceleration(
  Timestamp t0, Timestamp t1, Timestamp dT) const
{
  std::vector<Vec6d> relativePoses;
  relativePoses.reserve(_poses.size());
  for (Timestamp t = t0 + dT; t < t1; t += dT) {
    relativePoses.push_back(
      algorithm::computeRelativeTransform(poseAt(t - dT)->pose(), poseAt(t)->pose()).log() /
      static_cast<double>(dT));
  }
  std::vector<Vec6d> accelerations(relativePoses.size() - 1);
  for (size_t i = 1; i < relativePoses.size(); i++) {
    accelerations[i - 1] = algorithm::computeRelativeTransform(
                             SE3d::exp(relativePoses[i - 1]), SE3d::exp(relativePoses[i]))
                             .log() /
                           static_cast<double>(dT);
  }
  return computeMean(accelerations);
}
PoseWithCovariance::ConstShPtr Trajectory::meanAcceleration(Timestamp dT) const
{
  return meanAcceleration(tStart(), tEnd(), dT);
}
PoseWithCovariance::ConstShPtr Trajectory::computeMean(const std::vector<Vec6d> & poses) const
{
  if (poses.size() < 2) {
    throw pd::Exception(format("Not enough available poses to compute statistics."));
  }
  Vec6d sum = Vec6d::Zero();
  for (size_t i = 0; i < poses.size(); i++) {
    sum += poses[i];
  }
  const Vec6d mean = sum / poses.size();
  Matd<6, 6> cov = Matd<6, 6>::Zero();
  for (size_t i = 0; i < poses.size(); i++) {
    const auto diff = poses[i] - mean;
    cov += diff * diff.transpose();
  }
  cov /= poses.size() - 1;
  return std::make_shared<PoseWithCovariance>(SE3d::exp(mean), cov);
}

}  // namespace pd::vslam
