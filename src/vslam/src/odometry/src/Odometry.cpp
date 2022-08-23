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

#include "Odometry.h"
#include "utils/utils.h"
#define LOG_ODOM(level) CLOG(level, "odometry")
namespace pd::vslam
{
OdometryRgbd::OdometryRgbd(
  double minGradient, least_squares::Solver::ShPtr solver, least_squares::Loss::ShPtr loss,
  Map::ConstShPtr map)
: _aligner(std::make_shared<SE3Alignment>(minGradient, solver, loss, true)),
  _map(map),
  _includeKeyFrame(true),
  _trackKeyFrame(false)
{
  Log::get("odometry");
}
void OdometryRgbd::update(Frame::ConstShPtr frame)
{
  if (_map->lastFrame()) {
    try {
      if (_includeKeyFrame && _map->lastKf() && _map->lastKf() != _map->lastFrame()) {
        _pose = _aligner->align({_map->lastKf(), _map->lastFrame()}, frame);

      } else if (_trackKeyFrame && _map->lastKf()) {
        _pose = _aligner->align(_map->lastKf(), frame);

      } else {
        _pose = _aligner->align(_map->lastFrame(), frame);
      }
      auto dT = frame->t() - _map->lastFrame()->t();
      _speed = std::make_shared<PoseWithCovariance>(
        SE3d::exp(
          algorithm::computeRelativeTransform(_map->lastFrame()->pose().pose(), _pose->pose())
            .log() /
          (static_cast<double>(dT) / 1e9)),
        _pose->cov());

    } catch (const std::runtime_error & e) {
      LOG_ODOM(ERROR) << e.what();
      _pose = std::make_shared<PoseWithCovariance>(frame->pose());
      _speed = std::make_shared<PoseWithCovariance>();
    }
  } else {
    _pose = std::make_shared<PoseWithCovariance>(frame->pose());
    _speed = std::make_shared<PoseWithCovariance>();
    LOG_ODOM(DEBUG) << "Processing first frame";
  }
}

OdometryIcp::OdometryIcp(int level, int maxIterations, Map::ConstShPtr map)
: _aligner(std::make_shared<IterativeClosestPoint>(level, maxIterations)), _map(map)
{
}
void OdometryIcp::update(Frame::ConstShPtr frame)
{
  if (_map->lastFrame()) {
    LOG_ODOM(DEBUG) << "Processing frame";
    _pose = _aligner->align(_map->lastFrame(), frame);
    auto dT = frame->t() - _map->lastFrame()->t();
    _speed = std::make_shared<PoseWithCovariance>(
      SE3d::exp(
        algorithm::computeRelativeTransform(_map->lastFrame()->pose().pose(), _pose->pose()).log() /
        (static_cast<double>(dT) / 1e9)),
      _pose->cov());

  } else {
    _pose = std::make_shared<PoseWithCovariance>(frame->pose());
    _speed = std::make_shared<PoseWithCovariance>();
    LOG_ODOM(DEBUG) << "Processing first frame";
  }
}

}  // namespace pd::vslam
