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

#ifndef VSLAM_ODOMETRY
#define VSLAM_ODOMETRY

#include <core/core.h>

#include "direct_image_alignment/RgbdAlignmentOpenCv.h"
#include "direct_image_alignment/SE3Alignment.h"
#include "iterative_closest_point/IterativeClosestPoint.h"
#include "iterative_closest_point/IterativeClosestPointOcv.h"
#include "mapping/Map.h"

namespace pd::vslam
{
class Odometry
{
public:
  typedef std::shared_ptr<Odometry> ShPtr;
  typedef std::unique_ptr<Odometry> UnPtr;
  typedef std::shared_ptr<const Odometry> ConstShPtr;
  typedef std::unique_ptr<const Odometry> ConstUnPtr;

  virtual void update(Frame::ConstShPtr frame) = 0;

  virtual PoseWithCovariance::ConstShPtr pose() const = 0;
  virtual PoseWithCovariance::ConstShPtr speed() const = 0;

  static ShPtr make();
};

class OdometryRgbd : public Odometry
{
public:
  typedef std::shared_ptr<OdometryRgbd> ShPtr;
  typedef std::unique_ptr<OdometryRgbd> UnPtr;
  typedef std::shared_ptr<const OdometryRgbd> ConstShPtr;
  typedef std::unique_ptr<const OdometryRgbd> ConstUnPtr;

  OdometryRgbd(
    double minGradient, least_squares::Solver::ShPtr solver, least_squares::Loss::ShPtr loss,
    Map::ConstShPtr map, bool includeKeyFrame = false, bool trackKeyFrame = false);

  void update(Frame::ConstShPtr frame) override;

  PoseWithCovariance::ConstShPtr pose() const override { return _pose; }
  PoseWithCovariance::ConstShPtr speed() const override { return _speed; }

protected:
  const SE3Alignment::ConstShPtr _aligner;
  const Map::ConstShPtr _map;
  const bool _includeKeyFrame, _trackKeyFrame;
  PoseWithCovariance::ConstShPtr _speed;
  PoseWithCovariance::ConstShPtr _pose;
};
class OdometryIcp : public Odometry
{
public:
  typedef std::shared_ptr<OdometryIcp> ShPtr;
  typedef std::unique_ptr<OdometryIcp> UnPtr;
  typedef std::shared_ptr<const OdometryIcp> ConstShPtr;
  typedef std::unique_ptr<const OdometryIcp> ConstUnPtr;

  OdometryIcp(int level, int maxIterations, Map::ConstShPtr map);

  void update(Frame::ConstShPtr frame) override;

  PoseWithCovariance::ConstShPtr pose() const override { return _pose; }
  PoseWithCovariance::ConstShPtr speed() const override { return _speed; }

protected:
  const IterativeClosestPoint::ConstShPtr _aligner;
  PoseWithCovariance::ConstShPtr _speed;
  PoseWithCovariance::ConstShPtr _pose;
  const Map::ConstShPtr _map;
};

}  // namespace pd::vslam
#endif  // VSLAM_ODOMETRY
