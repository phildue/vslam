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

#ifndef VSLAM_BUNDLE_ADJUSTMENT_H__
#define VSLAM_BUNDLE_ADJUSTMENT_H__
#include <ceres/ceres.h>

#include <vector>

#include "core/core.h"

namespace pd::vslam::mapping
{
class BundleAdjustment
{
public:
  typedef std::shared_ptr<BundleAdjustment> ShPtr;
  typedef std::unique_ptr<BundleAdjustment> UnPtr;
  typedef std::shared_ptr<const BundleAdjustment> ConstShPtr;
  typedef std::unique_ptr<const BundleAdjustment> ConstUnPtr;

  BundleAdjustment();
  void optimize();

  void setFrame(std::uint64_t frameId, const SE3d & pose, const Mat3d & K);
  void setPoint(std::uint64_t pointId, const Vec3d & point);
  void setObservation(std::uint64_t pointId, std::uint64_t frameId, const Vec2d & observation);

  SE3d getPose(std::uint64_t frameId) const;
  Vec3d getPoint(std::uint64_t pointId) const;
  double computeError() const;
  size_t nPoints() const { return _points.size(); }
  size_t nPoses() const { return _poses.size(); }

private:
  //compute error is not const?
  mutable ceres::Problem _problem;
  std::map<std::uint64_t, SE3d> _poses;
  std::map<std::uint64_t, Mat3d> _Ks;
  std::map<std::uint64_t, Vec3d> _points;
};
}  // namespace pd::vslam::mapping

#endif
