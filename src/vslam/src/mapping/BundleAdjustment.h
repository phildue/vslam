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

  struct Results
  {
    typedef std::shared_ptr<Results> ShPtr;
    typedef std::unique_ptr<Results> UnPtr;
    typedef std::shared_ptr<const Results> ConstShPtr;
    typedef std::unique_ptr<const Results> ConstUnPtr;

    std::map<std::uint64_t, PoseWithCovariance> poses;
    std::map<std::uint64_t, Vec3d> positions;
    double errorBefore;
    double errorAfter;
  };

  BundleAdjustment(size_t maxIterations = 50, double huberConstant = 1.43);

  Results::ConstUnPtr optimize(
    const Frame::VecConstShPtr & frames, const Frame::VecConstShPtr & fixedFrames = {}) const;

private:
  const size_t _maxIterations;
  const double _huberConstant;
};
}  // namespace pd::vslam::mapping

#endif
