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

#ifndef VSLAM_MAP_OPTIMIZATION_H__
#define VSLAM_MAP_OPTIMIZATION_H__
#include <vector>

#include "BundleAdjustment.h"
#include "core/core.h"

namespace pd::vslam::mapping
{
class MapOptimization
{
public:
  typedef std::shared_ptr<MapOptimization> ShPtr;
  typedef std::unique_ptr<MapOptimization> UnPtr;
  typedef std::shared_ptr<const MapOptimization> ConstShPtr;
  typedef std::unique_ptr<const MapOptimization> ConstUnPtr;

  MapOptimization();

  void optimize(
    const std::vector<Frame::ShPtr> & frames, const std::vector<Point3D::ShPtr> & points) const;

private:
};
}  // namespace pd::vslam::mapping

#endif
