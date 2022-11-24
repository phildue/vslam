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

#ifndef VSLAM_ICP_H__
#define VSLAM_ICP_H__

#include <pcl/console/time.h>  // TicToc
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/pcl_visualizer.h>

#include "RgbdAlignment.h"
#include "core/core.h"
#include "utils/utils.h"
namespace pd::vslam
{
class IterativeClosestPoint : public RgbdAlignment
{
public:
  typedef std::shared_ptr<IterativeClosestPoint> ShPtr;
  typedef std::unique_ptr<IterativeClosestPoint> UnPtr;
  typedef std::shared_ptr<const IterativeClosestPoint> ConstShPtr;
  typedef std::unique_ptr<const IterativeClosestPoint> ConstUnPtr;

  IterativeClosestPoint(size_t level, int maxIterations)
  : _level(level), _maxIterations(maxIterations)
  {
    Log::get("odometry");
  }

  PoseWithCovariance::UnPtr align(Frame::ConstShPtr from, Frame::ConstShPtr to) const override;

protected:
  size_t _level;
  int _maxIterations;
};
}  // namespace pd::vslam
#endif  // VSLAM_ICP_H__
