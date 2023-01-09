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

#include "PoseWithCovariance.h"
namespace pd::vslam
{
PoseWithCovariance operator*(const SE3d & p1, const PoseWithCovariance & p0)
{
  //https://robotics.stackexchange.com/questions/2556/how-to-rotate-covariance
  const auto C = p0.cov();
  Matd<6, 6> R = Matd<6, 6>::Zero();
  R.block(0, 0, 3, 3) = p1.rotationMatrix();
  R.block(3, 3, 3, 3) = p1.rotationMatrix();

  return PoseWithCovariance(p1 * p0.pose(), R * C * R.transpose());
}
PoseWithCovariance operator*(const SE3d & p1, const PoseWithCovariance::ConstUnPtr & p0)
{
  return p1 * (*p0);
}
PoseWithCovariance operator*(const SE3d & p1, const PoseWithCovariance::ConstShPtr & p0)
{
  return p1 * (*p0);
}
PoseWithCovariance operator*(const SE3d & p1, const PoseWithCovariance::ShPtr & p0)
{
  return p1 * (*p0);
}
PoseWithCovariance operator*(const SE3d & p1, const PoseWithCovariance::UnPtr & p0)
{
  return p1 * (*p0);
}
}  // namespace pd::vslam
