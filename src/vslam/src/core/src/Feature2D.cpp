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

//
// Created by phil on 30.06.21.
//

#include "Feature2D.h"

namespace pd::vslam
{
std::uint64_t Feature2D::_idCtr = 0U;

Feature2D::Feature2D(
  const Eigen::Vector2d & position, std::shared_ptr<Frame> frame, size_t level, double response,
  const Descriptor & descriptor, std::shared_ptr<Point3D> p3d)
: _position(position),
  _frame(frame),
  _level(level),
  _response(response),
  _descriptor(descriptor),
  _point(p3d),
  _id(_idCtr++)
{
}

}  // namespace pd::vslam
