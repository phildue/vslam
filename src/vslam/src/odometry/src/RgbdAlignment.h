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

#ifndef VSLAM_ALIGNER_H__
#define VSLAM_ALIGNER_H__

#include <core/core.h>
namespace pd::vslam
{
class RgbdAlignment
{
public:
  typedef std::shared_ptr<RgbdAlignment> ShPtr;
  typedef std::unique_ptr<RgbdAlignment> UnPtr;
  typedef std::shared_ptr<const RgbdAlignment> ConstShPtr;
  typedef std::unique_ptr<const RgbdAlignment> ConstUnPtr;

  virtual PoseWithCovariance::UnPtr align(Frame::ConstShPtr from, Frame::ConstShPtr to) const = 0;
  virtual PoseWithCovariance::UnPtr align(
    const Frame::VecConstShPtr & UNUSED(from), Frame::ConstShPtr UNUSED(to)) const
  {
    throw pd::Exception("Method not implemented.");
  }
};
}  // namespace pd::vslam

#endif  //VSLAM_ALIGNER_H__
