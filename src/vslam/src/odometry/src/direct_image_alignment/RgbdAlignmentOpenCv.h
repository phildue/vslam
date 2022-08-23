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

#ifndef VSLAM_RGBD_ALIGNMENT_OPENCV
#define VSLAM_RGBD_ALIGNMENT_OPENCV

#include "AlignmentSE3.h"
#include "core/core.h"
#include "lukas_kanade/lukas_kanade.h"

namespace pd::vslam
{
class RgbdAlignmentOpenCv : public AlignmentSE3
{
public:
  typedef std::shared_ptr<RgbdAlignmentOpenCv> ShPtr;
  typedef std::unique_ptr<RgbdAlignmentOpenCv> UnPtr;
  typedef std::shared_ptr<const RgbdAlignmentOpenCv> ConstShPtr;
  typedef std::unique_ptr<const RgbdAlignmentOpenCv> ConstUnPtr;

  RgbdAlignmentOpenCv();

  PoseWithCovariance::UnPtr align(Frame::ConstShPtr from, Frame::ConstShPtr to) const;

protected:
};
}  // namespace pd::vslam
#endif  // VSLAM_SE3_ALIGNMENT
