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

#ifndef VSLAM_RGBD_ALIGNMENT_OCV_H__
#define VSLAM_RGBD_ALIGNMENT_OCV_H__

#include <core/core.h>
#include <least_squares/least_squares.h>

#include "RgbdAlignment.h"
namespace pd::vslam
{
class RgbdAlignmentOcv : public RgbdAlignment
{
public:
  typedef std::shared_ptr<RgbdAlignmentOcv> ShPtr;
  typedef std::unique_ptr<RgbdAlignmentOcv> UnPtr;
  typedef std::shared_ptr<const RgbdAlignmentOcv> ConstShPtr;
  typedef std::unique_ptr<const RgbdAlignmentOcv> ConstUnPtr;

  RgbdAlignmentOcv(
    bool initializeOnPrediction = true, int nLevels = 4,
    const std::vector<double> & minGradient = {0, 0, 0, 0, 0}, double minDepth = 0.1,
    double maxDepth = 50, double maxDepthDiff = 0.1,
    const std::vector<double> & maxPointsPart = {1.0, 1.0, 1.0, 1.0, 1.0},
    const std::string & odometryType = "RgbdOdometry");

  Pose align(Frame::ConstShPtr from, Frame::ConstShPtr to) const override;
  Pose align(const Frame::VecConstShPtr & from, Frame::ConstShPtr to) const override;
  void preprocessReference(Frame::ShPtr frame) const override;
  void preprocessTarget(Frame::ShPtr frame) const override;

protected:
  std::string _odometryType;
};

}  // namespace pd::vslam

#endif  //VSLAM_ALIGNER_H__
