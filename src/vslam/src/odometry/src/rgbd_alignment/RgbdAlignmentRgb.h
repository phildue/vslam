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

#ifndef VSLAM_ALIGNERRGB_H__
#define VSLAM_ALIGNERRGB_H__

#include <core/core.h>
#include <least_squares/least_squares.h>

#include "RgbdAlignment.h"
namespace pd::vslam
{
class RgbdAlignmentRgb : public RgbdAlignment
{
public:
  typedef std::shared_ptr<RgbdAlignmentRgb> ShPtr;
  typedef std::unique_ptr<RgbdAlignmentRgb> UnPtr;
  typedef std::shared_ptr<const RgbdAlignmentRgb> ConstShPtr;
  typedef std::unique_ptr<const RgbdAlignmentRgb> ConstUnPtr;

  RgbdAlignmentRgb(
    vslam::least_squares::Solver::ShPtr solver, vslam::least_squares::Loss::ShPtr loss,
    bool includePrior = false, bool initializeOnPrediction = true,
    const std::vector<double> & minGradient = {0, 0, 0, 0, 0}, double minDepth = 0.1,
    double maxDepth = 50, const std::vector<double> & maxPointsPart = {1.0, 1.0, 1.0, 1.0, 1.0});

  Pose align(Frame::ConstShPtr from, Frame::ConstShPtr to) const override;
  least_squares::Problem::UnPtr setupProblem(
    const Vec6d & twist, Frame::ConstShPtr from, Frame::ConstShPtr to, int level) const override;

protected:
  std::vector<Vec2i> selectInterestPoints(Frame::ConstShPtr frame, int level) const override;
};

}  // namespace pd::vslam

#endif  //VSLAM_ALIGNER_H__
