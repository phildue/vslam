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

#ifndef VSLAM_RGBD_ALIGNER
#define VSLAM_RGBD_ALIGNER

#include "RgbdAlignment.h"
#include "SE3Alignment.h"
#include "core/core.h"
#include "least_squares/least_squares.h"
#include "lukas_kanade/lukas_kanade.h"
namespace pd::vslam
{
class RGBDAligner : public RgbdAlignment
{
public:
  typedef std::shared_ptr<RGBDAligner> ShPtr;
  typedef std::unique_ptr<RGBDAligner> UnPtr;
  typedef std::shared_ptr<const RGBDAligner> ConstShPtr;
  typedef std::unique_ptr<const RGBDAligner> ConstUnPtr;

  RGBDAligner(
    double minGradient, vslam::least_squares::Solver::ShPtr solver,
    vslam::least_squares::Loss::ShPtr loss);

  PoseWithCovariance::UnPtr align(Frame::ConstShPtr from, Frame::ConstShPtr to) const override;

protected:
  const double _minGradient;
  const vslam::least_squares::Loss::ShPtr _loss;
  const vslam::least_squares::Solver::ShPtr _solver;
  SE3Alignment::InterestPointSelection::ConstUnPtr _interestPointSelection;
};
}  // namespace pd::vslam
#endif  // VSLAM_SE3_ALIGNMENT
