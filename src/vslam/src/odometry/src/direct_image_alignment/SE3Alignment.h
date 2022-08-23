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

#ifndef VSLAM_SE3_ALIGNMENT
#define VSLAM_SE3_ALIGNMENT

#include "AlignmentSE3.h"
#include "core/core.h"
#include "least_squares/least_squares.h"
#include "lukas_kanade/lukas_kanade.h"
namespace pd::vslam
{
class SE3Alignment : public AlignmentSE3
{
public:
  typedef std::shared_ptr<SE3Alignment> ShPtr;
  typedef std::unique_ptr<SE3Alignment> UnPtr;
  typedef std::shared_ptr<const SE3Alignment> ConstShPtr;
  typedef std::unique_ptr<const SE3Alignment> ConstUnPtr;

  SE3Alignment(
    double minGradient, vslam::least_squares::Solver::ShPtr solver,
    vslam::least_squares::Loss::ShPtr loss, bool includePrior = false);

  PoseWithCovariance::UnPtr align(Frame::ConstShPtr from, Frame::ConstShPtr to) const override;
  PoseWithCovariance::UnPtr align(
    const std::vector<Frame::ConstShPtr> & from, Frame::ConstShPtr to) const;

protected:
  const double _minGradient2;
  const vslam::least_squares::Loss::ShPtr _loss;
  const vslam::least_squares::Solver::ShPtr _solver;
  const bool _includePrior;
};
}  // namespace pd::vslam
#endif  // VSLAM_SE3_ALIGNMENT
