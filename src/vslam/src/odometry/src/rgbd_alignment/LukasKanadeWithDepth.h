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

#ifndef VSLAM_LUKAS_KANADE_WITH_DEPTH_H__
#define VSLAM_LUKAS_KANADE_WITH_DEPTH_H__
#include <memory>

#include "IterativeClosestPoint.h"
#include "core/core.h"
#include "least_squares/least_squares.h"
#include "lukas_kanade/lukas_kanade.h"
namespace pd::vslam
{
class LukasKanadeWithDepth : public least_squares::Problem
{
public:
  LukasKanadeWithDepth(
    const SE3d & se3, Frame::ConstShPtr fRef, Frame::ConstShPtr fTo,
    const std::vector<Eigen::Vector2i> & interestPoints, int level, double w = 0.1,
    double maxDepthDiff = 0.1,
    least_squares::Loss::ShPtr = std::make_shared<least_squares::QuadraticLoss>());

  virtual ~LukasKanadeWithDepth() = default;

  void updateX(const Eigen::VectorXd & dx) override;
  void setX(const Eigen::VectorXd & x) override;

  Eigen::VectorXd x() const override { return _se3.log(); }

  least_squares::NormalEquations::UnPtr computeNormalEquations() override;

protected:
  const double _w;
  const double _sqrtw;
  const lukas_kanade::WarpSE3::ShPtr _warp;
  const lukas_kanade::InverseCompositional::UnPtr _lk;
  const IterativeClosestPoint::UnPtr _icp;
  SE3d _se3;
};

}  // namespace pd::vslam
#endif
