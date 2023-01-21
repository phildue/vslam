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

#include <algorithm>

#include "IterativeClosestPoint.h"
#include "LukasKanadeWithDepth.h"
#include "core/core.h"
#include "utils/utils.h"
namespace pd::vslam
{
LukasKanadeWithDepth::LukasKanadeWithDepth(
  const SE3d & se3, Frame::ConstShPtr fRef, Frame::ConstShPtr fTo,
  const std::vector<Eigen::Vector2i> & interestPoints, int level, double w, double maxDepthDiff,
  least_squares::Loss::ShPtr l)
: least_squares::Problem(6),
  _w(w),
  _sqrtw(_w),
  _warp(std::make_shared<lukas_kanade::WarpSE3>(
    se3, fRef->pcl(level, false), fRef->width(level), fRef->camera(level), fTo->camera(level))),
  _lk(std::make_unique<lukas_kanade::InverseCompositional>(
    fRef->intensity(level), fRef->dIx(level), fRef->dIy(level), fTo->intensity(level), _warp,
    interestPoints, l)),
  _icp(std::make_unique<IterativeClosestPoint>(
    se3, fRef, fTo, interestPoints, level, maxDepthDiff, l)),
  _se3(se3)
{
}

void LukasKanadeWithDepth::updateX(const Eigen::VectorXd & dx)
{
  _lk->updateX(dx);
  _icp->updateX(dx);
  _se3 = _se3 * SE3d::exp(-dx);
}
void LukasKanadeWithDepth::setX(const Eigen::VectorXd & x)
{
  _lk->setX(x);
  _icp->setX(x);
  _se3 = SE3d::exp(x);
}

least_squares::NormalEquations::UnPtr LukasKanadeWithDepth::computeNormalEquations()
{
  auto neLk = _lk->computeNormalEquations();
  auto neIcp = _icp->computeNormalEquations();
  auto A = neIcp->A() + _w * neLk->A();
  auto b = neIcp->b() + _sqrtw * neLk->b();
  auto chi2 = neIcp->chi2() + _w * neLk->chi2();
  auto nConstraints = neIcp->nConstraints() + neLk->nConstraints();
  return std::make_unique<least_squares::NormalEquations>(A, b, chi2, nConstraints);
}

}  // namespace pd::vslam
