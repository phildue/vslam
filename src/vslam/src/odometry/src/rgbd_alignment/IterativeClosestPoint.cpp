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
#include "core/core.h"
#include "utils/utils.h"
namespace pd::vslam
{
IterativeClosestPoint::IterativeClosestPoint(
  const SE3d & se3, Frame::ConstShPtr fRef, Frame::ConstShPtr fTo,
  const std::vector<Eigen::Vector2i> & interestPoints, int level, double maxDepthDiff,
  least_squares::Loss::ShPtr l)
: least_squares::Problem(6),
  _level(level),
  _maxDepthDiff(maxDepthDiff),
  _f0(fRef),
  _f1(fTo),
  _loss(l),
  _uv0(interestPoints.size()),
  _se3(se3)
{
  size_t idx = 0U;
  std::for_each(interestPoints.begin(), interestPoints.end(), [&](auto kp) {
    _uv0[idx] = {idx, kp};
    idx++;
  });
  _uv0.resize(idx);
}

void IterativeClosestPoint::updateX(const Eigen::VectorXd & dx) { _se3 = _se3 * SE3d::exp(-dx); }
least_squares::NormalEquations::ConstShPtr IterativeClosestPoint::computeNormalEquations()
{
  MatXd R = MatXd::Zero(_f0->height(_level), _f0->width(_level));
  MatXd W = MatXd::Zero(_f0->height(_level), _f0->width(_level));
  VecXd r = VecXd::Zero(_uv0.size());
  MatXd J = MatXd::Zero(_uv0.size(), 6);
  VecXd w = VecXd::Zero(_uv0.size());

  std::for_each(_uv0.begin(), _uv0.end(), [&](auto uv0) {
    const Vec3d & p0 = _f0->p3d(uv0.pos.y(), uv0.pos.x(), _level);
    Vec2d uv1 = _f1->camera2image(_se3 * p0, _level);
    const Vec3d & n0 = _f0->normal(uv0.pos.y(), uv0.pos.x(), _level);
    const Vec3d & p1t = _se3.inverse() * _f1->p3d(uv1.y(), uv1.x(), _level);
    const Vec3d v = p0 - p1t;
    R(uv0.pos.y(), uv0.pos.x()) = n0[0] * v.x() + n0[1] * v.y() + n0[2] * v.z();

    if (
      std::isfinite(n0.x()) && std::isfinite(uv1.x()) && _f1->withinImage(uv1, 7, _level) &&
      std::abs(p1t.z() - p0.z()) <= _maxDepthDiff) {
      Vec3d pxn = p1t.cross(n0);
      Vec6d Ji;
      Ji << n0, -p1t.z() * n0[1] + p1t.y() * n0[2], p1t.z() * n0[0] - p1t.x() * n0[2],
        -p1t.y() * n0[0] + p1t.x() * n0[1];
      J.row(uv0.idx) = Ji;
      W(uv0.pos.y(), uv0.pos.x()) = 1.0;
      w(uv0.idx) = W(uv0.pos.y(), uv0.pos.x());

      r(uv0.idx) = R(uv0.pos.y(), uv0.pos.x());
    }
  });

  auto s = _loss->computeScale(r);
  std::for_each(_uv0.begin(), _uv0.end(), [&](auto uv0) {
    W(uv0.pos.y(), uv0.pos.x()) *= _loss->computeWeight((r(uv0.idx) - s.offset) / s.scale);
    w(uv0.idx) = W(uv0.pos.y(), uv0.pos.x());
  });
  auto ne = std::make_shared<least_squares::NormalEquations>(J, r, w);

  LOG_IMG("Residual") << R;
  LOG_IMG("Weights") << W;

  return ne;
}

}  // namespace pd::vslam
