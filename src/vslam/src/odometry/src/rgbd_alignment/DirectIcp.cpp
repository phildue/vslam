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

#include <fmt/core.h>
#include <fmt/ostream.h>

using fmt::format;
using fmt::print;

#include <algorithm>
#include <execution>
using std::for_each;
using std::transform;
using std::execution::par_unseq;
#include "DirectIcp.h"
#include "core/core.h"
using pd::vslam::algorithm::insertionSort;
using pd::vslam::algorithm::median;
using pd::vslam::linalg::stddev;

#include "utils/utils.h"
template <typename T>
struct fmt::formatter<T, std::enable_if_t<std::is_base_of_v<Eigen::DenseBase<T>, T>, char>>
: ostream_formatter
{
};
namespace pd::vslam
{
DirectIcp::DirectIcp(
  const SE3d & se3, Frame::ConstShPtr fRef, Frame::ConstShPtr fTo,
  const std::vector<Eigen::Vector2i> & interestPoints, int level, least_squares::Loss::ShPtr l)
: least_squares::Problem(6),
  _level(level),
  _f0(fRef),
  _f1(fTo),
  _loss(l),
  _I1(fTo->intensity(_level)),
  _Z1(fTo->depth(_level)),
  _constraints(interestPoints.size()),
  _se3(se3)
{
  size_t idx = 0U;
  for_each(interestPoints.begin(), interestPoints.end(), [&](auto kp) {
    const Vec3d & p3d = _f0->p3d(kp.y(), kp.x(), _level);
    const Vec6d Jx_ = J_T_x(p3d);
    const Vec6d Jy_ = J_T_y(p3d);
    auto c = std::make_shared<Constraint>();
    c->idx = idx;
    c->u = kp.x();
    c->v = kp.y();
    c->p = p3d;
    c->intensity = (double)fRef->intensity(_level)(kp.y(), kp.x());
    c->JiJpJt = fRef->dIdx(_level)(kp.y(), kp.x()) * Jx_ + fRef->dIdy(_level)(kp.y(), kp.x()) * Jy_;
    c->JzJpJt = fRef->dZdx(_level)(kp.y(), kp.x()) * Jx_ + fRef->dZdy(_level)(kp.y(), kp.x()) * Jy_;
    c->weight = 0.;
    _constraints[idx] = c;
    idx++;
  });
  _constraints.resize(idx);

  //TODO(me): avoid recomputation
  const double stdri = median(_f0->intensity(_level));
  const double stdrz = median(_f0->depth(_level));
  _wz = std::abs(stdri / stdrz);

  LOG(INFO) << format(
    "Computed wz = {}/{} = {} from #{} constraints", stdri, stdrz, _wz, _constraints.size());
}

void DirectIcp::updateX(const Eigen::VectorXd & dx) { _se3 = SE3d::exp(-dx) * _se3; }
least_squares::NormalEquations::UnPtr DirectIcp::computeNormalEquations()
{
  const SE3d T = _se3 * _f0->pose().SE3().inverse();
  const SE3d Tinv = T.inverse();
  std::vector<Constraint::ConstShPtr> constraints(_constraints.size());
  VecXd r = VecXd::Zero(_constraints.size());
  MatXd J = MatXd::Zero(_constraints.size(), 6);
  LOG(INFO) << "Computing Residual..";
  transform(
    par_unseq, _constraints.cbegin(), _constraints.cend(), constraints.begin(),
    [&](Constraint::ConstShPtr c) -> Constraint::ConstShPtr {
      const Vec3d p0t = T * c->p;
      Vec2d uv0t = _f1->camera2image(p0t, _level);

      if (!std::isfinite(uv0t.x()) || !_f1->withinImage(uv0t, 7, _level)) return c;

      const Vec2d iz = interpolate(uv0t.x(), uv0t.y());

      if (!std::isfinite(iz.norm())) return c;

      auto cc = std::make_shared<Constraint>(*c);
      cc->I1Wxp = iz(0);
      cc->ri = cc->I1Wxp - c->intensity;

      Vec3d p1 = _f1->image2camera(uv0t, iz(1), _level);
      Vec3d p1t = Tinv * p1;
      cc->Z1Wxp = p1t.z();
      cc->rz = cc->Z1Wxp - c->p.z();

      cc->Jtz = J_T_z(p1t);

      if (!std::isfinite(cc->Jtz.norm())) return cc;

      cc->weight = 1.;
      cc->residual = cc->ri + _wz * cc->rz;
      cc->J = cc->JiJpJt + _wz * (cc->JzJpJt - cc->Jtz);

      r(c->idx) = cc->residual;
      J.row(c->idx) = cc->J;

      return cc;
    });

  auto s = _loss->computeScale(r);
  VecXd w = VecXd::Zero(_constraints.size());
  LOG(INFO) << "Computing weights";
  transform(
    par_unseq, constraints.begin(), constraints.end(), constraints.begin(),
    [&](Constraint::ConstShPtr c) -> Constraint::ConstShPtr {
      if (c->weight < 1.) return c;
      auto cc = std::make_shared<Constraint>(*c);
      cc->weight = _loss->computeWeight(c->residual / s.scale);
      w(c->idx) = cc->weight;
      return cc;
    });
  auto ne = std::make_unique<least_squares::NormalEquations>(J, r, w);

  ne->A() /= ne->nConstraints();
  ne->b() /= ne->nConstraints();
  ne->chi2() /= ne->nConstraints();

  //TODO move this to a Drawable
  LOG(INFO) << "Drawing..";

  MatXd ZWxp = MatXd::Zero(_f0->height(_level), _f0->width(_level));
  Image IWxp = Image::Zero(_f0->height(_level), _f0->width(_level));
  MatXd R = MatXd::Zero(_f0->height(_level), _f0->width(_level));
  MatXd Rz = MatXd::Zero(_f0->height(_level), _f0->width(_level));
  MatXd Ri = MatXd::Zero(_f0->height(_level), _f0->width(_level));
  MatXd W = MatXd::Zero(_f0->height(_level), _f0->width(_level));
  for_each(par_unseq, constraints.begin(), constraints.end(), [&](Constraint::ConstShPtr c) {
    if (c) {
      R(c->v, c->u) = c->residual;
      Rz(c->v, c->u) = c->rz;
      Ri(c->v, c->u) = c->ri;
      W(c->v, c->u) = c->weight;
      ZWxp(c->v, c->u) = c->Z1Wxp;
      IWxp(c->v, c->u) = static_cast<image_value_t>(c->I1Wxp);
    };
  });
  LOG(INFO) << format("|rz| = {} |ri| = {} wz = {}", Rz.norm(), Ri.norm(), _wz);
  LOG_IMG("ResidualIntensity") << Ri;
  LOG_IMG("ResidualDepth") << Rz;
  LOG_IMG("Residual") << R;
  LOG_IMG("Weights") << W;
  LOG_IMG("DepthWarped") << ZWxp;
  LOG_IMG("ImageWarped") << IWxp;
  return ne;
}  // namespace pd::vslam

Vec6d DirectIcp::J_T_x(const Vec3d & p)
{
  const double & x = p.x();
  const double & y = p.y();
  const double z_inv = 1. / p.z();
  const double z_inv_2 = z_inv * z_inv;

  Vec6d J;
  J(0) = z_inv;
  J(1) = 0.0;
  J(2) = -x * z_inv_2;
  J(3) = y * J(2);
  J(4) = 1.0 - x * J(2);
  J(5) = -y * z_inv;
  J *= _f0->camera(_level)->fx();
  return J;
}
Vec6d DirectIcp::J_T_y(const Vec3d & p)
{
  const double & x = p.x();
  const double & y = p.y();
  const double z_inv = 1. / p.z();
  const double z_inv_2 = z_inv * z_inv;

  Vec6d J;
  J(0) = 0.0;
  J(1) = z_inv;
  J(2) = -y * z_inv_2;
  J(3) = -1.0 + y * J(2);
  J(4) = -J(3);
  J(5) = x * z_inv;
  J *= _f0->camera(_level)->fy();

  return J;
}
Vec6d DirectIcp::J_T_z(const Vec3d & p)
{
  Vec6d J;
  J(0) = 0.0;
  J(1) = 0.0;
  J(2) = 1.0;
  J(3) = p(1);
  J(4) = -p(0);
  J(5) = 0.0;

  return J;
}

Vec2d DirectIcp::interpolate(double u, double v) const
{
  /*We want to interpolate P
        * http://supercomputingblog.com/graphics/coding-bilinear-interpolation/
        *
        * v2 |Q12    R2          Q22
        *    |
        * y  |       P
        *    |
        * v1 |Q11    R1          Q21
        *    _______________________
        *    u1      x            u2
        * */
  auto mat = [&](int v, int u) -> Vec2d {
    Vec2d m;
    m << _I1(v, u), _Z1(v, u);
    return m;
  };
  const double u1 = std::floor(u);
  const double u2 = std::ceil(u);
  const double v1 = std::floor(v);
  const double v2 = std::ceil(v);
  const Vec2d Q11 = mat(v1, u1);
  const Vec2d Q12 = mat(v1, u2);
  const Vec2d Q21 = mat(v2, u1);
  const Vec2d Q22 = mat(v2, u2);

  Vec2d R1 = Vec2d::Zero(), R2 = Vec2d::Zero();
  if (u2 == u1) {
    R1 = Q11;
    R2 = Q12;
  } else {
    R1 = ((u2 - u) / (u2 - u1)) * Q11 + ((u - u1) / (u2 - u1)) * Q21;
    R2 = ((u2 - u) / (u2 - u1)) * Q12 + ((u - u1) / (u2 - u1)) * Q22;
  }

  Vec2d P = Vec2d::Zero();
  if (v2 == v1) {
    P = R1;
  } else {
    // After the two R values are calculated, the value of P can finally be calculated by a weighted average of R1 and R2.
    P = ((v2 - v) / (v2 - v1)) * R1 + ((v - v1) / (v2 - v1)) * R2;
  }

  return P;
}
}  // namespace pd::vslam
