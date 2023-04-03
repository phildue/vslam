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

#include "DirectIcp.h"
#include "core/core.h"
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
  std::for_each(interestPoints.begin(), interestPoints.end(), [&](auto kp) {
    const Vec3d & p3d = _f0->p3d(kp.y(), kp.x(), _level);
    const Vec6d Jx_ = J_T_x(p3d);
    const Vec6d Jy_ = J_T_y(p3d);
    _constraints[idx] = {
      idx,
      kp.x(),
      kp.y(),
      p3d,
      (double)fRef->intensity(_level)(kp.y(), kp.x()),
      0.,
      (double)fRef->depth(_level)(kp.y(), kp.x()),
      0.,
      Vec6d::Zero(),
      fRef->dIdx(_level)(kp.y(), kp.x()) * Jx_ + fRef->dIdy(_level)(kp.y(), kp.x()) * Jy_,
      fRef->dZdx(_level)(kp.y(), kp.x()) * Jx_ + fRef->dZdy(_level)(kp.y(), kp.x()) * Jy_,
      0.,
      0.,
      0.,
      0.};
    idx++;
  });
  _constraints.resize(idx);
}

void DirectIcp::updateX(const Eigen::VectorXd & dx) { _se3 = SE3d::exp(-dx) * _se3; }
least_squares::NormalEquations::UnPtr DirectIcp::computeNormalEquations()
{
  VecXd r = VecXd::Zero(_constraints.size());
  MatXd J = MatXd::Zero(_constraints.size(), 6);
  VecXd w = VecXd::Zero(_constraints.size());
  const SE3d T = _se3 * _f0->pose().SE3().inverse();
  const SE3d Tinv = T.inverse();
  const double wi = 0.0;
  const double wz = 1.0 - wi;
  std::for_each(
    std::execution::par_unseq, _constraints.begin(), _constraints.end(), [&](Constraint & c) {
      //TODO create a new instance each time instead of reusing
      c.r_I = 0.;
      c.r_Z = 0.;
      c.I1Wxp = 0;
      c.Z1Wxp = 0;
      c.weight = 0.;
      c.residual = 0.;
      c.J = Vec6d::Zero();

      const Vec3d p0t = T * c.p;
      Vec2d uv0t = _f1->camera2image(p0t, _level);

      if (!std::isfinite(uv0t.x()) || !_f1->withinImage(uv0t, 7, _level)) return;

      interpolate(uv0t.x(), uv0t.y(), c);

      if (!std::isfinite(c.Z1Wxp)) return;

      Vec3d p1 = _f1->image2camera(uv0t, c.Z1Wxp, _level);
      Vec3d p1t = Tinv * p1;
      c.Z1Wxp = p1t.z();
      c.r_I = c.I1Wxp - c.I0;
      c.r_Z = c.Z1Wxp - c.p.z();  //looks reasonable

      Vec6d Jz_ = J_T_z(p1t);

      if (!std::isfinite(Jz_.norm())) return;

      c.residual = wi * c.r_I + wz * c.r_Z;
      c.J = wi * c.J_I + wz * (c.J_Z - Jz_);
      c.weight = 1.0;

      w(c.idx) = c.weight;
      r(c.idx) = c.residual;
      J.row(c.idx) = c.J;
    });

  auto s = _loss->computeScale(r);
  std::for_each(
    std::execution::par_unseq, _constraints.begin(), _constraints.end(), [&](Constraint & c) {
      c.weight *= _loss->computeWeight(c.residual / s.scale);
      w(c.idx) = c.weight;
    });
  auto ne = std::make_unique<least_squares::NormalEquations>(J, r, w);

  ne->A() /= ne->nConstraints();
  ne->b() /= ne->nConstraints();
  ne->chi2() /= ne->nConstraints();

  //TODO move this to a Drawable
  MatXd ZWxp = MatXd::Zero(_f0->height(_level), _f0->width(_level));
  Image IWxp = Image::Zero(_f0->height(_level), _f0->width(_level));
  MatXd R = MatXd::Zero(_f0->height(_level), _f0->width(_level));
  MatXd Rz = MatXd::Zero(_f0->height(_level), _f0->width(_level));
  MatXd Ri = MatXd::Zero(_f0->height(_level), _f0->width(_level));
  MatXd W = MatXd::Zero(_f0->height(_level), _f0->width(_level));
  std::for_each(
    std::execution::par_unseq, _constraints.begin(), _constraints.end(), [&](Constraint & c) {
      R(c.v, c.u) = c.residual;
      Rz(c.v, c.u) = c.r_Z;
      Ri(c.v, c.u) = c.r_I;
      W(c.v, c.u) = c.weight;
      ZWxp(c.v, c.u) = c.Z1Wxp;
      IWxp(c.v, c.u) = static_cast<image_value_t>(c.I1Wxp);
    });
  LOG(INFO) << format(
    "Residual Depth Total: {} Residual Intensity Total: {}", Rz.norm(), Ri.norm());
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

void DirectIcp::interpolate(double u, double v, DirectIcp::Constraint & c) const
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

  c.I1Wxp = P(0);
  c.Z1Wxp = P(1);
}
}  // namespace pd::vslam
