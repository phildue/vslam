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
#include "Overlays.h"
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
  _I0(fRef->intensity(_level).cast<double>()),
  _I1(fTo->intensity(_level).cast<double>()),
  _Z0(fRef->depth(_level)),
  _Z1(fTo->depth(_level)),
  _constraints(interestPoints.size()),
  _se3(se3),
  _scale(Mat2d::Identity()),
  _iteration(0)
{
  _pCcs0 = Matd<-1, 3>::Zero(interestPoints.size(), 3);
  _JIJpJt = Matd<-1, 6>::Zero(interestPoints.size(), 6);
  _JZJpJt = Matd<-1, 6>::Zero(interestPoints.size(), 6);

  LOG(INFO) << "Precomputing..";
  size_t nConstraints = 0U;
  std::for_each(interestPoints.begin(), interestPoints.end(), [&](auto kp) {
    const Vec3d & p3d = _f0->p3d(kp.y(), kp.x(), _level);
    const Vec6d Jx_ = J_T_x(p3d);
    const Vec6d Jy_ = J_T_y(p3d);
    Vec6d JI = fRef->dIdx(_level)(kp.y(), kp.x()) * Jx_ + fRef->dIdy(_level)(kp.y(), kp.x()) * Jy_;
    Vec6d JZ = fRef->dZdx(_level)(kp.y(), kp.x()) * Jx_ + fRef->dZdy(_level)(kp.y(), kp.x()) * Jy_;

    if (!std::isfinite(p3d.norm()) || !std::isfinite(JI.norm()) || !std::isfinite(JZ.norm()))
      return;

    _constraints[nConstraints] = {nConstraints, kp.x(), kp.y()};
    _pCcs0.row(nConstraints) = p3d;
    _JZJpJt.row(nConstraints) = JZ;
    _JIJpJt.row(nConstraints) = JI;
    nConstraints++;
  });
  _constraints.resize(nConstraints);
  _pCcs0.conservativeResize(nConstraints, Eigen::NoChange);
  _JIJpJt.conservativeResize(nConstraints, Eigen::NoChange);
  _JZJpJt.conservativeResize(nConstraints, Eigen::NoChange);
  LOG(INFO) << format("Precomputed: {}", nConstraints);
}

void DirectIcp::updateX(const Eigen::VectorXd & dx) { _se3 = SE3d::exp(-dx) * _se3; }
least_squares::NormalEquations::UnPtr DirectIcp::computeNormalEquations()
{
  computeResidualsAndJacobians();
  /*
  LOG(INFO) << format(
    "Computing Weights from r {} -> {} |r| = {}", _r.minCoeff(), _r.maxCoeff(), _r.norm());
  auto s = _loss->computeScale(_r);
  std::for_each(
    std::execution::par_unseq, _constraints.begin(), _constraints.end(),
    [&](const Constraint & c) { _w(c.idx) *= _loss->computeWeight(_r(c.idx) / s.scale); });
  _scale(1, 1) = 1. / s.scale;
  */
  estimateScaleAndWeights();

  auto ne = _computeNormalEquations();
  _iteration++;
  //TODO move this to a Drawable
  MatXd ZWxp = MatXd::Zero(_f0->height(_level), _f0->width(_level));
  Image IWxp = Image::Zero(_f0->height(_level), _f0->width(_level));
  MatXd R = MatXd::Zero(_f0->height(_level), _f0->width(_level));
  MatXd Rz = MatXd::Zero(_f0->height(_level), _f0->width(_level));
  MatXd Ri = MatXd::Zero(_f0->height(_level), _f0->width(_level));
  MatXd W = MatXd::Zero(_f0->height(_level), _f0->width(_level));
  std::for_each(
    std::execution::par_unseq, _constraints.begin(), _constraints.end(), [&](Constraint & c) {
      R(c.v, c.u) = _r(c.idx);
      Rz(c.v, c.u) = _rIZ(c.idx, 1);
      Ri(c.v, c.u) = _rIZ(c.idx, 0);
      W(c.v, c.u) = _w(c.idx);
    });
  LOG(INFO) << format(
    "Residual Depth Total: {} Residual Intensity Total: {}", Rz.norm(), Ri.norm());
  LOG_IMG("ResidualIntensity") << Ri;
  LOG_IMG("ResidualDepth") << Rz;
  LOG_IMG("Residual") << R;
  LOG_IMG("Weights") << W;
  LOG_IMG("DepthWarped") << _Z1Wxp;
  LOG_IMG("ImageWarped") << _I1Wxp;
  LOG_IMG("ResidualWeighted") << std::make_shared<OverlayWeightedResidual>(
    _f0->height(_level), _f0->width(_level), _constraints, _r, _w);
  LOG_IMG("PlotResidual") << std::make_shared<PlotResiduals>(
    _f0->t(), _iteration, _constraints, _r, _w);

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

void DirectIcp::estimateScaleAndWeights()
{
  LOG(INFO) << "Estimating scale and weights..";
  double l2scale = std::numeric_limits<double>::max();
  int i = 0;
  for (; i < 50 && l2scale > 1e-4; i++) {
    MatXd scale_i = MatXd::Zero(2, 2);
    int nValid = 0;
    for (int n = 0; n < _rIZ.rows(); n++) {
      if (_w(n) > 0.) {
        const Vec2d ri = Vec2d(_rIZ.row(n));
        _w(n) = computeWeight(ri.transpose() * _scale * ri);
        scale_i += _w(n) * ri * ri.transpose();
        nValid++;
      }
    }
    scale_i /= nValid;
    scale_i = scale_i.inverse();
    l2scale = (_scale - scale_i).norm();
    _scale = scale_i;

    //LOG(INFO) << format("Iteration = {}, scale = \n{}, \nl2 = {}", i, _scale, l2scale);
  }
  LOG(INFO) << format("Iterations: {}, scale = \n{}, \nl2 = {}", i, _scale, l2scale);
}
double DirectIcp::computeWeight(double residual) const { return (5. + 2.) / (5. + residual); }

void DirectIcp::computeResidualsAndJacobians()
{
  LOG(INFO) << "Computing Residuals..";
  _JZJpJt_Jtz = Matd<-1, 6>::Zero(_constraints.size(), 6);
  _rIZ = Matd<-1, 2>::Zero(_constraints.size(), 2);
  _r = VecXd::Zero(_constraints.size());
  _w = VecXd::Zero(_constraints.size());
  _I1Wxp = MatXd::Zero(_I1.rows(), _I1.cols());
  _Z1Wxp = MatXd::Zero(_Z1.rows(), _Z1.cols());
  const SE3d T = _se3 * _f0->pose().SE3().inverse();
  const SE3d Tinv = T.inverse();
  std::for_each(
    std::execution::par_unseq, _constraints.begin(), _constraints.end(), [&](const Constraint & c) {
      Vec2d uv0t = _f1->camera2image(T * Vec3d(_pCcs0.row(c.idx)), _level);

      if (!std::isfinite(uv0t.x()) || !_f1->withinImage(uv0t, 7, _level)) return;

      Vec2d iz = interpolate(uv0t.x(), uv0t.y());

      if (!std::isfinite(iz.norm())) return;
      _I1Wxp(c.v, c.u) = iz(0);
      Vec3d p1 = _f1->image2camera(uv0t, iz(1), _level);
      Vec3d p1t = Tinv * p1;
      _Z1Wxp(c.v, c.u) = p1t.z();

      Vec6d J_t_z = J_T_z(p1t);

      if (!std::isfinite(J_t_z.norm())) return;

      _JZJpJt_Jtz.row(c.idx) = VecXd(_JZJpJt.row(c.idx)) - J_t_z;
      _rIZ(c.idx, 0) = _I1Wxp(c.v, c.u) - _I0(c.v, c.u);
      _rIZ(c.idx, 1) = _Z1Wxp(c.v, c.u) - _Z0(c.v, c.u);

      _w(c.idx) = 1.0;
    });
}
least_squares::NormalEquations::UnPtr DirectIcp::_computeNormalEquations()
{
  LOG(INFO) << "Computing Normal Equations..";
  auto ne = std::make_unique<least_squares::NormalEquations>(6);
  for_each(_constraints.cbegin(), _constraints.cend(), [&](const Constraint & c) {
    if (_w(c.idx) > 0.) {
      Matd<6, 2> J;
      J << Vec6d(_JIJpJt.row(c.idx)), Vec6d(_JZJpJt_Jtz.row(c.idx));
      ne->A().noalias() += _w(c.idx) * J * _scale * J.transpose();

      //LOG_EVERY_N(1000, INFO) << format("{} * {} * {} * {}", _w(c.idx), J, _scale, J.transpose());
      ne->b().noalias() += _w(c.idx) * J * _scale * Vec2d(_rIZ.row(c.idx));
      //LOG_EVERY_N(1000, INFO) << format(
      //  "{} * {} * {} * {}", _w(c.idx), J, _scale, Vec2d(_rIZ.row(c.idx)));
      _r(c.idx) = Vec2d(_rIZ.row(c.idx)).transpose() * _scale * Vec2d(_rIZ.row(c.idx));

      ne->chi2() += _w(c.idx) * _r(c.idx);
      ne->nConstraints() += 1;
    }
  });
  ne->A() /= ne->nConstraints();
  ne->b() /= ne->nConstraints();
  ne->chi2() /= ne->nConstraints();
  return ne;
}
}  // namespace pd::vslam
