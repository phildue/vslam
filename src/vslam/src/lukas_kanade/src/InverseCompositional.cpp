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

#include "InverseCompositional.h"
#include "core/core.h"
#include "utils/utils.h"
namespace pd::vslam::lukas_kanade
{
constexpr double NORMALIZER = 1 / (255.0 * 255.0);
InverseCompositional::InverseCompositional(
  const Image & templ, const MatXd & dTx, const MatXd & dTy, const Image & image,
  std::shared_ptr<Warp> w0, least_squares::Loss::ShPtr l, double minGradient)
: least_squares::Problem(w0->nParameters()),
  _T(templ),
  _I(image),
  _w(w0),
  _loss(l),
  _J(Eigen::MatrixXd::Zero(_T.rows() * _T.cols(), w0->nParameters()))
{
  // TODO(unknown): this could come from some external feature selector
  // TODO(unknown): move dTx, dTy computation outside
  std::vector<Eigen::Vector2i> interestPoints;
  interestPoints.reserve(_T.rows() * _T.cols());
  for (int32_t v = 0; v < _T.rows(); v++) {
    for (int32_t u = 0; u < _T.cols(); u++) {
      if (std::sqrt(dTx(v, u) * dTx(v, u) + dTy(v, u) * dTy(v, u)) >= minGradient) {
        interestPoints.emplace_back(u, v);
      }
    }
  }
  Eigen::MatrixXd steepestDescent = Eigen::MatrixXd::Zero(_T.rows(), _T.cols());
  size_t idx = 0U;
  std::for_each(interestPoints.begin(), interestPoints.end(), [&](auto kp) {
    const Eigen::MatrixXd Jw = _w->J(kp.x(), kp.y());
    _J.row(idx) = Jw.row(0) * dTx(kp.y(), kp.x()) + Jw.row(1) * dTy(kp.y(), kp.x());
    const double Jnorm = _J.row(kp.y() * _T.cols() + kp.x()).norm();
    steepestDescent(kp.y(), kp.x()) = std::isfinite(Jnorm) ? Jnorm : 0.0;
    if (std::isfinite(Jnorm)) {
      _interestPoints.push_back({idx++, kp});
    }
  });
  _J.conservativeResize(idx, Eigen::NoChange);
  LOG_IMG("SteepestDescent") << steepestDescent;
}

InverseCompositional::InverseCompositional(
  const Image & templ, const MatXd & dTx, const MatXd & dTy, const Image & image,
  std::shared_ptr<Warp> w0, const std::vector<Eigen::Vector2i> & interestPoints,
  least_squares::Loss::ShPtr l)
: least_squares::Problem(w0->nParameters()),
  _T(templ),
  _I(image),
  _w(w0),
  _loss(l),
  _J(Eigen::MatrixXd::Zero(interestPoints.size(), w0->nParameters())),
  _interestPoints(interestPoints.size())
{
  Eigen::MatrixXd steepestDescent = Eigen::MatrixXd::Zero(_T.rows(), _T.cols());
  size_t idx = 0U;
  std::for_each(interestPoints.begin(), interestPoints.end(), [&](auto kp) {
    const auto Jw = _w->J(kp.x(), kp.y());
    const auto Jwi = Jw.row(0) * dTx(kp.y(), kp.x()) + Jw.row(1) * dTy(kp.y(), kp.x());
    const double Jwin = Jwi.norm();
    if (std::isfinite(Jwin)) {
      _J.row(idx) = Jwi;
      steepestDescent(kp.y(), kp.x()) = Jwin;
      _interestPoints[idx] = {idx, kp};
      idx++;
    }
  });
  _J.conservativeResize(idx, Eigen::NoChange);
  _interestPoints.resize(idx);

  LOG_IMG("SteepestDescent") << steepestDescent;
}
InverseCompositional::InverseCompositional(
  const Image & templ, const Image & image, std::shared_ptr<Warp> w0,
  std::shared_ptr<least_squares::Loss> l, double minGradient)
: InverseCompositional(
    templ, algorithm::gradX(templ).cast<double>(), algorithm::gradY(templ).cast<double>(), image,
    w0, l, minGradient)
{
}

void InverseCompositional::updateX(const Eigen::VectorXd & dx) { _w->updateCompositional(-dx); }
least_squares::NormalEquations::UnPtr InverseCompositional::computeNormalEquations()
{
  Image IWxp = Image::Zero(_I.rows(), _I.cols());
  MatXd R = MatXd::Zero(_I.rows(), _I.cols());
  MatXd W = MatXd::Zero(_I.rows(), _I.cols());
  VecXd r = VecXd::Zero(_interestPoints.size());
  VecXd w = VecXd::Zero(_interestPoints.size());

  std::for_each(_interestPoints.begin(), _interestPoints.end(), [&](auto kp) {
    IWxp(kp.pos.y(), kp.pos.x()) = _w->apply(_I, kp.pos.x(), kp.pos.y());
    if (std::isfinite(IWxp(kp.pos.y(), kp.pos.x()))) {
      R(kp.pos.y(), kp.pos.x()) =
        (double)IWxp(kp.pos.y(), kp.pos.x()) - (double)_T(kp.pos.y(), kp.pos.x());
      W(kp.pos.y(), kp.pos.x()) = 1.0;
      r(kp.idx) = R(kp.pos.y(), kp.pos.x());
      w(kp.idx) = W(kp.pos.y(), kp.pos.x());
    }
  });
  if (!_scale) _scale = std::make_unique<least_squares::Scaler::Scale>(_loss->computeScale(r));

  std::for_each(_interestPoints.begin(), _interestPoints.end(), [&](auto kp) {
    if (w(kp.idx) > 0.0) {
      W(kp.pos.y(), kp.pos.x()) =
        _loss->computeWeight((r(kp.idx) - _scale->offset) / _scale->scale);
      w(kp.idx) = W(kp.pos.y(), kp.pos.x());
    }
  });
  auto ne = std::make_unique<least_squares::NormalEquations>(_J, r, w);
  /*ne->A() /= ne->nConstraints();
  ne->b() /= ne->nConstraints();
  ne->chi2() /= ne->nConstraints();*/
  LOG_IMG("ImageWarped") << IWxp;
  LOG_IMG("Residual") << R;
  LOG_IMG("Weights") << W;

  return ne;
}

}  // namespace pd::vslam::lukas_kanade
