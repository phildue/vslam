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

#include <execution>

#include "ForwardAdditive.h"
#include "core/core.h"
#include "utils/utils.h"

namespace pd::vslam::lukas_kanade
{
ForwardAdditive::ForwardAdditive(
  const Image & templ, const MatXd & dIdx, const MatXd & dIdy, const Image & image,
  std::shared_ptr<Warp> w0, least_squares::Loss::ShPtr l, double minGradient,
  least_squares::Prior::ConstShPtr prior)
: least_squares::Problem(w0->nParameters()),
  _T(templ),
  _Iref(image),
  _dIdx(dIdx),
  _dIdy(dIdy),
  _w(w0),
  _loss(l),
  _minGradient(minGradient),
  _prior(prior)
{
  // TODO(unknown): this could come from some external feature selector
  _interestPoints.reserve(_T.rows() * _T.cols());
  for (int32_t v = 0; v < _T.rows(); v++) {
    for (int32_t u = 0; u < _T.cols(); u++) {
      if (std::sqrt(_dIdx(v, u) * _dIdx(v, u) + _dIdy(v, u) * _dIdy(v, u)) >= _minGradient) {
        _interestPoints.push_back({u, v});
      }
    }
  }
}

void ForwardAdditive::updateX(const Eigen::VectorXd & dx) { _w->updateAdditive(dx); }

least_squares::NormalEquations::UnPtr ForwardAdditive::computeNormalEquations()
{
  Eigen::MatrixXd steepestDescent = Eigen::MatrixXd::Zero(_T.rows(), _T.cols());
  Eigen::MatrixXd dIxWp = Eigen::MatrixXd::Zero(_Iref.rows(), _Iref.cols());
  Eigen::MatrixXd dIyWp = Eigen::MatrixXd::Zero(_Iref.rows(), _Iref.cols());
  Eigen::MatrixXd J = Eigen::MatrixXd::Zero(_Iref.rows() * _Iref.cols(), _w->nParameters());

  std::for_each(_interestPoints.begin(), _interestPoints.end(), [&](auto kp) {
    Eigen::Vector2d uvWarped = _w->apply(kp.x(), kp.y());
    if (
      1 < uvWarped.x() && uvWarped.x() < _Iref.cols() - 1 && 1 < uvWarped.y() &&
      uvWarped.y() < _Iref.rows() - 1) {
      dIxWp(kp.y(), kp.x()) = algorithm::bilinearInterpolation(_dIdx, uvWarped.x(), uvWarped.y());
      dIyWp(kp.y(), kp.x()) = algorithm::bilinearInterpolation(_dIdy, uvWarped.x(), uvWarped.y());

      const Eigen::MatrixXd Jwarp = _w->J(kp.x(), kp.y());

      J.row(kp.y() * _Iref.cols() + kp.x()) =
        (dIxWp(kp.y(), kp.x()) * Jwarp.row(0) + dIyWp(kp.y(), kp.x()) * Jwarp.row(1));
      steepestDescent(kp.y(), kp.x()) = J.row(kp.y() * _Iref.cols() + kp.x()).norm();
    }
  });
  LOG_IMG("Gradient_X_Warped") << dIxWp;
  LOG_IMG("Gradient_Y_Warped") << dIyWp;
  LOG_IMG("SteepestDescent") << steepestDescent;

  Image IWxp = Image::Zero(_Iref.rows(), _Iref.cols());
  std::vector<Eigen::Vector2i> interestPointsVisible(_interestPoints.size());
  auto it = std::copy_if(
    std::execution::par_unseq, _interestPoints.begin(), _interestPoints.end(),
    interestPointsVisible.begin(), [&](auto kp) {
      Eigen::Vector2d uvI = _w->apply(kp.x(), kp.y());
      const bool visible = 1 < uvI.x() && uvI.x() < _Iref.cols() - 1 && 1 < uvI.y() &&
                           uvI.y() < _Iref.rows() - 1 && std::isfinite(uvI.x());
      if (visible) {
        IWxp(kp.y(), kp.x()) = algorithm::bilinearInterpolation(_Iref, uvI.x(), uvI.y());
      }
      return visible;
    });
  interestPointsVisible.resize(std::distance(interestPointsVisible.begin(), it));

  if (interestPointsVisible.size() < _w->nParameters()) {
    throw std::runtime_error("Not enough valid interest points!");
  }

  const MatXd R = _T.cast<double>() - IWxp.cast<double>();

  std::vector<double> r(interestPointsVisible.size());
  std::transform(
    std::execution::unseq, interestPointsVisible.begin(), interestPointsVisible.end(), r.begin(),
    [&](auto kp) { return R(kp.y(), kp.x()); });

  if (_loss) {
    _loss->computeScale(Eigen::Map<Eigen::VectorXd>(r.data(), r.size()));
  }

  auto ne = std::make_unique<least_squares::NormalEquations>(_w->nParameters());
  Eigen::MatrixXd W = Eigen::MatrixXd::Zero(_T.rows(), _T.cols());
  std::for_each(
    std::execution::unseq, interestPointsVisible.begin(), interestPointsVisible.end(),
    [&](auto kp) {
      W(kp.y(), kp.x()) = _loss ? _loss->computeWeight(R(kp.y(), kp.x())) : 1.0;

      if (
        !std::isfinite(J.row(kp.y() * _T.cols() + kp.x()).norm()) ||
        !std::isfinite(R(kp.y(), kp.x())) || !std::isfinite(W(kp.y(), kp.x()))) {
        std::stringstream ss;
        ss << "NaN during LK with: R = " << R(kp.y(), kp.x()) << " W = " << W(kp.y(), kp.x())
           << " J = " << J.row(kp.y() * _T.cols() + kp.x()) << " at: " << kp.transpose();
        throw std::runtime_error(ss.str());
      }
      ne->addConstraint(J.row(kp.y() * _T.cols() + kp.x()), R(kp.y(), kp.x()), W(kp.y(), kp.x()));
    });
  ne->A().noalias() = ne->A() / static_cast<double>(ne->nConstraints());
  ne->b().noalias() = ne->b() / static_cast<double>(ne->nConstraints());

  LOG_IMG("ImageWarped") << IWxp;
  LOG_IMG("Residual") << R;
  LOG_IMG("Weights") << W;

  return ne;
}

}  // namespace pd::vslam::lukas_kanade
