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

#ifndef VSLAM_LUKAS_KANADE_INVERSE_COMPOSITIONAL_H__
#define VSLAM_LUKAS_KANADE_INVERSE_COMPOSITIONAL_H__
#include <memory>

#include "Warp.h"
#include "core/core.h"
#include "least_squares/least_squares.h"
namespace pd::vslam::lukas_kanade
{
/*
Compute parameters p of the warp W by incrementally warping the image I to the template T.
For inverse compositional we switch the role of I and T:

chi2 = rho(|T(W(x,p+h)) - I(W(x,p))|^2)
with rho being a robust loss function reducing the influence of outliers
and h the parameter update.

This results in normal equations:

(J^T * W * J)h = -J*W*r           (1)

where Ji =  [dTix * JiWx, dTiy * JiWy]
with i corresponding to the ith pixel / feature / row in J
dTi* being the image derivative in x/y direction of the ith pixel / feature
JiW* being the partial derivatives of the warping function to its parameters.

Given Jw is independent of h, J can be precomputed as its not changing during a parameter update.

After solving (1) we obtain the parameter update that would warp the template to the image!
Hence, we update p by applying the *inverse compositional*: W_new = W(-h,W(p)).
*/

class InverseCompositional : public least_squares::Problem
{
public:
  InverseCompositional(
    const Image & templ, const MatXd & dX, const MatXd & dY, const Image & image,
    std::shared_ptr<Warp> w0,
    least_squares::Loss::ShPtr = std::make_shared<least_squares::QuadraticLoss>(),
    double minGradient = 0);

  InverseCompositional(
    const Image & templ, const MatXd & dX, const MatXd & dY, const Image & image,
    std::shared_ptr<Warp> w0, const std::vector<Eigen::Vector2i> & interestPoints,
    least_squares::Loss::ShPtr = std::make_shared<least_squares::QuadraticLoss>());

  virtual ~InverseCompositional() = default;
  std::shared_ptr<const Warp> warp() { return _w; }

  void updateX(const Eigen::VectorXd & dx) override;
  void setX(const Eigen::VectorXd & x) override { _w->setX(x); }

  Eigen::VectorXd x() const override { return _w->x(); }

  least_squares::NormalEquations::UnPtr computeNormalEquations() override;
  least_squares::NormalEquations::UnPtr computeNormalEquationsScaled();
  least_squares::NormalEquations::UnPtr computeNormalEquationsAndScale();

protected:
  const Image _T;
  const Image _I;
  const std::shared_ptr<Warp> _w;
  const std::shared_ptr<least_squares::Loss> _loss;
  Eigen::MatrixXd _J;
  struct IndexedKeyPoint
  {
    size_t idx;
    Eigen::Vector2i pos;
  };
  std::vector<IndexedKeyPoint> _interestPoints;
  std::unique_ptr<least_squares::Scaler::Scale> _scale;
  uint64_t _iteration;
};

}  // namespace pd::vslam::lukas_kanade
#endif
