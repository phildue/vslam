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

#ifndef VSLAM_LOSS_H__
#define VSLAM_LOSS_H__

#include <eigen3/Eigen/Dense>
#include <memory>

#include "Scaler.h"
#include "core/core.h"
// computeWeights assigns weights corresponding to L'(r) the first order derivative of the loss function
// There are many other weights based on the expected distribution of errors (e.g. t distribution)[Image Gradient-based Joint Direct Visual Odometry for Stereo Camera]

namespace pd::vslam::least_squares
{
class Loss
{
public:
  typedef std::shared_ptr<Loss> ShPtr;
  typedef std::unique_ptr<Loss> UnPtr;
  typedef std::shared_ptr<const Loss> ConstShPtr;
  typedef std::unique_ptr<const Loss> ConstUnPtr;

  virtual ~Loss() = default;
  Loss(Scaler::ShPtr scaler = std::make_shared<Scaler>()) : _scaler(scaler) {}
  //l(r)
  virtual double compute(double r) const = 0;
  //dl/dr
  virtual double computeDerivative(double r) const = 0;
  //w(r) = dl/dr (r) * 1/r
  virtual double computeWeight(double r) const { return computeDerivative(r) / r; }

  virtual Scaler::Scale computeScale(const VecXd & residuals) const
  {
    return _scaler->compute(residuals);
  }

private:
  Scaler::ConstShPtr _scaler;
};

class QuadraticLoss : public Loss
{
public:
  QuadraticLoss(Scaler::ShPtr scaler = std::make_shared<Scaler>()) : Loss(scaler) {}

  //l(r)
  double compute(double r) const override { return 0.5 * r * r; }
  //dl/dr
  double computeDerivative(double r) const override { return r; }
  //w(r) = dl/dr (r) * 1/r
  double computeWeight(double UNUSED(r)) const override { return 1.0; }
};

class TukeyLoss : public Loss
{
public:
  inline constexpr static double C =
    4.6851;  //<constant from paper corresponding to the 95% asymptotic efficiency on the standard normal distribution
  inline constexpr static double C2_6 = C / 6.0;

  TukeyLoss(Scaler::ShPtr scaler = std::make_shared<MedianScaler>()) : Loss(scaler) {}

  //w(r) = dl/dr (r) * 1/r
  double computeWeight(double r) const override;
  //dl/dr
  double computeDerivative(double r) const override;
  //l(r)
  double compute(double r) const override;
};

class HuberLoss : public Loss
{
public:
  HuberLoss(Scaler::ShPtr scaler = std::make_shared<MeanScaler>(), double c = 1.345f)
  : Loss(scaler), _c(c)
  {
  }
  const double _c;
  //w(r) = dl/dr (r) * 1/r
  double computeWeight(double r) const override;
  //dl/dr
  double computeDerivative(double r) const override;
  //l(r)
  double compute(double r) const override;
};

class LossTDistribution : public Loss
{
public:
  LossTDistribution(Scaler::ShPtr scaler = std::make_shared<ScalerTDistribution>(), double v = 5.0)
  : Loss(scaler), _v(v)
  {
  }
  //w(r) = dl/dr (r) * 1/r
  double computeWeight(double r) const override;
  //dl/dr
  double computeDerivative(double r) const override;
  //l(r)
  double compute(double r) const override;

private:
  const double _v;
};

}  // namespace pd::vslam::least_squares
#endif
