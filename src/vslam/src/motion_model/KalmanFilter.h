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

#ifndef VSLAM_KALMAN_FILTER_H__
#define VSLAM_KALMAN_FILTER_H__
//https://thekalmanfilter.com/kalman-filter-explained-simply/
#include "core/core.h"
#include "utils/utils.h"
namespace pd::vslam
{
/// Bayesian filtering for a linear system following:
/// x_t1 = A x_t0 + B u_t0
/// With measurement model:
/// z_t = C x_t
/// Assuming gaussian noise on system
/// p(x_t) = N(x_t,P)
/// Assuming gaussian noise on measurement:
/// z_t = N(z_1,R)
template <int N, int M>
class KalmanFilter
{
public:
  struct Prediction
  {
    Matd<N, 1> state;
    Matd<N, N> cov;
  };
  KalmanFilter(
    const Matd<N, N> & P0, const Matd<N, 1> & x0,
    std::uint64_t t0 = std::numeric_limits<uint64_t>::max())
  : _P(P0), _K(), _t(t0), _x(x0), _Q()
  {
    Log::get("kalman");
  }
  Prediction predict(std::uint64_t t)
  {
    if (_t < std::numeric_limits<uint64_t>::max()) {
      const auto At = A(t - _t);
      return {At * _x, At * _P * At.transpose() + _Q};
    } else {
      return {_x, _P + _Q};
    }
  }

  void update(std::uint64_t t, const Matd<M, 1> & z, const Matd<M, M> & R)
  {
    if (_t < std::numeric_limits<uint64_t>::max()) {
      const auto pred = predict(t);
      const auto H_ = H(t - _t);

      _K = pred.cov * H_.transpose() * (H_ * pred.cov * H_.transpose() + R).inverse();

      _x = pred.state + _K * (z - H_ * pred.state);
      _P = pred.cov - _K * H_ * pred.cov;
    }

    _t = t;
    CLOG(DEBUG, "kalman") << "z: " << z.transpose() << "R: " << R;
    CLOG(DEBUG, "kalman") << "x: " << _x.transpose() << "P: " << _P;
  }

  //System model matrix
  virtual Matd<N, N> A(std::uint64_t dT) const = 0;

  //State to measurement matrix
  virtual Matd<M, N> H(std::uint64_t dT) const = 0;

protected:
  Matd<N, N> _P;     //< State Covariance n x n
  Matd<N, M> _K;     //< Kalman gain
  std::uint64_t _t;  //< last update
  Matd<N, 1> _x;     //< state n x 1
  Matd<N, N> _Q;     //< process noise
};
}  // namespace pd::vslam
#endif  //VSLAM_KALMAN_H__
