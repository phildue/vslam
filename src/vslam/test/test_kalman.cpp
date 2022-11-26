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

//
// Created by phil on 10.10.20.
//

#include <gtest/gtest.h>

#include "core/core.h"
#include "kalman/kalman.h"
#include "utils/utils.h"
using namespace testing;
using namespace pd;
using namespace pd::vslam;

#define LOG_TEST(level) LOG(level)
#ifdef TEST_VISUALIZE
#define VISUALIZE true
#else
#define VISUALIZE false
#endif
class KalmanFilter2D : public KalmanFilter<4, 2>
{
  ///state = [px py vx vy]
  ///measurement = [px py]

public:
  KalmanFilter2D(const Matd<4, 1> & x0, std::uint64_t t0)
  : KalmanFilter<4, 2>(Eigen::MatrixXd::Identity(4, 4), x0, t0)
  {
    _Q(2, 2) = 1.5;
    _Q(3, 3) = 1.5;
  }
  Matd<4, 4> A(std::uint64_t dt) const override
  {
    Matd<4, 4> M = Matd<4, 4>::Identity(4, 4);
    M(0, 2) = dt;
    M(1, 3) = dt;
    return M;
  }
  Matd<2, 4> H(std::uint64_t dt) const override
  {
    Matd<2, 4> M = Matd<2, 4>::Zero();
    M(0, 0) = 1;
    M(1, 1) = 1;
    return M;
  }
};

void plot(const std::vector<Eigen::Vector2d> & traj, std::string name)
{
  std::vector<double> x(traj.size()), y(traj.size());
  for (size_t i = 0; i < traj.size(); i++) {
    x[i] = traj[i].x();
    y[i] = traj[i].y();
  }
  vis::plt::named_plot(name.c_str(), x, y);
}

TEST(KalmanFilterTest, Motion2DTestSanity)
{
  Eigen::Vector2d velTrue;
  velTrue << 1, 2;
  Eigen::Vector2d accTrue;
  accTrue << 0.1, 0.4;
  Eigen::Vector4d x0;
  x0 << 0, 0, 1.0, 2.0;
  const int dt = 1;
  Eigen::Vector2d covTrue;
  covTrue.x() = 15.5;
  covTrue.y() = 15.5;

  Eigen::Vector2d covAssum;
  covAssum.x() = 10.1;
  covAssum.y() = 10.1;

  std::shared_ptr<KalmanFilter2D> kalman = std::make_shared<KalmanFilter2D>(x0, 0U);

  std::vector<Eigen::Vector2d> trajTrue(100);
  std::vector<Eigen::Vector2d> trajNoise(100);
  std::vector<Eigen::Vector2d> trajKalman(100);
  trajKalman[0].setZero();
  trajNoise[0].setZero();
  trajTrue[0].setZero();

  for (int t = 1; t < 100; t += dt) {
    auto pred = kalman->predict(t);
    trajKalman[t](0) = pred.state(0);
    trajKalman[t](1) = pred.state(1);
    velTrue += static_cast<double>(dt) * accTrue;
    trajTrue[t] = trajTrue[t - 1] + static_cast<double>(dt) * velTrue;

    ASSERT_NEAR(trajKalman[t].x(), trajKalman[t].x(), 0.1);
    ASSERT_NEAR(trajTrue[t].y(), trajTrue[t].y(), 0.1);

    trajNoise[t] = trajTrue[t] + random::N(covTrue.asDiagonal());
    kalman->update(t, trajNoise[t], covAssum.asDiagonal());
    if (VISUALIZE) {
      LOG_TEST(DEBUG) << t << "----";
      LOG_TEST(DEBUG) << "v: " << velTrue.transpose();
      LOG_TEST(DEBUG) << "Kalman: " << trajKalman[t].transpose();
      LOG_TEST(DEBUG) << "True: " << trajTrue[t].transpose();
      LOG_TEST(DEBUG) << "Noise: " << trajNoise[t].transpose();
      LOG_TEST(DEBUG) << "State:\n " << pred.state.transpose();
      LOG_TEST(DEBUG) << "Uncertainty:\n " << pred.cov;
    }
  }
  if (VISUALIZE) {
    vis::plt::figure();
    plot(trajTrue, "True");
    plot(trajNoise, "Noise");
    plot(trajKalman, "Kalman");

    vis::plt::legend();
    vis::plt::show();
  }
}
