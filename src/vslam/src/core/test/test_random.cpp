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

#include <gtest/gtest.h>

#include "core/core.h"
using namespace testing;
using namespace pd;
using namespace pd::vslam;

struct normal_random_variable
{
  //Thanks to: https://stackoverflow.com/questions/6142576/sample-from-multivariate-normal-gaussian-distribution-in-c
  explicit normal_random_variable(Eigen::MatrixXd const & covar)
  : normal_random_variable(Eigen::VectorXd::Zero(covar.rows()), covar)
  {
  }

  normal_random_variable(Eigen::VectorXd const & mean, Eigen::MatrixXd const & covar) : mean(mean)
  {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(covar);
    transform = eigenSolver.eigenvectors() * eigenSolver.eigenvalues().cwiseSqrt().asDiagonal();
  }

  Eigen::VectorXd mean;
  Eigen::MatrixXd transform;

  Eigen::VectorXd operator()() const
  {
    static std::mt19937 gen{std::random_device{}()};
    static std::normal_distribution<> dist;

    return mean + transform * Eigen::VectorXd{mean.size()}.unaryExpr(
                                [&](auto UNUSED(x)) { return dist(gen); });
  }
};

TEST(RandomTest, MultivariateGaussian)
{
  int size = 2;
  Eigen::MatrixXd covar(size, size);
  covar << 1, .5, .5, 1;

  normal_random_variable sample{covar};
  for (int i = 0; i < 10; i++) {
    const Eigen::Vector2d s = sample();
    EXPECT_LE(s.x(), covar(0, 0) * 9) << "Drawn sample should be within 9 stddev";
    EXPECT_LE(s.y(), covar(1, 1) * 9) << "Drawn sample should be within 9 stddev";
    std::cout << "Sample: " << i << ": " << s.transpose() << std::endl;
  }
}

TEST(RandomTest, MultivariateGaussianFunction)
{
  int size = 2;
  Eigen::MatrixXd covar(size, size);
  covar << 1, .5, .5, 1;

  for (int i = 0; i < 10; i++) {
    const Eigen::Vector2d s = random::N(covar);
    EXPECT_LE(s.x(), std::sqrt(covar(0, 0)) * 9) << "Drawn sample should be within 9 stddev";
    EXPECT_LE(s.y(), std::sqrt(covar(1, 1)) * 9) << "Drawn sample should be within 9 stddev";
    std::cout << "Sample: " << i << ": " << s.transpose() << std::endl;
  }
}

TEST(RandomTest, MultivariateGaussianFunction6d)
{
  Eigen::MatrixXd covar(6, 6);
  covar << 0.0004, -0., 0., 0., 0.0003, 0.0002, -0., 0.0003, -0., -0.0001, -0., 0., 0., -0., 0.0004,
    -0.0002, -0., 0., 0., -0.0001, -0.0002, 0.0007, 0., -0., 0.0003, -0., -0., 0., 0.0006, 0.0001,
    0.0002, 0., 0., -0., 0.0001, 0.0003;

  for (int i = 0; i < 10; i++) {
    const Eigen::Vector6d s = random::N(covar);
    for (int j = 0; j < 6; j++) {
      EXPECT_LE(s(j), std::sqrt(covar(j, j)) * 9) << "Drawn sample should be within 9 stddev";
    }
    std::cout << "Sample: " << i << ": " << s.transpose() << std::endl;
  }
}
