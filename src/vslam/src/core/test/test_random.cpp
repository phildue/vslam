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

class Student_t
{
public:
  Student_t(const VecXd & mean, const MatXd & cov, int dof);
  VecXd draw() const { return pd::vslam::random::student_t(_mean, _scale, _dof); }
  VecXd operator()() const { return draw(); }
  static Student_t fit(
    const MatXd & data, int dof, const MatXd & scale0, int maxIterations = 30,
    double convergenceThreshold = 1e-7);
  const MatXd & scale() const { return _scale; }
  const MatXd & scaleInv() const { return _scaleInv; }
  const VecXd & mean() const { return _mean; }
  const int & dof() { return _dof; }
  const int & nDims() { return _nDims; }

private:
  VecXd _mean;
  MatXd _scale, _scaleInv;
  int _dof;
  int _nDims;
};

Student_t::Student_t(const VecXd & mean, const MatXd & scale, int dof)
: _mean(mean), _scale(scale), _scaleInv(scale.inverse()), _dof(dof), _nDims(scale.rows())
{
  if (mean.rows() != scale.rows()) throw pd::Exception("Unequal dimensions");
}

Student_t Student_t::fit(
  const MatXd & data, int dof, const MatXd & scale0, int maxIterations, double convergenceThreshold)
{
  const int nSamples = data.rows();
  const int nDims = data.cols();

  MatXd scale = scale0;
  VecXd mean;
  double stepSize = std::numeric_limits<double>::max();
  for (int i = 0; i < maxIterations && stepSize > convergenceThreshold; i++) {
    mean = VecXd::Zero(nDims);
    VecXd u1 = VecXd::Zero(nSamples);
    const MatXd scaleInv = scale.inverse();
    for (int n = 0; n < nSamples; n++) {
      VecXd r_n(data.row(n));
      u1[n] = (dof + nDims) / (dof + r_n.transpose() * scaleInv * r_n);
      mean += r_n * u1[n];
    }
    mean /= u1.sum();

    MatXd scale_i = MatXd::Zero(nDims, nDims);
    MatXd diff = data.rowwise() - mean.transpose();
    for (int n = 0; n < nSamples; n++) {
      VecXd diff_n(diff.row(n));
      scale_i += u1[n] * diff_n * diff_n.transpose();
    }
    scale_i /= nSamples;
    stepSize = (scale - scale_i).norm();
    scale = scale_i;
  }
  return Student_t(mean, scale, dof);
}

TEST(RandomTest, MultivariateStudent_t)
{
  VecXd meanGt = VecXd::Zero(2);
  MatXd covGt = MatXd::Identity(2, 2) * 10.0;
  int nSamples = 1000000;
  MatXd X = MatXd::Zero(nSamples, 2);
  for (int i = 0; i < nSamples; i++) {
    X.row(i) = random::student_t(meanGt, covGt, 5);
  }
  auto distr = std::make_shared<Student_t>(Student_t::fit(X, 5, MatXd::Identity(2, 2), 100, 1e-9));

  EXPECT_NEAR((distr->mean() - meanGt).norm(), 0., 1e-2) << "Mean* = " << distr->mean();
  EXPECT_NEAR((distr->scale() - covGt).norm(), 0., 1e-1) << "Cov* = " << distr->scale();
}