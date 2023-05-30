#include "macros.h"
#include "random.h"
namespace vslam::random
{
static std::default_random_engine eng(0);

template <typename T = double>
T U(T min, T max)
{
  std::uniform_real_distribution<double> distr(min, max);
  return static_cast<T>(distr(eng));
}
double U(double min, double max) { return U<double>(min, max); }
uint64_t U(uint64_t min, uint64_t max) { return U<uint64_t>(min, max); }

template <typename T = double>
T chi2(double dof)
{
  std::chi_squared_distribution<T> distr{dof};
  return static_cast<T>(distr(eng));
}

double chi2(double dof) { return chi2<double>(dof); }

int sign() { return U(-1, 1) > 0 ? 1 : -1; }

template <typename T = double>
T N(T mean, T stddev)
{
  std::normal_distribution<double> distr(mean, stddev);
  return static_cast<T>(distr(eng));
}

double N(double stddev) { return N<double>(0., stddev); }
double N(double mean, double stddev) { return N<double>(mean, stddev); }

Eigen::VectorXd N(const Eigen::MatrixXd & cov)
{
  std::normal_distribution<> dist;
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(cov);
  auto transform = eigenSolver.eigenvectors() * eigenSolver.eigenvalues().cwiseSqrt().asDiagonal();

  return transform *
         Eigen::VectorXd{cov.cols()}.unaryExpr([&](auto UNUSED(x)) { return dist(eng); });
}

Eigen::VectorXd N(const VecXd & mean, const Eigen::MatrixXd & cov) { return mean + N(cov); }

double student_t(double mean, double cov, int dof)
{
  const double s = chi2(dof) / dof;
  const double x = N(cov);
  return mean + x / std::sqrt(s);
}

VecXd student_t(VecXd mean, MatXd cov, int dof)
{
  const double s = chi2(dof) / dof;
  const VecXd x = N(cov);
  return mean + x / std::sqrt(s);
}

}  // namespace vslam::random