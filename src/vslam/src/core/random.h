#ifndef VSLAM_RANDOM_H__
#define VSLAM_RANDOM_H__
#include "types.h"

namespace vslam::random
{
template <int dim>
struct Gaussian
{
  Matd<dim, dim> cov;
  Matd<dim, 1> mean;
};

double U(double min, double max);
uint64_t U(uint64_t min, uint64_t max);

int sign();

double N(double stddev);
double N(double mean, double stddev);
Eigen::VectorXd N(const Eigen::MatrixXd & cov);
Eigen::VectorXd N(const VecXd & mean, const Eigen::MatrixXd & cov);

double chi2(double dof);
double student_t(double mean, double cov, int dof);
VecXd student_t(VecXd mean, MatXd cov, int dof);

}  // namespace vslam::random

#endif