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

#ifndef VSLAM_LEAST_SQUARES_NORMAL_EQUATIONS_H__
#define VSLAM_LEAST_SQUARES_NORMAL_EQUATIONS_H__
#include <Eigen/Dense>
#include <memory>
#include <vector>

#include "core/core.h"

/*TODO:
- remove all constructors expect A / b one
- remove nContraints?
- remove addConstraint?
*/
namespace pd::vslam::least_squares
{
class NormalEquations
{
public:
  typedef std::shared_ptr<NormalEquations> ShPtr;
  typedef std::unique_ptr<NormalEquations> UnPtr;
  typedef std::shared_ptr<const NormalEquations> ConstShPtr;
  typedef std::unique_ptr<const NormalEquations> ConstUnPtr;

  NormalEquations(size_t nParameters);
  NormalEquations(const std::vector<NormalEquations> & normalEquations);
  NormalEquations(const std::vector<NormalEquations::ConstShPtr> & normalEquations);
  NormalEquations(const Eigen::MatrixXd & J, const Eigen::VectorXd & r, const Eigen::VectorXd & w);
  NormalEquations(
    const Eigen::MatrixXd & A, const Eigen::VectorXd & b, double chi2, size_t nConstraints);

  void addConstraint(const Eigen::VectorXd & J, double r, double w);
  void combine(const NormalEquations & that);

  std::string toString() const;

  Eigen::MatrixXd & A() { return _A; }
  Eigen::VectorXd & b() { return _b; }
  double & chi2() { return _chi2; }
  size_t & nConstraints() { return _nConstraints; }

  const Eigen::MatrixXd & A() const { return _A; }
  const Eigen::VectorXd & b() const { return _b; }
  const double & chi2() const { return _chi2; }
  const size_t & nConstraints() const { return _nConstraints; }

private:
  Eigen::MatrixXd _A;
  Eigen::VectorXd _b;
  double _chi2;
  size_t _nConstraints;
};

}  // namespace pd::vslam::least_squares
#endif
