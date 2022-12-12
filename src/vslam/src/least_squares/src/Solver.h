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

#ifndef VSLAM_SOLVER_H__
#define VSLAM_SOLVER_H__

#include "NormalEquations.h"
namespace pd::vslam::least_squares
{
class Problem
{
public:
  typedef std::shared_ptr<Problem> ShPtr;
  typedef std::unique_ptr<Problem> UnPtr;
  typedef std::shared_ptr<const Problem> ConstShPtr;
  typedef std::unique_ptr<const Problem> ConstUnPtr;

  size_t nParameters() const { return _nParameters; }
  Problem(size_t nParameters) : _nParameters(nParameters) {}

  virtual void updateX(const Eigen::VectorXd & dx) = 0;
  virtual void setX(const Eigen::VectorXd & x) = 0;
  virtual Eigen::VectorXd x() const = 0;
  virtual NormalEquations::ConstShPtr computeNormalEquations() = 0;

private:
  size_t _nParameters;
};

class Solver
{
public:
  typedef std::shared_ptr<Solver> ShPtr;
  typedef std::unique_ptr<Solver> UnPtr;
  typedef std::shared_ptr<const Solver> ConstShPtr;
  typedef std::unique_ptr<const Solver> ConstUnPtr;
  struct Results
  {
    typedef std::shared_ptr<Results> ShPtr;
    typedef std::unique_ptr<Results> UnPtr;
    typedef std::shared_ptr<const Results> ConstShPtr;
    typedef std::unique_ptr<const Results> ConstUnPtr;

    Eigen::VectorXd chi2, stepSize;
    Eigen::MatrixXd x;
    std::vector<NormalEquations::ConstShPtr> normalEquations;
    size_t iteration;
    VecXd solution() const;
    MatXd covariance() const;
  };

  virtual Results::ConstUnPtr solve(std::shared_ptr<Problem> problem) = 0;
};
}  // namespace pd::vslam::least_squares
#endif
