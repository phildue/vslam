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

#ifndef VSLAM_GAUSS_NEWTON_H__
#define VSLAM_GAUSS_NEWTON_H__
#include <core/core.h>

#include <Eigen/Dense>
#include <memory>

#include "Solver.h"

namespace pd::vslam::least_squares
{
class GaussNewton : public Solver
{
public:
  typedef std::shared_ptr<GaussNewton> ShPtr;
  typedef std::unique_ptr<GaussNewton> UnPtr;
  typedef std::shared_ptr<const GaussNewton> ConstShPtr;
  typedef std::unique_ptr<const GaussNewton> ConstUnPtr;

  GaussNewton(double minStepSize, size_t maxIterations);

  typename Solver::Results::ConstUnPtr solve(std::shared_ptr<Problem> problem) override;

private:
  const double _minStepSize;
  const double _minGradient;
  const double _minReduction;
  const size_t _maxIterations;
};

}  // namespace pd::vslam::least_squares
#endif
