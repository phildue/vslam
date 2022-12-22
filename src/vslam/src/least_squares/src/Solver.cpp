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

#include "Solver.h"
namespace pd::vslam::least_squares
{
bool Solver::Results::hasSolution() const { return iteration > 0; }
VecXd Solver::Results::solution() const
{
  if (hasSolution()) {
    return x.row(iteration - 1);
  } else {
    return VecXd::Zero(1);
  }
}
MatXd Solver::Results::covariance() const
{
  if (hasSolution()) {
    return normalEquations[iteration - 1]->A().inverse();
  } else {
    return MatXd::Zero(1, 1);
  }
}
}  // namespace pd::vslam::least_squares
