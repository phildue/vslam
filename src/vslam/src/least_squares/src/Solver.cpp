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
VecXd Solver::Results::solution(int iter) const
{
  return iter < 0 ? x.row(iter + iteration) : x.row(iter);
}
MatXd Solver::Results::covariance(int iter) const
{
  auto idx = iter < 0 ? iter + iteration : iter;
  //https://stats.stackexchange.com/questions/482985/non-linear-least-squares-covariance-estimate
  auto normalizer = (normalEquations[idx]->chi2()) /
                    (normalEquations[idx]->nConstraints() - normalEquations[idx]->nParameters());
  //auto normalizer = 1.0;
  return normalizer * normalEquations[idx]->A().inverse();
}
}  // namespace pd::vslam::least_squares
