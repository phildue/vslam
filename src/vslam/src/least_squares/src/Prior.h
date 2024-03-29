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

#ifndef VSLAM_SOLVER_PRIOR
#define VSLAM_SOLVER_PRIOR
#include <core/core.h>

#include "GaussNewton.h"
namespace pd::vslam::least_squares
{
class Prior
{
public:
  typedef std::shared_ptr<Prior> ShPtr;
  typedef std::unique_ptr<Prior> UnPtr;
  typedef std::shared_ptr<const Prior> ConstShPtr;
  typedef std::unique_ptr<const Prior> ConstUnPtr;

  virtual void apply(typename NormalEquations::ShPtr ne, const Eigen::VectorXd & x) const = 0;
};

}  // namespace pd::vslam::least_squares

#endif
