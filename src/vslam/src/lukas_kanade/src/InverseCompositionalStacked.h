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

#ifndef VSLAM_LUKAS_KANADE_INVERSE_COMPOSITIONAL_STACKED_H__
#define VSLAM_LUKAS_KANADE_INVERSE_COMPOSITIONAL_STACKED_H__

#include <memory>
#include <vector>

#include "InverseCompositional.h"
#include "core/core.h"
#include "least_squares/least_squares.h"

namespace pd::vslam::lukas_kanade
{
class InverseCompositionalStacked : public least_squares::Problem
{
public:
  virtual ~InverseCompositionalStacked() = default;
  InverseCompositionalStacked(const std::vector<std::shared_ptr<InverseCompositional>> & frames);
  std::shared_ptr<const Warp> warp() { return _frames[0]->warp(); }

  void updateX(const Eigen::VectorXd & dx) override;

  Eigen::VectorXd x() const override { return _frames[0]->x(); }
  void setX(const Eigen::VectorXd & x) override;

  least_squares::NormalEquations::ConstShPtr computeNormalEquations() override;

protected:
  std::vector<std::shared_ptr<InverseCompositional>> _frames;
};

}  // namespace pd::vslam::lukas_kanade
#endif
