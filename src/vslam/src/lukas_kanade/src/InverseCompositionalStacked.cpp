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

#include <execution>
#include <vector>

#include "InverseCompositionalStacked.h"
#include "core/core.h"
#include "utils/utils.h"
namespace pd::vslam::lukas_kanade
{
InverseCompositionalStacked::InverseCompositionalStacked(
  const std::vector<std::shared_ptr<InverseCompositional>> & frames)
: least_squares::Problem(frames[0]->nParameters()), _frames(frames)
{
}

void InverseCompositionalStacked::updateX(const Eigen::VectorXd & dx)
{
  std::for_each(_frames.begin(), _frames.end(), [&dx](auto f) { f->updateX(dx); });
}
void InverseCompositionalStacked::setX(const Eigen::VectorXd & x)
{
  std::for_each(_frames.begin(), _frames.end(), [&x](auto f) { f->setX(x); });
}
least_squares::NormalEquations::UnPtr InverseCompositionalStacked::computeNormalEquations()
{
  std::vector<least_squares::NormalEquations::ConstShPtr> nes(_frames.size());
  std::transform(_frames.begin(), _frames.end(), nes.begin(), [&](auto f) {
    return f->computeNormalEquations();
  });
  auto ne = std::make_unique<least_squares::NormalEquations>(_frames[0]->nParameters());
  std::for_each(nes.begin(), nes.end(), [&](auto n) { ne->combine(*n); });
  return ne;
}

}  // namespace pd::vslam::lukas_kanade
