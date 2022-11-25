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

#include "NormalEquations.h"
namespace pd::vslam::least_squares
{
NormalEquations::NormalEquations(size_t nParameters)
: _A(Eigen::MatrixXd::Zero(nParameters, nParameters)),
  _b(Eigen::VectorXd::Zero(nParameters)),
  _chi2(0),
  _nConstraints(0)
{
}
NormalEquations::NormalEquations(const std::vector<NormalEquations> & normalEquations)
{
  _A = normalEquations[0].A();
  _b = normalEquations[0].b();
  _nConstraints = normalEquations[0].nConstraints();
  _chi2 = normalEquations[0].chi2();
  for (size_t i = 1; i < normalEquations.size(); i++) {
    _A.noalias() += normalEquations[i].A();
    _b.noalias() += normalEquations[i].b();
    _chi2 += normalEquations[i].chi2();
    _nConstraints += normalEquations[i].nConstraints();
  }
}
NormalEquations::NormalEquations(const std::vector<NormalEquations::ConstShPtr> & normalEquations)
{
  _A = normalEquations[0]->A();
  _b = normalEquations[0]->b();
  _nConstraints = normalEquations[0]->nConstraints();
  _chi2 = normalEquations[0]->chi2();
  for (size_t i = 1; i < normalEquations.size(); i++) {
    _A.noalias() += normalEquations[i]->A();
    _b.noalias() += normalEquations[i]->b();
    _chi2 += normalEquations[i]->chi2();
    _nConstraints += normalEquations[i]->nConstraints();
  }
}

NormalEquations::NormalEquations(
  const Eigen::MatrixXd & J, const Eigen::VectorXd & r, const Eigen::VectorXd & w)
{
  auto Jtw = J.transpose() * w.asDiagonal();
  _A = Jtw * J;
  _b = Jtw * r;
  _chi2 = (w.asDiagonal() * r).transpose() * r;
  _nConstraints = r.rows();
}

void NormalEquations::addConstraint(const Eigen::VectorXd & J, double r, double w)
{
  _A.noalias() += J * J.transpose() * w;
  _b.noalias() += J * r * w;
  _chi2 += r * r * w;
  _nConstraints++;
}
void NormalEquations::combine(const NormalEquations & that)
{
  _A.noalias() += that.A();
  _b.noalias() += that.b();
  _chi2 += that.chi2();
  _nConstraints += that.nConstraints();
}

std::string NormalEquations::toString() const
{
  std::stringstream ss;
  ss << "A=\n"
     << _A << "\nb=\n"
     << _b.transpose() << "\nnConstraints=" << _nConstraints << " chi2=" << _chi2;
  return ss.str();
}

}  // namespace pd::vslam::least_squares
