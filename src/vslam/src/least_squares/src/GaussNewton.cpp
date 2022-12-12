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

#include <memory>

#include "GaussNewton.h"
#include "core/core.h"
#include "utils/utils.h"
namespace pd::vslam::least_squares
{
GaussNewton::GaussNewton(double minStepSize, size_t maxIterations)
: _minStepSize(minStepSize),
  _minGradient(minStepSize),
  _minReduction(minStepSize),
  _maxIterations(maxIterations)
{
  Log::get("solver");
}

Solver::Results::ConstUnPtr GaussNewton::solve(std::shared_ptr<Problem> problem)
{
  SOLVER(INFO) << "Solving Problem for " << problem->nParameters() << " parameters.";
  TIMED_FUNC(timerF);

  auto r = std::make_unique<Solver::Results>();

  r->chi2 = Eigen::VectorXd::Zero(_maxIterations);
  r->stepSize = Eigen::VectorXd::Zero(_maxIterations);
  r->x = Eigen::MatrixXd::Zero(_maxIterations, problem->nParameters());
  r->normalEquations.reserve(_maxIterations);
  for (size_t i = 0; i < _maxIterations; i++) {
    TIMED_SCOPE(timerI, "solve ( " + std::to_string(i) + " )");

    // We want to solve dx = (JWJ)^(-1)*JWr
    // This can be solved with cholesky decomposition (Ax = b)
    // Where A = (JWJ + lambda * I), x = dx, b = JWr
    auto ne = problem->computeNormalEquations();

    const double det = ne->A().determinant();
    if (ne->nConstraints() < problem->nParameters()) {
      SOLVER(WARNING) << i << " > "
                      << "STOP. Not enough constraints: " << ne->nConstraints() << " / "
                      << problem->nParameters();
      break;
    }
    if (!std::isfinite(det) || std::abs(det) < 1e-6) {
      SOLVER(WARNING) << i << " > "
                      << "STOP. Bad Hessian. det| H | = " << det;
      break;
    }

    SOLVER(DEBUG) << i << " > " << ne->toString();

    r->chi2(i) = ne->chi2();

    const double dChi2 = i > 0 ? r->chi2(i) - r->chi2(i - 1) : 0;
    if (i > 0 && dChi2 > 0) {
      SOLVER(INFO) << i << " > "
                   << "CONVERGED. No improvement";
      problem->setX(r->x.row(i - 1));
      break;
    }
    const VecXd dx = ne->A().ldlt().solve(ne->b());
    problem->updateX(dx);
    r->x.row(i) = problem->x();
    r->stepSize(i) = dx.norm();
    r->normalEquations.push_back(ne);
    r->iteration = i;

    SOLVER(INFO) << "Iteration: " << i << " chi2: " << r->chi2(i) << " dChi2: " << dChi2
                 << " stepSize: " << r->stepSize(i) << " Points: " << ne->nConstraints()
                 << "\nx: " << problem->x().transpose() << "\ndx: " << dx.transpose();
    if (
      i > 0 && (r->stepSize(i) < _minStepSize || std::abs(ne->b().maxCoeff()) < _minGradient ||
                std::abs(dChi2) < _minReduction)) {
      SOLVER(INFO) << i << " > " << r->stepSize(i) << "/" << _minStepSize << " CONVERGED. ";
      break;
    }

    if (!std::isfinite(r->stepSize(i))) {
      SOLVER(ERROR) << i << " > "
                    << "STOP. NaN during optimization.";
      problem->setX(r->x.row(i - 1));
      break;
    }
  }
  LOG_IMG("SolverGN") << std::make_shared<vis::PlotGaussNewton>(r->iteration, r->chi2, r->stepSize);
  return r;
}

}  // namespace pd::vslam::least_squares
