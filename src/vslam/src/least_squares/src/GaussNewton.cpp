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
#include "PlotSolver.h"
#include "core/core.h"
#include "utils/utils.h"
namespace pd::vslam::least_squares
{
GaussNewton::GaussNewton(double minStepSize, size_t maxIterations)
: _minStepSize(minStepSize),
  _minGradient(minStepSize),
  _minReduction(minStepSize),
  _maxIterations(maxIterations),
  _maxIncrease(minStepSize)
{
  Log::get("solver");
  LOG_IMG("Solver");
}

GaussNewton::GaussNewton(
  double minStepSize, size_t maxIterations, double minGradient, double minReduction,
  double maxIncrease)
: _minStepSize(minStepSize),
  _minGradient(minGradient),
  _minReduction(std::min(1.0, minReduction)),
  _maxIterations(maxIterations),
  _maxIncrease(std::max(1.0, maxIncrease))
{
  Log::get("solver");
  LOG_IMG("Solver");
}

Solver::Results::ConstUnPtr GaussNewton::solve(std::shared_ptr<Problem> problem) const
{
  SOLVER(INFO) << "Solving Problem for " << problem->nParameters() << " parameters.";
  //TIMED_FUNC(timerF);

  auto r = std::make_unique<Solver::Results>();

  r->chi2 = Eigen::VectorXd::Zero(_maxIterations);
  r->stepSize = Eigen::VectorXd::Zero(_maxIterations);
  r->x = Eigen::MatrixXd::Zero(_maxIterations, problem->nParameters());
  r->normalEquations.reserve(_maxIterations);
  r->iteration = 0;
  r->convergenceCriteria = ConvergenceCriteria::MAX_ITERATIONS_EXCEEDED;

  for (size_t i = 0; i < _maxIterations; i++) {
    //TIMED_SCOPE(timerI, "solve ( " + std::to_string(i) + " )");

    // We want to solve dx = (JWJ)^(-1)*JWr
    // This can be solved with cholesky decomposition (Ax = b)
    // Where A = (JWJ + lambda * I), x = dx, b = JWr
    auto ne = problem->computeNormalEquations();

    const double det = ne->A().determinant();
    if (ne->nConstraints() < problem->nParameters()) {
      SOLVER(WARNING) << i << " > "
                      << "STOP. Not enough constraints: " << ne->nConstraints() << " / "
                      << problem->nParameters();
      r->convergenceCriteria = ConvergenceCriteria::NOT_ENOUGH_CONSTRAINTS;
      break;
    }
    if (!std::isfinite(det) || std::abs(det) < 1e-6) {
      SOLVER(WARNING) << i << " > "
                      << "STOP. Bad Hessian. det| H | = " << det << " \n"
                      << ne->toString();
      r->convergenceCriteria = ConvergenceCriteria::HESSIAN_SINGULAR;
      break;
    }

    SOLVER(DEBUG) << i << " > " << ne->toString();

    r->chi2(i) = ne->chi2();

    const double dChi2 = i > 1 ? r->chi2(i) / r->chi2(i - 1) : 0;
    if (i > 1 && dChi2 > _maxIncrease) {
      SOLVER(INFO) << i << " > "
                   << "CONVERGED. No improvement"
                   << " dChi2: " << dChi2 << "/" << _maxIncrease;
      problem->setX(r->x.row(i - 1));
      r->convergenceCriteria = ConvergenceCriteria::ERROR_INCREASED;
      break;
    }
    const VecXd dx = ne->A().ldlt().solve(ne->b());
    const auto gradient = std::abs(ne->b().maxCoeff());
    problem->updateX(dx);

    SOLVER(INFO) << "Iteration: " << i << " chi2: " << r->chi2(i) << " dChi2: " << dChi2
                 << " stepSize: " << r->stepSize(i) << " Points: " << ne->nConstraints()
                 << "\nx: " << problem->x().transpose() << "\ndx: " << dx.transpose();

    r->x.row(i) = problem->x();
    r->stepSize(i) = dx.norm();
    r->normalEquations.push_back(std::move(ne));
    r->iteration = i + 1;

    if (i > 1) {
      if (r->stepSize(i) < _minStepSize) {
        r->convergenceCriteria = ConvergenceCriteria::PARAMETER_THRESHOLD_REACHED;
        SOLVER(INFO) << to_string(r->convergenceCriteria);
        break;
      }
      if (gradient < _minGradient) {
        r->convergenceCriteria = ConvergenceCriteria::GRADIENT_THRESHOLD_REACHED;
        SOLVER(INFO) << to_string(r->convergenceCriteria);
        break;
      }
      if (1.0 > dChi2 && dChi2 > _minReduction) {
        r->convergenceCriteria = ConvergenceCriteria::BELOW_MIN_ERROR_REDUCTION;
        SOLVER(INFO) << to_string(r->convergenceCriteria);
        break;
      }
    }

    if (!std::isfinite(r->stepSize(i))) {
      SOLVER(ERROR) << i << " > "
                    << "STOP. NaN during optimization.";
      problem->setX(r->x.row(i - 1));
      r->convergenceCriteria = ConvergenceCriteria::NAN_DURING_OPTIMIZATION;
      break;
    }
  }
  LOG_IMG("Solver") << std::make_shared<PlotGaussNewton>(r->iteration - 1, r->chi2, r->stepSize);
  return r;
}

}  // namespace pd::vslam::least_squares
