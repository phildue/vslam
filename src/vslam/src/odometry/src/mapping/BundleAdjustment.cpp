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

#include <sophus/ceres_manifold.hpp>

#include "BundleAdjustment.h"
#include "utils/utils.h"
#define LOG_MAPPING(level) CLOG(level, "mapping")

namespace pd::vslam::mapping
{
class ReprojectionErrorManifold
{
public:
  ReprojectionErrorManifold(const Eigen::Vector2d & observation, const Eigen::Matrix3d & K)
  : _obs{observation}, _K{K}
  {
  }

  template <typename T>
  bool operator()(const T * const poseData, const T * const pointData, T * residuals) const
  {
    Eigen::Map<Sophus::SE3<T> const> const pose(poseData);
    Eigen::Map<Eigen::Matrix<T, 3, 1> const> pWcs(pointData);

    auto pCcs = pose * pWcs;

    if (pCcs[2] > 0.1) {
      auto pIcs = _K * pCcs;

      residuals[0] = pIcs.x() / pIcs.z() - T(_obs.x());
      residuals[1] = pIcs.y() / pIcs.z() - T(_obs.y());
    } else {
      residuals[0] = T(0.0);
      residuals[1] = T(0.0);
    }
    return true;
  }
  static ceres::CostFunction * Create(
    const Eigen::Vector2d & observation, const Eigen::Matrix3d & K)
  {
    return new ceres::AutoDiffCostFunction<
      ReprojectionErrorManifold, 2, Sophus::SE3d::num_parameters, 3>(
      new ReprojectionErrorManifold(observation, K));
  }

private:
  const Eigen::Vector2d _obs;
  const Eigen::Matrix3d _K;
};
BundleAdjustment::BundleAdjustment(size_t maxIterations) : _maxIterations(maxIterations)
{
  Log::get("mapping");
}

BundleAdjustment::Results::ConstUnPtr BundleAdjustment::optimize(
  const std::vector<Frame::ConstShPtr> & frames) const
{
  Results::UnPtr results = std::make_unique<Results>();
  ceres::Problem problem;

  for (const auto & f : frames) {
    results->poses[f->id()] = f->pose();
    problem.AddParameterBlock(
      results->poses[f->id()].pose().data(), SE3d::num_parameters,
      new Sophus::Manifold<Sophus::SE3>());

    for (const auto & ft : f->featuresWithPoints()) {
      auto pointId = ft->point()->id();
      results->positions[pointId] = ft->point()->position();
      problem.AddResidualBlock(
        ReprojectionErrorManifold::Create(ft->position(), f->camera()->K()),
        nullptr /* squared loss */, results->poses[f->id()].pose().data(),
        results->positions[pointId].data());
    }
  }
  ceres::Problem::EvaluateOptions evalOptions;
  problem.Evaluate(evalOptions, &results->errorBefore, nullptr, nullptr, nullptr);

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.minimizer_progress_to_stdout = true;
  options.max_num_iterations = _maxIterations;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  LOG_MAPPING(DEBUG) << summary.FullReport();
  problem.Evaluate(evalOptions, &results->errorAfter, nullptr, nullptr, nullptr);

  return results;
}

}  // namespace pd::vslam::mapping
