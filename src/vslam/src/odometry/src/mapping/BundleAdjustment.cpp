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
BundleAdjustment::BundleAdjustment() { Log::get("mapping"); }

void BundleAdjustment::setFrame(std::uint64_t frameId, const SE3d & pose, const Mat3d & K)
{
  _poses[frameId] = pose;
  _Ks[frameId] = K;
  _problem.AddParameterBlock(
    _poses[frameId].data(), SE3d::num_parameters, new Sophus::Manifold<Sophus::SE3>());
}
void BundleAdjustment::setPoint(std::uint64_t pointId, const Vec3d & position)
{
  _points[pointId] = position;
}
void BundleAdjustment::setObservation(
  std::uint64_t pointId, std::uint64_t frameId, const Vec2d & observation)
{
  auto itF = _Ks.find(frameId);
  if (itF == _Ks.end()) {
    throw pd::Exception("No corresponding frame found.");
  }
  auto itP = _points.find(pointId);
  if (itP == _points.end()) {
    throw pd::Exception("No corresponding point found.");
  }

  auto & K = itF->second;
  auto & pose = _poses.find(frameId)->second;
  auto & point = itP->second;
  _problem.AddResidualBlock(
    ReprojectionErrorManifold::Create(observation, K), nullptr /* squared loss */, pose.data(),
    point.data());
}

void BundleAdjustment::optimize()
{
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.minimizer_progress_to_stdout = true;
  options.max_num_iterations = 200;
  //double errorPrev = computeReprojectionError();
  ceres::Solver::Summary summary;
  ceres::Solve(options, &_problem, &summary);

  LOG_MAPPING(DEBUG) << summary.FullReport();
  //double errorAfter = computeReprojectionError();
  //std::cout << "Before: " << errorPrev << " -->  " << errorAfter << std::endl;
}
SE3d BundleAdjustment::getPose(std::uint64_t frameId) const
{
  auto it = _poses.find(frameId);
  if (it != _poses.end()) {
    return it->second;
  } else {
    throw pd::Exception("Did not find corresponding pose.");
  }
}
Vec3d BundleAdjustment::getPoint(std::uint64_t pointId) const
{
  auto it = _points.find(pointId);
  if (it != _points.end()) {
    return it->second;
  } else {
    throw pd::Exception("Did not find corresponding point.");
  }
}

double BundleAdjustment::computeError() const
{
  double error;
  ceres::Problem::EvaluateOptions options;
  _problem.Evaluate(options, &error, nullptr, nullptr, nullptr);
  return error;
}

}  // namespace pd::vslam::mapping
