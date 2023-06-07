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

//
// Created by phil on 10.10.20.
//

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <core/core.h>
#include <gtest/gtest.h>
#include <utils/utils.h>

#include <sophus/ceres_manifold.hpp>

#include "BundleAdjustment.h"
using namespace testing;
using namespace pd;
using namespace pd::vslam;

class ReprojectionErrorManifold
{
public:
  ReprojectionErrorManifold(double observedX, double observedY, const Eigen::Matrix3d & K)
  : _observedX{observedX}, _observedY{observedY}, _K{K}
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

      residuals[0] = pIcs.x() / pIcs.z() - T(_observedX);
      residuals[1] = pIcs.y() / pIcs.z() - T(_observedY);
    } else {
      residuals[0] = T(0.0);
      residuals[1] = T(0.0);
    }
    return true;
  }
  static ceres::CostFunction * Create(double observed_x, double observed_y, const Eigen::MatrixXd K)
  {
    return new ceres::AutoDiffCostFunction<
      ReprojectionErrorManifold, 2, Sophus::SE3d::num_parameters, 3>(
      new ReprojectionErrorManifold(observed_x, observed_y, K));
  }

private:
  double _observedX, _observedY;
  const Eigen::Matrix3d _K;
};

class ReprojectionError
{
public:
  ReprojectionError(double observedX, double observedY, const Eigen::Matrix3d & K)
  : _observedX{observedX}, _observedY{observedY}, _K{K}
  {
  }

  template <typename T>
  bool operator()(const T * const pose, const T * const pWcs, T * residuals) const
  {
    T pCcs[3];
    ceres::AngleAxisRotatePoint(pose, pWcs, pCcs);
    pCcs[0] += pose[3];
    pCcs[1] += pose[4];
    pCcs[2] += pose[5];

    if (pCcs[2] > 0.1) {
      auto pIcsX = _K(0, 0) * pCcs[0] + _K(0, 1) * pCcs[1] + _K(0, 2) * pCcs[2];
      auto pIcsY = _K(1, 0) * pCcs[0] + _K(1, 1) * pCcs[1] + _K(1, 2) * pCcs[2];
      auto pIcsZ = pCcs[2];

      residuals[0] = pIcsX / pIcsZ - T(_observedX);
      residuals[1] = pIcsY / pIcsZ - T(_observedY);
    } else {
      residuals[0] = T(0.0);
      residuals[1] = T(0.0);
    }

    return true;
  }

  static ceres::CostFunction * Create(double observed_x, double observed_y, const Eigen::MatrixXd K)
  {
    return new ceres::AutoDiffCostFunction<ReprojectionError, 2, 6, 3>(
      new ReprojectionError(observed_x, observed_y, K));
  }

private:
  double _observedX, _observedY;
  const Eigen::Matrix3d _K;
};

class TestBundleAdjustment : public Test
{
public:
  TestBundleAdjustment()
  {
    _poses.resize(2);
    _posesv.resize(_poses.size());
    _poses[0] = SE3d();
    _posesv[0] = _poses[0].log();
    MatXd Rt = utils::loadMatCsv<Eigen::Matrix<double, 4, 4>>(TEST_RESOURCE "/Rt.csv");
    std::cout << "Rt: " << Rt << std::endl;
    _poses[1] = SE3d(Rt);
    _posesv[1] = _poses[1].log();

    auto points = utils::loadMatCsv<Eigen::Matrix<double, 100, 3>>(TEST_RESOURCE "/points3d.csv");

    for (int i = 0; i < points.rows(); ++i) {
      _pcl.push_back(points.row(i));
      _pcl[i].x() += random::U(-1.0, 1.0);
      _pcl[i].y() += random::U(-1.0, 1.0);
      _pcl[i].z() += random::U(-1.0, 1.0);
    }
    _observations.resize(_poses.size());
    _observations[0] =
      utils::loadMatCsv<Eigen::Matrix<double, 100, 2>>(TEST_RESOURCE "/observations1.csv");
    _observations[1] =
      utils::loadMatCsv<Eigen::Matrix<double, 100, 2>>(TEST_RESOURCE "/observations2.csv");

    _cam = std::make_shared<Camera>(525.0, 525.0, 319.5, 239.5);
  }
  double computeReprojectionError() const
  {
    double error = 0.0;
    for (size_t idxF = 0; idxF < _poses.size(); ++idxF) {
      for (size_t idxP = 0; idxP < _pcl.size(); ++idxP) {
        auto pCcs = _poses[idxF] * _pcl[idxP];
        if (pCcs.z() > 0.1) {
          Eigen::Vector2d pIcs = _cam->camera2image(pCcs);
          auto e = _observations[idxF].row(idxP).transpose() - pIcs;
          error += e.norm();
        }
      }
    }
    return error;
  }

protected:
  std::vector<SE3d> _poses;
  std::vector<Vec6d> _posesv;
  std::vector<Vec3d> _pcl;
  std::vector<Eigen::Matrix<double, 100, 2>> _observations;
  Camera::ConstShPtr _cam;
};

TEST_F(TestBundleAdjustment, DISABLED_BA)
{
  ceres::Problem problem;

  for (size_t idxF = 0; idxF < _poses.size(); ++idxF) {
    for (int idxP = 0; idxP < _observations[idxF].rows(); ++idxP) {
      problem.AddResidualBlock(
        ReprojectionError::Create(
          _observations[idxF].row(idxP).x(), _observations[idxF].row(idxP).y(), _cam->K()),
        nullptr /* squared loss */, _posesv[idxF].data(), _pcl[idxP].data());
    }
  }
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.minimizer_progress_to_stdout = true;
  options.max_num_iterations = 200;
  double errorPrev = computeReprojectionError();
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.FullReport() << "\n";
  double errorAfter = computeReprojectionError();
  EXPECT_LT(errorAfter, errorPrev) << "Error should decrease by optimization";
  std::cout << "Before: " << errorPrev << " -->  " << errorAfter << std::endl;
}

TEST_F(TestBundleAdjustment, BAManifold)
{
  ceres::Problem problem;

  for (size_t i = 0; i < _poses.size(); ++i) {
    problem.AddParameterBlock(
      _poses[i].data(), SE3d::num_parameters, new Sophus::Manifold<Sophus::SE3>());
  }

  for (size_t idxF = 0; idxF < _poses.size(); ++idxF) {
    for (int idxP = 0; idxP < _observations[idxF].rows(); ++idxP) {
      problem.AddResidualBlock(
        ReprojectionErrorManifold::Create(
          _observations[idxF].row(idxP).x(), _observations[idxF].row(idxP).y(), _cam->K()),
        nullptr /* squared loss */, _poses[idxF].data(), _pcl[idxP].data());
    }
  }
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.minimizer_progress_to_stdout = true;
  options.max_num_iterations = 200;
  double errorPrev = computeReprojectionError();
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.FullReport() << "\n";
  double errorAfter = computeReprojectionError();
  EXPECT_LT(errorAfter, errorPrev) << "Error should decrease by optimization";
  std::cout << "Before: " << errorPrev << " -->  " << errorAfter << std::endl;
}

TEST_F(TestBundleAdjustment, BAClass)
{
  std::vector<Frame::ShPtr> frames;

  std::vector<Point3D::ShPtr> points;
  for (size_t i = 0U; i < _poses.size(); ++i) {
    frames.push_back(std::make_shared<Frame>(
      Image::Zero(480, 640), _cam, i, PoseWithCovariance(_poses[i], MatXd::Identity(6, 6))));
  }
  for (size_t i = 0U; i < _pcl.size(); ++i) {
    std::shared_ptr<Point3D> point;
    for (size_t j = 0U; j < _observations.size(); ++j) {
      auto feature = std::make_shared<Feature2D>(_observations[j].row(i), frames[j]);
      frames[j]->addFeature(feature);
      if (!point) {
        point = std::make_shared<Point3D>(_pcl[i], feature);
      } else {
        point->addFeature(feature);
      }
      feature->point() = point;
    }
  }
  mapping::BundleAdjustment ba;
  auto results = ba.optimize(std::vector<Frame::ConstShPtr>(frames.begin(), frames.end()));
  EXPECT_LT(results->errorAfter, results->errorBefore) << "Error should decrease by optimization";
  std::cout << "Before: " << results->errorBefore << " -->  " << results->errorAfter << std::endl;
}
