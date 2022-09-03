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

#include "odometry/odometry.h"
using namespace testing;
using namespace pd;
using namespace pd::vslam;

class ReprojectionErrorManifold
{
public:
  ReprojectionErrorManifold(double observedX, double observedY)
  : _observedX{observedX}, _observedY{observedY}
  {
  }

  template <typename T>
  bool operator()(
    const T * const poseData, const T * const intrinsics, const T * const pointData,
    T * residuals) const
  {
    Eigen::Map<Sophus::SE3<T> const> const pose(poseData);
    Eigen::Map<Eigen::Matrix<T, 3, 1> const> point(pointData);

    auto pCcs = pose * point;
    // Compute the center of distortion. The sign change comes from
    // the camera model that Noah Snavely's Bundler assumes, whereby
    // the camera coordinate system has a negative z axis.
    T xp = -pCcs[0] / pCcs[2];
    T yp = -pCcs[1] / pCcs[2];

    // Apply second and fourth order radial distortion.
    const T & l1 = intrinsics[1];
    const T & l2 = intrinsics[2];
    T r2 = xp * xp + yp * yp;
    T distortion = 1.0 + r2 * (l1 + l2 * r2);

    // Compute final projected point position.
    const T & focal = intrinsics[0];
    T predicted_x = focal * distortion * xp;
    T predicted_y = focal * distortion * yp;
    residuals[0] = predicted_x - T(_observedX);
    residuals[1] = predicted_y - T(_observedY);

    return true;
  }
  static ceres::CostFunction * Create(double observed_x, double observed_y)
  {
    return new ceres::AutoDiffCostFunction<
      ReprojectionErrorManifold, 2, Sophus::SE3d::num_parameters, 3, 3>(
      new ReprojectionErrorManifold(observed_x, observed_y));
  }

private:
  double _observedX, _observedY;
  const Eigen::Matrix3d _K;
};

class ReprojectionError
{
public:
  ReprojectionError(double observedX, double observedY)
  : _observedX{observedX}, _observedY{observedY}
  {
  }

  template <typename T>
  bool operator()(
    const T * const pose, const T * const intrinsics, const T * const pWcs, T * residuals) const
  {
    T pCcs[3];
    ceres::AngleAxisRotatePoint(pose, pWcs, pCcs);
    pCcs[0] += pose[3];
    pCcs[1] += pose[4];
    pCcs[2] += pose[5];

    // Compute the center of distortion. The sign change comes from
    // the camera model that Noah Snavely's Bundler assumes, whereby
    // the camera coordinate system has a negative z axis.
    T xp = -pCcs[0] / pCcs[2];
    T yp = -pCcs[1] / pCcs[2];

    // Apply second and fourth order radial distortion.
    const T & l1 = intrinsics[1];
    const T & l2 = intrinsics[2];
    T r2 = xp * xp + yp * yp;
    T distortion = 1.0 + r2 * (l1 + l2 * r2);

    // Compute final projected point position.
    const T & focal = intrinsics[0];
    T predicted_x = focal * distortion * xp;
    T predicted_y = focal * distortion * yp;

    residuals[0] = predicted_x - T(_observedX);
    residuals[1] = predicted_y - T(_observedY);

    return true;
  }

  static ceres::CostFunction * Create(double observed_x, double observed_y)
  {
    return new ceres::AutoDiffCostFunction<ReprojectionError, 2, 6, 3, 3>(
      new ReprojectionError(observed_x, observed_y));
  }

private:
  double _observedX, _observedY;
  const Eigen::Matrix3d _K;
};

class BALProblem
{
public:
  ~BALProblem()
  {
    delete[] point_index_;
    delete[] camera_index_;
    delete[] observations_;
    delete[] parameters_;
  }

  int num_observations() const { return num_observations_; }
  int num_cameras() const { return num_cameras_; }
  const double * observations() const { return observations_; }
  double * mutable_cameras() { return parameters_; }
  double * mutable_points() { return parameters_ + 9 * num_cameras_; }
  double * mutable_poses(int i) { return _poses[i].data(); }
  double * mutable_pose_for_observation(int i) { return _poses[camera_index_[i]].data(); }

  double * mutable_camera_for_observation(int i)
  {
    return mutable_cameras() + camera_index_[i] * 9;
  }
  double * mutable_point_for_observation(int i) { return mutable_points() + point_index_[i] * 3; }

  bool LoadFile(const char * filename)
  {
    FILE * fptr = fopen(filename, "r");
    if (fptr == nullptr) {
      return false;
    }

    FscanfOrDie(fptr, "%d", &num_cameras_);
    FscanfOrDie(fptr, "%d", &num_points_);
    FscanfOrDie(fptr, "%d", &num_observations_);

    point_index_ = new int[num_observations_];
    camera_index_ = new int[num_observations_];
    observations_ = new double[2 * num_observations_];

    num_parameters_ = 9 * num_cameras_ + 3 * num_points_;
    parameters_ = new double[num_parameters_];

    for (int i = 0; i < num_observations_; ++i) {
      FscanfOrDie(fptr, "%d", camera_index_ + i);
      FscanfOrDie(fptr, "%d", point_index_ + i);
      for (int j = 0; j < 2; ++j) {
        FscanfOrDie(fptr, "%lf", observations_ + 2 * i + j);
      }
    }

    for (int i = 0; i < num_parameters_; ++i) {
      FscanfOrDie(fptr, "%lf", parameters_ + i);
    }
    for (int i = 0; i < num_cameras_; ++i) {
      Eigen::Map<Eigen::Vector6d> v(parameters_ + i * 9);
      _poses.push_back(Sophus::SE3d::exp(v));
    }
    return true;
  }

private:
  template <typename T>
  void FscanfOrDie(FILE * fptr, const char * format, T * value)
  {
    int num_scanned = fscanf(fptr, format, value);
    if (num_scanned != 1) {
      LOG(FATAL) << "Invalid UW data file.";
    }
  }

  int num_cameras_;
  int num_points_;
  int num_observations_;
  int num_parameters_;

  int * point_index_;
  int * camera_index_;
  double * observations_;
  double * parameters_;
  std::vector<SE3d> _poses;
};

// Templated pinhole camera model for used with Ceres.  The camera is
// parameterized using 9 parameters: 3 for rotation, 3 for translation, 1 for
// focal length and 2 for radial distortion. The principal point is not modeled
// (i.e. it is assumed be located at the image center).
struct SnavelyReprojectionError
{
  SnavelyReprojectionError(double observed_x, double observed_y)
  : observed_x(observed_x), observed_y(observed_y)
  {
  }

  template <typename T>
  bool operator()(const T * const camera, const T * const point, T * residuals) const
  {
    // camera[0,1,2] are the angle-axis rotation.
    T p[3];
    ceres::AngleAxisRotatePoint(camera, point, p);

    // camera[3,4,5] are the translation.
    p[0] += camera[3];
    p[1] += camera[4];
    p[2] += camera[5];

    // Compute the center of distortion. The sign change comes from
    // the camera model that Noah Snavely's Bundler assumes, whereby
    // the camera coordinate system has a negative z axis.
    T xp = -p[0] / p[2];
    T yp = -p[1] / p[2];

    // Apply second and fourth order radial distortion.
    const T & l1 = camera[7];
    const T & l2 = camera[8];
    T r2 = xp * xp + yp * yp;
    T distortion = 1.0 + r2 * (l1 + l2 * r2);

    // Compute final projected point position.
    const T & focal = camera[6];
    T predicted_x = focal * distortion * xp;
    T predicted_y = focal * distortion * yp;

    // The error is the difference between the predicted and observed position.
    residuals[0] = predicted_x - observed_x;
    residuals[1] = predicted_y - observed_y;

    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction * Create(const double observed_x, const double observed_y)
  {
    return new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 9, 3>(
      new SnavelyReprojectionError(observed_x, observed_y));
  }

  double observed_x;
  double observed_y;
};

class TestBundleAdjustment : public Test
{
public:
  TestBundleAdjustment()
  {
    _K.setIdentity();
    if (!_ba_data.LoadFile(TEST_RESOURCE "/ba.txt")) {
      std::cerr << "ERROR: unable to open file " << TEST_RESOURCE "/ba.txt"
                << "\n";
    }
  }

protected:
  BALProblem _ba_data;
  Eigen::Matrix3d _K;
};

//Example from ceres repo
TEST_F(TestBundleAdjustment, CeresExample)
{
  const double * observations = _ba_data.observations();

  ceres::Problem problem;
  for (int i = 0; i < _ba_data.num_observations(); ++i) {
    ceres::CostFunction * cost_function =
      SnavelyReprojectionError::Create(observations[2 * i + 0], observations[2 * i + 1]);
    problem.AddResidualBlock(
      cost_function, nullptr /* squared loss */, _ba_data.mutable_camera_for_observation(i),
      _ba_data.mutable_point_for_observation(i));
  }

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.minimizer_progress_to_stdout = true;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.FullReport() << "\n";
}

TEST_F(TestBundleAdjustment, BA)
{
  ceres::Problem problem;

  const double * observations = _ba_data.observations();

  for (int i = 0; i < _ba_data.num_observations(); ++i) {
    ceres::CostFunction * cost_function =
      ReprojectionError::Create(observations[2 * i + 0], observations[2 * i + 1]);
    problem.AddResidualBlock(
      cost_function, nullptr /* squared loss */, _ba_data.mutable_camera_for_observation(i),
      _ba_data.mutable_camera_for_observation(i) + 6, _ba_data.mutable_point_for_observation(i));
  }

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.minimizer_progress_to_stdout = true;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.FullReport() << "\n";
}

TEST_F(TestBundleAdjustment, BAManifold)
{
  ceres::Problem problem;

  for (int i = 0; i < _ba_data.num_cameras(); ++i) {
    ceres::Manifold * se3Manif = new Sophus::Manifold<Sophus::SE3>();
    problem.AddParameterBlock(_ba_data.mutable_poses(i), SE3d::num_parameters, se3Manif);
  }

  const double * observations = _ba_data.observations();

  for (int i = 0; i < _ba_data.num_observations(); ++i) {
    ceres::CostFunction * cost_function =
      ReprojectionError::Create(observations[2 * i + 0], observations[2 * i + 1]);
    problem.AddResidualBlock(
      cost_function, nullptr /* squared loss */, _ba_data.mutable_camera_for_observation(i),
      _ba_data.mutable_camera_for_observation(i) + 6, _ba_data.mutable_point_for_observation(i));
  }

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.minimizer_progress_to_stdout = true;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.FullReport() << "\n";
}
