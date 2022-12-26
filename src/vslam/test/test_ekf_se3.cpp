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
// Created by phil on 25.11.22.
//

#include <fmt/chrono.h>
#include <fmt/core.h>
#include <fmt/ostream.h>

#include <Eigen/Dense>
#include <iostream>
using fmt::format;
using fmt::print;
template <typename T>
struct fmt::formatter<T, std::enable_if_t<std::is_base_of_v<Eigen::DenseBase<T>, T>, char>>
: ostream_formatter
{
};
//  ^ code unit type
#include <evaluation/evaluation.h>
#include <gtest/gtest.h>

#include <iomanip>
#include <iostream>

#include "core/core.h"
#include "manif/SE3.h"
#include "odometry/odometry.h"
#include "utils/utils.h"
using std::cout;
using std::endl;

using namespace testing;
using namespace pd;
using namespace pd::vslam;
using namespace pd::vslam::evaluation;

class TestKalmanSE3 : public Test
{
public:
  TestKalmanSE3()
  {
    LOG_IMG("Kalman")->set(TEST_VISUALIZE);

    _kalman = std::make_shared<EKFConstantVelocitySE3>(Matd<12, 12>::Identity(), 0);

    for (auto name : {"odometry"}) {
      el::Loggers::getLogger(name);
      el::Configurations defaultConf;
      defaultConf.setToDefault();
      defaultConf.set(el::Level::Debug, el::ConfigurationType::Format, "%datetime %level %msg");
      defaultConf.set(el::Level::Debug, el::ConfigurationType::Enabled, "false");
      defaultConf.set(el::Level::Info, el::ConfigurationType::Enabled, "false");
      el::Loggers::reconfigureLogger(name, defaultConf);
    }
    LOG_IMG("Test")->set(TEST_VISUALIZE, false);
  }

protected:
  EKFConstantVelocitySE3::ShPtr _kalman;
  std::map<std::string, Trajectory::ShPtr> _trajectories;
  std::map<std::string, RelativePoseError::ConstShPtr> _rpes;
};

TEST_F(TestKalmanSE3, DISABLED_RunWithGt)
{
  Trajectory::ConstShPtr trajectoryGt =
    utils::loadTrajectory(TEST_RESOURCE "/rgbd_dataset_freiburg1_floor-groundtruth.txt", true);

  std::uint16_t fNo = 0;
  _kalman->covarianceProcess() = Matd<12, 12>::Identity() * 1e-15;
  _kalman->reset(
    trajectoryGt->tStart(), {trajectoryGt->poseAt(trajectoryGt->tStart())->twist(), Vec6d::Zero(),
                             Matd<12, 12>::Identity() * 1e9});

  SE3d poseRef = trajectoryGt->poseAt(trajectoryGt->tStart())->SE3();
  Trajectory::ShPtr trajectory = std::make_shared<Trajectory>();
  Trajectory::ShPtr trajectoryLastMotion = std::make_shared<Trajectory>();
  SE3d poseLastMotion = poseRef;
  Vec6d lastVel = Vec6d::Zero();
  Timestamp t_1 = trajectoryGt->tStart();
  std::vector<double> dts;
  std::vector<double> timestamps;

  for (auto t_p : trajectoryGt->poses()) {
    try {
      const auto t = t_p.first;
      const double dT = t - t_1;
      const auto pose = t_p.second->SE3();
      poseLastMotion = poseLastMotion * SE3d::exp(lastVel * dT);
      auto pred = _kalman->predict(t);
      auto motionGt = (poseRef.inverse() * pose).log();

      _kalman->update(motionGt, Matd<6, 6>::Identity(), t);
      trajectory->append(t, std::make_shared<Pose>(pred->pose(), pred->covPose()));
      trajectoryLastMotion->append(
        t, std::make_shared<Pose>(poseLastMotion, Matd<6, 6>::Identity()));
      dts.push_back(dT);
      timestamps.push_back(t - trajectoryGt->tStart());
      poseRef = pose;
      t_1 = t;
      if (dT > 0) {
        lastVel = motionGt / dT;
      }
    } catch (const pd::Exception & e) {
      print("{}\n", e.what());
    }
    fNo++;
  }
  std::map<std::string, Trajectory::ConstShPtr> trajectories;
  trajectories["GroundTruth"] = trajectoryGt;
  trajectories["Kalman"] = trajectory;
  trajectories["LastMotion"] = trajectoryLastMotion;
  std::map<std::string, evaluation::RelativePoseError::ConstShPtr> rpes;

  evaluation::RelativePoseError::ConstShPtr rpe =
    evaluation::RelativePoseError::compute(trajectory, trajectoryGt, 0.05);
  evaluation::RelativePoseError::ConstShPtr rpeLastMotion =
    evaluation::RelativePoseError::compute(trajectoryLastMotion, trajectoryGt, 0.05);

  EXPECT_NEAR(rpe->translation().rmse, rpeLastMotion->translation().rmse, 0.001)
    << "With high state uncertainty kalman should always use the last velocity for prediction and "
       "thus achieve similar error.";
  EXPECT_NEAR(rpe->angle().rmse, rpeLastMotion->angle().rmse, 0.001)
    << "With high state uncertainty kalman should always use the last velocity for prediction and "
       "thus achieve similar error.";
  rpes["Kalman"] = rpe;
  rpes["LastMotion"] = rpeLastMotion;
  print("LastMotion\n: {}\n", rpeLastMotion->toString());
  print("Kalman:\n{}\n", rpe->toString());

  LOG_IMG("Test") << std::make_shared<evaluation::PlotTrajectory>(trajectories);
  LOG_IMG("Test") << std::make_shared<evaluation::PlotTrajectoryCovariance>(trajectories);
  LOG_IMG("Test") << std::make_shared<evaluation::PlotRPE>(rpes);
  vis::plt::figure();
  vis::plt::title("dts");

  vis::plt::plot(timestamps, dts);
  vis::plt::show();
}

TEST_F(TestKalmanSE3, RunWithAlgo)
{
  Trajectory::ShPtr trajectoryGt =
    utils::loadTrajectory(TEST_RESOURCE "/rgbd_dataset_freiburg1_floor-groundtruth.txt", true);
  Trajectory::ShPtr trajectoryAlgo =
    utils::loadTrajectory(TEST_RESOURCE "/rgbd_dataset_freiburg1_floor-algo.txt", true);

  auto statsAlgo = trajectoryAlgo->meanMotion();
  print(
    "Velocity statistics [Algo]: Sample time: {}s, n_poses: {} \nmean: {}\n cov: {}\n",
    trajectoryAlgo->averageSampleTime() / 1e9, trajectoryAlgo->size(),
    statsAlgo->mean().transpose(), statsAlgo->twistCov());
  auto stats =
    trajectoryGt->meanMotion(static_cast<Timestamp>(trajectoryAlgo->averageSampleTime()));
  print(
    "Velocity statistics [Gt]: Sample time: {}s mean: {}\n cov: {}\n",
    trajectoryAlgo->averageSampleTime() / 1e9, stats->mean().transpose(), stats->twistCov());
  std::uint16_t fNo = 0;
  _kalman->covarianceProcess() = Matd<12, 12>::Identity() * 1e-15;
  _kalman->covarianceProcess().block(6, 6, 6, 6) = stats->twistCov();
  _kalman->reset(
    trajectoryAlgo->tStart(), {trajectoryAlgo->poseAt(trajectoryAlgo->tStart())->twist(),
                               Vec6d::Zero(), Matd<12, 12>::Identity() * 1e9});
  SE3d poseRef = trajectoryAlgo->poseAt(trajectoryAlgo->tStart())->SE3();
  _trajectories["LastMotion"] = std::make_shared<Trajectory>();
  _trajectories["Kalman"] = std::make_shared<Trajectory>();
  _trajectories["KalmanSmooth"] = std::make_shared<Trajectory>();
  _trajectories["GroundTruth"] = std::make_shared<Trajectory>();
  _trajectories["Algorithm"] = std::make_shared<Trajectory>();
  SE3d poseLastMotion = poseRef;
  Vec6d lastVel = Vec6d::Zero();
  Timestamp t_1 = trajectoryGt->tStart();
  std::vector<double> dts;
  std::vector<double> timestamps;
  const std::uint16_t nFrames = 2000;
  for (auto t_p : trajectoryAlgo->poses()) {
    try {
      const auto t = t_p.first;
      const double dT = t - t_1;
      const auto pose = t_p.second->SE3();
      poseLastMotion = poseLastMotion * SE3d::exp(lastVel * dT);
      auto pred = _kalman->predict(t);
      auto motion = (poseRef.inverse() * pose).log();

      _kalman->update(motion, Matd<6, 6>::Identity(), t);
      _trajectories["KalmanSmooth"]->append(
        t, std::make_shared<Pose>(pred->pose(), pred->covPose()));
      _trajectories["Kalman"]->append(
        t, std::make_shared<Pose>(_kalman->state()->pose(), _kalman->state()->covPose()));
      _trajectories["LastMotion"]->append(
        t, std::make_shared<Pose>(poseLastMotion, Matd<6, 6>::Identity()));
      _trajectories["GroundTruth"]->append(t, trajectoryGt->poseAt(t));
      _trajectories["Algorithm"]->append(t, trajectoryAlgo->poseAt(t));

      dts.push_back(dT);
      timestamps.push_back(t - trajectoryAlgo->tStart());
      poseRef = pose;
      t_1 = t;
      if (dT > 0) {
        lastVel = motion / dT;
      }
    } catch (const pd::Exception & e) {
      print("{}\n", e.what());
    }
    if (fNo++ > nFrames) {
      break;
    }
  }
  for (auto n_t : _trajectories) {
    _rpes[n_t.first] = RelativePoseError::compute(n_t.second, trajectoryGt, 0.1);
    print("{}\n{}\n", n_t.first, _rpes[n_t.first]->toString());
  }

  LOG_IMG("Test") << std::make_shared<evaluation::PlotTrajectory>(_trajectories);
  LOG_IMG("Test") << std::make_shared<evaluation::PlotTrajectoryCovariance>(_trajectories);
  LOG_IMG("Test") << std::make_shared<evaluation::PlotTrajectoryMotion>(_trajectories);
  LOG_IMG("Test") << std::make_shared<evaluation::PlotRPE>(_rpes);
  vis::plt::figure();
  vis::plt::title("dts");

  vis::plt::plot(timestamps, dts);
  vis::plt::show();
}

typedef Eigen::Matrix<double, 12, 12> Mat12d;
typedef Eigen::Matrix<double, 6, 6> Mat6d;
typedef Eigen::Array<double, 3, 1> Array3d;
typedef Eigen::Array<double, 6, 1> Array6d;
typedef Eigen::Matrix<double, 6, 1> Vector6d;
typedef Eigen::Matrix<double, 6, 6> Matrix6d;
TEST_F(TestKalmanSE3, DISABLED_SyntheticData)
{
  /*State (Simulated, Without Filtering)*/
  manif::SE3d Xp_simulation, Xp_unfiltered;
  manif::SE3d Xv_simulation, Xv_unfiltered;

  Xp_simulation.setIdentity();
  Xv_simulation.setIdentity();
  Xp_unfiltered.setIdentity();
  Xv_unfiltered.setIdentity();

  _kalman->covarianceProcess() =
    Mat12d::Identity() * 1e-15;  //Trade-Off between adaptation and smoothing, hand tuned

  _kalman->reset(0UL, {Vec6d::Zero(), Vec6d::Zero(), Mat12d::Identity() * 1000});

  // Define a velocity vector and its noise and covariance
  Vec6d velocity;
  velocity << 0.1, 0.0, 0.0, 0.0, 0.0, 0.05;
  Xv_simulation = manif::SE3Tangentd(velocity).exp();
  Array6d measurement_sigmas;
  measurement_sigmas << 0.01, 0.01, 0.01, 0.00, 0.00, 0.01;

  // Covariance matrix of the measurements
  Matrix6d R = (measurement_sigmas * measurement_sigmas).matrix().asDiagonal();

  _trajectories["Unfiltered"] = std::make_shared<Trajectory>();
  _trajectories["Kalman"] = std::make_shared<Trajectory>();
  _trajectories["GroundTruth"] = std::make_shared<Trajectory>();
  for (int t = 0; t < 200; t++) {
    //// I. Simulation ###############################################################################

    /// move robot by true velocity - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Xp_simulation =
      Xp_simulation + manif::SE3Tangentd(velocity);  // overloaded X.rplus(u) = X * exp(u)
    _trajectories["GroundTruth"]->append(t, Pose(Xp_simulation.log().coeffs()));

    /// simulate noise on velocity
    auto measurement_noise = measurement_sigmas * Array6d::Random();  // control noise
    auto velocity_noisy = velocity + measurement_noise.matrix();      // noisy control

    // move also an unfiltered version for comparison purposes
    Xp_unfiltered = Xp_unfiltered + manif::SE3Tangentd(velocity_noisy);
    Xv_unfiltered = manif::SE3Tangentd(velocity_noisy).exp();
    _trajectories["Unfiltered"]->append(t, Pose(Xp_unfiltered.log().coeffs()));
    //// II. Estimation ###############################################################################

    _kalman->update(velocity, R, t);

    _trajectories["Kalman"]->append(t, Pose(_kalman->state()->pose(), _kalman->state()->covPose()));
  }

  EXPECT_NEAR((_kalman->state()->velocity() - velocity).norm(), 0.0, 0.0001)
    << "After 200 iterations filter should converge to the true velocity.";

  if (TEST_VISUALIZE) {
    LOG_IMG("Test")->set(TEST_VISUALIZE, false);
    LOG_IMG("Test") << std::make_shared<evaluation::PlotTrajectory>(_trajectories);
    LOG_IMG("Test") << std::make_shared<evaluation::PlotTrajectoryCovariance>(_trajectories);
    vis::plt::show();
  }
}

using namespace Eigen;

TEST(EKFSE3, DISABLED_PoC)
{
  /**
     * Adapted example from manif library.
     * State consists of [pose, velocity] .
     * Where pose is the integrated velocity 
     * and the velocity is assumed to be constant
     * with some noise to allow for derivations.
     * pose(t) = pose(t-1) * exp(vel*dt) 
     * vel(t) = vel(t-1) + w(t-1)
     * Simple test on synthetic data.
    */
  std::srand((unsigned int)std::time(0));

  /*State (Estimated, Simulated, Without Filtering)*/
  manif::SE3d Xp, Xp_simulation, Xp_unfiltered;
  manif::SE3d Xv, Xv_simulation, Xv_unfiltered;

  Xp_simulation.setIdentity();
  Xp.setIdentity();
  Xp_unfiltered.setIdentity();
  Xv.setIdentity();
  Xv_unfiltered.setIdentity();
  /*Estimated uncertainty*/
  Mat12d P;
  //Initialize with high value to give high weight on uncertainties at the beginning
  P.setIdentity() * 1000;

  /*Process noise*/
  Mat12d Q = Mat12d::Identity() * 1e-15;  //Trade-Off between adaptation and smoothing, hand tuned

  // Define a velocity vector and its noise and covariance
  Vec6d velocity;
  velocity << 0.1, 0.0, 0.0, 0.0, 0.0, 0.05;
  Xv_simulation = manif::SE3Tangentd(velocity).exp();
  Array6d measurement_sigmas;
  measurement_sigmas << 0.01, 0.01, 0.01, 0.00, 0.00, 0.01;

  // Covariance matrix of the measurements
  Matrix6d R = (measurement_sigmas * measurement_sigmas).matrix().asDiagonal();

  // Declare the Jacobian of the measurements wrt to robot state
  Matd<6, 12> J_h_x;
  J_h_x << MatXd::Zero(6, 6), MatXd::Identity(6, 6);

  LOG_IMG("Kalman")->set(TEST_VISUALIZE);
  auto plot = PlotKalman::make();
  std::map<std::string, Trajectory::ShPtr> trajectories;
  trajectories["Unfiltered"] = std::make_shared<Trajectory>();
  trajectories["Kalman"] = std::make_shared<Trajectory>();
  trajectories["GroundTruth"] = std::make_shared<Trajectory>();
  for (int t = 0; t < 200; t++) {
    //// I. Simulation ###############################################################################

    /// move robot by true velocity - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Xp_simulation =
      Xp_simulation + manif::SE3Tangentd(velocity);  // overloaded X.rplus(u) = X * exp(u)
    trajectories["GroundTruth"]->append(t, Pose(Xp_simulation.log().coeffs()));

    /// simulate noise on velocity
    auto measurement_noise = measurement_sigmas * Array6d::Random();  // control noise
    auto velocity_noisy = velocity + measurement_noise.matrix();      // noisy control

    // move also an unfiltered version for comparison purposes
    Xp_unfiltered = Xp_unfiltered + manif::SE3Tangentd(velocity_noisy);
    Xv_unfiltered = manif::SE3Tangentd(velocity_noisy).exp();
    trajectories["Unfiltered"]->append(t, Pose(Xp_unfiltered.log().coeffs()));
    //// II. Estimation ###############################################################################

    /// First we move - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    manif::SE3d::Jacobian J_xp, J_xv;
    Xp = Xp.plus(Xv.log(), J_xp, J_xv);  // X * exp(u), with Jacobians
    Mat12d J_x = Mat12d::Zero();
    J_x.block(0, 0, 6, 6) = J_xp;
    J_x.block(6, 0, 6, 6) = J_xv;
    P = J_x * P * J_x.transpose() + Q;

    // expectation
    Vec6d e = Xv.log().coeffs();
    Mat6d E = J_h_x * P * J_h_x.transpose();

    // innovation
    Vec6d z = velocity_noisy - e;
    Mat6d Z = E + R;

    // Kalman gain
    auto K = P * J_h_x.transpose() * Z.inverse();  // K = P * H.tr * ( H * P * H.tr + R).inv

    // Correction step
    Vec12d dx = K * z;  // dx is in the tangent space at X
    manif::SE3Tangentd dxp = dx.block(0, 0, 6, 1);
    manif::SE3Tangentd dxv = dx.block(6, 0, 6, 1);

    // Update
    Xp = Xp + dxp;  // overloaded X.rplus(dx) = X * exp(dx)
    Xv = Xv + dxv;  // overloaded X.rplus(dx) = X * exp(dx)
    P = P - K * Z * K.transpose();
    Vec12d state;
    state << Xp.log().coeffs(), Xv.log().coeffs();
    //plot->append(t, {state, e, velocity_noisy, z, dx, P, E, R, K});
    trajectories["Kalman"]->append(t, Pose(Xp.log().coeffs(), P.block(0, 0, 6, 6)));
  }

  EXPECT_NEAR((Xv.log().coeffs() - velocity).norm(), 0.0, 0.0001)
    << "After 200 iterations filter should converge to the true velocity.";

  if (TEST_VISUALIZE) {
    LOG_IMG("Test")->set(TEST_VISUALIZE, false);
    //LOG_IMG("Test") << plot;
    LOG_IMG("Test") << std::make_shared<evaluation::PlotTrajectory>(trajectories);
    LOG_IMG("Test") << std::make_shared<evaluation::PlotTrajectoryCovariance>(trajectories);
    vis::plt::show();
  }
}
