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

#include "core/core.h"
#include "odometry/odometry.h"
#include "utils/utils.h"
using namespace testing;
using namespace pd;
using namespace pd::vslam;
using namespace pd::vslam::evaluation;

class TestKalmanSE3 : public Test
{
public:
  TestKalmanSE3()
  {
    LOG_PLT("Kalman")->set(TEST_VISUALIZE);

    _kalman = std::make_shared<odometry::EKFConstantVelocitySE3>(Matd<12, 12>::Identity(), 0);

    for (auto name : {"odometry"}) {
      el::Loggers::getLogger(name);
      el::Configurations defaultConf;
      defaultConf.setToDefault();
      defaultConf.set(el::Level::Debug, el::ConfigurationType::Format, "%datetime %level %msg");
      defaultConf.set(el::Level::Debug, el::ConfigurationType::Enabled, "false");
      defaultConf.set(el::Level::Info, el::ConfigurationType::Enabled, "false");
      el::Loggers::reconfigureLogger(name, defaultConf);
    }
    LOG_PLT("Test")->set(TEST_VISUALIZE, false);
  }

protected:
  odometry::EKFConstantVelocitySE3::ShPtr _kalman;
  std::map<std::string, Trajectory::ShPtr> _trajectories;
  std::map<std::string, RelativePoseError::ConstShPtr> _rpes;
};

TEST_F(TestKalmanSE3, DISABLED_RunWithGt)
{
  Trajectory::ConstShPtr trajectoryGt =
    utils::loadTrajectory(TEST_RESOURCE "/rgbd_dataset_freiburg1_floor-groundtruth.txt", true);

  std::uint16_t fNo = 0;
  _kalman->pose() = trajectoryGt->poseAt(trajectoryGt->tStart())->twist();
  _kalman->covProcess().block(0, 0, 6, 6) = Matd<6, 6>::Identity() * 1e-9;
  _kalman->covProcess().block(6, 6, 6, 6) = Matd<6, 6>::Identity() * 1e3;
  //_kalman->covProcess().setZero();
  _kalman->t() = trajectoryGt->tStart();
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
      trajectory->append(t, std::make_shared<Pose>(pred->pose, pred->covPose));
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

  LOG_PLT("Test") << std::make_shared<evaluation::PlotTrajectory>(trajectories);
  LOG_PLT("Test") << std::make_shared<evaluation::PlotTrajectoryCovariance>(trajectories);
  LOG_PLT("Test") << std::make_shared<evaluation::PlotRPE>(rpes);
  LOG_PLT("Test") << _kalman->plot();
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

  std::uint16_t fNo = 0;
  _kalman->pose() = trajectoryAlgo->poseAt(trajectoryAlgo->tStart())->twist();
  _kalman->covProcess().block(0, 0, 6, 6) = Matd<6, 6>::Identity() * 1e-9;
  _kalman->covProcess().block(6, 6, 3, 3) = Matd<3, 3>::Identity() * 1e-15;
  _kalman->covProcess().block(9, 9, 3, 3) = Matd<3, 3>::Identity() * 1e-15;

  //_kalman->covProcess().setZero();
  _kalman->t() = trajectoryAlgo->tStart();
  SE3d poseRef = trajectoryAlgo->poseAt(trajectoryAlgo->tStart())->SE3();
  _trajectories["LastMotion"] = std::make_shared<Trajectory>();
  _trajectories["Kalman"] = std::make_shared<Trajectory>();
  _trajectories["GroundTruth"] = std::make_shared<Trajectory>();
  _trajectories["Algorithm"] = std::make_shared<Trajectory>();
  SE3d poseLastMotion = poseRef;
  Vec6d lastVel = Vec6d::Zero();
  Timestamp t_1 = trajectoryGt->tStart();
  std::vector<double> dts;
  std::vector<double> timestamps;
  const std::uint16_t nFrames = 5000;
  for (auto t_p : trajectoryAlgo->poses()) {
    try {
      const auto t = t_p.first;
      const double dT = t - t_1;
      const auto pose = t_p.second->SE3();
      poseLastMotion = poseLastMotion * SE3d::exp(lastVel * dT);
      auto pred = _kalman->predict(t);
      auto motion = (poseRef.inverse() * pose).log();

      _kalman->update(motion, Matd<6, 6>::Identity(), t);
      _trajectories["Kalman"]->append(t, std::make_shared<Pose>(pred->pose, pred->covPose));
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

  LOG_PLT("Test") << std::make_shared<evaluation::PlotTrajectory>(_trajectories);
  LOG_PLT("Test") << std::make_shared<evaluation::PlotTrajectoryCovariance>(_trajectories);
  LOG_PLT("Test") << std::make_shared<evaluation::PlotTrajectoryMotion>(_trajectories);
  LOG_PLT("Test") << std::make_shared<evaluation::PlotRPE>(_rpes);
  LOG_PLT("Test") << _kalman->plot();
  vis::plt::figure();
  vis::plt::title("dts");

  vis::plt::plot(timestamps, dts);
  vis::plt::show();
}

TEST(EKFSE3, DISABLED_SyntheticData)
{
  auto kalman = std::make_shared<odometry::EKFConstantVelocitySE3>(Matd<12, 12>::Identity(), 0UL);

  for (auto name : {"odometry"}) {
    el::Loggers::getLogger(name);
    el::Configurations defaultConf;
    defaultConf.setToDefault();
    defaultConf.set(el::Level::Debug, el::ConfigurationType::Format, "%datetime %level %msg");
    defaultConf.set(el::Level::Debug, el::ConfigurationType::Enabled, "true");
    defaultConf.set(el::Level::Info, el::ConfigurationType::Enabled, "false");
    el::Loggers::reconfigureLogger(name, defaultConf);
  }
  std::vector<double> timestamps;
  Trajectory::ShPtr trajectory = std::make_shared<Trajectory>();
  Trajectory::ShPtr trajectoryGt = std::make_shared<Trajectory>();
  const Timestamp dT = 1;
  std::uint16_t fNo = 0;
  SE3d motionGtSE3;
  motionGtSE3.translation().x() = 0.1;
  Vec6d motionGt = motionGtSE3.log();
  SE3d poseGt;
  for (Timestamp t = 0UL; t < 100; t += dT) {
    try {
      auto pred = kalman->predict(t);

      EXPECT_NEAR(
        (SE3d::exp(motionGt).inverse() * SE3d::exp(pred->velocity * dT)).log().norm(), 0.0, 0.01)
        << "With high state uncertainty kalman should always use the last measured velocity.";

      poseGt = poseGt * SE3d::exp(motionGt);

      EXPECT_NEAR((poseGt.inverse() * SE3d::exp(pred->pose)).log().norm(), 0.0, 0.01)
        << "The predicted pose should be the last pose updated with the current velocity * dt.\n"
        << "Gt: " << poseGt.log().transpose() << "\n"
        << "Pred: " << pred->pose.transpose() << "\n";

      kalman->update(poseGt.log(), Matd<6, 6>::Identity(), t);
      trajectoryGt->append(t, std::make_shared<Pose>(poseGt, Matd<6, 6>::Identity()));
      trajectory->append(t, std::make_shared<Pose>(pred->pose, pred->covPose));
      timestamps.push_back((t) / 1e9);
    } catch (const pd::Exception & e) {
      print("{}\n", e.what());
    }
    fNo++;
  }
  std::map<std::string, Trajectory::ConstShPtr> trajectories;
  trajectories["GroundTruth"] = trajectoryGt;
  trajectories["Kalman"] = trajectory;
  LOG_PLT("Test")->set(TEST_VISUALIZE, false);
  LOG_PLT("Test") << std::make_shared<evaluation::PlotTrajectory>(trajectories);
  LOG_PLT("Test") << std::make_shared<evaluation::PlotTrajectoryCovariance>(trajectories);
  vis::plt::show();
}