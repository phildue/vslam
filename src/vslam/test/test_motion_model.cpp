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
#include <gtest/gtest.h>

#include "core/core.h"
#include "evaluation/evaluation.h"
#include "odometry/odometry.h"
#include "utils/utils.h"
using namespace testing;
using namespace pd;
using namespace pd::vslam;
using namespace pd::vslam::evaluation;

class PlotEgomotion : public vis::Plot
{
public:
  PlotEgomotion(
    const std::vector<std::vector<Vec6d>> & egomotion, const std::vector<double> & ts,
    const std::vector<std::string> & names)
  : _twists(egomotion), _names(names), _ts(ts)
  {
  }

  void plot() const override
  {
    vis::plt::figure();
    vis::plt::subplot(2, 1, 1);
    vis::plt::title("Translational Velocity");
    vis::plt::ylabel("$m$");
    vis::plt::xlabel("$t-t_0 [s]$");
    vis::plt::ylim(0.0, 0.1);
    for (size_t i = 0; i < _names.size(); i++) {
      std::vector<double> v(_ts.size());
      std::transform(
        _twists[i].begin(), _twists[i].end(), v.begin(), [](auto tw) { return tw.head(3).norm(); });

      vis::plt::named_plot(_names[i], _ts, v, ".--");
    }
    vis::plt::legend();

    vis::plt::subplot(2, 1, 2);
    vis::plt::title("Angular Velocity");
    vis::plt::ylabel("$\\circ$");
    vis::plt::xlabel("$t-t_0 [s]$");
    vis::plt::ylim(0.0, 5.0);

    //vis::plt::xticks(_ts);
    for (size_t i = 0; i < _names.size(); i++) {
      std::vector<double> va(_ts.size());
      std::transform(_twists[i].begin(), _twists[i].end(), va.begin(), [](auto tw) {
        return tw.tail(3).norm() / M_PI * 180.0;
      });

      vis::plt::named_plot(_names[i], _ts, va, ".--");
    }
    vis::plt::legend();
  }
  std::string csv() const override { return ""; }

private:
  std::vector<std::vector<Vec6d>> _twists;
  std::vector<std::string> _names;
  std::vector<double> _ts;
};
class PlotEgomotionTranslation : public vis::Plot
{
public:
  PlotEgomotionTranslation(
    const std::vector<std::vector<Vec6d>> & egomotion, const std::vector<double> & ts,
    const std::vector<std::string> & names)
  : _egomotion(egomotion), _names(names), _ts(ts)
  {
  }

  void plot() const override
  {
    vis::plt::figure();
    const std::vector<std::string> dims = {"x", "y", "z"};
    for (size_t i = 0U; i < dims.size(); i++) {
      vis::plt::subplot(dims.size(), 1, i + 1);
      vis::plt::title(format("$\\Delta t_{}$", dims[i]));
      vis::plt::ylabel("$m$");
      vis::plt::xlabel("$t-t_0 [s]$");
      vis::plt::ylim(-0.25, 0.25);
      for (size_t j = 0; j < _names.size(); j++) {
        std::vector<double> v(_ts.size());
        std::transform(
          _egomotion[j].begin(), _egomotion[j].end(), v.begin(), [&](auto tw) { return tw(i); });

        vis::plt::named_plot(_names[j], _ts, v, ".--");
      }
    }

    vis::plt::legend();
  }
  std::string csv() const override { return ""; }

private:
  std::vector<std::vector<Vec6d>> _egomotion;
  std::vector<std::string> _names;
  std::vector<double> _ts;
};

TEST(MotionModel, Compare)
{
  Trajectory::ConstShPtr trajectoryGt =
    utils::loadTrajectory(TEST_RESOURCE "/rgbd_dataset_freiburg2_desk-groundtruth.txt", true);
  Trajectory::ConstShPtr trajectoryAlgo =
    utils::loadTrajectory(TEST_RESOURCE "/rgbd_dataset_freiburg2_desk-algo.txt", true);

  using time::to_time_point;

  print(
    "GT time range [{:%Y-%m-%d %H:%M:%S}] -> [{:%Y-%m-%d %H:%M:%S}]\n",
    to_time_point(trajectoryGt->tStart()), to_time_point(trajectoryGt->tEnd()));
  auto meanAcceleration = trajectoryGt->meanAcceleration(0.1 * 1e9);
  print(
    "Acceleration Statistics\ndT = {:.3f} [f/s] Mean = {}\n Cov = {}\n", 0.1,
    meanAcceleration->mean().transpose(), meanAcceleration->cov());
  Matd<12, 12> covProcess = Matd<12, 12>::Identity();
  //covProcess.block(6, 6, 6, 6) = Matd<6, 6>::Identity() * 1e10;
  covProcess.block(0, 0, 6, 6) = Matd<6, 6>::Identity() * 0.001;
  covProcess.block(6, 6, 6, 6) = Matd<6, 6>::Identity() * 1e3;
  auto kalman =
    std::make_shared<MotionModelConstantSpeedKalman>(covProcess, Matd<12, 12>::Identity() * 200);
  auto constantSpeed = std::make_shared<MotionModelConstantSpeed>();
  auto noMotion = std::make_shared<MotionModelNoMotion>();
  auto movingAverage = std::make_shared<MotionModelMovingAverage>(3 * 1e8);

  std::map<std::string, MotionModel::ShPtr> motionModels = {
    {"NoMotion", noMotion}, {"ConstantSpeed", constantSpeed}, {"Kalman", kalman},
    //{"MovingAverage", movingAverage}
  };
  for (auto name : {"odometry"}) {
    el::Loggers::getLogger(name);
    el::Configurations defaultConf;
    defaultConf.setToDefault();
    defaultConf.set(el::Level::Debug, el::ConfigurationType::Format, "%datetime %level %msg");
    defaultConf.set(el::Level::Debug, el::ConfigurationType::Enabled, "false");
    defaultConf.set(el::Level::Info, el::ConfigurationType::Enabled, "false");
    el::Loggers::reconfigureLogger(name, defaultConf);
  }

  const Timestamp t0 = trajectoryAlgo->tStart();
  std::vector<double> timestamps;
  std::map<std::string, Trajectory::ShPtr> trajectories;
  for (const auto n_m : motionModels) {
    trajectories[n_m.first] = std::make_shared<Trajectory>();
    n_m.second->update(*trajectoryAlgo->poseAt(t0), t0);
  }
  std::uint16_t fNo = 0;
  for (auto t_p : trajectoryAlgo->poses()) {
    try {
      const Timestamp t = t_p.first;
      const auto poseGt = trajectoryGt->poseAt(t);
      const auto poseAlgo = t_p.second;
      for (const auto & n_m : motionModels) {
        const auto model = n_m.second;
        trajectories[n_m.first]->append(t, model->predictPose(t));
        model->update(*poseAlgo, t);
      }
      timestamps.push_back((t - t0) / 1e9);
    } catch (const pd::Exception & e) {
      print("{}\n", e.what());
    }
    fNo++;
  }

  std::map<std::string, evaluation::RelativePoseError::ConstShPtr> rpes;
  for (const auto & n_m : trajectories) {
    auto rpe = evaluation::RelativePoseError::compute(n_m.second, trajectoryGt, 0.05);
    std::cout << n_m.first << "\n" << rpe->toString() << std::endl;
    rpes[n_m.first] = std::move(rpe);
  }
  std::map<std::string, Trajectory::ConstShPtr> trajsPlot;
  trajsPlot["GroundTruth"] = trajectoryGt;
  trajsPlot["Kalman"] = trajectories["Kalman"];
  trajsPlot["ConstantSpeed"] = trajectories["ConstantSpeed"];
  LOG_IMG("TEST")->set(TEST_VISUALIZE, TEST_VISUALIZE);
  LOG_IMG("TEST") << std::make_shared<evaluation::PlotTrajectory>(trajsPlot);
  LOG_IMG("TEST") << std::make_shared<evaluation::PlotRPE>(rpes);
  //auto plot = std::make_shared<PlotEgomotion>(egomotions, timestamps, names2);
  // plot->plot();
  //auto plotTranslation = std::make_shared<PlotEgomotionTranslation>(egomotions, timestamps, names2);
  //plotTranslation->plot();

  vis::plt::show();
}