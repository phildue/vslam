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
class PlotTrajectory : public vis::Plot
{
public:
  PlotTrajectory(const std::map<std::string, Trajectory::ConstShPtr> & trajectories)
  : _trajectories(trajectories)
  {
  }
  PlotTrajectory(const std::map<std::string, Trajectory::ShPtr> & trajectories)
  {
    for (auto n_t : trajectories) {
      _trajectories[n_t.first] = n_t.second;
    }
  }
  void plot() const override
  {
    vis::plt::figure();
    for (auto traj : _trajectories) {
      std::vector<double> x, y;
      for (auto p : traj.second->poses()) {
        x.push_back(p.second->SE3().translation().x());
        y.push_back(p.second->SE3().translation().y());
      }
      vis::plt::named_plot(traj.first, x, y);
    }
    vis::plt::axis("equal");

    vis::plt::legend();
  }
  std::string csv() const override { return ""; }

private:
  std::map<std::string, Trajectory::ConstShPtr> _trajectories;
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
  auto it = trajectoryAlgo->poses().begin();
  const Timestamp t0 = it->first;
  auto itPrev = it;
  ++it;
  Matd<12, 12> covProcess = Matd<12, 12>::Identity();
  covProcess.block(6, 6, 6, 6) = meanAcceleration->cov();
  auto kalman = std::make_shared<MotionModelConstantSpeedKalman>(covProcess);
  auto constantSpeed = std::make_shared<MotionModelConstantSpeed>();
  auto noMotion = std::make_shared<MotionModelNoMotion>();
  auto movingAverage = std::make_shared<MotionModelMovingAverage>(3 * 1e8);

  std::map<std::string, MotionModel::ShPtr> motionModels = {
    {"NoMotion", noMotion},
    //{"ConstantSpeed", constantSpeed},
    {"Kalman", kalman},
    //{"MovingAverage", movingAverage}
  };
  std::vector<std::string> names2 = {"GroundTruth",   "Algorithm", "NoMotion",
                                     "ConstantSpeed", "Kalman",    "MovingAverage"};

  for (auto name : {"odometry"}) {
    el::Loggers::getLogger(name);
    el::Configurations defaultConf;
    defaultConf.setToDefault();
    defaultConf.set(el::Level::Debug, el::ConfigurationType::Format, "%datetime %level %msg");
    defaultConf.set(el::Level::Debug, el::ConfigurationType::Enabled, "false");
    defaultConf.set(el::Level::Info, el::ConfigurationType::Enabled, "false");
    el::Loggers::reconfigureLogger(name, defaultConf);
  }
  std::vector<std::vector<Vec6d>> egomotions(motionModels.size() + 2);

  std::vector<double> timestamps;
  SE3d poseGtPrev;
  SE3d poseAlgoPrev;
  std::map<std::string, Trajectory::ShPtr> trajectories;
  for (const auto n_m : motionModels) {
    trajectories[n_m.first] = std::make_shared<Trajectory>();
    n_m.second->update(trajectoryAlgo->poseAt(t0), t0);
  }
  std::uint16_t fNo = 0;
  for (; it != trajectoryAlgo->poses().end(); ++it) {
    const Timestamp t = it->first;
    try {
      const auto poseGt = trajectoryGt->poseAt(t);
      const auto poseAlgo = it->second;
      auto motionGt = algorithm::computeRelativeTransform(poseGtPrev, poseGt->pose());
      auto motionAlgo = algorithm::computeRelativeTransform(poseAlgoPrev, poseAlgo->pose());
      egomotions[0].push_back(motionGt.log());
      egomotions[1].push_back(motionAlgo.log());
      int i = 0;
      for (const auto & n_m : motionModels) {
        const auto name = n_m.first;
        const auto model = n_m.second;
        Pose::ConstShPtr predictedPose = model->predictPose(t);
        trajectories[n_m.first]->append(t, predictedPose);
        auto motionPred = algorithm::computeRelativeTransform(poseAlgoPrev, predictedPose->SE3());
        egomotions[i + 2].push_back(motionPred.log());
        model->update(poseAlgo, t);
        i++;
      }
      poseGtPrev = poseGt->pose();
      poseAlgoPrev = poseAlgo->pose();
      timestamps.push_back((t - t0) / 1e9);
    } catch (const pd::Exception & e) {
      print("{}\n", e.what());
    }
    fNo++;
    ++itPrev;
  }

  std::map<std::string, evaluation::RelativePoseError::ConstShPtr> rpes;
  for (const auto & n_m : trajectories) {
    auto rpe = evaluation::RelativePoseError::compute(n_m.second, trajectoryGt, 0.05);
    std::cout << n_m.first << "\n" << rpe->toString() << std::endl;
    rpes[n_m.first] = std::move(rpe);
  }
  evaluation::PlotRPE(rpes).plot();
  PlotTrajectory({
                   {"GroundTruth", trajectoryGt},
                   {"Algorithm", trajectoryAlgo},
                   {"NoMotion", trajectories["NoMotion"]},
                   {"Kalman", trajectories["Kalman"]},
                 })
    .plot();
  //auto plot = std::make_shared<PlotEgomotion>(egomotions, timestamps, names2);
  // plot->plot();
  //auto plotTranslation = std::make_shared<PlotEgomotionTranslation>(egomotions, timestamps, names2);
  //plotTranslation->plot();

  vis::plt::show();
}