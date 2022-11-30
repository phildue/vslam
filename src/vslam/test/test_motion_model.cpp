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
class PlotRMSE : public vis::Plot
{
public:
  PlotRMSE(const std::vector<std::string> & names) : _names(names), _errors(names.size()) {}
  void plot() const override
  {
    vis::plt::figure();
    vis::plt::subplot(2, 1, 1);
    vis::plt::title("$Translational Error$");
    vis::plt::ylabel("$|t|_2 [m]$");
    vis::plt::xlabel("$t-t_0 [s]$");
    vis::plt::grid(true);

    vis::plt::ylim(0.0, 0.1);
    for (size_t i = 0; i < _names.size(); i++) {
      std::vector<double> c(_ts.size());
      std::transform(_errors[i].begin(), _errors[i].end(), c.begin(), [](auto c) {
        return c.translation().norm();
      });
      Eigen::Map<VecXd> rmseT(c.data(), c.size());
      std::cout << _names[i] << "\n |Translation|"
                << "\n |RMSE: " << std::sqrt(rmseT.dot(rmseT) / c.size())
                << "\n |Max: " << rmseT.maxCoeff() << "\n |Mean: " << rmseT.mean()
                << "\n |Min: " << rmseT.minCoeff() << std::endl;

      vis::plt::named_plot(_names[i], _ts, c, ".--");
    }
    vis::plt::legend();
    vis::plt::subplot(2, 1, 2);
    vis::plt::title("$Rotational Error$");
    vis::plt::ylabel("$|\\theta|_2   [°]$");
    vis::plt::xlabel("$t-t_0 [s]$");
    vis::plt::grid(true);
    vis::plt::ylim(0, 1);
    for (size_t i = 0; i < _names.size(); i++) {
      std::vector<double> c(_ts.size());
      std::transform(_errors[i].begin(), _errors[i].end(), c.begin(), [](auto c) {
        return c.log().tail(3).norm() / M_PI * 180.0;
      });
      Eigen::Map<VecXd> rmse(c.data(), c.size());
      std::cout << _names[i] << "\n |Rotation|"
                << "\n |RMSE: " << std::sqrt(rmse.dot(rmse) / c.size())
                << "\n |Max: " << rmse.maxCoeff() << "\n |Mean: " << rmse.mean()
                << "\n |Min: " << rmse.minCoeff() << std::endl;
      vis::plt::named_plot(_names[i], _ts, c, ".--");
    }
    vis::plt::legend();
  }
  void extend(double t, const std::vector<SE3d> & errors)
  {
    _ts.push_back(t);
    for (size_t i = 0; i < _names.size(); i++) {
      _errors[i].push_back(errors[i]);
    }
  }
  std::string csv() const override { return ""; }

private:
  const std::vector<std::string> _names;
  std::vector<std::vector<SE3d>> _errors;
  std::vector<double> _ts;
};

TEST(MotionModel, Compare)
{
  auto trajectoryGt = std::make_shared<Trajectory>(
    utils::loadTrajectory(TEST_RESOURCE "/rgbd_dataset_freiburg2_desk-groundtruth.txt"));
  auto trajectoryAlgo = std::make_shared<Trajectory>(
    utils::loadTrajectory(TEST_RESOURCE "/rgbd_dataset_freiburg2_desk-algo.txt"));

  using time::to_time_point;

  print(
    "GT time range [{:%Y-%m-%d %H:%M:%S}] -> [{:%Y-%m-%d %H:%M:%S}]\n",
    to_time_point(trajectoryGt->tStart()), to_time_point(trajectoryGt->tEnd()));
  auto meanAcceleration = trajectoryGt->meanAcceleration(0.1 * 1e9);
  print(
    "Acceleration Statistics\ndT = {:.3f} [f/s] Mean = {}\n Cov = {}", 0.1,
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

  std::vector<MotionModel::ShPtr> motionModels = {noMotion, constantSpeed, kalman};
  //auto movingAverage = std::make_shared<MotionModelMovingAverage>();
  std::vector<std::string> names = {"NoMotion", "ConstantSpeed", "Kalman"};
  std::vector<std::string> names2 = {"GroundTruth", "NoMotion", "ConstantSpeed", "Kalman"};

  for (auto name : {"odometry"}) {
    el::Loggers::getLogger(name);
    el::Configurations defaultConf;
    defaultConf.setToDefault();
    defaultConf.set(el::Level::Debug, el::ConfigurationType::Format, "%datetime %level %msg");
    defaultConf.set(el::Level::Debug, el::ConfigurationType::Enabled, "false");
    defaultConf.set(el::Level::Info, el::ConfigurationType::Enabled, "false");
    el::Loggers::reconfigureLogger(name, defaultConf);
  }
  std::vector<std::vector<Vec6d>> egomotions(motionModels.size() + 1);
  auto plotRmse = std::make_shared<PlotRMSE>(names);

  std::vector<double> timestamps;
  SE3d poseGtPrev;
  for (; it != trajectoryAlgo->poses().end(); ++it) {
    const Timestamp t = it->first;
    const Timestamp tp = itPrev->first;
    try {
      const auto poseGt = trajectoryGt->poseAt(t);
      //auto motionGt = trajectoryGt->motionBetween(tp, t)->pose(); std::vector<std::string>({"GroundTruth", "ConstantSpeed"})
      auto motionGt = algorithm::computeRelativeTransform(poseGtPrev, poseGt->pose());
      egomotions[0].push_back(motionGt.log());
      std::vector<SE3d> errors(motionModels.size());
      for (size_t i = 0U; i < motionModels.size(); i++) {
        auto predictedPose = motionModels[i]->predictPose(t);
        auto motionPred = algorithm::computeRelativeTransform(poseGtPrev, predictedPose->pose());
        auto error = algorithm::computeRelativeTransform(motionGt, motionPred);
        egomotions[i + 1].push_back(motionPred.log());
        errors[i] = error;
        motionModels[i]->update(poseGt, t);
      }
      plotRmse->extend((t - t0) / 1e9, errors);
      poseGtPrev = poseGt->pose();
      timestamps.push_back((t - t0) / 1e9);
    } catch (const pd::Exception & e) {
      print("{}\n", e.what());
    }

    ++itPrev;
  }
  auto plot = std::make_shared<PlotEgomotion>(egomotions, timestamps, names2);
  plot->plot();
  auto plotTranslation = std::make_shared<PlotEgomotionTranslation>(egomotions, timestamps, names2);
  plotTranslation->plot();
  plotRmse->plot();
  vis::plt::show();
}