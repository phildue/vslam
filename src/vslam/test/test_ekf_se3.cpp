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
    vis::plt::subplot(1, 2, 1);
    vis::plt::title("Translational Velocity");
    vis::plt::ylabel("$m$");
    vis::plt::xlabel("$t-t_0 [s]$");
    //vis::plt::ylim(0.0, 5.0);
    for (size_t i = 0; i < _twists.size(); i++) {
      std::vector<double> v(_ts.size());
      std::transform(
        _twists[i].begin(), _twists[i].end(), v.begin(), [](auto tw) { return tw.head(3).norm(); });

      vis::plt::named_plot(_names[i], _ts, v, ".--");
    }
    vis::plt::legend();

    vis::plt::subplot(1, 2, 2);
    vis::plt::title("Angular Velocity");
    vis::plt::ylabel("$\\circ$");
    vis::plt::xlabel("$t-t_0 [s]$");
    //vis::plt::ylim(0.0, 5.0);

    //vis::plt::xticks(_ts);
    for (size_t i = 0; i < _twists.size(); i++) {
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
      vis::plt::subplot(1, dims.size(), i + 1);
      vis::plt::title(format("$\\Delta t_{}$", dims[i]));
      vis::plt::ylabel("$m$");
      vis::plt::xlabel("$t-t_0 [s]$");
      vis::plt::ylim(-0.25, 0.25);
      for (size_t j = 0; j < _egomotion.size(); j++) {
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
class PlotCovariances : public vis::Plot
{
public:
  PlotCovariances(
    const std::vector<std::vector<MatXd>> & covariances, const std::vector<double> & ts,
    const std::vector<std::string> & names)
  : _covariances(covariances), _names(names), _ts(ts)
  {
  }

  void plot() const override
  {
    vis::plt::figure();
    vis::plt::subplot(1, 2, 1);
    vis::plt::title("Covariances");
    vis::plt::ylabel("$|\\Sigma|$");
    vis::plt::xlabel("$t-t_0 [s]$");
    for (size_t i = 0; i < _covariances.size(); i++) {
      std::vector<double> c(_ts.size());
      std::transform(_covariances[i].begin(), _covariances[i].end(), c.begin(), [](auto c) {
        return c.determinant();
      });

      vis::plt::named_plot(_names[i], _ts, c, ".--");
    }
    vis::plt::legend();
  }
  std::string csv() const override { return ""; }

private:
  std::vector<std::vector<MatXd>> _covariances;
  std::vector<std::string> _names;
  std::vector<double> _ts;
};

TEST(EKFSE3, RunWithGt)
{
  auto trajectoryGt =
    utils::loadTrajectory(TEST_RESOURCE "/rgbd_dataset_freiburg2_desk-groundtruth.txt");
  auto trajectoryAlgo =
    utils::loadTrajectory(TEST_RESOURCE "/rgbd_dataset_freiburg2_desk-algo.txt");

  using time::to_time_point;

  print(
    "GT time range [{:%Y-%m-%d %H:%M:%S}] -> [{:%Y-%m-%d %H:%M:%S}]\n",
    to_time_point(trajectoryGt->tStart()), to_time_point(trajectoryGt->tEnd()));
  auto meanAcceleration = trajectoryGt->meanAcceleration(0.1 * 1e9);
  print(
    "Acceleration Statistics\ndT = {:.3f} [f/s] Mean = {}\n Cov = {}", 0.1,
    meanAcceleration->mean().transpose(), meanAcceleration->cov());

  std::vector<Vec6d> twistsKalman, twistsGt, twistsAlgo;
  twistsKalman.reserve(trajectoryGt->poses().size());
  twistsGt.reserve(trajectoryGt->poses().size());
  std::vector<double> timestamps;
  timestamps.reserve(trajectoryGt->poses().size());
  std::vector<MatXd> covsMeasurement, covsState, covsProcess;

  auto it = trajectoryAlgo->poses().begin();
  const Timestamp t0 = it->first;
  auto itPrev = it;
  ++it;
  Matd<12, 12> covProcess = Matd<12, 12>::Identity();
  covProcess.block(6, 6, 6, 6) = meanAcceleration->cov();
  auto kalman =
    std::make_shared<odometry::EKFConstantVelocitySE3>(covProcess, t0, Matd<12, 12>::Identity());
  for (auto name : {"odometry"}) {
    el::Loggers::getLogger(name);
    el::Configurations defaultConf;
    defaultConf.setToDefault();
    defaultConf.set(el::Level::Debug, el::ConfigurationType::Format, "%datetime %level %msg");
    defaultConf.set(el::Level::Debug, el::ConfigurationType::Enabled, "false");
    defaultConf.set(el::Level::Info, el::ConfigurationType::Enabled, "false");
    el::Loggers::reconfigureLogger(name, defaultConf);
  }
  int i = 0;
  for (; it != trajectoryAlgo->poses().end(); ++it) {
    const Timestamp t = it->first;
    const auto p = it->second;
    const Timestamp tp = itPrev->first;
    const auto pp = itPrev->second;
    const double dt = static_cast<double>(t - tp) / 1e9;
    try {
      auto dxAlgo = algorithm::computeRelativeTransform(pp->pose(), p->pose()).log();
      auto dxGt = trajectoryGt->motionBetween(tp, t)->pose().log();
      auto covMeasurement = MatXd::Identity(6, 6);
      kalman->update(dxAlgo, covMeasurement, t);
      // print("twist_gt = {}\n", (dxGt / dt).transpose());
      twistsKalman.push_back(kalman->velocity() * 1e9 * dt);
      twistsGt.push_back(dxGt);
      twistsAlgo.push_back(dxAlgo);

      covsMeasurement.push_back(covMeasurement);
      covsProcess.push_back(kalman->covProcess());
      covsState.push_back(kalman->covState());
      timestamps.push_back(static_cast<double>(t - t0) / 1e9);
    } catch (const pd::Exception & e) {
      print("{}", e.what());
    }

    ++itPrev;
    if (i++ > 100) {
      break;
    }
  }
  print("Computed twist for {} timestamps.\n", timestamps.size());
  auto plot = std::make_shared<PlotEgomotion>(
    std::vector<std::vector<Vec6d>>({twistsGt, twistsKalman, twistsAlgo}), timestamps,
    std::vector<std::string>({"GroundTruth", "Kalman", "Algo"}));
  plot->plot();
  auto plotTranslation = std::make_shared<PlotEgomotionTranslation>(
    std::vector<std::vector<Vec6d>>({twistsGt, twistsKalman, twistsAlgo}), timestamps,
    std::vector<std::string>({"GroundTruth", "Kalman", "Algo"}));
  plotTranslation->plot();
  auto plotCov = std::make_shared<PlotCovariances>(
    std::vector<std::vector<MatXd>>({covsState}), timestamps, std::vector<std::string>({"State"}));
  plotCov->plot();
  vis::plt::show();
}