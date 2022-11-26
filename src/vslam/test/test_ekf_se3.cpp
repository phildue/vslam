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

#include <fmt/core.h>
#include <gtest/gtest.h>

#include "core/core.h"
#include "odometry/odometry.h"
#include "utils/utils.h"
using namespace testing;
using namespace pd;
using namespace pd::vslam;

class PlotSE3Velocity : public vis::Plot
{
public:
  PlotSE3Velocity(
    const std::vector<std::vector<Vec6d>> & twists, const std::vector<Timestamp> & ts,
    const std::vector<std::string> & names)
  : _twists(twists), _names(names), _ts(ts)
  {
  }

  void plot() const override
  {
    vis::plt::figure();
    vis::plt::subplot(1, 2, 1);
    vis::plt::title("Translational Velocity");
    vis::plt::ylabel("$\\frac{m}{s}$");
    vis::plt::xlabel("$t-t_0 [s]$");
    for (size_t i = 0; i < _twists.size(); i++) {
      std::vector<double> v(_ts.size());
      std::transform(
        _twists[i].begin(), _twists[i].end(), v.begin(), [](auto tw) { return tw.head(3).norm(); });

      vis::plt::named_plot(_names[i], _ts, v);
    }
    vis::plt::legend();

    vis::plt::subplot(1, 2, 2);
    vis::plt::title("Angular Velocity");
    vis::plt::ylabel("$\\frac{\\circ}{s}$");
    vis::plt::xlabel("$t-t_0 [s]$");
    //vis::plt::xticks(_ts);
    for (size_t i = 0; i < _twists.size(); i++) {
      std::vector<double> va(_ts.size());
      std::transform(_twists[i].begin(), _twists[i].end(), va.begin(), [](auto tw) {
        return tw.tail(3).norm() / M_PI * 180.0;
      });

      vis::plt::named_plot(_names[i], _ts, va);
    }
    vis::plt::legend();
  }
  std::string csv() const override { return ""; }

private:
  std::vector<std::vector<Vec6d>> _twists;
  std::vector<std::string> _names;
  std::vector<Timestamp> _ts;
};

TEST(EKFSE3, RunWithGt)
{
  auto trajectoryGt = std::make_shared<Trajectory>(
    utils::loadTrajectory(TEST_RESOURCE "/rgbd_dataset_freiburg2_desk-groundtruth.txt"));

  auto trajectoryAlgo = std::make_shared<Trajectory>(
    utils::loadTrajectory(TEST_RESOURCE "/rgbd_dataset_freiburg2_desk-algo.txt"));

  auto kalman = std::make_shared<odometry::EKFConstantVelocitySE3>(Matd<12, 12>::Identity());

  Timestamp tLast = 0UL;
  SE3d poseRef;
  std::vector<Vec6d> twistsAlgo, twistsKalman, twistsGt;
  twistsAlgo.reserve(trajectoryAlgo->poses().size());
  twistsKalman.reserve(trajectoryAlgo->poses().size());
  twistsGt.reserve(trajectoryAlgo->poses().size());
  std::vector<Timestamp> timestamps;
  Timestamp t0 = 0UL;
  for (const auto & t_pose : trajectoryAlgo->poses()) {
    if (t0 == 0UL) {
      poseRef = t_pose.second->pose();
      tLast = t_pose.first;
      t0 = t_pose.first;
      timestamps.push_back(0UL);
      continue;
    }
    auto dt = (t_pose.first - tLast) / 1e9;
    try {
      auto dxGt = trajectoryGt->motionBetween(tLast, t_pose.first)->pose().log();
      tLast = t_pose.first;

      twistsGt.push_back(dxGt / dt);

      auto dxAlgo = algorithm::computeRelativeTransform(poseRef, t_pose.second->pose()).log();
      twistsAlgo.push_back(dxAlgo / dt);

      kalman->update(dxGt, MatXd::Identity(6, 6), t_pose.first);
      twistsKalman.push_back(kalman->predict(0)->velocity);

      timestamps.push_back((t_pose.first - t0) / 1e9);
    } catch (const pd::Exception & e) {
      std::cerr << e.what() << std::endl;
    }
  }
  std::cout << fmt::format("Computed speed for {} timestamps.", timestamps.size());
  auto plot = std::make_shared<PlotSE3Velocity>(
    std::vector<std::vector<Vec6d>>({twistsGt, twistsAlgo}), timestamps,
    std::vector<std::string>({"Visual", "GroundTruth"}));
  plot->plot();
  vis::plt::show();
}