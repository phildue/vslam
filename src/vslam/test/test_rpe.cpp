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

TEST(MotionModel, Compare)
{
  Trajectory::ConstShPtr trajectoryGt =
    utils::loadTrajectory(TEST_RESOURCE "/rgbd_dataset_freiburg2_desk-groundtruth.txt", true);
  Trajectory::ConstShPtr trajectoryAlgo =
    utils::loadTrajectory(TEST_RESOURCE "/rgbd_dataset_freiburg2_desk-algo.txt", true);
  using time::to_time_point;

  print(
    "GT time range [{:%Y-%m-%d %H:%M:%S}] -> [{:%Y-%m-%d %H:%M:%S}], {} poses\n",
    to_time_point(trajectoryGt->tStart()), to_time_point(trajectoryGt->tEnd()),
    trajectoryGt->poses().size());
  print(
    "Algo time range [{:%Y-%m-%d %H:%M:%S}] -> [{:%Y-%m-%d %H:%M:%S}], {} poses\n",
    to_time_point(trajectoryAlgo->tStart()), to_time_point(trajectoryAlgo->tEnd()),
    trajectoryAlgo->poses().size());
  RelativePoseError::ConstShPtr rpe = RelativePoseError::compute(trajectoryAlgo, trajectoryGt, 1.0);

  //values from:
  //python3 script/tum/evaluate_rpe.py src/vslam/test/resource/rgbd_dataset_freiburg2_desk-groundtruth.txt src/vslam/test/resource/rgbd_dataset_freiburg2_desk-algo.txt --fixed_delta --verbose

  EXPECT_NEAR(rpe->translation().rmse, 0.036, 0.001);
  EXPECT_NEAR(rpe->translation().mean, 0.021, 0.001);
  EXPECT_NEAR(rpe->translation().median, 0.012, 0.001);
  EXPECT_NEAR(rpe->translation().stddev, 0.030, 0.001);
  EXPECT_NEAR(rpe->translation().min, 0.000, 0.001);
  EXPECT_NEAR(rpe->translation().max, 0.292, 0.001);

  EXPECT_NEAR(rpe->angle().rmse, 1.941, 0.1);
  EXPECT_NEAR(rpe->angle().mean, 1.041, 0.1);
  EXPECT_NEAR(rpe->angle().median, 0.551, 0.1);
  EXPECT_NEAR(rpe->angle().stddev, 1.638, 0.1);
  EXPECT_NEAR(rpe->angle().min, 0.030, 0.1);
  EXPECT_NEAR(rpe->angle().max, 18.112, 0.1);

  print("{}\n", rpe->toString());

  auto plot = std::make_shared<PlotRPE>(
    std::map<std::string, RelativePoseError::ConstShPtr>({{"rgbdAlignment", rpe}}));
  if (TEST_VISUALIZE) {
    plot->plot();
    vis::plt::show();
  }
}