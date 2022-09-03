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

#include <core/core.h>
#include <gtest/gtest.h>
#include <utils/utils.h>

#include <opencv2/highgui.hpp>

#include "odometry/odometry.h"
using namespace testing;
using namespace pd;
using namespace pd::vslam;

TEST(TrajectoryTest, Create)
{
  auto poseGraph =
    std::make_shared<Trajectory>(utils::loadTrajectory(TEST_RESOURCE "/trajectory.txt"));
}
TEST(TrajectoryTest, Interpolate)
{
  SE3d pose0, pose2;
  pose2.translation().x() += 2.0;
  std::map<Timestamp, SE3d> poses;
  poses[0] = pose0;
  poses[2] = pose2;
  auto poseGraph = std::make_shared<Trajectory>(poses);
  auto pose1 = poseGraph->poseAt(1);
  EXPECT_NEAR(pose1->pose().translation().x(), 1.0, 1e-7);
}
