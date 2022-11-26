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

#include <gtest/gtest.h>

#include "core/core.h"
#include "kalman/kalman.h"
#include "utils/utils.h"
using namespace testing;
using namespace pd;
using namespace pd::vslam;

TEST(EKFSE3, RunWithGt)
{
  auto trajectoryGt =
    std::make_shared<Trajectory>(utils::loadTrajectory(TEST_RESOURCE "/trajectoryGt.txt"));

  auto trajectoryAlgo =
    std::make_shared<Trajectory>(utils::loadTrajectory(TEST_RESOURCE "/trajectoryAlgo.txt"));

  auto kalman = std::make_shared<EKFConstantVelocitySE3>();

  Timestamp tLast = 0UL;
  SE3d poseRef;
  std::vector<Vec6d> twistsAlgo, twistsKalman, twistsGt;
  twistsAlgo.reserve(trajectoryAlgo->poses().size());
  twistsKalman.reserve(trajectoryAlgo->poses().size());
  twistsGt.reserve(trajectoryAlgo->poses().size());

  for (const auto & t_pose : trajectoryAlgo->poses()) {
    if (tLast == 0UL) {
      tLast = t_pose.first;
      poseRef = t_pose.second;
      continue;
    }
    auto dt = t_pose.first - tLast;
    auto dxAlgo = algorithm::computeRelativePose(poseRef, t_pose.second).log();
    twistsAlgo.push_back(dxAlgo / dt);
    auto dxGt = trajectoryGt.motionBetween(tLast, t_pose.first);
    twistsGt.push_back(dxGt / dt);
    kalman->update(dx, t_pose.first);
    twistsKalman.push_back(kalman->predict(0)->stateVel);
  }
}