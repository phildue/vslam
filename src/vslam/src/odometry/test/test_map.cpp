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

#include <Eigen/Dense>
#include <opencv2/highgui.hpp>
#include <opencv4/opencv2/core/eigen.hpp>

#include "odometry/odometry.h"
using namespace testing;
using namespace pd;
using namespace pd::vslam;

TEST(MappingTest, FillAndDelete)
{
  auto cam = std::make_shared<Camera>(525.0, 525.0, 319.5, 239.5);

  auto f0 = std::make_shared<Frame>(
    utils::loadImage(TEST_RESOURCE "/1311868164.363181.png"),
    utils::loadDepth(TEST_RESOURCE "/1311868164.338541.png") / 5000.0, cam, 1311868164363181000U);

  auto f1 = std::make_shared<Frame>(
    utils::loadImage(TEST_RESOURCE "/1311868165.499179.png"),
    utils::loadDepth(TEST_RESOURCE "/1311868165.409979.png") / 5000.0, cam, 1311868165499179000U);

  auto f2 = std::make_shared<Frame>(
    utils::loadImage(TEST_RESOURCE "/1311868166.763333.png"),
    utils::loadDepth(TEST_RESOURCE "/1311868166.715787.png") / 5000.0, cam, 1311868166763333000U);

  auto trajectoryGt =
    std::make_shared<Trajectory>(utils::loadTrajectory(TEST_RESOURCE "/trajectory.txt"));
  f0->set(trajectoryGt->poseAt(f0->t())->inverse());
  f1->set(trajectoryGt->poseAt(f1->t())->inverse());
  f2->set(trajectoryGt->poseAt(f2->t())->inverse());
  auto tracking = std::make_shared<FeatureTracking>(
    std::make_shared<vslam::Matcher>(vslam::Matcher::reprojectionHamming, 10, 0.8));

  tracking->track(f0, {});

  auto points1 = tracking->track(f1, {f0});
  for (auto p : points1) {
    EXPECT_NE(f0->observationOf(p->id()), nullptr);
    EXPECT_NE(f1->observationOf(p->id()), nullptr);
  }
  auto points2 = tracking->track(f2, {f0, f1});
  auto map = std::make_shared<Map>(1, 1);
  map->insert(f0, true);
  map->insert(f1, false);
  map->insert(points1);
  map->insert(f2, true);
  map->insert(points2);
}
