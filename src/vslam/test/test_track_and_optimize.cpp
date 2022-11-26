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

#include "mapping/mapping.h"
#include "odometry/odometry.h"
using namespace testing;
using namespace pd;
using namespace pd::vslam;

TEST(TrackingTest, TrackAndOptimize)
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

  /*Adding noise*/
  auto pose = f1->pose().pose();
  pose.translation().x() += 0.1;
  pose.translation().y() -= 0.1;
  f1->set(PoseWithCovariance(pose, f1->pose().cov()));
  auto pose2 = f2->pose().pose();
  pose2.translation().x() += 0.1;
  pose2.translation().y() -= 0.1;
  f2->set(PoseWithCovariance(pose2, f2->pose().cov()));

  tracking->track(f0, {});

  auto points = tracking->track(f1, {f0});
  auto points1 = tracking->track(f2, {f1, f0});
  for (auto p : points1) {
    points.push_back(p);
  }
  LOG(INFO) << "#Matches: " << points.size();

  LOG_IMG("TrackAndOptimizeBefore")->set(TEST_VISUALIZE, false);
  LOG_IMG("TrackAndOptimizeBefore") << std::make_shared<OverlayFeatureDisplacement>(
    std::vector<Frame::ConstShPtr>({f0, f1, f2}),
    Point3D::VecConstShPtr(points.begin(), points.end()));

  mapping::BundleAdjustment ba(100);
  auto results = ba.optimize(std::vector<Frame::ConstShPtr>({f0, f1, f2}));
  EXPECT_GT(results->errorBefore, 0);
  EXPECT_LT(results->errorAfter, results->errorBefore);

  for (auto f : {f0, f1, f2}) {
    f->set(results->poses.find(f->id())->second);
  }
  for (auto p : points) {
    p->position() = results->positions.find(p->id())->second;
  }

  LOG(INFO) << "Pose: " << f1->pose().pose().matrix();
  LOG_IMG("TrackAndOptimizeAfter")->set(TEST_VISUALIZE, TEST_VISUALIZE);
  LOG_IMG("TrackAndOptimizeAfter") << std::make_shared<OverlayFeatureDisplacement>(
    std::vector<Frame::ConstShPtr>({f0, f1, f2}),
    Point3D::VecConstShPtr(points.begin(), points.end()));
}
