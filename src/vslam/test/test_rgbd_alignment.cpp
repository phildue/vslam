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
#include <fmt/chrono.h>
#include <fmt/core.h>
#include <fmt/ostream.h>
#include <gtest/gtest.h>
#include <utils/utils.h>

#include <opencv2/rgbd.hpp>
#include <opencv4/opencv2/core/eigen.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/opencv.hpp>

#include "odometry/odometry.h"
#include "utils/utils.h"

using fmt::format;
using fmt::print;

using namespace testing;
using namespace pd;
using namespace pd::vslam;

class TestRgbdAlignment : public Test
{
public:
  TestRgbdAlignment()
  {
    _loss = std::make_shared<least_squares::QuadraticLoss>();
    least_squares::Scaler::ShPtr scaler;

    _solver = std::make_shared<least_squares::GaussNewton>(1e-9, 50, 1e-9, 1e-9, 0);

    _aligners["NoPrior"] = std::make_shared<RgbdAlignment>(_solver, _loss, false, false);
    _aligners["InitOnPrior"] = std::make_shared<RgbdAlignment>(_solver, _loss, false, true);
    _aligners["IncludePrior"] = std::make_shared<RgbdAlignment>(_solver, _loss, true, false);
    _aligners["InitOnAndIncludePrior"] =
      std::make_shared<RgbdAlignment>(_solver, _loss, true, true);

    for (auto name : {"odometry"}) {
      el::Loggers::getLogger(name);
      el::Configurations defaultConf;
      defaultConf.setToDefault();
      defaultConf.set(el::Level::Debug, el::ConfigurationType::Format, "%datetime %level %msg");
      defaultConf.set(el::Level::Debug, el::ConfigurationType::Enabled, "true");
      defaultConf.set(el::Level::Info, el::ConfigurationType::Enabled, "true");
      el::Loggers::reconfigureLogger(name, defaultConf);
    }
    for (auto name : {"solver"}) {
      el::Loggers::getLogger(name);
      el::Configurations defaultConf;
      defaultConf.setToDefault();
      defaultConf.set(el::Level::Debug, el::ConfigurationType::Format, "%datetime %level %msg");
      defaultConf.set(el::Level::Debug, el::ConfigurationType::Enabled, "false");
      defaultConf.set(el::Level::Info, el::ConfigurationType::Enabled, "true");
      el::Loggers::reconfigureLogger(name, defaultConf);
    }

    _dl = std::make_unique<tum::DataLoader>(
      "/mnt/dataset/tum_rgbd/rgbd_dataset_freiburg2_desk/rgbd_dataset_freiburg2_desk");
    loadFrames();
  }

  void loadFrames()
  {
    size_t fId = 100;  // random::U(0, _dl->nFrames());
    for (size_t i = fId; i < fId + 6; i += 2) {
      auto f = _dl->loadFrame(i);
      f->computePyramid(4);
      f->computeDerivatives();
      f->computePcl();
      _frames.push_back(std::move(f));
    }
  }

protected:
  std::map<std::string, std::shared_ptr<RgbdAlignment>> _aligners;
  tum::DataLoader::ConstShPtr _dl;
  Frame::VecShPtr _frames;
  least_squares::Loss::ShPtr _loss;
  least_squares::GaussNewton::ShPtr _solver;
};

TEST_F(TestRgbdAlignment, DISABLED_RoughEstimateLowConfidence)
{
  for (auto f : {_frames[0], _frames[1]}) {
    f->set(*_dl->trajectoryGt()->poseAt(f->t()));
  }
  auto T01 =
    algorithm::computeRelativeTransform(_frames[0]->pose().SE3(), _frames[1]->pose().SE3());
  _frames[2]->set(Pose(T01 * _frames[1]->pose().SE3(), Matd<6, 6>::Identity() * 1e3));
  auto errInit = algorithm::computeRelativeTransform(
    T01 * _frames[1]->pose().SE3(), _dl->trajectoryGt()->poseAt(_frames[2]->t())->SE3());
  print("Err Estimate. Err_t = {}\n", errInit.translation().norm());

  for (auto n_aligner : _aligners) {
    auto pose2 = n_aligner.second->align(_frames[1], _frames[2]);
    auto err = algorithm::computeRelativeTransform(
      pose2->pose(), _dl->trajectoryGt()->poseAt(_frames[2]->t())->SE3());

    print("{}. Err_t = {}\n", n_aligner.first, err.translation().norm());
  }
}

TEST_F(TestRgbdAlignment, DISABLED_RoughEstimateHighConfidence)
{
  for (auto f : {_frames[0], _frames[1]}) {
    f->set(*_dl->trajectoryGt()->poseAt(f->t()));
  }
  auto T01 =
    algorithm::computeRelativeTransform(_frames[0]->pose().SE3(), _frames[1]->pose().SE3());
  _frames[2]->set(Pose(T01 * _frames[1]->pose().SE3(), Matd<6, 6>::Identity() * 1e-3));
  auto errInit = algorithm::computeRelativeTransform(
    T01 * _frames[1]->pose().SE3(), _dl->trajectoryGt()->poseAt(_frames[2]->t())->SE3());
  print("Err Estimate. Err_t = {}\n", errInit.translation().norm());

  for (auto n_aligner : _aligners) {
    auto pose2 = n_aligner.second->align(_frames[1], _frames[2]);
    auto err = algorithm::computeRelativeTransform(
      pose2->pose(), _dl->trajectoryGt()->poseAt(_frames[2]->t())->SE3());

    print("{}. Err_t = {}\n", n_aligner.first, err.translation().norm());
  }
}

TEST_F(TestRgbdAlignment, DISABLED_GtHighConfidence)
{
  for (auto f : {_frames[0], _frames[1]}) {
    f->set(*_dl->trajectoryGt()->poseAt(f->t()));
  }
  _frames[2]->set(
    {_dl->trajectoryGt()->poseAt(_frames[2]->t())->SE3(), Matd<6, 6>::Identity() * 1e-3});

  for (auto n_aligner : _aligners) {
    auto pose2 = n_aligner.second->align(_frames[1], _frames[2]);
    auto err = algorithm::computeRelativeTransform(
      pose2->pose(), _dl->trajectoryGt()->poseAt(_frames[2]->t())->SE3());

    print("{}. Err_t = {}\n", n_aligner.first, err.translation().norm());
  }
}

TEST_F(TestRgbdAlignment, DISABLED_GtLowConfidence)
{
  for (auto f : {_frames[0], _frames[1]}) {
    f->set(*_dl->trajectoryGt()->poseAt(f->t()));
  }
  _frames[2]->set(
    {_dl->trajectoryGt()->poseAt(_frames[2]->t())->SE3(), Matd<6, 6>::Identity() * 1e3});

  for (auto n_aligner : _aligners) {
    auto pose2 = n_aligner.second->align(_frames[1], _frames[2]);
    auto err = algorithm::computeRelativeTransform(
      pose2->pose(), _dl->trajectoryGt()->poseAt(_frames[2]->t())->SE3());

    print("{}. Err_t = {}\n", n_aligner.first, err.translation().norm());
  }
}

TEST_F(TestRgbdAlignment, DISABLED_RgbdAlignerOpenCv)
{
  for (auto f : {_frames[0], _frames[1]}) {
    f->set(*_dl->trajectoryGt()->poseAt(f->t()));
  }
  auto from = _frames[1];
  auto to = _frames[2];

  cv::Mat camMat, srcImage, srcDepth, dstImage, dstDepth, guess;
  cv::eigen2cv(from->camera()->K(), camMat);
  auto relativePose = algorithm::computeRelativeTransform(from->pose().pose(), to->pose().pose());
  cv::eigen2cv(relativePose.matrix(), guess);

  cv::rgbd::RgbdOdometry estimator(
    camMat, cv::rgbd::Odometry::DEFAULT_MIN_DEPTH(), cv::rgbd::Odometry::DEFAULT_MAX_DEPTH(),
    cv::rgbd::Odometry::DEFAULT_MAX_DEPTH_DIFF(), {}, {}, 1.0);

  cv::eigen2cv(from->intensity(), srcImage);
  cv::eigen2cv(from->depth(), srcDepth);
  cv::eigen2cv(to->intensity(), dstImage);
  cv::eigen2cv(to->depth(), dstDepth);
  srcImage.convertTo(srcImage, CV_8UC1);
  srcDepth.convertTo(srcDepth, CV_32FC1);
  dstImage.convertTo(dstImage, CV_8UC1);
  dstDepth.convertTo(dstDepth, CV_32FC1);
  auto odomFrameFrom = cv::rgbd::OdometryFrame::create(srcImage, srcDepth);
  auto odomFrameTo = cv::rgbd::OdometryFrame::create(dstImage, dstDepth);
  cv::Mat RtCv;
  MatXd Rt;
  estimator.compute(odomFrameFrom, odomFrameTo, RtCv);
  cv::cv2eigen(RtCv, Rt);
  auto err0 = algorithm::computeRelativeTransform(
    _frames[2]->pose().SE3(), _dl->trajectoryGt()->poseAt(_frames[2]->t())->SE3());
  print("{}. Before Err_t = {}\n", "rgbd aligner opencv", err0.translation().norm());

  SE3d se3(Rt);
  auto pose2 =
    std::make_unique<PoseWithCovariance>(se3 * from->pose().pose(), MatXd::Identity(6, 6));
  auto err = algorithm::computeRelativeTransform(
    pose2->pose(), _dl->trajectoryGt()->poseAt(_frames[2]->t())->SE3());
  print("{}. After Err_t = {}\n", "rgbd aligner opencv", err.translation().norm());
}

TEST_F(TestRgbdAlignment, DISABLED_RgbdAlignerInverse)
{
  for (auto f : {_frames[0], _frames[1]}) {
    f->set(*_dl->trajectoryGt()->poseAt(f->t()));
  }
  auto aligner = std::make_shared<RgbdAlignment>(_solver, _loss, false, false);
  LOG_IMG("Correspondences")->set(false, false);
  auto pose2 = aligner->align(_frames[1], _frames[2]);
  auto err = algorithm::computeRelativeTransform(
    pose2->pose(), _dl->trajectoryGt()->poseAt(_frames[2]->t())->SE3());
  auto err0 = algorithm::computeRelativeTransform(
    _frames[2]->pose().SE3(), _dl->trajectoryGt()->poseAt(_frames[2]->t())->SE3());
  print("{}. Before Err_t = {}\n", "rgbd aligner inverse", err0.translation().norm());

  print("{}. Err_t = {}\n", "rgbd aligner inverse", err.translation().norm());
}

TEST(FrameTest, DISABLED_CompareToOpenCV)
{
  DepthMap depth = utils::loadDepth(TEST_RESOURCE "/depth.png") / 5000.0;
  Image img = utils::loadImage(TEST_RESOURCE "/rgb.png");
  int nLevels = 4;
  auto cam = tum::Camera();
  cv::Mat camMat;
  cv::eigen2cv(cam->K(), camMat);
  auto f = std::make_shared<Frame>(img, depth, cam, 0);
  f->computePyramid(nLevels);
  f->computeDerivatives();
  f->computePcl();
  cv::rgbd::RgbdOdometry estimator(
    camMat, cv::rgbd::Odometry::DEFAULT_MIN_DEPTH(), cv::rgbd::Odometry::DEFAULT_MAX_DEPTH(),
    cv::rgbd::Odometry::DEFAULT_MAX_DEPTH_DIFF(), {}, {}, 1.0);
  cv::Mat img_cv, depth_cv;
  cv::eigen2cv(img, img_cv);
  cv::eigen2cv(depth, depth_cv);
  img_cv.convertTo(img_cv, CV_8UC1);
  depth_cv.convertTo(depth_cv, CV_32FC1);
  auto f_cv = cv::rgbd::OdometryFrame::create(img_cv, depth_cv);
  estimator.prepareFrameCache(f_cv, cv::rgbd::OdometryFrame::CACHE_SRC);
  estimator.prepareFrameCache(f_cv, cv::rgbd::OdometryFrame::CACHE_DST);

  EXPECT_EQ(f_cv->pyramidImage.size(), nLevels);
  EXPECT_EQ(f_cv->pyramidDepth.size(), nLevels);
  EXPECT_EQ(f_cv->pyramidMask.size(), nLevels);
  EXPECT_EQ(f_cv->pyramidCloud.size(), nLevels);
  EXPECT_EQ(f_cv->pyramid_dI_dx.size(), nLevels);
  EXPECT_EQ(f_cv->pyramid_dI_dy.size(), nLevels);
  EXPECT_EQ(f_cv->pyramidNormals.size(), 0);
  EXPECT_EQ(f_cv->pyramidTexturedMask.size(), nLevels);
  EXPECT_EQ(f->nLevels(), nLevels);
  for (int i = 0; i < nLevels; i++) {
    MatXd _intensity, _depth, _dI_dx, _dI_dy;
    cv::cv2eigen(f_cv->pyramidImage[i], _intensity);
    cv::cv2eigen(f_cv->pyramidDepth[i], _depth);
    cv::cv2eigen(f_cv->pyramid_dI_dx[i], _dI_dx);
    cv::cv2eigen(f_cv->pyramid_dI_dy[i], _dI_dy);

    LOG(INFO) << "##################Level: " << i;

    EXPECT_NEAR((_intensity.cast<std::uint8_t>() - f->intensity(i)).norm(), 0.0, 1e-4);
    EXPECT_NEAR((_depth.cast<double>() - f->depth(i)).norm(), 0.0, 1e-4);
    EXPECT_NEAR((_dI_dx.cast<double>() - f->dIx(i)).norm(), 0.0, 1e-4);
    EXPECT_NEAR((_dI_dy.cast<double>() - f->dIy(i)).norm(), 0.0, 1e-4);
  }
}
