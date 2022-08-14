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
#include <lukas_kanade/lukas_kanade.h>
#include <utils/utils.h>

#include <opencv2/highgui.hpp>

#include "odometry/odometry.h"

using namespace testing;
using namespace pd;
using namespace pd::vslam;
using namespace pd::vslam::least_squares;
using namespace pd::vslam::lukas_kanade;

#ifdef TEST_VISUALIZE
#define VISUALIZE true
#else
#define VISUALIZE false
#endif

class TestSE3Alignment : public Test
{
public:
  TestSE3Alignment()
  {
    auto solver = std::make_shared<GaussNewton>(1e-7, 10);
    auto loss = std::make_shared<QuadraticLoss>();
    auto scaler = std::make_shared<Scaler>();
    if (VISUALIZE) {
      LOG_IMG("ImageWarped")->_show = true;
      LOG_IMG("Depth")->_show = true;
      LOG_IMG("Residual")->_show = true;
      LOG_IMG("Image")->_show = true;
      LOG_IMG("Template")->_show = true;
      LOG_IMG("Depth")->_show = true;
      LOG_IMG("Weights")->_show = true;
      // LOG_PLT("MedianScaler")->_show = true;
      // LOG_PLT("MedianScaler")->_block = true;
      LOG_IMG("Residual")->_block = true;
    }
    _aligner = std::make_shared<RgbdAlignmentOpenCv>();

    // tum depth format:
    // https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats
    _depth0 = utils::loadDepth(TEST_RESOURCE "/depth.jpg") / 5000.0;
    _img0 = utils::loadImage(TEST_RESOURCE "/rgb.jpg");
    _depth1 = _depth0;
    _img1 = _img0;
    _cam = std::make_shared<Camera>(525.0, 525.0, 319.5, 239.5);

    /* Max Relative Poses within 0.03 seconds estimated from
    rgbd_dataset_freiburg1_desk2: tx ty tz rx ry rz*/
    _noise = {
      {-0.0145, 0.046, 0.0267, -0.2531, -0.0278, 0.0078},
      {-0.0145, 0.0453, 0.027, -0.2425, -0.027, 0.009},
      {-0.0045, 0.0285, 0.0169, -0.1262, -0.0149, 0.0259},
      {-0.0255, 0.0066, 0.0122, -0.018, -0.0327, 0.0056},
      {-0.0258, 0.0067, 0.0113, -0.0172, -0.0318, 0.0054},
      {-0.0252, 0.0063, 0.0128, -0.0134, -0.0327, 0.0052},
      {-0.025, 0.0061, 0.0129, -0.0141, -0.0326, 0.005},
      {-0.0252, 0.0068, 0.0119, -0.0177, -0.0311, 0.0051},
      {-0.0245, 0.0057, 0.0135, -0.0081, -0.0311, 0.0049},
      {-0.0255, 0.0066, 0.0108, -0.0148, -0.0306, 0.0042},

    };
  }

protected:
  std::shared_ptr<RgbdAlignmentOpenCv> _aligner;
  std::vector<std::vector<double>> _noise;
  Image _img0, _img1;
  DepthMap _depth0, _depth1;
  Camera::ConstShPtr _cam;
};

TEST_F(TestSE3Alignment, DISABLED_TestOnSyntheticDataTranslation)
{
  SE3d refPose(transforms::euler2quaternion(0, 0, 0), {0, 0, 0});
  for (size_t i = 1; i < _noise.size(); i++) {
    size_t ri = _noise.size() - i;
    SE3d deltaPoseGt(
      transforms::euler2quaternion(0, 0, 0), {_noise[ri][0], _noise[ri][1], _noise[ri][2]});

    auto warpGt = std::make_shared<WarpSE3>(deltaPoseGt * refPose, _depth0, _cam, _cam, refPose);
    _img1 = warpGt->apply(_img0);
    _depth1 = warpGt->apply(_depth0);

    auto fRef = std::make_shared<FrameRgbd>(
      _img0, _depth0, _cam, 3, 0, PoseWithCovariance(refPose, MatXd::Identity(6, 6)));
    auto fCur = std::make_shared<FrameRgbd>(
      _img1, _depth1, _cam, 3, 1, PoseWithCovariance(refPose, MatXd::Identity(6, 6)));

    auto result = _aligner->align(fRef, fCur)->pose().inverse().log();
    auto angleAxis = result.tail(3);
    const double eps = 0.01;
    EXPECT_NEAR(result.x(), deltaPoseGt.log().x(), eps) << "Failed in: " << ri;
    EXPECT_NEAR(result.y(), deltaPoseGt.log().y(), eps) << "Failed in: " << ri;
    EXPECT_NEAR(result.z(), deltaPoseGt.log().z(), eps) << "Failed in: " << ri;
    EXPECT_NEAR(angleAxis.norm(), deltaPoseGt.log().tail(3).norm(), eps) << "Failed in: " << ri;
  }
}
TEST_F(TestSE3Alignment, DISABLED_TestOnSyntheticDataRotation)
{
  SE3d refPose(transforms::euler2quaternion(0, 0, 0), {0, 0, 0});
  for (size_t i = 1; i < _noise.size(); i++) {
    size_t ri = _noise.size() - i;
    SE3d deltaPoseGt(
      transforms::euler2quaternion(_noise[ri][3], _noise[ri][4], _noise[ri][5]), {0, 0, 0});

    auto warpGt = std::make_shared<WarpSE3>(deltaPoseGt * refPose, _depth0, _cam, _cam, refPose);
    _img1 = warpGt->apply(_img0);
    _depth1 = warpGt->apply(_depth0);

    auto fRef = std::make_shared<FrameRgbd>(
      _img0, _depth0, _cam, 3, 0, PoseWithCovariance(refPose, MatXd::Identity(6, 6)));
    auto fCur = std::make_shared<FrameRgbd>(
      _img1, _depth1, _cam, 3, 1, PoseWithCovariance(refPose, MatXd::Identity(6, 6)));

    auto result = _aligner->align(fRef, fCur)->pose().inverse().log();
    auto angleAxis = result.tail(3);
    const double eps = 0.01;
    EXPECT_NEAR(result.x(), deltaPoseGt.log().x(), eps) << "Failed in: " << ri;
    EXPECT_NEAR(result.y(), deltaPoseGt.log().y(), eps) << "Failed in: " << ri;
    EXPECT_NEAR(result.z(), deltaPoseGt.log().z(), eps) << "Failed in: " << ri;
    EXPECT_NEAR(angleAxis.norm(), deltaPoseGt.log().tail(3).norm(), eps) << "Failed in: " << ri;
  }
}

TEST_F(TestSE3Alignment, DISABLED_TestOnSyntheticData)
{
  SE3d refPose(transforms::euler2quaternion(0, 0, 0), {0, 0, 0});
  for (size_t i = 1; i < _noise.size(); i++) {
    size_t ri = _noise.size() - i;
    SE3d deltaPoseGt(
      transforms::euler2quaternion(_noise[ri][3], _noise[ri][4], _noise[ri][5]),
      {_noise[ri][0], _noise[ri][1], _noise[ri][2]});

    auto warpGt = std::make_shared<WarpSE3>(deltaPoseGt * refPose, _depth0, _cam, _cam, refPose);
    _img1 = warpGt->apply(_img0);
    _depth1 = warpGt->apply(_depth0);

    auto fRef = std::make_shared<FrameRgbd>(
      _img0, _depth0, _cam, 3, 0, PoseWithCovariance(refPose, MatXd::Identity(6, 6)));
    auto fCur = std::make_shared<FrameRgbd>(
      _img1, _depth1, _cam, 3, 1, PoseWithCovariance(refPose, MatXd::Identity(6, 6)));

    auto result = _aligner->align(fRef, fCur)->pose().inverse().log();
    auto angleAxis = result.tail(3);
    const double eps = 0.01;
    EXPECT_NEAR(result.x(), deltaPoseGt.log().x(), eps) << "Failed in: " << ri;
    EXPECT_NEAR(result.y(), deltaPoseGt.log().y(), eps) << "Failed in: " << ri;
    EXPECT_NEAR(result.z(), deltaPoseGt.log().z(), eps) << "Failed in: " << ri;
    EXPECT_NEAR(angleAxis.norm(), deltaPoseGt.log().tail(3).norm(), eps) << "Failed in: " << ri;
  }
}

TEST_F(TestSE3Alignment, DISABLED_TestOnSyntheticDataTranslationAbsolute)
{
  SE3d refPose(transforms::euler2quaternion(0, 0, 0), {3.0, 4.0, 1.0});
  for (size_t i = 1; i < _noise.size(); i++) {
    size_t ri = _noise.size() - i;
    SE3d initialPose(
      transforms::euler2quaternion(0, 0, 0), {_noise[ri][0], _noise[ri][1], _noise[ri][2]});
    // SE3d
    // initialPose(transforms::euler2quaternion(0.03,0.03,0.03),{0.03,0.05,0.03});
    auto fRef = std::make_shared<FrameRgbd>(
      _img0, _depth0, _cam, 3, 0, PoseWithCovariance(refPose, MatXd::Identity(6, 6)));
    auto fCur = std::make_shared<FrameRgbd>(
      _img1, _depth1, _cam, 3, 1, PoseWithCovariance(initialPose * refPose, MatXd::Identity(6, 6)));

    auto result = _aligner->align(fRef, fCur)->pose().log();
    auto angleAxis = result.tail(3);
    const double eps = 0.01;
    EXPECT_NEAR(result.x(), refPose.log().x(), eps) << "Failed in: " << ri;
    EXPECT_NEAR(result.y(), refPose.log().y(), eps) << "Failed in: " << ri;
    EXPECT_NEAR(result.z(), refPose.log().z(), eps) << "Failed in: " << ri;
    EXPECT_NEAR(angleAxis.norm(), refPose.log().tail(3).norm(), eps) << "Failed in: " << ri;
  }
}
