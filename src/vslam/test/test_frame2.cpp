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
#include <opencv2/rgbd.hpp>
#include <opencv4/opencv2/core/eigen.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/opencv.hpp>

#include "odometry/odometry.h"

using namespace testing;
using namespace pd;
using namespace pd::vslam;
using namespace pd::vslam::lukas_kanade;

TEST(FrameTest, DISABLED_CreatePyramid)
{
  DepthMap depth = utils::loadDepth(TEST_RESOURCE "/depth.jpg") / 5000.0;
  Image img = utils::loadImage(TEST_RESOURCE "/rgb.jpg");

  auto cam = std::make_shared<Camera>(525.0, 525.0, 319.5, 239.5);
  auto f = std::make_shared<Frame>(img, depth, cam, 0);
  for (size_t i = 0; i < f->nLevels(); i++) {
    auto pcl = f->pcl(i, true);
    DepthMap depthReproj = algorithm::resize(depth, std::pow(0.5, i));

    for (const auto & p : pcl) {
      const Eigen::Vector2i uv = f->camera2image(p, i).cast<int>();
      EXPECT_GT(uv.x(), 0);
      EXPECT_GT(uv.y(), 0);
      EXPECT_LT(uv.x(), f->width(i));
      EXPECT_LT(uv.y(), f->height(i));

      depthReproj(uv.y(), uv.x()) = p.z();
    }

    EXPECT_NEAR((depthReproj - f->depth(i)).norm(), 0.0, 1e-6);

    depthReproj = algorithm::resize(depth, std::pow(0.5, i));

    pcl = f->pcl(i, false);
    for (const auto & p : pcl) {
      const Eigen::Vector2i uv = f->camera2image(p, i).cast<int>();
      if (
        0 <= uv.x() && uv.x() < depthReproj.cols() && 0 <= uv.y() && uv.y() < depthReproj.cols()) {
        depthReproj(uv.y(), uv.x()) = p.z();
      }
    }

    EXPECT_NEAR((depthReproj - f->depth(i)).norm(), 0.0, 1e-6);

    if (TEST_VISUALIZE) {
      cv::imshow("Image", vis::drawMat(f->intensity(i)));
      cv::imshow("dIx", vis::drawAsImage(f->dIx(i).cast<double>()));
      cv::imshow("dIy", vis::drawAsImage(f->dIy(i).cast<double>()));
      cv::imshow("Depth Reproj", vis::drawAsImage(depthReproj));
      cv::imshow("Depth", vis::drawAsImage(f->depth(i)));
      cv::waitKey(0);
    }
  }
}
