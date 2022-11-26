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
using namespace testing;
using namespace pd;
using namespace pd::vslam;
using namespace pd::vslam::lukas_kanade;

TEST(FrameTest, CreatePyramid)
{
  DepthMap depth = utils::loadDepth(TEST_RESOURCE "/depth.jpg") / 5000.0;
  Image img = utils::loadImage(TEST_RESOURCE "/rgb.jpg");

  auto cam = std::make_shared<Camera>(525.0, 525.0, 319.5, 239.5);
  auto f = std::make_shared<Frame>(img, depth, cam, 3, 0);
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

TEST(WarpTest, DISABLED_Warp)
{
  DepthMap depth = utils::loadDepth(TEST_RESOURCE "/depth.jpg") / 5000.0;
  Image img = utils::loadImage(TEST_RESOURCE "/rgb.jpg");

  auto cam = std::make_shared<Camera>(525.0, 525.0, 319.5, 239.5);
  auto f0 = std::make_shared<Frame>(img, depth, cam, 3, 0);
  auto f1 = std::make_shared<Frame>(img, depth, cam, 3, 0);

  for (size_t i = 0; i < f0->nLevels(); i++) {
    auto w = std::make_shared<WarpSE3>(
      f0->pose().pose(), f0->pcl(i, false), f0->width(i), f0->camera(i), f1->camera(i),
      f1->pose().pose());

    auto & img = f0->intensity(i);
    Image iwxp = w->apply(img);
    Eigen::MatrixXd steepestDescent = Eigen::MatrixXd::Zero(iwxp.rows(), iwxp.cols());
    std::vector<MatXd> Js(6, MatXd::Zero(iwxp.rows(), iwxp.cols()));
    for (int v = 0; v < steepestDescent.rows(); v++) {
      for (int u = 0; u < steepestDescent.cols(); u++) {
        const Eigen::Matrix<double, 2, 6> Jw = w->J(u, v);
        const Eigen::Matrix<double, 1, 6> Jwi =
          Jw.row(0) * f0->dIx(i)(v, u) + Jw.row(1) * f0->dIx(i)(v, u);
        // std::cout << "J = " << Jwi << std::endl;
        for (int j = 0; j < 6; j++) {
          Js[j](v, u) = Jwi(j);
        }

        steepestDescent(v, u) = std::isfinite(Jwi.norm()) ? Jwi.norm() : 0.0;
      }
    }
    if (TEST_VISUALIZE) {
      for (int j = 0; j < 6; j++) {
        cv::imshow("J" + std::to_string(j), vis::drawAsImage(Js.at(j)));
      }
      cv::imshow("Image", vis::drawMat(f0->intensity(i)));
      cv::imshow("Iwxp", vis::drawMat(iwxp));
      cv::imshow("J", vis::drawAsImage(steepestDescent));
      cv::waitKey(0);
    }
  }
}
