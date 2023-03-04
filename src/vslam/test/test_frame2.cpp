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
// https://forum.kde.org/viewtopic.php?f=74&t=96407#
template <typename Derived>
MatXd conv2d(const Eigen::MatrixBase<Derived> & mat, const Eigen::MatrixBase<Derived> & kernel)
{
  // TODO(unknown): is this the most efficient way? add padding
  MatXd res = MatXd::Zero(mat.rows(), mat.cols());
  const int kX_2 = static_cast<int>(std::floor(static_cast<double>(kernel.cols()) / 2.0));
  const int kY_2 = static_cast<int>(std::floor(static_cast<double>(kernel.rows()) / 2.0));
  for (int i = 0; i < res.rows(); i++) {
    for (int j = 0; j < res.cols(); j++) {
      double sum = 0.0;
      for (int ki = 0; ki < kernel.rows(); ki++) {
        for (int kj = 0; kj < kernel.cols(); kj++) {
          const int idxY = std::min<int>(res.rows() - 1, std::max<int>(0, i - kY_2 + ki));
          const int idxX = std::min<int>(res.cols() - 1, std::max<int>(0, j - kX_2 + kj));
          const double kv = kernel(ki, kj);
          const double mv = mat(idxY, idxX);
          sum += kv * mv;
        }
      }
      res(i, j) = (static_cast<double>(sum));
    }
  }

  return res;
}

TEST(FrameTest, Sobel)
{
  Image img = utils::loadImage(TEST_RESOURCE "/rgb.png");
  cv::Mat matDiDx, matDiDy;
  MatXd dIdx, dIdy;
  {
    TIMED_SCOPE(timerCv, "Sobel-OpenCV-Total");
    cv::Mat mat(img.rows(), img.cols(), CV_32F);
    matDiDx = cv::Mat(img.rows(), img.cols(), CV_32F);
    matDiDy = cv::Mat(img.rows(), img.cols(), CV_32F);
    cv::eigen2cv(img, mat);
    TIMED_SCOPE(timerCv2, "Sobel-OpenCV");
    cv::Sobel(mat, matDiDx, CV_32F, 1, 0, 3);
    cv::Sobel(mat, matDiDy, CV_32F, 0, 1, 3);
  }
  {
    TIMED_SCOPE(timerVslam, "Sobel-VSLAM");
    dIdx = conv2d<MatXd>(MatXd(img.cast<double>()), Kernel2d<double>::sobelX());
    dIdy = conv2d<MatXd>(MatXd(img.cast<double>()), Kernel2d<double>::sobelY());
  }
  MatXd eigDiDx, eigDiDy;
  cv::cv2eigen(matDiDx, eigDiDx);
  cv::cv2eigen(matDiDy, eigDiDy);
  MatXd diffx = eigDiDx.block(2, 2, eigDiDx.rows() - 4, eigDiDx.cols() - 4) -
                dIdx.block(2, 2, eigDiDx.rows() - 4, eigDiDx.cols() - 4);
  MatXd diffy = eigDiDy.block(2, 2, eigDiDx.rows() - 4, eigDiDx.cols() - 4) -
                dIdy.block(2, 2, eigDiDx.rows() - 4, eigDiDx.cols() - 4);

  EXPECT_NEAR(diffx.norm(), 0.0, 0.0001);
  EXPECT_NEAR(diffy.norm(), 0.0, 0.0001);

  vis::imshow("Diffx", diffx.cast<image_value_t>());
  vis::imshow("Diffy", diffy.cast<image_value_t>(), 0);
}
#if false

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
      cv::imshow("dIx", vis::drawAsImage(f->dIdx(i).cast<double>()));
      cv::imshow("dIy", vis::drawAsImage(f->dIdy(i).cast<double>()));
      cv::imshow("Depth Reproj", vis::drawAsImage(depthReproj));
      cv::imshow("Depth", vis::drawAsImage(f->depth(i)));
      cv::waitKey(0);
    }
  }
}
#endif
