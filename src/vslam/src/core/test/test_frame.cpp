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

using namespace testing;
using namespace pd;
using namespace pd::vslam;

TEST(FrameTest, BadDimensions)
{
  Image img = Image::Random(640, 480);

  auto cam = std::make_shared<Camera>(525.0, 525.0, 319.5, 239.5);
  EXPECT_ANY_THROW(std::make_shared<Frame>(img, cam))
    << "Should throw because image dimensions don't match with camera parameters";

  DepthMap depth = DepthMap::Ones(520, 640) * 20;
  Image img1 = Image::Random(480, 640);
  EXPECT_ANY_THROW(std::make_shared<Frame>(img1, depth, cam))
    << "Should throw because image dimensions don't match depth dimensions";
}
TEST(FrameTest, BadAccessDepth)
{
  Image img = Image::Random(480, 640);
  auto cam = std::make_shared<Camera>(525.0, 525.0, 319.5, 239.5);
  auto f = std::make_shared<Frame>(img, cam);

  EXPECT_EQ(-1, f->depth()(10, 10));
}

TEST(FrameTest, GoodAccessDepth)
{
  DepthMap depth = DepthMap::Ones(480, 640) * 20;
  Image img = Image::Random(480, 640);

  auto cam = std::make_shared<Camera>(525.0, 525.0, 319.5, 239.5);
  auto f = std::make_shared<Frame>(img, depth, cam);

  EXPECT_EQ(20, f->depth()(10, 10));
}

TEST(FrameTest, AccessPcl)
{
  DepthMap depth = DepthMap::Ones(480, 640) * 20;
  Image img = Image::Random(480, 640);

  auto cam = std::make_shared<Camera>(525.0, 525.0, 319.5, 239.5);
  auto f = std::make_shared<Frame>(img, depth, cam);

  EXPECT_ANY_THROW(f->pcl());

  f->computePcl();
  EXPECT_FALSE(f->pcl().empty());
}

TEST(FrameTest, Pyramid)
{
  DepthMap depth = DepthMap::Ones(480, 640) * 20;
  Image img = Image::Random(480, 640);

  auto cam = std::make_shared<Camera>(525.0, 525.0, 319.5, 239.5);
  auto f = std::make_shared<Frame>(img, cam);

  EXPECT_ANY_THROW(f->intensity(2));
  f->computePyramid(3);
  f->intensity(2);
}

TEST(FrameTest, BadAccessDerivative)
{
  DepthMap depth = DepthMap::Ones(480, 640) * 20;
  Image img = Image::Random(480, 640);

  auto cam = std::make_shared<Camera>(525.0, 525.0, 319.5, 239.5);
  auto f = std::make_shared<Frame>(img, cam);

  EXPECT_ANY_THROW(f->dIx());
}
TEST(FrameTest, GoodAccessDerivative)
{
  DepthMap depth = DepthMap::Ones(480, 640) * 20;
  Image img = Image::Random(480, 640);

  auto cam = std::make_shared<Camera>(525.0, 525.0, 319.5, 239.5);
  auto f = std::make_shared<Frame>(img, cam);
  f->computeDerivatives();
  f->dIx();
}
TEST(FrameTest, Pcl)
{
  DepthMap depth = DepthMap::Ones(480, 640) * 20;
  Image img = Image::Random(480, 640);

  auto cam = std::make_shared<Camera>(525.0, 525.0, 319.5, 239.5);
  auto f = std::make_shared<Frame>(img, depth, cam);
  f->computePyramid(3);
  f->computePcl();
  auto pclFull = f->pcl();
  ASSERT_NEAR((pclFull[100 * 640 + 100] - f->p3d(100, 100)).norm(), 0, 0.001);
  for (size_t i = 0; i < f->nLevels(); i++) {
    auto pcl = f->pcl(i, true);
    EXPECT_FALSE(pcl.empty());

    for (const auto & p : pcl) {
      auto uv = f->camera2image(p, i);
      EXPECT_GE(uv.x(), 0);
      EXPECT_GE(uv.y(), 0);
      EXPECT_LE(uv.x(), f->width(i));
      EXPECT_LE(uv.y(), f->height(i));
      const double z = f->depth(i)(uv.y(), uv.x());
      EXPECT_TRUE(std::isfinite(z) && z > 0);
      EXPECT_NEAR(p.z(), z, 0.001);
      EXPECT_NEAR((f->image2camera(uv, z, i) - p).norm(), 0.0, 0.001);

      EXPECT_NEAR((f->p3d(uv.y(), uv.x(), i) - p).norm(), 0.0, 0.001)
        << "Failed for uv=" << uv.transpose();
      EXPECT_NEAR((f->world2image(f->image2world(uv, z, i), i) - uv).norm(), 0.0, 0.001);
    }
  }
}