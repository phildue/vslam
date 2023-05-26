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

#include <gtest/gtest.h>

#include "algorithm.h"
using namespace testing;
using namespace pd;
using namespace pd::vslam;

TEST(MathTest, BilinearInterpolation)
{
  Eigen::Matrix<std::uint8_t, 3, 3> m;
  m << 128, 128, 128, 255, 255, 255, 255, 255, 255;
  const std::uint8_t r = algorithm::bilinearInterpolation(m, 0.5, 0.5);
  EXPECT_EQ(r, (255 + 128) / 2);
}

TEST(AlgorithmTest, DISABLED_Gradient)
{
  Eigen::Matrix<std::uint8_t, 3, 3> m;
  m << 128, 128, 128, 255, 128, 255, 255, 255, 255;

  const Eigen::MatrixXi ix = algorithm::gradX(m);
  const Eigen::MatrixXi iy = algorithm::gradY(m);
  const Image r = algorithm::gradient(m);

  EXPECT_EQ(ix(0, 0), 0);
  EXPECT_EQ(ix(1, 0), -127);
  EXPECT_EQ(ix(2, 0), 0);

  EXPECT_EQ(iy(0, 0), 127);
  EXPECT_EQ(iy(1, 0), 0);
  EXPECT_EQ(iy(2, 0), 0);

  EXPECT_EQ(r(0, 0), 127);
  EXPECT_EQ(r(1, 0), 127);
  EXPECT_EQ(r(2, 0), 0);
}

TEST(AlgorithmTest, Resize)
{
  Eigen::Matrix<std::uint8_t, 4, 4> m;
  m << 128, 128, 128, 128, 128, 128, 255, 255, 255, 128, 255, 255, 255, 255, 255, 255;

  const Image mRes = algorithm::resize<std::uint8_t>(m, 0.5);
  EXPECT_EQ(mRes(0, 0), 128U);
  EXPECT_EQ(mRes(1, 1), 255U);

  EXPECT_EQ(mRes.rows(), 2);
  EXPECT_EQ(mRes.cols(), 2);
}

TEST(AlgorithmTest, Conv2d)
{
  Eigen::Matrix<std::uint8_t, 4, 4> m;
  m << 128, 128, 128, 128, 128, 128, 255, 255, 255, 128, 255, 255, 255, 255, 255, 255;

  const auto mRes = algorithm::conv2d(m.cast<double>(), Kernel2d<double>::gaussian()).cast<int>();
  std::cout << "out:\n" << mRes << std::endl;
  EXPECT_EQ(
    mRes(1, 1), (128 + 2 * 128 + 128 + 2 * 128 + 4 * 128 + 2 * 255 + 255 + 2 * 128 + 255) / 16);
}
TEST(AlgorithmTest, Normalize)
{
  Eigen::Matrix<double, 3, 3> m;
  m << 128, 128, 128, 255, 255, 255, 255, 255, 255;
  const auto r = algorithm::normalize(m);
  EXPECT_NEAR(r.maxCoeff(), 1.0, 0.0001);
  EXPECT_NEAR(r.minCoeff(), 0.0, 0.0001);
}

TEST(AlgorithmTest, InsertionSort)
{
  std::vector<double> v = {4, 2, 1, 2, 8, 9};

  std::vector<double> vs;
  for (const auto e : v) {
    algorithm::insertionSort(vs, e);
  }
  EXPECT_EQ(vs[0], 1);
  EXPECT_EQ(vs[1], 2);
  EXPECT_EQ(vs[2], 2);
  EXPECT_EQ(vs[3], 4);
  EXPECT_EQ(vs[4], 8);
  EXPECT_EQ(vs[5], 9);
}

TEST(AlgorithmTest, InsertionSortSingle)
{
  std::vector<double> vs;
  algorithm::insertionSort(vs, 0.5);
  EXPECT_EQ(vs[0], 0.5);
}
