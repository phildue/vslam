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

#include "visuals.h"
using namespace testing;
using namespace pd;
using namespace pd::vslam;

TEST(LogTest, Plot)
{
  vis::plt::plot({1, 3, 2, 4});
  vis::plt::figure();
  vis::plt::hist<double>({1.0, 3.0, 2.0, 4.0});

  if (TEST_VISUALIZE) {
    vis::plt::show();
  }
}

TEST(LogTest, Histogram)
{
  Eigen::Matrix<double, 1, 9> v;
  v << -1, -1, 1, 3, 3, 3, 0, 0, 0;
  auto h = std::make_shared<vis::Histogram>(v, "TestHistogram");
  h->plot();
  if (TEST_VISUALIZE) {
    vis::plt::show();
  }
}

TEST(LogTest, HistogramLarge)
{
  Eigen::VectorXd v(20000);
  for (int i = -5000; i < 15000; i++) {
    v(i + 5000) = i;
  }
  auto h = std::make_shared<vis::Histogram>(v, "TestHistogram");
  h->plot();
  if (TEST_VISUALIZE) {
    vis::plt::show();
  }
}
