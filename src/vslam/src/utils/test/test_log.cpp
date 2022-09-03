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

#include "utils/utils.h"
using namespace testing;
using namespace pd;
using namespace pd::vslam;

TEST(LogTest, Plot)
{
  LOG_IMG("image");
  LOG_PLT("plot");
  auto imageLoggers = Log::registeredLogsImage();
  auto plotLoggers = Log::registeredLogsPlot();

  EXPECT_NE(std::find(imageLoggers.begin(), imageLoggers.end(), "image"), imageLoggers.end());
  EXPECT_EQ(std::find(imageLoggers.begin(), imageLoggers.end(), "plot"), imageLoggers.end());
}