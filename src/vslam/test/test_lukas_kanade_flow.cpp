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

#include "core/core.h"
#include "least_squares/least_squares.h"
#include "lukas_kanade/lukas_kanade.h"
#include "utils/utils.h"

using namespace testing;
using namespace pd;
using namespace pd::vslam;
using namespace pd::vslam::least_squares;
using namespace pd::vslam::lukas_kanade;

class LukasKanadeOpticalFlowTest : public Test
{
public:
  Image img0, img1;
  Eigen::Matrix3d A;
  int _nRuns = 20;
  int _nFailed = 0;
  LukasKanadeOpticalFlowTest()
  {
    img0 = utils::loadImage(TEST_RESOURCE "/person.jpg", 50, 50, true);
    A = Eigen::Matrix3d::Identity();
    img1 = img0;
    algorithm::warpAffine(img0, A, img1);
  }
};

TEST_F(LukasKanadeOpticalFlowTest, DISABLED_LukasKanadeOpticalFlow)
{
  for (int i = 0; i < _nRuns; i++) {
    Eigen::Vector2d x;
    x << random::U(5.0, 6.0) * random::sign(), random::U(5.0, 6.0) * random::sign();
    auto w = std::make_shared<WarpOpticalFlow>(x);
    auto gn = std::make_shared<GaussNewton>(1e-7, 100);
    auto lk = std::make_shared<InverseCompositional>(img1, img0, w);

    if (TEST_VISUALIZE) {
      LOG_IMG("ImageWarped")->show() = true;
      LOG_IMG("Depth")->show() = true;
      LOG_IMG("Residual")->show() = true;
      LOG_IMG("Image")->show() = true;
      LOG_IMG("Depth")->show() = true;
      LOG_IMG("Weights")->show() = true;
    }

    ASSERT_GT(w->x().norm(), 1.0) << "Noise should be greater than that.";

    gn->solve(lk);

    if (w->x().norm() > 1.0) {
      _nFailed++;
    }
  }

  EXPECT_LE((double)_nFailed / (double)_nRuns, 0.05) << "Majority of test cases should pass.";
}
