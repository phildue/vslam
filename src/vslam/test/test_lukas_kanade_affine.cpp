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

#include "core/algorithm.h"
#include "core/types.h"
#include "lukas_kanade/LukasKanade.h"
#include "lukas_kanade/LukasKanadeInverseCompositional.h"
#include "solver/Loss.h"
#include "utils/Exceptions.h"
#include "utils/Log.h"
#include "utils/utils.h"
#include "utils/visuals.h"
using namespace testing;
using namespace pd;
using namespace pd::vision;
using namespace pd::vslam::lukas_kanade;

class LukasKanadeAffineTest : public TestWithParam<int>
{
public:
  Image img0, img1;
  Eigen::Matrix3d A;
  Eigen::Vector6d x;
  LukasKanadeAffineTest()
  {
    img0 = utils::loadImage(TEST_RESOURCE "/person.jpg", 50, 50, true);
    A = Eigen::Matrix3d::Identity();
    img1 = img0;
    algorithm::warpAffine(img0, A, img1);
    const Eigen::Matrix3d Anoisy = transforms::createdTransformMatrix2D(
      random::U(1, 2) * random::sign(), random::U(1, 2) * random::sign(),
      random::U(0.025 * M_PI, 0.05 * M_PI) * random::sign());
    x(0) = Anoisy(0, 0) - 1;
    x(1) = Anoisy(1, 0);
    x(2) = Anoisy(0, 1);
    x(3) = Anoisy(1, 1) - 1;
    x(4) = Anoisy(0, 2);
    x(5) = Anoisy(1, 2);
  }
};
TEST_P(LukasKanadeAffineTest, DISABLED_LukasKanadeAffine)
{
  auto mat0 = vis::drawMat(img0);
  auto mat1 = vis::drawMat(img1);

  Log::getImageLog("I")->append(mat0);
  Log::getImageLog("T")->append(mat1);

  auto w = std::make_shared<WarpAffine>(x, img0.cols() / 2, img0.rows() / 2);
  auto gn = std::make_shared<GaussNewton<ForwardAdditive>>(0.1, 1e-3, 100);
  auto lk = std::make_shared<ForwardAdditive>(img1, img0, w);

  EXPECT_GT(w->x().norm(), 0.5);

  gn->solve(lk);

  EXPECT_LE(w->x().norm(), 0.5);
}

TEST_P(LukasKanadeAffineTest, LukasKanadeAffineInverseCompositional)
{
  auto mat0 = vis::drawMat(img0);
  auto mat1 = vis::drawMat(img1);

  Log::getImageLog("I")->append(mat0);
  Log::getImageLog("T")->append(mat1);

  auto w = std::make_shared<WarpAffine>(x, img0.cols() / 2, img0.rows() / 2);
  auto gn = std::make_shared<GaussNewton<InverseCompositional>>(0.1, 1e-3, 100);
  auto lk = std::make_shared<InverseCompositional>(img1, img0, w);

  EXPECT_GT(w->x().norm(), 0.5);

  gn->solve(lk);

  EXPECT_LE(w->x().norm(), 0.5);
}
INSTANTIATE_TEST_CASE_P(Instantiation, LukasKanadeAffineTest, ::testing::Range(1, 11));
