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
#include <solver/solver.h>
#include <utils/utils.h>

#include "lukas_kanade/LukasKanade.h"

using namespace testing;
using namespace pd;
using namespace pd::vision;
using namespace pd::vslam::least_squares;
using namespace pd::vslam::lukas_kanade;

class LukasKanadeSE3Test : public TestWithParam<int>
{
public:
  Image _img0, _img1;
  Sophus::SE3d _pose;
  Eigen::Vector6d _x;
  Eigen::MatrixXd _depth;
  std::shared_ptr<Camera> _camera;
  LukasKanadeSE3Test()
  {
    _img0 = utils::loadImage(TEST_RESOURCE "/sim.png", 0, 0, true);
    _depth = utils::loadDepth(TEST_RESOURCE "/sim.exr");

    _img0 = algorithm::resize(_img0, 0.25);
    _depth = algorithm::resize(_depth, 0.25);
    //img0 = utils::loadImage(TEST_RESOURCE"/person.jpg",50,50,true);
    //depth = Eigen::MatrixXd::Ones(img0.rows(),img0.cols())*110;
    _camera = std::make_shared<Camera>(381 / 4, _img0.cols() / 2, _img0.rows() / 2);
    _img1 = _img0;
    _x << random::U(0.1, 0.11) * random::sign(), random::U(0.1, 0.11) * random::sign(),
      random::U(0.01, 0.011) * random::sign(), random::U(0.001, 0.0011) * random::sign(),
      random::U(0.001, 0.0011) * random::sign(), random::U(0.001, 0.0011) * random::sign();
    _pose = Sophus::SE3d::exp(_x);
  }
};

TEST_P(LukasKanadeSE3Test, LukasKanadeSE3)
{
  auto mat0 = vis::drawMat(_img0);
  auto mat1 = vis::drawMat(_img1);

  Log::getImageLog("I")->append(mat0);
  Log::getImageLog("T")->append(mat1);

  auto w = std::make_shared<WarpSE3>(_x, _depth, _camera);
  auto gn = std::make_shared<GaussNewton<LukasKanade>>(0.1, 1e-3, 100);
  auto lk = std::make_shared<LukasKanade>(_img1, _img0, w);

  ASSERT_GT(w->x().norm(), 0.1);

  gn->solve(lk);

  EXPECT_LE(w->x().norm(), 0.1);
}
INSTANTIATE_TEST_CASE_P(Instantiation, LukasKanadeSE3Test, ::testing::Range(1, 11));
