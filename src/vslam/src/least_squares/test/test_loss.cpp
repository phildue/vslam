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
#include <utils/utils.h>

#include "Loss.h"
using namespace testing;
using namespace pd;
using namespace pd::vslam;
using namespace pd::vslam::least_squares;
TEST(LossTest, TukeyLoss)
{
  std::vector<double> l, l_, r, w;
  auto tk = std::make_shared<TukeyLoss>();
  for (int i = -100; i < 100; i++) {
    const double ri = static_cast<double>(i) / 10.0;
    r.push_back(ri);
    l.push_back(tk->compute(ri));
    l_.push_back(tk->computeDerivative(ri));
    w.push_back(tk->computeWeight(ri) * ri * ri);
  }

  if (TEST_VISUALIZE) {
    vis::plt::figure();
    vis::plt::title("TukeyLoss");
    vis::plt::named_plot("l(r)", r, l);
    vis::plt::named_plot("l'(r)", r, l_);
    vis::plt::named_plot("$w(r)r^2$'", r, w);
    vis::plt::legend();
    vis::plt::xlabel("r");
    vis::plt::show();
  }
}