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
#include <fmt/chrono.h>
#include <fmt/core.h>
#include <fmt/ostream.h>
#include <gtest/gtest.h>
#include <utils/utils.h>

#include <opencv2/rgbd.hpp>
#include <opencv4/opencv2/core/eigen.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/opencv.hpp>

#include "evaluation/evaluation.h"
#include "odometry/odometry.h"
#include "utils/utils.h"
using fmt::format;
using fmt::print;

using namespace testing;
using namespace pd;
using namespace pd::vslam;

TEST(RgbdAlignmentUncertainty, Plot)
{
  using namespace pd::vslam::evaluation;
  Trajectory::ConstShPtr trajectoryGt =
    utils::loadTrajectory(TEST_RESOURCE "/rgbd_dataset_freiburg1_floor-groundtruth.txt", true);
  Trajectory::ConstShPtr trajectoryAlgo =
    utils::loadTrajectory(TEST_RESOURCE "/rgbd_dataset_freiburg1_floor-algo.txt", true);

  RelativePoseError::ConstShPtr rpe = RelativePoseError::compute(trajectoryAlgo, trajectoryGt, 1.0);

  print("{}\n", rpe->toString());

  auto plot = std::make_shared<PlotRPE>(
    std::map<std::string, RelativePoseError::ConstShPtr>({{"rgbdAlignment", rpe}}));
  auto plotCov = std::make_shared<PlotTrajectoryCovariance>(
    std::map<std::string, Trajectory::ConstShPtr>({{"rgbdAlignment", trajectoryAlgo}}));
  if (TEST_VISUALIZE) {
    plot->plot();
    plotCov->plot();
    vis::plt::show();
  }
}

class TestRgbdAlignment : public Test
{
public:
  TestRgbdAlignment()
  {
    _loss = std::make_shared<least_squares::QuadraticLoss>();
    least_squares::Scaler::ShPtr scaler;

    _solver = std::make_shared<least_squares::GaussNewton>(1e-9, 50, 1e-9, 1e-9, 0);

    _aligners["NoPrior"] = std::make_shared<RgbdAlignment>(_solver, _loss, false, false);
    _aligners["InitOnPrior"] = std::make_shared<RgbdAlignment>(_solver, _loss, false, true);
    _aligners["IncludePrior"] = std::make_shared<RgbdAlignment>(_solver, _loss, true, false);
    _aligners["InitOnAndIncludePrior"] =
      std::make_shared<RgbdAlignment>(_solver, _loss, true, true);

    for (auto name : {"odometry"}) {
      el::Loggers::getLogger(name);
      el::Configurations defaultConf;
      defaultConf.setToDefault();
      defaultConf.set(el::Level::Debug, el::ConfigurationType::Format, "%datetime %level %msg");
      defaultConf.set(el::Level::Debug, el::ConfigurationType::Enabled, "true");
      defaultConf.set(el::Level::Info, el::ConfigurationType::Enabled, "true");
      el::Loggers::reconfigureLogger(name, defaultConf);
    }
    for (auto name : {"solver"}) {
      el::Loggers::getLogger(name);
      el::Configurations defaultConf;
      defaultConf.setToDefault();
      defaultConf.set(el::Level::Debug, el::ConfigurationType::Format, "%datetime %level %msg");
      defaultConf.set(el::Level::Debug, el::ConfigurationType::Enabled, "false");
      defaultConf.set(el::Level::Info, el::ConfigurationType::Enabled, "true");
      el::Loggers::reconfigureLogger(name, defaultConf);
    }

    _dl = std::make_unique<tum::DataLoader>(
      "/mnt/dataset/tum_rgbd/rgbd_dataset_freiburg2_desk/rgbd_dataset_freiburg2_desk");
    loadFrames();
  }

  void loadFrames()
  {
    size_t fId = 100;  // random::U(0, _dl->nFrames());
    for (size_t i = fId; i < fId + 6; i += 2) {
      auto f = _dl->loadFrame(i);
      f->computePyramid(4);
      f->computeDerivatives();
      f->computePcl();
      _frames.push_back(std::move(f));
    }
  }

protected:
  std::map<std::string, std::shared_ptr<RgbdAlignment>> _aligners;
  tum::DataLoader::ConstShPtr _dl;
  Frame::VecShPtr _frames;
  least_squares::Loss::ShPtr _loss;
  least_squares::GaussNewton::ShPtr _solver;
};
