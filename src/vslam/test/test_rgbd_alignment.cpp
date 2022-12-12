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

#include "odometry/odometry.h"
#include "utils/utils.h"

using fmt::format;
using fmt::print;
using namespace testing;
using namespace pd;
using namespace pd::vslam;

class TestRgbdAlignment : public Test
{
public:
  TestRgbdAlignment()
  {
    least_squares::Loss::ShPtr loss = std::make_shared<least_squares::QuadraticLoss>();
    least_squares::Scaler::ShPtr scaler;

    auto solver = std::make_shared<least_squares::GaussNewton>(1e-6, 50);

    _aligners["NoPrior"] = std::make_shared<SE3Alignment>(18, solver, loss, false, false);
    _aligners["InitOnPrior"] = std::make_shared<SE3Alignment>(18, solver, loss, false, true);
    _aligners["IncludePrior"] = std::make_shared<SE3Alignment>(18, solver, loss, true, false);
    _aligners["InitOnAndIncludePrior"] =
      std::make_shared<SE3Alignment>(18, solver, loss, true, true);

    for (auto name : {"odometry"}) {
      el::Loggers::getLogger(name);
      el::Configurations defaultConf;
      defaultConf.setToDefault();
      defaultConf.set(el::Level::Debug, el::ConfigurationType::Format, "%datetime %level %msg");
      defaultConf.set(el::Level::Debug, el::ConfigurationType::Enabled, "false");
      defaultConf.set(el::Level::Info, el::ConfigurationType::Enabled, "false");
      el::Loggers::reconfigureLogger(name, defaultConf);
    }
    for (auto name : {"solver"}) {
      el::Loggers::getLogger(name);
      el::Configurations defaultConf;
      defaultConf.setToDefault();
      defaultConf.set(el::Level::Debug, el::ConfigurationType::Format, "%datetime %level %msg");
      defaultConf.set(el::Level::Debug, el::ConfigurationType::Enabled, "false");
      defaultConf.set(el::Level::Info, el::ConfigurationType::Enabled, "false");
      el::Loggers::reconfigureLogger(name, defaultConf);
    }

    _dl = std::make_unique<tum::DataLoader>(
      "/mnt/dataset/tum_rgbd/rgbd_dataset_freiburg2_desk/rgbd_dataset_freiburg2_desk");
    loadFrames();
  }

  void loadFrames()
  {
    auto fId = 100;  // random::U(0, _dl->nFrames());
    for (size_t i = fId; i < fId + 6; i += 2) {
      auto f = _dl->loadFrame(i);
      f->computePyramid(3);
      f->computeDerivatives();
      f->computePcl();
      _frames.push_back(std::move(f));
    }
  }

protected:
  std::map<std::string, std::shared_ptr<SE3Alignment>> _aligners;
  tum::DataLoader::ConstShPtr _dl;
  Frame::VecShPtr _frames;
};

TEST_F(TestRgbdAlignment, RoughEstimateLowConfidence)
{
  for (auto f : {_frames[0], _frames[1]}) {
    f->set(*_dl->trajectoryGt()->poseAt(f->t()));
  }
  auto T01 =
    algorithm::computeRelativeTransform(_frames[0]->pose().SE3(), _frames[1]->pose().SE3());
  _frames[2]->set(Pose(T01 * _frames[1]->pose().SE3(), Matd<6, 6>::Identity() * 1e3));
  auto errInit = algorithm::computeRelativeTransform(
    T01 * _frames[1]->pose().SE3(), _dl->trajectoryGt()->poseAt(_frames[2]->t())->SE3());
  print("Err Estimate. Err_t = {}\n", errInit.translation().norm());

  for (auto n_aligner : _aligners) {
    auto pose2 = n_aligner.second->align(_frames[1], _frames[2]);
    auto err = algorithm::computeRelativeTransform(
      pose2->pose(), _dl->trajectoryGt()->poseAt(_frames[2]->t())->SE3());

    print("{}. Err_t = {}\n", n_aligner.first, err.translation().norm());
  }
}

TEST_F(TestRgbdAlignment, RoughEstimateHighConfidence)
{
  for (auto f : {_frames[0], _frames[1]}) {
    f->set(*_dl->trajectoryGt()->poseAt(f->t()));
  }
  auto T01 =
    algorithm::computeRelativeTransform(_frames[0]->pose().SE3(), _frames[1]->pose().SE3());
  _frames[2]->set(Pose(T01 * _frames[1]->pose().SE3(), Matd<6, 6>::Identity() * 1e-3));
  auto errInit = algorithm::computeRelativeTransform(
    T01 * _frames[1]->pose().SE3(), _dl->trajectoryGt()->poseAt(_frames[2]->t())->SE3());
  print("Err Estimate. Err_t = {}\n", errInit.translation().norm());

  for (auto n_aligner : _aligners) {
    auto pose2 = n_aligner.second->align(_frames[1], _frames[2]);
    auto err = algorithm::computeRelativeTransform(
      pose2->pose(), _dl->trajectoryGt()->poseAt(_frames[2]->t())->SE3());

    print("{}. Err_t = {}\n", n_aligner.first, err.translation().norm());
  }
}

TEST_F(TestRgbdAlignment, GtHighConfidence)
{
  for (auto f : {_frames[0], _frames[1]}) {
    f->set(*_dl->trajectoryGt()->poseAt(f->t()));
  }
  _frames[2]->set(
    {_dl->trajectoryGt()->poseAt(_frames[2]->t())->SE3(), Matd<6, 6>::Identity() * 1e-3});

  for (auto n_aligner : _aligners) {
    auto pose2 = n_aligner.second->align(_frames[1], _frames[2]);
    auto err = algorithm::computeRelativeTransform(
      pose2->pose(), _dl->trajectoryGt()->poseAt(_frames[2]->t())->SE3());

    print("{}. Err_t = {}\n", n_aligner.first, err.translation().norm());
  }
}

TEST_F(TestRgbdAlignment, GtLowConfidence)
{
  for (auto f : {_frames[0], _frames[1]}) {
    f->set(*_dl->trajectoryGt()->poseAt(f->t()));
  }
  _frames[2]->set(
    {_dl->trajectoryGt()->poseAt(_frames[2]->t())->SE3(), Matd<6, 6>::Identity() * 1e3});

  for (auto n_aligner : _aligners) {
    auto pose2 = n_aligner.second->align(_frames[1], _frames[2]);
    auto err = algorithm::computeRelativeTransform(
      pose2->pose(), _dl->trajectoryGt()->poseAt(_frames[2]->t())->SE3());

    print("{}. Err_t = {}\n", n_aligner.first, err.translation().norm());
  }
}