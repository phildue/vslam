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
#include <lukas_kanade/lukas_kanade.h>
#include <utils/utils.h>

#include <opencv2/highgui.hpp>

#include "odometry/odometry.h"

using namespace testing;
using namespace pd;
using namespace pd::vslam;
using namespace pd::vslam::least_squares;

void readAssocTextfile(
  std::string filename, std::vector<std::string> & inputRGBPaths,
  std::vector<std::string> & inputDepthPaths, std::vector<Timestamp> & timestamps)
{
  std::string line;
  std::ifstream in_stream(filename.c_str());
  if (!in_stream.is_open()) {
    std::runtime_error("Could not open file at: " + filename);
  }

  while (!in_stream.eof()) {
    std::getline(in_stream, line);
    std::stringstream ss(line);
    std::string buf;
    int c = 0;
    while (ss >> buf) {
      c++;
      if (c == 1) {
        timestamps.push_back(static_cast<Timestamp>(std::stod(ss.str()) * 1e9));
      } else if (c == 2) {
        inputDepthPaths.push_back(buf);
      } else if (c == 4) {
        inputRGBPaths.push_back(buf);
      }
    }
  }
  in_stream.close();
}

class TestSE3Alignment : public Test
{
public:
  TestSE3Alignment()
  {
    if (TEST_VISUALIZE) {
      LOG_IMG("ImageWarped")->_show = true;
      LOG_IMG("Depth")->_show = true;
      LOG_IMG("Residual")->_show = true;
      LOG_IMG("Image")->_show = true;
      LOG_IMG("Template")->_show = true;
      LOG_IMG("Depth")->_show = true;
      LOG_IMG("Weights")->_show = true;
      LOG_IMG("SteepestDescent")->_show = true;
      //LOG_PLT("MedianScaler")->_show = true;
      //LOG_PLT("MedianScaler")->_block = true;
      //LOG_IMG("Residual")->_block = true;
    }

    // tum depth format: https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats
    _datasetPath = TEST_RESOURCE "/rgbd_dataset_freiburg2_desk";
    _cam = std::make_shared<Camera>(525.0, 525.0, 319.5, 239.5);
    readAssocTextfile(_datasetPath + "/assoc.txt", _imgFilenames, _depthFilenames, _timestamps);
    _trajectoryGt =
      std::make_shared<Trajectory>(utils::loadTrajectory(_datasetPath + "/groundtruth.txt"));

    auto solver = std::make_shared<GaussNewton>(1e-9, 30);
    auto loss = std::make_shared<HuberLoss>(std::make_shared<MeanScaler>());
    _aligner = std::make_shared<SE3Alignment>(5, solver, loss, true);
  }

  Frame::ShPtr loadFrame(size_t fNo)
  {
    // tum depth format: https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats
    auto f = std::make_shared<Frame>(
      utils::loadImage(_datasetPath + "/" + _imgFilenames.at(fNo)),
      utils::loadDepth(_datasetPath + "/" + _depthFilenames.at(fNo)) / 5000.0, _cam,
      _timestamps.at(fNo));
    f->computeDerivatives();
    f->computePyramid(3);
    return f;
  }

protected:
  SE3Alignment::ShPtr _aligner;
  std::vector<std::string> _depthFilenames;
  std::vector<std::string> _imgFilenames;
  std::vector<Timestamp> _timestamps;
  Trajectory::ConstShPtr _trajectoryGt;
  Camera::ConstShPtr _cam;
  std::string _datasetPath;
};

TEST_F(TestSE3Alignment, DISABLED_Comparison)
{
  const double maxError = 0.01;
  const int nFrames = 50;
  std::vector<int> fIds(nFrames, 0);
  std::transform(fIds.begin(), fIds.end(), fIds.begin(), [&](auto UNUSED(p)) {
    return random::U(0, _timestamps.size());
  });
  Eigen::Vector6d error = Eigen::Vector6d::Zero();
  for (int i = 0; i < nFrames; i++) {
    const int fId = i;  //fIds[i];
    auto fRef = loadFrame(fId);
    auto fRef1 = loadFrame(fId + 1);
    auto fCur = loadFrame(fId + 2);
    auto poseGt = _trajectoryGt->poseAt(fCur->t())->pose().inverse();
    fRef->set(_trajectoryGt->poseAt(fRef->t())->inverse());
    fRef1->set(_trajectoryGt->poseAt(fRef1->t())->inverse());
    //fCur->set(_trajectoryGt->poseAt(fCur->t())->inverse());
    fCur->set(fRef1->pose());

    auto result =
      _aligner->align(std::vector<Frame::ConstShPtr>({fRef, fRef1}), fCur)->pose().log();
    //auto result = _aligner->align(fRef1,fCur)->pose().log();
    fCur->set(result);
    error += (fCur->pose().pose().inverse() * poseGt).log();
    std::cout << fId << ": " << error.transpose() / static_cast<double>(i + 1) << std::endl;
  }
  std::cout << "AVG RMSE: " << (error / static_cast<double>(nFrames)).norm()
            << "\nAVG RMSE Translation: " << (error.head(3) / static_cast<double>(nFrames)).norm()
            << "\nAVG RMSE Rotation: " << (error.tail(3) / static_cast<double>(nFrames)).norm()
            << std::endl;
  EXPECT_LT((error / (double)nFrames).norm(), maxError);
}
