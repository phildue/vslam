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

class EvaluationOdometry : public Test
{
public:
  EvaluationOdometry()
  {
    // tum depth format: https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats
    _datasetPath = TEST_RESOURCE "/rgbd_dataset_freiburg2_desk";
    _cam = std::make_shared<Camera>(525.0, 525.0, 319.5, 239.5);
    readAssocTextfile(_datasetPath + "/assoc.txt", _imgFilenames, _depthFilenames, _timestamps);
    _trajectoryGt =
      std::make_shared<Trajectory>(utils::loadTrajectory(_datasetPath + "/groundtruth.txt"));

    auto solver = std::make_shared<GaussNewton>(1e-9, 100);
    auto loss = std::make_shared<LossTDistribution>(std::make_shared<ScalerTDistribution>());
    _keyFrameSelection = std::make_shared<KeyFrameSelectionIdx>(5);
    _map = std::make_shared<Map>();
    _prediction = std::make_shared<MotionPredictionConstant>();
    _odometry = std::make_shared<OdometryRgbd>(30, solver, loss, _map);

    for (auto log : Log::registeredLogsImage()) {
      LOG_IMG(log)->show() = TEST_VISUALIZE;
    }
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
  Odometry::ShPtr _odometry;
  KeyFrameSelection::ShPtr _keyFrameSelection;
  MotionPrediction::ShPtr _prediction;
  Map::ShPtr _map;

  std::vector<std::string> _depthFilenames;
  std::vector<std::string> _imgFilenames;
  std::vector<Timestamp> _timestamps;
  Trajectory::ConstShPtr _trajectoryGt;
  Camera::ConstShPtr _cam;
  std::string _datasetPath;
};

TEST_F(EvaluationOdometry, DISABLED_Sequential)
{
  const int fId0 = 0;
  const int nFrames = _imgFilenames.size();

  Trajectory traj;
  for (int fId = fId0; fId < nFrames; fId++) {
    auto fCur = loadFrame(fId);

    fCur->set(*_prediction->predict(fCur->t()));

    _odometry->update(fCur);

    fCur->set(*_odometry->pose());

    _prediction->update(_odometry->pose(), fCur->t());

    _keyFrameSelection->update(fCur);

    _map->insert(fCur, _keyFrameSelection->isKeyFrame());
    traj.append(fCur->t(), std::make_shared<PoseWithCovariance>(fCur->pose().inverse()));
    if (_map->lastKf()) {
      auto relativePose =
        algorithm::computeRelativeTransform(_map->lastKf()->pose().pose(), fCur->pose().pose())
          .inverse();
      auto relativePoseGt = _trajectoryGt->motionBetween(_map->lastKf()->t(), fCur->t())->inverse();
      auto error = (relativePose.inverse() * relativePoseGt.pose()).log();

      std::cout << fId << "/" << nFrames << ": "
                << ": " << fCur->pose().pose().log().transpose()
                << "\n Error Translation: " << error.head(3).norm()
                << "\n Error Angle: " << error.tail(3).norm() << std::endl;
    }

    utils::writeTrajectory(traj, "trajectory.txt");
  }
  // TODO(unknown): call evaluation script?
}
