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
// Created by phil on 08.08.21.
//

#include <fmt/chrono.h>
#include <fmt/core.h>
using fmt::format;
using fmt::print;
#include <Eigen/Dense>
#include <filesystem>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv4/opencv2/core/eigen.hpp>

#include "Exceptions.h"
#include "utils/utils.h"
namespace fs = std::filesystem;
namespace pd::vslam
{
void utils::throw_if_nan(const Eigen::MatrixXd & mat, const std::string & msg)
{
  auto result = mat.norm();
  if (std::isnan(mat.norm()) || std::isinf(result)) {
    std::stringstream ss;
    ss << mat;
    throw pd::Exception(msg + " contains nan: \n" + ss.str());
  }
}

Image utils::loadImage(const fs::path & path, int height, int width, bool grayscale)
{
  auto mat = cv::imread(path.string(), grayscale ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR);

  if (std::is_floating_point<image_value_t>::value) {
    mat.convertTo(mat, CV_32F);
  } else {
    mat.convertTo(mat, CV_8U);
  }

  if (mat.cols == 0 || mat.rows == 0) {
    throw pd::Exception("Failure during load of: " + path.string());
  }

  if (height > 0 && width > 0) {
    cv::resize(mat, mat, cv::Size(width, height));
  }

  Image img;
  cv::cv2eigen(mat, img);
  return img;
}

Eigen::MatrixXd utils::loadDepth(const fs::path & path, int height, int width)
{
  auto mat = cv::imread(path.string(), cv::IMREAD_ANYDEPTH);
  mat.convertTo(mat, CV_32F);

  if (mat.cols == 0 || mat.rows == 0) {
    throw pd::Exception("Failure during load of: " + path.string());
  }

  if (height > 0 && width > 0) {
    cv::resize(mat, mat, cv::Size(width, height));
  }

  Eigen::MatrixXd img;
  cv::cv2eigen(mat, img);
  return img.array().isNaN().select(0, img);
}
Trajectory::UnPtr utils::loadTrajectory(const fs::path & path, bool invertPoses)
{
  if (!fs::exists(path)) {
    throw pd::Exception(fmt::format("Could not find [{}]", path.string()));
  }
  std::ifstream gtFile;
  gtFile.open(path.string());

  if (!gtFile.is_open()) {
    throw std::runtime_error("Could not open file at: " + path.string());
  }

  //TODO shouldn't this better be objects or unique ptrs
  std::map<Timestamp, PoseWithCovariance::ConstShPtr> poses;

  std::string line;
  while (getline(gtFile, line)) {
    std::vector<std::string> elements;
    std::string s;
    std::istringstream lines(line);
    while (getline(lines, s, ' ')) {
      elements.push_back(s);
    }

    if (elements[0] == "#") {
      continue;
    }  //Skip comments

    //Pose
    Eigen::Vector3d trans;
    trans << std::stod(elements[1]), std::stod(elements[2]), std::stod(elements[3]);
    Eigen::Quaterniond q(
      std::stod(elements[7]), std::stod(elements[4]), std::stod(elements[5]),
      std::stod(elements[6]));
    auto se3 = invertPoses ? SE3d(q, trans) : SE3d(q, trans).inverse();

    //Covariance
    Eigen::Matrix<double, 6, 6> cov = Eigen::Matrix<double, 6, 6>::Identity();
    if (elements.size() >= 8 + 36) {
      for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
          cov(i, j) = std::stod(elements[8 + i * 6 + j]);
        }
      }
    }

    //Timestamp
    std::vector<std::string> tElements;
    std::string st;
    std::istringstream tLine(elements[0]);
    while (getline(tLine, st, '.')) {
      tElements.push_back(st);
    }
    auto sec = std::stoull(tElements[0]);
    auto nanosec = std::stoull(tElements[1]) * std::pow(10, 9 - tElements[1].size());

    poses.insert({sec * 1e9 + nanosec, std::make_shared<PoseWithCovariance>(se3, cov)});
  }
  return std::make_unique<Trajectory>(poses);
}

void utils::saveImage(const Image & img, const fs::path & path)
{
  cv::Mat mat;
  cv::eigen2cv(img, mat);
  mat.convertTo(mat, CV_8UC3);
  cv::imwrite(path.string() + ".png", mat);
}
void utils::saveDepth(const Eigen::MatrixXd & img, const fs::path & path)
{
  cv::Mat mat;
  cv::eigen2cv(img, mat);
  mat.convertTo(mat, CV_32F);
  cv::imwrite(path.string() + ".exr", mat);
}

void utils::writeTrajectory(const Trajectory & traj, const fs::path & path, bool writeCovariance)
{
  if (!fs::is_directory(path.parent_path())) {
    fs::create_directories(path.parent_path());
  }
  std::fstream algoFile;
  algoFile.open(path, std::ios_base::out);
  algoFile << "# Algorithm Trajectory\n";
  algoFile << "# file: " << path << "\n";
  algoFile << "# timestamp tx ty tz qx qy qz qw\n";
  if (!algoFile.is_open()) {
    std::runtime_error("Could not open file at: " + path.string());
  }

  for (const auto & pose : traj.poses()) {
    std::string ts = format("{}", pose.first);
    ts = format("{}.{}", ts.substr(0, 10), ts.substr(10));

    ts = ts.substr(0, ts.size() - 3);

    const auto t = pose.second->pose().translation();
    const auto q = pose.second->pose().unit_quaternion();
    algoFile << ts << " " << t.x() << " " << t.y() << " " << t.z() << " " << q.x() << " " << q.y()
             << " " << q.z() << " " << q.w();
    if (writeCovariance) {
      for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
          algoFile << " " << pose.second->cov()(i, j);
        }
      }
    }

    algoFile << "\n";
  }
}

}  // namespace pd::vslam
