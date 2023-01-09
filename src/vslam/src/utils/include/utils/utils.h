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

#ifndef VSLAM_UTILS_H__
#define VSLAM_UTILS_H__

#include <core/core.h>

#include <Eigen/Dense>
#include <filesystem>
#include <map>
#include <string>

#include "Exceptions.h"
#include "Log.h"
#include "tum.h"
#include "visuals.h"
namespace pd::vslam::utils
{
void throw_if_nan(const Eigen::MatrixXd & mat, const std::string & msg);
Image loadImage(
  const std::filesystem::path & path, int height = -1, int width = -1, bool grayscale = true);
Eigen::MatrixXd loadDepth(const std::filesystem::path & path, int height = -1, int width = -1);

/**
   * @brief Load trajectory from file (TUM RGBD Format)
   *
   * @param path filepath
   * @return std::map<Timestamp,SE3d>
   */
Trajectory::UnPtr loadTrajectory(const std::filesystem::path & path, bool invertPoses = false);

/**
   * @brief Write trajectory to txt file (TUM RGBD Format)
   *
   * @param traj
   * @param path
   * @param writeCovariance
   */
void writeTrajectory(
  const Trajectory & traj, const std::filesystem::path & path, bool writeCovariance = false);

void saveImage(const Image & img, const std::filesystem::path & path);
void saveDepth(const Eigen::MatrixXd & img, const std::filesystem::path & path);

/**
   * @brief Load eigen matrix from csv
   *
   * Usage:
   * Matrix3d A = load_csv<MatrixXd>("A.csv");
   * Matrix3d B = load_csv<Matrix3d>("B.csv");
   * Source: https://stackoverflow.com/questions/34247057/how-to-read-csv-file-and-assign-to-eigen-matrix
   *
   * @param path to csv
   * @return Matrix
   */
template <typename M>
M loadMatCsv(const std::filesystem::path & path, char delim = ',')
{
  std::ifstream indata;
  indata.open(path);
  std::string line;
  std::vector<double> values;
  uint rows = 0;
  while (std::getline(indata, line)) {
    std::stringstream lineStream(line);
    std::string cell;
    while (std::getline(lineStream, cell, delim)) {
      values.push_back(std::stod(cell));
    }
    ++rows;
  }
  return Eigen::Map<const Eigen::Matrix<
    typename M::Scalar, M::RowsAtCompileTime, M::ColsAtCompileTime, Eigen::RowMajor>>(
    values.data(), rows, values.size() / rows);
}
template <typename Derived>
std::string toCsv(const Eigen::DenseBase<Derived> & mat, const std::string & delim = ",")
{
  std::stringstream ss;
  for (int i = 0; i < mat.rows(); i++) {
    for (int j = 0; j < mat.cols(); j++) {
      if (i > 0 || j > 0) {
        ss << delim;
      }
      ss << mat(i, j);
    }
  }
  return ss.str();
}
}  // namespace pd::vslam::utils
#endif
