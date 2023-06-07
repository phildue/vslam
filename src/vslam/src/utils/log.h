#ifndef VSLAM_LOG_H__
#define VSLAM_LOG_H__
#include <Eigen/Dense>
#include <filesystem>

#include "easylogging++.h"
namespace vslam::log
{
void initialize(const std::string & logfile, bool clean = false);

void create(const std::string & name);

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

}  // namespace vslam::log
#endif