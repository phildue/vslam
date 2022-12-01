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
// Created by phil on 02.07.21.
//
#include "Exceptions.h"
#include "Kernel2d.h"
#include "algorithm.h"
#include "macros.h"
namespace pd::vslam
{
namespace algorithm
{
double rmse(const Eigen::MatrixXi & patch1, const Eigen::MatrixXi & patch2)
{
  if (patch1.rows() != patch2.rows() || patch1.cols() != patch2.cols()) {
    throw pd::Exception("rmse:: Patches have unequal dimensions!");
  }

  double sum = 0.0;
  for (int i = 0; i < patch1.rows(); i++) {
    for (int j = 0; j < patch2.cols(); j++) {
      sum += std::pow(patch1(i, j) - patch2(i, j), 2);
    }
  }
  return std::sqrt(sum / (patch1.rows() * patch1.cols()));
}

double sad(const Eigen::MatrixXi & patch1, const Eigen::MatrixXi & patch2)
{
  if (patch1.rows() != patch2.rows() || patch1.cols() != patch2.cols()) {
    throw pd::Exception("sad:: Patches have unequal dimensions!");
  }

  double sum = 0.0;
  for (int i = 0; i < patch1.rows(); i++) {
    for (int j = 0; j < patch2.cols(); j++) {
      sum += std::abs(patch1(i, j) - patch2(i, j));
    }
  }
  return sum;
}

Image resize(const Image & mat, double scale) { return resize<std::uint8_t>(mat, scale); }

Eigen::MatrixXd resize(const Eigen::MatrixXd & mat, double scale)
{
  return resize<double>(mat, scale);
}

Image gradient(const Image & image)
{
  const auto ix = gradX(image);
  const auto iy = gradY(image);
  const auto grad = ix.array().pow(2) + iy.array().pow(2);
  return grad.array().sqrt().cast<std::uint8_t>();
}

Eigen::MatrixXi gradX(const Image & image)
{
  return conv2d(image.cast<double>(), Kernel2d<double>::scharrX()).cast<int>();
}

Eigen::MatrixXi gradY(const Image & image)
{
  return conv2d(image.cast<double>(), Kernel2d<double>::scharrY()).cast<int>();
}

Sophus::SE3d computeRelativeTransform(const Sophus::SE3d & t0, const Sophus::SE3d & t1)
{
  return t1 * t0.inverse();
}
Eigen::MatrixXd normalize(const Eigen::MatrixXd & mat)
{
  return normalize(mat, mat.minCoeff(), mat.maxCoeff());
}
Eigen::MatrixXd normalize(const Eigen::MatrixXd & mat, double min, double max)
{
  Eigen::MatrixXd matImage = mat;
  matImage.array() -= min;
  matImage /= (max - min);
  return matImage;
}

double median(const Eigen::VectorXd & d, bool isSorted)
{
  // TODO(unknown): do this without copy?
  std::vector<double> r;
  r.reserve(d.rows());
  for (int i = 0; i < d.rows(); i++) {
    r.push_back(d(i));
  }
  return median(r, isSorted);
}

double median(const std::vector<double> & v, bool isSorted)
{
  const int n = v.size();
  std::vector<double> vv = v;
  if (isSorted) {
    std::sort(vv.begin(), vv.end());
  }
  if (n % 2 == 0) {
    return (vv[n / 2 - 1] + vv[n / 2 + 1]) / 2;
  } else {
    return vv[n / 2];
  }
}

Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> conv2d(
  const Eigen::Matrix<double, -1, -1> & mat, const Eigen::Matrix<double, -1, -1> & kernel)
{
  typedef int Idx;
  // TODO(unknown): is this the most efficient way? add padding
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> res(mat.rows(), mat.cols());
  res.setZero();
  const Idx kX_2 = static_cast<Idx>(std::floor(static_cast<double>(kernel.cols()) / 2.0));
  const Idx kY_2 = static_cast<Idx>(std::floor(static_cast<double>(kernel.rows()) / 2.0));
  for (Idx i = kY_2; i < res.rows() - kY_2; i++) {
    for (Idx j = kX_2; j < res.cols() - kX_2; j++) {
      double sum = 0.0;
      double norm = 0.0;
      for (Idx ki = 0; ki < kernel.rows(); ki++) {
        for (Idx kj = 0; kj < kernel.cols(); kj++) {
          Idx idxY = i - kY_2 + ki;
          Idx idxX = j - kX_2 + kj;
          double kv = kernel(ki, kj);
          double mv = mat(idxY, idxX);
          sum += kv * mv;
          norm += std::abs(kv);
        }
      }
      res(i, j) = (sum / norm);
    }
  }
  return res;
}
MatXd computeF(const Mat3d & Kref, const Sophus::SE3d & Rt, const Mat3d & Kcur)
{
  const Vec3d t = Rt.translation();
  const Mat3d R = Rt.rotationMatrix();
  Mat3d tx;
  tx << 0, -t.z(), t.y(), t.x(), 0, t.z(), -t.y(), t.x(), 0;
  const Mat3d E = tx * R;
  return Kcur.inverse().transpose() * E * Kref.inverse();
}
MatXd computeF(Frame::ConstShPtr frameRef, Frame::ConstShPtr frameCur)
{
  const SE3d Rt = computeRelativeTransform(frameRef->pose().pose(), frameCur->pose().pose());
  const Vec3d t = Rt.translation();
  const Mat3d R = Rt.rotationMatrix();
  Mat3d tx;
  tx << 0, -t.z(), t.y(), t.x(), 0, t.z(), -t.y(), t.x(), 0;
  const Mat3d E = tx * R;
  return frameCur->camera()->Kinv().transpose() * E * frameRef->camera()->Kinv();
}
}  // namespace algorithm
namespace transforms
{
Eigen::MatrixXd createdTransformMatrix2D(double x, double y, double angle)
{
  Eigen::Rotation2Dd rot(angle);
  Eigen::Matrix2d r = rot.toRotationMatrix();
  Eigen::Matrix3d m;
  m << r(0, 0), r(0, 1), x, r(1, 0), r(1, 1), y, 0, 0, 1;
  return m;
}

double deg2rad(double deg) { return deg / 180 * M_PI; }
double rad2deg(double rad) { return rad / M_PI * 180.0; }

Eigen::Quaterniond euler2quaternion(double rx, double ry, double rz)
{
  Eigen::AngleAxisd rxa(rx, Eigen::Vector3d::UnitX());
  Eigen::AngleAxisd rya(ry, Eigen::Vector3d::UnitY());
  Eigen::AngleAxisd rza(rz, Eigen::Vector3d::UnitZ());

  Eigen::Quaterniond q = rza * rya * rxa;
  q.normalize();
  return q;
}

}  // namespace transforms

namespace random
{
static std::default_random_engine eng(0);

template <typename T = double>
T U(T min, T max)
{
  std::uniform_real_distribution<double> distr(min, max);
  return static_cast<T>(distr(eng));
}
double U(double min, double max) { return U<double>(min, max); }
uint64_t U(uint64_t min, uint64_t max) { return U<uint64_t>(min, max); }

int sign() { return U(-1, 1) > 0 ? 1 : -1; }
Eigen::VectorXd N(const Eigen::MatrixXd & cov)
{
  std::normal_distribution<> dist;
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(cov);
  auto transform = eigenSolver.eigenvectors() * eigenSolver.eigenvalues().cwiseSqrt().asDiagonal();

  return transform *
         Eigen::VectorXd{cov.cols()}.unaryExpr([&](auto UNUSED(x)) { return dist(eng); });
}

}  // namespace random
namespace time
{
std::chrono::time_point<std::chrono::high_resolution_clock> to_time_point(Timestamp t)
{
  auto epoch = std::chrono::time_point<std::chrono::high_resolution_clock>();
  auto duration = std::chrono::nanoseconds(t);
  return epoch + duration;
}
}  // namespace time
}  // namespace pd::vslam
