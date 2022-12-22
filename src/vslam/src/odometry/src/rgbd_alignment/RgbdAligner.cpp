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

#include <fmt/core.h>
#include <fmt/ostream.h>

#include <Eigen/Dense>
#include <opencv2/rgbd.hpp>
#include <opencv4/opencv2/core/eigen.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/opencv.hpp>

#include "RgbdAligner.h"
#include "RgbdAlignmentOpenCv.h"
#include "utils/utils.h"

using fmt::format;
using fmt::print;
template <typename T>
struct fmt::formatter<T, std::enable_if_t<std::is_base_of_v<Eigen::DenseBase<T>, T>, char>>
: ostream_formatter
{
};
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv4/opencv2/core/eigen.hpp>

#define LOG_ODOM(level) CLOG(level, "odometry")
using namespace pd::vslam::least_squares;
namespace pd::vslam
{
class OverlayConstraints : public vis::Drawable
{
public:
  OverlayConstraints(
    Frame::ConstShPtr fRef, Frame::ConstShPtr fTo, const std::vector<Eigen::Vector2i> & ipRef,
    const std::vector<Eigen::Vector2i> & ipTo, int level, const std::vector<double> & weights)
  : _fRef(fRef), _fTo(fTo), _ipRef(ipRef), _ipTo(ipTo), _level(level), _weights(weights)
  {
  }
  cv::Mat draw() const override
  {
    cv::Mat matRef, matTo;
    cv::eigen2cv(_fRef->intensity(_level), matRef);
    cv::eigen2cv(_fTo->intensity(_level), matTo);

    cv::cvtColor(matRef, matRef, cv::COLOR_GRAY2BGR);
    cv::cvtColor(matTo, matTo, cv::COLOR_GRAY2BGR);
    constexpr int radius = 3;
    for (size_t i = 0U; i < _ipRef.size(); i++) {
      if (_weights[i] > 0.0) {
        cv::Scalar color(
          (double)std::rand() / RAND_MAX * 255, (double)std::rand() / RAND_MAX * 255,
          (double)std::rand() / RAND_MAX * 255);
        cv::circle(matRef, cv::Point(_ipRef[i].x(), _ipRef[i].y()), radius, color);
        cv::circle(matTo, cv::Point(_ipTo[i].x(), _ipTo[i].y()), radius, color);
      }
    }
    cv::Mat mat;
    cv::hconcat(std::vector<cv::Mat>({matRef, matTo}), mat);
    return mat;
  }

private:
  Frame::ConstShPtr _fRef, _fTo;
  std::vector<Eigen::Vector2i> _ipRef, _ipTo;
  int _level;
  std::vector<double> _weights;
};

class LeastSquaresProblem : public least_squares::Problem
{
public:
  LeastSquaresProblem(
    const SE3d & T_SE3, Frame::ConstShPtr fRef, Frame::ConstShPtr fTo,
    const std::vector<Eigen::Vector2i> & interestPoints, int level)
  : Problem(6),
    _T_SE3(T_SE3),
    _fRef(fRef),
    _fTo(fTo),
    _interestPoints(interestPoints),
    _level(level)
  {
    _jacobians.resize(_fTo->height(level) * _fTo->width(level));
    for (size_t i = 0U; i < interestPoints.size(); i++) {
      const int u1 = interestPoints[i](0), v1 = interestPoints[i](1);

      _jacobians[v1 * _fRef->width(_level) + u1] = computeJ_f_p(
        _fTo->dIx(_level)(v1, u1) * 1. / 8., _fTo->dIy(_level)(v1, u1) * 1. / 8.,
        _fTo->p3d(v1, u1, _level));  //TODO move sobel scale to gradient computation
    }
  }
  void updateX(const Eigen::VectorXd & dx) override { _T_SE3 = SE3d::exp(dx) * _T_SE3; }

  void setX(const Eigen::VectorXd & x) override { _T_SE3 = SE3d::exp(x); }

  Eigen::VectorXd x() const override { return _T_SE3.log(); }

  std::vector<Eigen::Vector4i> computeCorrespondences() const
  {
    std::vector<Eigen::Matrix<int, 5, 1>> correspondences(
      _fTo->depth(_level).rows() * _fTo->depth(_level).cols(),
      Eigen::Matrix<int, 5, 1>::Constant(-1));
    for (size_t i = 0; i < _interestPoints.size(); i++) {
      const double d1 = _fTo->depth(_level)(_interestPoints[i].y(), _interestPoints[i].x());
      Eigen::Vector3d uv1;
      uv1 << _interestPoints[i].cast<double>(), 1;
      Eigen::Vector3d pCcs1 = d1 * _fTo->camera(_level)->K().inverse() * uv1;
      Eigen::Vector3d pCcs0 = _T_SE3.inverse() * pCcs1;
      const double transformed_d1 = pCcs0.z();
      if (transformed_d1 > 0) {
        Eigen::Vector3d uv0 = 1. / transformed_d1 * _fTo->camera(_level)->K() * pCcs0;
        const int u0 = std::round(uv0.x());
        const int v0 = std::round(uv0.y());
        if (
          0 <= u0 && u0 < _fRef->depth(_level).cols() && 0 <= v0 &&
          v0 < _fRef->depth(_level).rows()) {
          const double d0 = _fRef->depth(_level)(v0, u0);
          if (std::abs(transformed_d1 - d0) < _maxDepthDiff && _minDepth < d0 && d0 < _maxDepth) {
            Eigen::Matrix<int, 5, 1> c;
            c << u0, v0, _interestPoints[i], transformed_d1;
            if (correspondences[v0 * _fTo->depth(_level).cols() + u0](0) == -1) {
              correspondences[v0 * _fTo->depth(_level).cols() + u0] = c;
            } else if (transformed_d1 < correspondences[v0 * _fTo->depth(_level).cols() + u0](4)) {
              correspondences[v0 * _fTo->depth(_level).cols() + u0] = c;
            }
          }
        }
      }
    }
    std::vector<Eigen::Vector4i> cs;
    cs.reserve(_interestPoints.size());
    for (auto c : correspondences) {
      if (c(0) > -1) {
        cs.push_back({c(0), c(1), c(2), c(3)});
      }
    }
    return cs;
  }
  Vec6d computeJ_f_p(double dIdx, double dIdy, const Vec3d & p) const
  {
    double invz = 1. / p.z(), v0 = dIdx * _fTo->camera(_level)->fx() * invz,
           v1 = dIdy * _fTo->camera(_level)->fy() * invz, v2 = -(v0 * p.x() + v1 * p.y()) * invz;
    Vec6d J = Vec6d::Zero();
    J[0] = v0;
    J[1] = v1;
    J[2] = v2;
    J[3] = -p.z() * v1 + p.y() * v2;
    J[4] = p.z() * v0 - p.x() * v2;
    J[5] = -p.y() * v0 + p.x() * v1;

    return J;
  }

  least_squares::NormalEquations::ConstShPtr computeNormalEquations() override
  {
    auto correspondences = computeCorrespondences();
    VecXd r = VecXd::Zero(correspondences.size());
    MatXd J = MatXd::Zero(correspondences.size(), 6);
    double sigma = 0.0;
    double nTotal = 0.0;
    for (size_t i = 0; i < correspondences.size(); i++) {
      const int u0 = correspondences[i](0), v0 = correspondences[i](1);
      const int u1 = correspondences[i](2), v1 = correspondences[i](3);

      // r_i = IWxp - T
      const double ri = static_cast<int>(_fRef->intensity(_level)(v0, u0)) -
                        static_cast<int>(_fTo->intensity(_level)(v1, u1));
      r(i) = ri;
      J.row(i) = _jacobians[v1 * _fTo->width(_level) + u1];
      sigma += ri * ri;
      nTotal += 1.0;
    }

    sigma = std::sqrt(sigma / nTotal);

    VecXd w = VecXd::Zero(correspondences.size());
    for (size_t i = 0; i < correspondences.size(); i++) {
      double wi = sigma + std::abs(r[i]);
      wi = wi > DBL_EPSILON ? 1. / wi : 1.;
      w(i) = wi;
    }
    return std::make_unique<least_squares::NormalEquations>(J, r, w);
  }

private:
  SE3d _T_SE3;  // p_to = T_SE3 * p_ref
  std::vector<Vec6d> _jacobians;
  Frame::ConstShPtr _fRef, _fTo;
  std::vector<Eigen::Vector2i> _interestPoints;
  int _level;
  const double _maxDepthDiff = 0.07;
  const double _minDepth = 0.0;
  const double _maxDepth = 4.0;
};

RGBDAligner::RGBDAligner(double minGradient, Solver::ShPtr solver, Loss::ShPtr loss)
: _minGradient(minGradient), _loss(loss), _solver(solver)
{
  std::vector<double> minGradients;
  for (int i = 0; i < 4; i++) {
    minGradients.push_back(minGradient);
  }

  _interestPointSelection =
    std::make_unique<SE3Alignment::InterestPointSelection>(minGradients, 0.0, 4.0, 1.0);

  Log::get("odometry");
  LOG_IMG("ImageWarped");
  LOG_IMG("Residual");
  LOG_IMG("Weights");
  LOG_IMG("Image");
  LOG_IMG("Template");
  LOG_IMG("Depth");
}

PoseWithCovariance::UnPtr RGBDAligner::align(Frame::ConstShPtr from, Frame::ConstShPtr to) const
{
  Vec6d twist = Vec6d::Zero();
  Mat<double, 6, 6> covariance = Mat<double, 6, 6>::Identity();

  for (int level = from->nLevels() - 1; level >= 0; level--) {
    TIMED_SCOPE(timerI, "align at level ( " + std::to_string(level) + " )");
    LOG_ODOM(INFO) << "Aligning " << level << " image size: [" << from->width(level) << ","
                   << from->height(level) << "].";

    LOG_IMG("Image") << to->intensity(level);
    LOG_IMG("Template") << from->intensity(level);
    LOG_IMG("Depth") << from->depth(level);

    std::vector<Eigen::Vector2i> interestPoints = _interestPointSelection->select(to, from, level);
    auto p =
      std::make_shared<LeastSquaresProblem>(SE3d::exp(twist), from, to, interestPoints, level);
    auto r = _solver->solve(p);
    if (r->hasSolution()) {
      twist = r->solution();
      covariance = r->covariance();
      //TODO threshold on maximum speed ?
    }
    LOG_ODOM(INFO) << format(
      "Aligned with {} iterations: {} +- {}", r->iteration, twist.transpose(),
      covariance.diagonal().transpose());
  }
  //TODO how to convert the covariance to the covariance of the absolute pose?
  return std::make_unique<PoseWithCovariance>(SE3d::exp(twist) * from->pose().SE3(), covariance);
}

}  // namespace pd::vslam
