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

#include <Eigen/Dense>
#include <iostream>
#include <map>
#include <opencv4/opencv2/calib3d.hpp>
#include <opencv4/opencv2/core/eigen.hpp>
#include <opencv4/opencv2/core/utility.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/rgbd.hpp>

#include "Exceptions.h"
#include "Frame.h"
#include "Point3D.h"
#include "algorithm.h"
namespace pd::vslam
{
std::uint64_t Frame::_idCtr = 0U;

Frame::Frame(
  const Image & intensity, Camera::ConstShPtr cam, const Timestamp & t, const Pose & pose)
: Frame(intensity, -1 * MatXd::Ones(intensity.rows(), intensity.cols()), cam, t, pose)
{
}
Eigen::Vector2d Frame::camera2image(const Eigen::Vector3d & pCamera, size_t level) const
{
  return _cam.at(level)->camera2image(pCamera);
}
Eigen::Vector3d Frame::image2camera(
  const Eigen::Vector2d & pImage, double depth, size_t level) const
{
  return _cam.at(level)->image2camera(pImage, depth);
}
Eigen::Vector2d Frame::world2image(const Eigen::Vector3d & pWorld, size_t level) const
{
  return camera2image(_pose.pose() * pWorld, level);
}
Eigen::Vector3d Frame::image2world(const Eigen::Vector2d & pImage, double depth, size_t level) const
{
  return _pose.pose().inverse() * image2camera(pImage, depth, level);
}
Feature2D::ConstShPtr Frame::observationOf(std::uint64_t pointId) const
{
  for (auto ft : _features) {
    if (ft->point() && ft->point()->id() == pointId) {
      return ft;
    }
  }
  return nullptr;
}

const MatXd & Frame::dIdx(size_t level) const
{
  if (level >= _dIdx.size()) {
    throw pd::Exception(
      "No dIdx available for level: " + std::to_string(level) +
      ". Available: " + std::to_string(_dIdx.size()));
  }
  return _dIdx[level];
}
const MatXd & Frame::dIdy(size_t level) const
{
  if (level >= _dIdy.size()) {
    throw pd::Exception(
      "No dIdy available for level: " + std::to_string(level) +
      ". Available: " + std::to_string(_dIdy.size()));
  }
  return _dIdy[level];
}

const MatXd & Frame::dZdx(size_t level) const
{
  if (level >= _dZdx.size()) {
    throw pd::Exception(
      "No dIdx available for level: " + std::to_string(level) +
      ". Available: " + std::to_string(_dZdx.size()));
  }
  return _dZdx[level];
}
const MatXd & Frame::dZdy(size_t level) const
{
  if (level >= _dZdy.size()) {
    throw pd::Exception(
      "No dIdy available for level: " + std::to_string(level) +
      ". Available: " + std::to_string(_dZdy.size()));
  }
  return _dZdy[level];
}

void Frame::addFeature(Feature2D::ShPtr ft)
{
  if (ft->frame() && ft->frame()->id() != _id) {
    throw pd::Exception(
      "Feature is alread associated with frame [" + std::to_string(ft->frame()->id()) + "]");
  }
  _features.push_back(ft);
}

void Frame::addFeatures(const std::vector<Feature2D::ShPtr> & features)
{
  _features.reserve(_features.size() + features.size());
  for (const auto & ft : features) {
    addFeature(ft);
  }
}
std::vector<Feature2D::ConstShPtr> Frame::features() const
{
  return std::vector<Feature2D::ConstShPtr>(_features.begin(), _features.end());
}
std::vector<Feature2D::ShPtr> Frame::featuresWithPoints()
{
  std::vector<Feature2D::ShPtr> fts;
  fts.reserve(_features.size());
  std::copy_if(_features.begin(), _features.end(), std::back_inserter(fts), [&](auto ft) {
    return ft->point();
  });
  return fts;
}
std::vector<Feature2D::ConstShPtr> Frame::featuresWithPoints() const
{
  std::vector<Feature2D::ConstShPtr> fts;
  fts.reserve(_features.size());
  std::copy_if(_features.begin(), _features.end(), std::back_inserter(fts), [&](auto ft) {
    return ft->point();
  });
  return fts;
}

void Frame::removeFeatures()
{
  for (auto ft : _features) {
    ft->frame() = nullptr;
    if (ft->point()) {
      ft->point()->removeFeature(ft);
    }
  }
  _features.clear();
}
void Frame::removeFeature(Feature2D::ShPtr ft)
{
  auto it = std::find(_features.begin(), _features.end(), ft);
  if (it == _features.end()) {
    throw pd::Exception(
      "Did not find feature: [" + std::to_string(ft->id()) + " ] in frame: [" +
      std::to_string(_id) + "]");
  }
  _features.erase(it);
  ft->frame() = nullptr;

  if (ft->point()) {
    ft->point()->removeFeature(ft);
  }
}

Frame::Frame(
  const Image & intensity, const MatXd & depth, Camera::ConstShPtr cam, const Timestamp & t,
  const Pose & pose)
: _id(_idCtr++), _intensity({intensity}), _cam({cam}), _t(t), _pose(pose), _depth({depth})
{
  if (
    intensity.cols() != depth.cols() ||
    std::abs(intensity.cols() / 2 - cam->principalPoint().x()) > 10 ||
    intensity.rows() != depth.rows()) {
    throw pd::Exception("Inconsistent camera parameters / image / depth dimensions detected.");
  }
}

std::vector<Vec3d> Frame::pcl(size_t level, bool removeInvalid) const
{
  if (level >= _pcl.size()) {
    throw pd::Exception(
      "No PCL available for level: " + std::to_string(level) +
      ". Available: " + std::to_string(_pcl.size()));
  }
  if (removeInvalid) {
    std::vector<Vec3d> pcl;
    pcl.reserve(_pcl.at(level).size());
    std::copy_if(_pcl.at(level).begin(), _pcl.at(level).end(), std::back_inserter(pcl), [](auto p) {
      return p.z() > 0 && std::isfinite(p.z());
    });
    return pcl;
  } else {
    return _pcl.at(level);
  }
}
std::vector<Vec3d> Frame::pclWorld(size_t level, bool removeInvalid) const
{
  if (level >= _pcl.size()) {
    throw pd::Exception(
      "No PCL available for level: " + std::to_string(level) +
      ". Available: " + std::to_string(_pcl.size()));
  }
  auto points = pcl(level, removeInvalid);
  std::transform(points.begin(), points.end(), points.begin(), [&](auto p) {
    return pose().pose().inverse() * p;
  });
  return points;
}

const Vec3d & Frame::p3d(int v, int u, size_t level) const
{
  if (level >= _pcl.size()) {
    throw pd::Exception(
      "No PCL available for level: " + std::to_string(level) +
      ". Available: " + std::to_string(_pcl.size()));
  }
  return _pcl.at(level)[v * width(level) + u];
}
Vec3d Frame::p3dWorld(int v, int u, size_t level) const
{
  if (level >= _pcl.size()) {
    throw pd::Exception(
      "No PCL available for level: " + std::to_string(level) +
      ". Available: " + std::to_string(_pcl.size()));
  }
  return pose().pose().inverse() * _pcl.at(level)[v * width() + u];
}

bool Frame::withinImage(const Vec2d & pImage, double border, size_t level) const
{
  return 0 + border < pImage.x() && pImage.x() < width(level) - border && 0 + border < pImage.y() &&
         pImage.y() < height(level) - border;
}

void Frame::computeIntensityDerivatives()
{
  _dIdx.resize(nLevels());
  _dIdy.resize(nLevels());

  // TODO(unknown): replace using custom implementation
  for (size_t i = 0; i < nLevels(); i++) {
#if true
    cv::Mat mat;
    cv::eigen2cv(intensity(i), mat);
    cv::Mat dIdx, dIdy;
    cv::Sobel(mat, dIdx, CV_16S, 1, 0, 3);
    cv::Sobel(mat, dIdy, CV_16S, 0, 1, 3);
    cv::cv2eigen(dIdx, _dIdx[i]);
    cv::cv2eigen(dIdy, _dIdy[i]);
#else
    _dIdx[i].resize(intensity(i).rows(), intensity(i).cols());
    for (int y = 0; y < intensity(i).rows(); ++y) {
      for (int x = 0; x < intensity(i).cols(); ++x) {
        int prev = std::max(x - 1, 0);
        int next = std::min<int>(x + 1, intensity(i).cols() - 1);
        _dIdx[i](y, x) = (double)(intensity(i)(y, next) - intensity(i)(y, prev)) * 0.5f;
      }
    }
    _dIdy[i].resize(intensity(i).rows(), intensity(i).cols());
    for (int y = 0; y < intensity(i).rows(); ++y) {
      for (int x = 0; x < intensity(i).cols(); ++x) {
        int prev = std::max(y - 1, 0);
        int next = std::min<int>(y + 1, intensity(i).rows() - 1);
        _dIdy[i](y, x) = (double)(intensity(i)(next, x) - intensity(i)(prev, x)) * 0.5f;
      }
    }
#endif
  }
}
void Frame::computeDepthDerivatives()
{
  _dZdx.resize(nLevels());
  _dZdy.resize(nLevels());

  // TODO(unknown): replace using custom implementation
  for (size_t i = 0; i < nLevels(); i++) {
    cv::Mat mat;
    cv::eigen2cv(depth(i), mat);
    cv::Mat dIdx, dIdy;
    cv::Sobel(mat, dIdx, CV_16S, 1, 0, 3);
    cv::Sobel(mat, dIdy, CV_16S, 0, 1, 3);
    cv::cv2eigen(dIdx, _dZdx[i]);
    cv::cv2eigen(dIdy, _dZdy[i]);
  }
}
void Frame::computeDerivatives()
{
  computeIntensityDerivatives();
  computeDepthDerivatives();
}

void Frame::computePcl()
{
  _pcl.resize(nLevels());

  auto depth2pcl = [](const DepthMap & d, Camera::ConstShPtr c) {
    std::vector<Vec3d> pcl(d.rows() * d.cols());
    for (int v = 0; v < d.rows(); v++) {
      for (int u = 0; u < d.cols(); u++) {
        if (std::isfinite(d(v, u)) && d(v, u) > 0.0) {
          pcl[v * d.cols() + u] = c->image2camera({u, v}, d(v, u));
        } else {
          pcl[v * d.cols() + u] = Eigen::Vector3d::Zero();
        }
      }
    }
    return pcl;
  };
  for (size_t i = 0; i < nLevels(); i++) {
    _pcl[i] = depth2pcl(depth(i), camera(i));
  }
}

void Frame::computeNormals()
{
  auto depth2normal = [](const DepthMap & depth) {
    std::vector<Vec3d> normals(depth.rows() * depth.cols());
    // TODO(unknown): replace using custom implementation
    cv::Mat mat;
    cv::eigen2cv(depth, mat);
    cv::Mat dZdxm, dZdym;
    cv::Sobel(mat, dZdxm, CV_16S, 1, 0, 3);
    cv::Sobel(mat, dZdym, CV_16S, 0, 1, 3);
    MatXd dZdx, dZdy;
    cv::cv2eigen(dZdxm, dZdx);
    cv::cv2eigen(dZdym, dZdy);
    for (int x = 1; x < depth.cols() - 1; ++x) {
      for (int y = 1; y < depth.rows() - 1; ++y) {
        Vec3d n(-dZdx(y, x), -dZdy(y, x), 1);
        normals[y * depth.cols() + x] = n / n.norm();
      }
    }
    return normals;
  };
  cv::Mat K;
  cv::eigen2cv(camera(0)->K(), K);
  cv::rgbd::RgbdNormals normalComputer(depth(0).rows(), depth(0).cols(), CV_64F, K);
  cv::Mat normals_cv;
  cv::Mat points(height(0), width(0), CV_64FC3);
  for (size_t y = 0; y < height(0); ++y) {
    for (size_t x = 0; x < width(0); ++x) {
      cv::Point3d p;
      const auto & pp = p3d(y, x);
      p.x = pp.x();
      p.y = pp.y();
      p.z = pp.z();
      points.at<cv::Point3d>(y, x) = p;
    }
  }
  normalComputer(points, normals_cv);
  std::vector<cv::Mat> normalsPyramid;
  cv::buildPyramid(normals_cv, normalsPyramid, nLevels());
  _normals.resize(nLevels());
  for (size_t i = 0; i < nLevels(); i++) {
    _normals[i].resize(height(i) * width(i));
    for (size_t y = 0; y < height(i); ++y) {
      for (size_t x = 0; x < width(i); ++x) {
        cv::Point3d p = normalsPyramid[i].at<cv::Point3d>(y, x);
        Vec3d n(p.x, p.y, p.z);
        _normals[i][y * width(i) + x] = n / n.norm();
      }
    }
  }
}

const Vec3d & Frame::normal(int v, int u, size_t level) const
{
  if (level >= _normals.size()) {
    throw pd::Exception(
      "No Normals available for level: " + std::to_string(level) +
      ". Available: " + std::to_string(_pcl.size()));
  }
  return _normals.at(level)[v * width(level) + u];
}

void Frame::computePyramid(size_t nLevels, double s)
{
  _intensity.resize(nLevels);
  _cam.resize(nLevels);

  // TODO(unknown): replace using custom implementation
  {
    cv::Mat mat;
    cv::eigen2cv(_intensity[0], mat);
    std::vector<cv::Mat> mats;
    cv::buildPyramid(mat, mats, nLevels - 1);
    for (size_t i = 0; i < mats.size(); i++) {
      cv::cv2eigen(mats[i], _intensity[i]);
      _cam[i] = Camera::resize(_cam[0], std::pow(s, i));
    }
  }
  _depth.resize(nLevels);

  for (size_t i = 1; i < nLevels; i++) {
    //DepthMap depthBlur =
    //  algorithm::medianBlur<double>(_depth[i - 1], 3, 3, [](double v) { return v <= 0.0; });
    // TODO(unknown): replace using custom implementation
    cv::Mat mat;
    cv::eigen2cv(depth(i - 1), mat);
    mat.convertTo(mat, CV_32F);
    cv::medianBlur(mat, mat, 3);
    MatXd depthBlur;
    cv::cv2eigen(mat, depthBlur);

    _depth[i] = algorithm::resize(depthBlur, s);
  }
}

}  // namespace pd::vslam
