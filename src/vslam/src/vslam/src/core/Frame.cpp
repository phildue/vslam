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

#include "Frame.h"
#include "Point3D.h"

namespace vslam
{
std::uint64_t Frame::_idCtr = 0U;

Frame::Frame(
  const cv::Mat & intensity, Camera::ConstShPtr cam, const Timestamp & t, const Pose & pose)
: Frame(intensity, -1 * cv::Mat(cv::Size(intensity.rows, intensity.cols), CV_32F), cam, t, pose)
{
}
Vec2d Frame::camera2image(const Vec3d & pCamera, size_t level) const
{
  return _cam.at(level)->project(pCamera);
}
Vec3d Frame::image2camera(const Vec2d & pImage, double depth, size_t level) const
{
  return _cam.at(level)->reconstruct(pImage, depth);
}
Vec2d Frame::world2image(const Vec3d & pWorld, size_t level) const
{
  return camera2image(_pose.pose() * pWorld, level);
}
Vec3d Frame::image2world(const Vec2d & pImage, double depth, size_t level) const
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

const cv::Mat & Frame::dI(size_t level) const
{
  if (level >= _dI.size()) {
    throw std::runtime_error(
      "No dI available for level: " + std::to_string(level) +
      ". Available: " + std::to_string(_dI.size()));
  }
  return _dI[level];
}

const cv::Mat & Frame::dZ(size_t level) const
{
  if (level >= _dZ.size()) {
    throw std::runtime_error(
      "No dZ available for level: " + std::to_string(level) +
      ". Available: " + std::to_string(_dZ.size()));
  }
  return _dZ[level];
}

void Frame::addFeature(Feature2D::ShPtr ft)
{
  if (ft->frame() && ft->frame()->id() != _id) {
    throw std::runtime_error(
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
    throw std::runtime_error(
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
  const cv::Mat & intensity, const cv::Mat & depth, Camera::ConstShPtr cam, const Timestamp & t,
  const Pose & pose)
: _id(_idCtr++), _intensity({intensity}), _cam({cam}), _t(t), _pose(pose), _depth({depth})
{
  if (
    intensity.cols != depth.cols || std::abs(intensity.cols / 2 - cam->cx()) > 50 ||
    intensity.rows != depth.rows) {
    throw std::runtime_error(format(
      "Inconsistent camera parameters / image / depth dimensions detected: I:{}x{}, Z:{}x{}, "
      "pp:{},{}",
      intensity.cols, intensity.rows, depth.cols, depth.rows, cam->cx(), cam->cy()));
  }
}

std::vector<Vec3d> Frame::pcl(size_t level, bool removeInvalid) const
{
  if (level >= _pcl.size()) {
    throw std::runtime_error(
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
    throw std::runtime_error(
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
    throw std::runtime_error(
      "No PCL available for level: " + std::to_string(level) +
      ". Available: " + std::to_string(_pcl.size()));
  }
  return _pcl.at(level)[v * width(level) + u];
}
Vec3d Frame::p3dWorld(int v, int u, size_t level) const
{
  if (level >= _pcl.size()) {
    throw std::runtime_error(
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
  if (_dI.size() == nLevels()) return;

  _dI.resize(nLevels());

  for (size_t i = 0; i < nLevels(); i++) {
    cv::Mat dIdx, dIdy;
    cv::Sobel(intensity(i), dIdx, CV_32F, 1, 0, 3, 1. / 8.);
    cv::Sobel(intensity(i), dIdy, CV_32F, 0, 1, 3, 1. / 8.);
    cv::merge(std::vector<cv::Mat>({dIdx, dIdy}), _dI[i]);
  }
}
void Frame::computeDepthDerivatives()
{
  if (_dZ.size() == nLevels()) return;

  _dZ.resize(nLevels());
  for (size_t i = 0; i < nLevels(); i++) {
    {
      cv::Mat dZdx(cv::Size(width(i), height(i)), CV_32F),
        dZdy(cv::Size(width(i), height(i)), CV_32F);

      auto validZ = [](float z0, float z1) {
        return std::isfinite(z0) && z0 > 0 && std::isfinite(z1) && z1 > 0;
      };

      for (int y = 0; y < depth(i).rows; ++y) {
        for (int x = 0; x < depth(i).cols; ++x) {
          const int y0 = std::max(y - 1, 0);
          const int y1 = std::min(y + 1, depth(i).rows - 1);
          const int x0 = std::max(x - 1, 0);
          const int x1 = std::min(x + 1, depth(i).cols - 1);
          const float zyx0 = depth(i).at<float>(y, x0);
          const float zyx1 = depth(i).at<float>(y, x1);
          const float zy0x = depth(i).at<float>(y0, x);
          const float zy1x = depth(i).at<float>(y1, x);

          dZdx.at<float>(y, x) =
            validZ(zyx0, zyx1) ? (zyx1 - zyx0) * 0.5f : std::numeric_limits<float>::quiet_NaN();
          dZdy.at<float>(y, x) =
            validZ(zy0x, zy1x) ? (zy1x - zy0x) * 0.5f : std::numeric_limits<float>::quiet_NaN();
        }
      }
      cv::merge(std::vector<cv::Mat>({dZdx, dZdy}), _dZ[i]);
    }
  }
}
void Frame::computeDerivatives()
{
  computeIntensityDerivatives();
  computeDepthDerivatives();
}

void Frame::computePcl()
{
  if (_pcl.size() == nLevels()) return;
  _pcl.resize(nLevels());

  auto depth2pcl = [](const cv::Mat & d, Camera::ConstShPtr c) {
    std::vector<Vec3d> pcl(d.rows * d.cols);
    for (int v = 0; v < d.rows; v++) {
      for (int u = 0; u < d.cols; u++) {
        if (std::isfinite(d.at<float>(v, u)) && d.at<float>(v, u) > 0.0) {
          pcl[v * d.cols + u] = c->reconstruct({u, v}, d.at<float>(v, u));
        } else {
          pcl[v * d.cols + u] = Vec3d::Zero();
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
  if (_normals.size() == nLevels()) return;
  _normals.resize(nLevels());
  cv::Mat K(3, 3, CV_32F);
  cv::eigen2cv(camera(0)->K(), K);
  cv::rgbd::RgbdNormals normalComputer(depth(0).rows, depth(0).cols, CV_64F, K);
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
    throw std::runtime_error(
      "No Normals available for level: " + std::to_string(level) +
      ". Available: " + std::to_string(_pcl.size()));
  }
  return _normals.at(level)[v * width(level) + u];
}

void Frame::computePyramid(size_t nLevels)
{
  if (_intensity.size() == nLevels) return;
  _intensity.resize(nLevels);
  _cam.resize(nLevels);
  _depth.resize(nLevels);

  auto pyrDownZ = [](const cv::Mat & Z) -> cv::Mat {
    cv::Mat out(cv::Size(Z.cols / 2, Z.rows / 2), CV_32F);
    for (int y = 0; y < out.rows; ++y) {
      for (int x = 0; x < out.cols; ++x) {
        out.at<float>(y, x) = Z.at<float>(y * 2, x * 2);
      }
    }
    return out;
  };

  for (size_t i = 1; i < nLevels; i++) {
    cv::pyrDown(_intensity[i - 1], _intensity[i]);
    _cam[i] = Camera::resize(_cam[i - 1], 0.5);
    _depth[i] = pyrDownZ(_depth[i - 1]);
  }
}

}  // namespace vslam
