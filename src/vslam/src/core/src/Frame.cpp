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
#include <opencv4/opencv2/calib3d.hpp>
#include <opencv4/opencv2/core/eigen.hpp>
#include <opencv4/opencv2/core/utility.hpp>
#include <opencv4/opencv2/imgproc.hpp>

#include "Exceptions.h"
#include "Frame.h"
#include "Point3D.h"
#include "algorithm.h"

#define USE_OPENCV
namespace pd::vslam
{
std::uint64_t Frame::_idCtr = 0U;

Frame::Frame(
  const Image & intensity, Camera::ConstShPtr cam, size_t nLevels, const Timestamp & t,
  const PoseWithCovariance & pose)
: Frame(intensity, -1 * MatXd::Ones(intensity.rows(), intensity.cols()), cam, nLevels, t, pose)
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
  for (const auto & ft : _features) {
    if (ft->point() && ft->point()->id() == pointId) {
      return ft;
    }
  }
  return nullptr;
}

const MatXd & Frame::dIx(size_t level) const
{
  if (level >= _dIx.size()) {
    throw pd::Exception(
      "No dIdx available for level: " + std::to_string(level) +
      ". Available: " + std::to_string(_dIx.size()));
  }
  return _dIx[level];
}
const MatXd & Frame::dIy(size_t level) const
{
  if (level >= _dIy.size()) {
    throw pd::Exception(
      "No dIdy available for level: " + std::to_string(level) +
      ". Available: " + std::to_string(_dIy.size()));
  }
  return _dIy[level];
}

void Frame::addFeature(Feature2D::ShPtr ft) { _features.push_back(ft); }

void Frame::addFeatures(const std::vector<Feature2D::ShPtr> & features)
{
  _features.reserve(_features.size() + features.size());
  for (const auto & ft : features) {
    _features.push_back(ft);
  }
}
std::vector<Feature2D::ConstShPtr> Frame::features() const
{
  std::vector<Feature2D::ConstShPtr> features;
  features.reserve(_features.size());
  std::for_each(_features.begin(), _features.end(), [&](auto ft) { features.push_back(ft); });
  return features;
}

void Frame::removeFeatures()
{
  for (const auto & ft : _features) {
    ft->frame() = nullptr;
    if (ft->point()) {
      ft->point()->removeFeature(ft);
      ft->point() = nullptr;
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

Frame::~Frame() { removeFeatures(); }

Frame::Frame(
  const Image & intensity, const MatXd & depth, Camera::ConstShPtr cam, size_t nLevels,
  const Timestamp & t, const PoseWithCovariance & pose)
: _id(_idCtr++), _t(t), _pose(pose)
{
  _intensity.resize(nLevels);
  _cam.resize(nLevels);
  const double s = 0.5;

  // TODO(unknown): replace using custom implementation
  cv::Mat mat;
  cv::eigen2cv(intensity, mat);
  std::vector<cv::Mat> mats;
  cv::buildPyramid(mat, mats, nLevels - 1);
  for (size_t i = 0; i < mats.size(); i++) {
    cv::cv2eigen(mats[i], _intensity[i]);
    _cam[i] = Camera::resize(cam, std::pow(s, i));
  }
  _depth.resize(nLevels);
  _depth[0] = depth;
  for (size_t i = 1; i < nLevels; i++) {
    DepthMap depthBlur =
      algorithm::medianBlur<double>(_depth[i - 1], 3, 3, [](double v) { return v <= 0.0; });
    _depth[i] = algorithm::resize(depthBlur, s);
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

void Frame::computeDerivatives()
{
  _dIx.resize(nLevels());
  _dIy.resize(nLevels());
  const double s = 0.5;

  // TODO(unknown): replace using custom implementation
  for (size_t i = 0; i < nLevels(); i++) {
    cv::Mat mat;
    cv::eigen2cv(intensity(i), mat);
    cv::Mat mati_blur;
    cv::GaussianBlur(mat, mati_blur, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);
    cv::Mat dIdx, dIdy;
    cv::Sobel(mati_blur, dIdx, CV_16S, 1, 0, 3);
    cv::Sobel(mati_blur, dIdy, CV_16S, 0, 1, 3);
    cv::cv2eigen(dIdx, _dIx[i]);
    cv::cv2eigen(dIdy, _dIy[i]);
  }
}
void Frame::computePcl()
{
  _pcl.resize(nLevels());

  auto depth2pcl = [](const DepthMap & d, Camera::ConstShPtr c) {
    std::vector<Vec3d> pcl(d.rows() * d.cols());
    for (int v = 0; v < d.rows(); v++) {
      for (int u = 0; u < d.cols(); u++) {
        if (std::isfinite(d(v, u)) && d(v, u) > 0.0) {
          pcl[v * d.cols() + u] = c->image2camera({u + 0.5, v + 0.5}, d(v, u));
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

}  // namespace pd::vslam
