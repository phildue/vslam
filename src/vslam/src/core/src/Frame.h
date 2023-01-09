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

#ifndef VSLAM_FRAME_H__
#define VSLAM_FRAME_H__
#include <memory>

#include "Camera.h"
#include "Feature2D.h"
#include "Kernel2d.h"
#include "PoseWithCovariance.h"
#include "types.h"
namespace pd::vslam
{
class Frame
{
public:
  typedef std::shared_ptr<Frame> ShPtr;
  typedef std::shared_ptr<const Frame> ConstShPtr;
  typedef std::unique_ptr<Frame> UnPtr;
  typedef std::unique_ptr<const Frame> ConstUnPtr;
  typedef std::vector<ShPtr> VecShPtr;
  typedef std::vector<ConstShPtr> VecConstShPtr;

  Frame(
    const Image & rgb, const DepthMap & depth, Camera::ConstShPtr cam, const Timestamp & t = 0U,
    const PoseWithCovariance & pose = {});

  Frame(
    const Image & intensity, Camera::ConstShPtr cam, const Timestamp & t = 0U,
    const PoseWithCovariance & pose = {});
  virtual ~Frame() = default;
  std::uint64_t id() const { return _id; }

  const Image & intensity(size_t level = 0) const { return _intensity.at(level); }
  const MatXd & dIx(size_t level = 0) const;
  const MatXd & dIy(size_t level = 0) const;

  const Pose & pose() const { return _pose; }

  const Timestamp & t() const { return _t; }
  Camera::ConstShPtr camera(size_t level = 0) const { return _cam.at(level); }
  size_t width(size_t level = 0) const { return _intensity.at(level).cols(); }
  size_t height(size_t level = 0) const { return _intensity.at(level).rows(); }
  size_t nLevels() const { return _intensity.size(); }
  bool withinImage(const Vec2d & pImage, double border = 7, size_t level = 0) const;

  std::vector<Feature2D::ConstShPtr> features() const;
  std::vector<Feature2D::ConstShPtr> featuresWithPoints() const;
  std::vector<Feature2D::ShPtr> features() { return _features; };
  std::vector<Feature2D::ShPtr> featuresWithPoints();
  Feature2D::ConstShPtr observationOf(std::uint64_t pointId) const;

  Eigen::Vector2d camera2image(const Eigen::Vector3d & pCamera, size_t level = 0) const;
  Eigen::Vector3d image2camera(
    const Eigen::Vector2d & pImage, double depth = 1.0, size_t level = 0) const;
  Eigen::Vector2d world2image(const Eigen::Vector3d & pWorld, size_t level = 0) const;
  Eigen::Vector3d image2world(
    const Eigen::Vector2d & pImage, double depth = 1.0, size_t level = 0) const;

  void set(const PoseWithCovariance & pose) { _pose = pose; }
  void addFeature(Feature2D::ShPtr ft);
  void addFeatures(const std::vector<Feature2D::ShPtr> & ft);
  void removeFeatures();
  void removeFeature(Feature2D::ShPtr f);
  void computeDerivatives();
  void computePcl();
  void computePyramid(size_t nLevels, double scale = 0.5);

  const DepthMap & depth(size_t level = 0) const { return _depth.at(level); }
  const Vec3d & p3d(int v, int u, size_t level = 0) const;
  Vec3d p3dWorld(int v, int u, size_t level = 0) const;
  std::vector<Vec3d> pcl(size_t level = 0, bool removeInvalid = false) const;
  std::vector<Vec3d> pclWorld(size_t level = 0, bool removeInvalid = false) const;

private:
  const std::uint64_t _id;
  ImageVec _intensity;
  MatXdVec _dIx, _dIy;
  Camera::ConstShPtrVec _cam;
  Timestamp _t;
  Pose _pose;  //<< Pf = pose * Pw
  std::vector<Feature2D::ShPtr> _features;
  DepthMapVec _depth;
  std::vector<std::vector<Vec3d>> _pcl;
  static std::uint64_t _idCtr;
};

}  // namespace pd::vslam

#endif
