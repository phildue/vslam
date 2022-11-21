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
// Created by phil on 30.06.21.
//

#ifndef VSLAM_POINT_H
#define VSLAM_POINT_H

#include <Eigen/Dense>
#include <memory>
#include <vector>

namespace pd::vslam
{
class Feature2D;

class Point3D
{
public:
  typedef std::shared_ptr<Point3D> ShPtr;
  typedef std::shared_ptr<const Point3D> ConstShPtr;
  typedef std::vector<ShPtr> VecShPtr;
  typedef std::vector<ConstShPtr> VecConstShPtr;

  Point3D(const Eigen::Vector3d & position, std::shared_ptr<Feature2D> ft);
  Point3D(
    const Eigen::Vector3d & position, const std::vector<std::shared_ptr<Feature2D>> & features);
  void addFeature(std::shared_ptr<Feature2D> ft);
  void removeFeatures();
  void removeFeature(std::shared_ptr<Feature2D> f);

  const Eigen::Vector3d & position() const { return _position; }
  Eigen::Vector3d & position() { return _position; }

  std::vector<std::shared_ptr<Feature2D>> features() { return _features; }
  std::vector<std::shared_ptr<const Feature2D>> features() const
  {
    return std::vector<std::shared_ptr<const Feature2D>>(_features.begin(), _features.end());
  }
  std::uint64_t id() const { return _id; }

private:
  const std::uint64_t _id;
  Eigen::Vector3d _position;
  std::vector<std::shared_ptr<Feature2D>> _features;
  static std::uint64_t _idCtr;
};

}  // namespace pd::vslam
#endif  //VSLAM_POINT_H
