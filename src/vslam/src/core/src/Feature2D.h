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

#ifndef VSLAM_FEATURE2D_H
#define VSLAM_FEATURE2D_H

#include <Eigen/Dense>
#include <memory>

#include "types.h"
namespace pd::vslam
{
class Point3D;
class Frame;

enum class DescriptorType { ORB = 0, BRISK, NONE };
class Descriptor
{
public:
  Descriptor(const VecXd & vec, DescriptorType type) : _vec(vec), _type(type) {}
  const VecXd & vec() const { return _vec; }
  DescriptorType type() const { return _type; }

private:
  const VecXd _vec;
  const DescriptorType _type;
};

class Feature2D
{
public:
  typedef std::shared_ptr<Feature2D> ShPtr;
  typedef std::shared_ptr<const Feature2D> ConstShPtr;
  typedef std::unique_ptr<Feature2D> UnPtr;
  typedef std::unique_ptr<const Feature2D> ConstUnPtr;
  typedef std::vector<ConstShPtr> VecConstShPtr;
  typedef std::vector<ShPtr> VecShPtr;

  Feature2D(
    const Vec2d & position, std::shared_ptr<Frame> frame = nullptr, size_t level = 0U,
    double response = 0.0,
    const Descriptor & descriptor = Descriptor(VecXd::Zero(2), DescriptorType::NONE),
    std::shared_ptr<Point3D> p3d = nullptr);

  std::shared_ptr<const Point3D> point() const { return _point; }
  std::shared_ptr<Point3D> & point() { return _point; }
  const Vec2d & position() const { return _position; }
  std::shared_ptr<Frame> frame() { return _frame; }
  std::shared_ptr<const Frame> frame() const { return _frame; }
  const VecXd & descriptor() const { return _descriptor.vec(); }
  DescriptorType descriptorType() const { return _descriptor.type(); }

  size_t level() const { return _level; }
  double response() const { return _response; }
  const std::uint64_t & id() const { return _id; }

private:
  const Vec2d _position;
  const std::shared_ptr<Frame> _frame;
  const size_t _level;
  const double _response;
  const Descriptor _descriptor;
  std::shared_ptr<Point3D> _point;
  const std::uint64_t _id;
  static std::uint64_t _idCtr;
};

}  // namespace pd::vslam

#endif  //VSLAM_FEATURE2D_H
