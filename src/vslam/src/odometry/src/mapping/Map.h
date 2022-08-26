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

#ifndef VSLAM_MAP_H__
#define VSLAM_MAP_H__
#include <deque>

#include "core/core.h"
namespace pd::vslam
{
class Map
{
public:
  typedef std::shared_ptr<Map> ShPtr;
  typedef std::unique_ptr<Map> UnPtr;
  typedef std::shared_ptr<const Map> ConstShPtr;
  typedef std::unique_ptr<const Map> ConstUnPtr;

  Map();

  virtual void insert(Frame::ShPtr frame, bool isKeyFrame);
  virtual void insert(Point3D::ShPtr point);
  virtual void insert(const std::vector<Point3D::ShPtr> & points);

  void updateFrame(std::uint64_t, const PoseWithCovariance & pose);
  void updateFrames(const std::map<std::uint64_t, PoseWithCovariance> & poses);
  void updatePoints(const std::map<std::uint64_t, Vec3d> & points);

  Frame::ConstShPtr lastKf(size_t idx = 0) const
  {
    return _keyFrames.size() <= idx ? nullptr : _keyFrames.at(idx);
  }
  Frame::ConstShPtr lastFrame(size_t idx = 0) const
  {
    return _frames.size() <= idx ? nullptr : _frames.at(idx);
  }

  std::vector<Frame::ShPtr> keyFrames();
  std::vector<Frame::ShPtr> frames();
  std::vector<Frame::ConstShPtr> keyFrames() const;
  std::vector<Frame::ConstShPtr> frames() const;

  std::vector<Point3D::ShPtr> points();
  std::vector<Point3D::ConstShPtr> points() const;

private:
  std::deque<Frame::ShPtr> _frames;
  std::deque<Frame::ShPtr> _keyFrames;
  std::map<std::uint64_t, Point3D::ShPtr> _points;
  const size_t _maxFrames, _maxKeyFrames;
};

}  // namespace pd::vslam
#endif  // VSLAM_MAP_H__
