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

#include "Map.h"
namespace pd::vslam
{
Map::Map() : _frames(), _keyFrames(), _maxFrames(7), _maxKeyFrames(7) {}
void Map::insert(Frame::ShPtr frame, bool isKeyFrame)
{
  if (_frames.size() >= _maxFrames) {
    _frames.pop_back();
  }
  _frames.push_front(frame);

  if (isKeyFrame) {
    if (_keyFrames.size() >= _maxKeyFrames) {
      _keyFrames.pop_back();
    }
    _keyFrames.push_front(frame);
  }
}
std::vector<Point3D::ShPtr> Map::points()
{
  std::vector<Point3D::ShPtr> ps;
  ps.reserve(_points.size());
  for (const auto & id_p : _points) {
    ps.push_back(id_p.second);
  }
  return ps;
}
std::vector<Point3D::ConstShPtr> Map::points() const
{
  std::vector<Point3D::ShPtr> ps;
  ps.reserve(_points.size());
  for (const auto & id_p : _points) {
    ps.push_back(id_p.second);
  }
  return ps;
}
std::vector<Frame::ShPtr> Map::frames()
{
  std::vector<Frame::ShPtr> fs;
  fs.reserve(_frames.size());
  for (const auto & f : _frames) {
    fs.push_back(f);
  }
  return fs;
}
std::vector<Frame::ConstShPtr> Map::frames() const
{
  std::vector<Frame::ConstShPtr> fs;
  fs.reserve(_frames.size());
  for (const auto & f : _frames) {
    fs.push_back(f);
  }
  return fs;
}
std::vector<Frame::ShPtr> Map::keyFrames()
{
  std::vector<Frame::ShPtr> fs;
  fs.reserve(_keyFrames.size());
  for (const auto & f : _keyFrames) {
    fs.push_back(f);
  }
  return fs;
}
std::vector<Frame::ConstShPtr> Map::keyFrames() const
{
  std::vector<Frame::ConstShPtr> fs;
  fs.reserve(_keyFrames.size());
  for (const auto & f : _keyFrames) {
    fs.push_back(f);
  }
  return fs;
}
void Map::insert(Point3D::ShPtr point) { _points[point->id()] = point; }

void Map::insert(const std::vector<Point3D::ShPtr> & points)
{
  for (const auto & p : points) {
    _points[p->id()] = p;
  }
}
void Map::updateFrame(std::uint64_t id, const PoseWithCovariance & pose)
{
  for (auto f : _keyFrames) {
    if (f->id() == id) {
      f->set(pose);
      return;
    }
  }
  for (auto f : _frames) {
    if (f->id() == id) {
      f->set(pose);
      return;
    }
  }
  throw pd::Exception("Frame not part of map: " + std::to_string(id));
}
void Map::updateFrames(const std::map<std::uint64_t, PoseWithCovariance> & poses)
{
  for (auto & id_pose : poses) {
    updateFrame(id_pose.first, id_pose.second);
  }
}

void Map::updatePoints(const std::map<std::uint64_t, Vec3d> & points)
{
  for (auto id_p : points) {
    auto it = _points.find(id_p.first);
    if (it != _points.end()) {
      it->second->position() = id_p.second;
    } else {
      throw pd::Exception("Point is not part of map: " + std::to_string(id_p.first));
    }
  }
}

}  // namespace pd::vslam
