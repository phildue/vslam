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

#include <list>

#include "Map.h"
#include "utils/utils.h"
#define LOG_MAPPING(level) CLOG(level, "mapping")

namespace pd::vslam
{
Map::Map(size_t nKeyFrames, size_t nFrames)
: _frames(), _keyFrames(), _maxFrames(nFrames), _maxKeyFrames(nKeyFrames)
{
  Log::get("mapping");
}
void Map::insert(Frame::ShPtr frame, bool isKeyFrame)
{
  LOG_MAPPING(DEBUG) << "Inserting new frame [" << frame->id() << "]";
  if (isKeyFrame) {
    LOG_MAPPING(DEBUG) << "Is keyframe [" << frame->id() << "]";
    if (_keyFrames.size() >= _maxKeyFrames) {
      removeLastKeyFrame();
    }
    while (!_frames.empty()) {
      removeLastFrame();
    }
    _keyFrames.push_front(frame);
  } else {
    if (_frames.size() >= _maxFrames) {
      removeLastFrame();
    }
    _frames.push_front(frame);
  }

  removeUnobservedPoints();
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
  std::vector<Point3D::ConstShPtr> ps;
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
std::vector<Frame::ShPtr> Map::keyFrames(size_t fromIdx, size_t toIdx)
{
  std::vector<Frame::ShPtr> fs;
  fs.reserve(toIdx - fromIdx);
  for (size_t i = fromIdx; i <= toIdx && i < _keyFrames.size(); i++) {
    fs.push_back(_keyFrames.at(i));
  }
  return fs;
}
std::vector<Frame::ConstShPtr> Map::keyFrames(size_t fromIdx, size_t toIdx) const
{
  std::vector<Frame::ConstShPtr> fs;
  fs.reserve(toIdx - fromIdx);
  for (size_t i = fromIdx; i <= toIdx && i < _keyFrames.size(); i++) {
    fs.push_back(_keyFrames.at(i));
  }
  return fs;
}
Frame::ConstShPtr Map::keyFrame(size_t idx) const
{
  return _keyFrames.size() <= idx ? nullptr : _keyFrames.at(idx);
}

Frame::ConstShPtr Map::frame(size_t idx) const
{
  return _frames.size() <= idx ? nullptr : _frames.at(idx);
}

Frame::ConstShPtr Map::lastKf() const { return keyFrame(0); }
Frame::ConstShPtr Map::lastFrame() const { return frame(0); }

Frame::ConstShPtr Map::oldestKf() const { return keyFrame(nKeyFrames() - 1); }

size_t Map::nKeyFrames() const { return _keyFrames.size(); }
size_t Map::nFrames() const { return _frames.size(); }

void Map::insert(Point3D::ShPtr point) { _points[point->id()] = point; }

void Map::insert(const std::vector<Point3D::ShPtr> & points)
{
  for (const auto & p : points) {
    _points[p->id()] = p;
  }
}
void Map::updatePose(std::uint64_t id, const PoseWithCovariance & pose)
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
  LOG_MAPPING(WARNING) << "Cannot update pose. Frame [" << id << "]is not part of map.";
}
void Map::updatePoses(const std::map<std::uint64_t, PoseWithCovariance> & poses)
{
  //TODO how to handle the effects on the remaining poses?
  for (auto & id_pose : poses) {
    updatePose(id_pose.first, id_pose.second);
  }
}

void Map::updatePoints(const std::map<std::uint64_t, Vec3d> & points)
{
  for (auto id_p : points) {
    auto it = _points.find(id_p.first);
    if (it != _points.end()) {
      it->second->position() = id_p.second;
    } else {
      LOG_MAPPING(WARNING) << "Cannot update point. Point [" << id_p.first
                           << "] is not part of map.";
    }
  }
}
void Map::updatePointsAndPoses(
  const std::map<std::uint64_t, PoseWithCovariance> & poses,
  const std::map<std::uint64_t, Vec3d> & points)
{
  updatePoses(poses);
  updatePoints(points);
}

void Map::removeUnobservedPoints()
{
  LOG_MAPPING(DEBUG) << "Removing unobserved points.";

  size_t nRemoved = 0U;
  for (auto id_p = _points.begin(); id_p != _points.end();) {
    auto p = id_p->second;
    if (p->features().size() < 2) {
      nRemoved++;
      p->removeFeatures();
      id_p = _points.erase(id_p);
      if (p.use_count() > 1) {
        LOG_MAPPING(WARNING) << "Point [" << p->id() << "] to delete still has [" << p.use_count()
                             << "] references.";
      }

    } else {
      id_p++;
    }
  }

  LOG_MAPPING(DEBUG) << "Removed: " << nRemoved << " points. Remaining: " << _points.size();
}

void Map::removeLastFrame()
{
  auto oldF = _frames.back();
  LOG_MAPPING(DEBUG) << "Deleting oldest frame: [" << oldF->id() << "].";
  _frames.pop_back();
  oldF->removeFeatures();
  if (oldF.use_count() > 1) {
    LOG_MAPPING(WARNING) << "Frame [" << oldF->id() << "] to delete still has [" << oldF.use_count()
                         << "] references.";
  }
  LOG_MAPPING(DEBUG) << "Deleting frame [" << oldF->id() << "] References: " << oldF.use_count()
                     << " #Frames: " << _frames.size();
}
void Map::removeLastKeyFrame()
{
  auto oldKf = _keyFrames.back();
  LOG_MAPPING(DEBUG) << "Deleting oldest frame: [" << oldKf->id() << "].";

  _keyFrames.pop_back();
  oldKf->removeFeatures();
  if (oldKf.use_count() > 1) {
    LOG_MAPPING(WARNING) << "Keyframe [" << oldKf->id() << "] to delete still has ["
                         << oldKf.use_count() << "] references.";
  }
  LOG_MAPPING(DEBUG) << "Deleting keyframe [" << oldKf->id()
                     << "] References: " << oldKf.use_count()
                     << " #Keyframes: " << _keyFrames.size();
}

}  // namespace pd::vslam
