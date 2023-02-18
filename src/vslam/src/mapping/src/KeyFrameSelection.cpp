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

#include "KeyFrameSelection.h"
#include "utils/utils.h"

#define LOG_MAPPING(level) CLOG(level, "mapping")

namespace pd::vslam
{
KeyFrameSelectionCustom::KeyFrameSelectionCustom(
  Map::ConstShPtr map, std::uint64_t minVisiblePoints, double maxTranslation, double maxRotation)
: KeyFrameSelection(),
  _minVisiblePoints(minVisiblePoints),
  _maxTranslation(maxTranslation),
  _maxRotation(maxRotation),
  _map(map),
  _visiblePoints(0)
{
}

void KeyFrameSelectionCustom::update(Frame::ConstShPtr frame)
{
  _visiblePoints = 0U;
  if (!_map->lastKf()) {
    return;
  }

  _relativePose =
    algorithm::computeRelativeTransform(_map->lastKf()->pose().pose(), frame->pose().pose());

  for (auto ft : _map->lastKf()->featuresWithPoints()) {
    if (frame->withinImage(frame->world2image(ft->point()->position()))) {
      _visiblePoints++;
    }
  }
}

bool KeyFrameSelectionCustom::isKeyFrame() const
{
  const double rotation = std::sqrt(
    std::pow(_relativePose.angleX(), 2) + std::pow(_relativePose.angleY(), 2) +
    std::pow(_relativePose.angleZ(), 2));
  return _relativePose.translation().norm() > _maxTranslation ||
         _visiblePoints < _minVisiblePoints || rotation > _maxRotation;
}

KeyFrameSelectionEntropy::KeyFrameSelectionEntropy(Map::ConstShPtr map, double maxEntropy)
: KeyFrameSelection(),
  _map(map),
  _maxEntropy(maxEntropy),
  _entropyRatio(0.0),
  _entropyRef(1.0),
  _refId(std::numeric_limits<uint64_t>::max())
{
}
void KeyFrameSelectionEntropy::update(Frame::ConstShPtr frame)
{
  if (!_map->lastKf()) {
    _entropyRatio = 1000.0;
  } else if (_refId != _map->lastKf()->id()) {
    _refId = _map->lastKf()->id();
    _entropyRatio = 1.0;
    _entropyRef = std::max(0.0001, std::log(frame->pose().twistCov().determinant()));

  } else {
    _entropyRatio = std::log(frame->pose().twistCov().determinant()) / _entropyRef;
  }

  LOG_MAPPING(INFO) << "Entropy ratio: " << _entropyRatio;
}
bool KeyFrameSelectionEntropy::isKeyFrame() const { return _entropyRatio > _maxEntropy; }
}  // namespace pd::vslam
