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
namespace pd::vslam
{
KeyFrameSelectionCustom::KeyFrameSelectionCustom(
  Map::ConstShPtr map, std::uint64_t minVisiblePoints, double maxTranslation)
: KeyFrameSelection(),
  _minVisiblePoints(minVisiblePoints),
  _maxTranslation(maxTranslation),
  _map(map),
  _visiblePoints(0)
{
}

void KeyFrameSelectionCustom::update(Frame::ConstShPtr frame)
{
  LOG(INFO) << "frame: " << frame->id();
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
  LOG(INFO) << "nVisiblePoints: " << _visiblePoints
            << " translation: " << _relativePose.translation().norm();

  return _relativePose.translation().norm() > _maxTranslation || _visiblePoints < _minVisiblePoints;
}
}  // namespace pd::vslam
