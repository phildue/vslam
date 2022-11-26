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

#ifndef VSLAM_KEY_FRAME_SELECTION
#define VSLAM_KEY_FRAME_SELECTION

#include "Map.h"
#include "core/core.h"
namespace pd::vslam
{
class KeyFrameSelection
{
public:
  typedef std::shared_ptr<KeyFrameSelection> ShPtr;
  typedef std::unique_ptr<KeyFrameSelection> UnPtr;
  typedef std::shared_ptr<const KeyFrameSelection> ConstShPtr;
  typedef std::unique_ptr<const KeyFrameSelection> ConstUnPtr;

  virtual void update(Frame::ConstShPtr frame) = 0;
  virtual bool isKeyFrame() const = 0;

  static ShPtr make();
};
class KeyFrameSelectionIdx : public KeyFrameSelection
{
public:
  typedef std::shared_ptr<KeyFrameSelectionIdx> ShPtr;
  typedef std::unique_ptr<KeyFrameSelectionIdx> UnPtr;
  typedef std::shared_ptr<const KeyFrameSelectionIdx> ConstShPtr;
  typedef std::unique_ptr<const KeyFrameSelectionIdx> ConstUnPtr;

  KeyFrameSelectionIdx(uint64_t period = 2) : KeyFrameSelection(), _period(period), _ctr(0U) {}
  void update(Frame::ConstShPtr UNUSED(frame)) override { _ctr++; }
  bool isKeyFrame() const override { return _ctr == 0 || _ctr % _period == 0; }

private:
  const uint64_t _period;
  uint64_t _ctr;
};

class KeyFrameSelectionCustom : public KeyFrameSelection
{
public:
  typedef std::shared_ptr<KeyFrameSelectionIdx> ShPtr;
  typedef std::unique_ptr<KeyFrameSelectionIdx> UnPtr;
  typedef std::shared_ptr<const KeyFrameSelectionIdx> ConstShPtr;
  typedef std::unique_ptr<const KeyFrameSelectionIdx> ConstUnPtr;

  KeyFrameSelectionCustom(
    Map::ConstShPtr map, std::uint64_t minVisiblePoints = 80, double maxTranslation = 0.2);
  void update(Frame::ConstShPtr frame) override;
  bool isKeyFrame() const override;

private:
  const uint64_t _minVisiblePoints;
  const double _maxTranslation;
  const Map::ConstShPtr _map;
  uint64_t _visiblePoints;
  SE3d _relativePose;
};
class KeyFrameSelectionNever : public KeyFrameSelection
{
public:
  typedef std::shared_ptr<KeyFrameSelectionNever> ShPtr;
  typedef std::unique_ptr<KeyFrameSelectionNever> UnPtr;
  typedef std::shared_ptr<const KeyFrameSelectionNever> ConstShPtr;
  typedef std::unique_ptr<const KeyFrameSelectionNever> ConstUnPtr;

  void update(Frame::ConstShPtr UNUSED(frame)) override {}
  bool isKeyFrame() const override { return false; }
};
}  // namespace pd::vslam
#endif  // VSLAM_RGBD_ODOMETRY
