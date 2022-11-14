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

#ifndef VSLAM_OVERLAY_FEATURES_H__
#define VSLAM_OVERLAY_FEATURES_H__
#include <opencv2/highgui/highgui.hpp>

#include "core/core.h"
#include "utils/utils.h"

namespace pd::vslam
{
class OverlayFeatures : public vis::Drawable
{
public:
  OverlayFeatures(Frame::ConstShPtr frame, double cellSize) : _frame(frame), _gridCellSize(cellSize)
  {
  }
  cv::Mat draw() const override;

private:
  const Frame::ConstShPtr _frame;
  const double _gridCellSize;
};
}  // namespace pd::vslam
#endif  //VSLAM_OVERLAY_FEATURES_H__