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

#ifndef VSLAM_FEATURE_PLOT_H__
#define VSLAM_FEATURE_PLOT_H__
#include <opencv2/highgui/highgui.hpp>

#include "core/core.h"
#include "utils/utils.h"

namespace pd::vslam
{
class FeaturePlot : public vis::Drawable
{
public:
  FeaturePlot(Frame::ConstShPtr frame, double cellSize) : _frame(frame), _gridCellSize(cellSize) {}
  cv::Mat draw() const;

private:
  const Frame::ConstShPtr _frame;
  const double _gridCellSize;
};
}  // namespace pd::vslam
#endif  //VSLAM_FEATURE_PLOT_H__