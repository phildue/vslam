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

#ifndef VSLAM_OVERLAY_MATCH_CANDIDATES_H__
#define VSLAM_OVERLAY_MATCH_CANDIDATES_H__
#include <opencv2/highgui/highgui.hpp>

#include "core/core.h"
#include "utils/utils.h"

namespace pd::vslam
{
class OverlayMatchCandidates : public vis::Drawable
{
public:
  OverlayMatchCandidates(
    Frame::ConstShPtr f0, Frame::ConstShPtr f1, const MatXd & mask, double maxMask, int idx)
  : _f0(f0), _f1(f1), _mask(mask), _maxMask(maxMask), _idx(idx)
  {
  }
  cv::Mat draw() const override;

private:
  void drawFeature(
    cv::Mat & mat, Feature2D::ConstShPtr ft, const std::string & annotation = "",
    double radius = 5) const;
  const Frame::ConstShPtr _f0, _f1;
  const MatXd _mask;
  const double _maxMask;
  const int _idx;
};
}  // namespace pd::vslam
#endif  //VSLAM_OVERLAY_MATCH_CANDIDATES_H__