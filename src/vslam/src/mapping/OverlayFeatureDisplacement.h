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

#ifndef VSLAM_OVERLAY_FEATURE_DISPLACEMENT_H__
#define VSLAM_OVERLAY_FEATURE_DISPLACEMENT_H__
#include <opencv2/highgui/highgui.hpp>

#include "core/core.h"
#include "utils/utils.h"

namespace pd::vslam
{
class OverlayFeatureDisplacement : public vis::Drawable
{
public:
  OverlayFeatureDisplacement(
    const Frame::VecConstShPtr & frames, const Point3D::VecConstShPtr & points,
    unsigned int maxWidth = 2560, unsigned int nRows = 2)
  : _frames(frames), _points(points), _maxWidth(maxWidth), _nRows(nRows)
  {
    std::sort(_frames.begin(), _frames.end(), [](auto f0, auto f1) { return f0->t() < f1->t(); });
  }
  OverlayFeatureDisplacement(
    const Frame::VecConstShPtr & frames, unsigned int maxWidth = 2560, unsigned int nRows = 2)
  : _frames(frames), _maxWidth(maxWidth), _nRows(nRows)
  {
    std::sort(_frames.begin(), _frames.end(), [](auto f0, auto f1) { return f0->t() < f1->t(); });

    for (auto f : frames) {
      for (auto ft : f->featuresWithPoints()) {
        if (std::find(_points.begin(), _points.end(), ft->point()) == _points.end()) {
          _points.push_back(ft->point());
        }
      }
    }
  }

  cv::Mat draw() const override;

private:
  Frame::VecConstShPtr _frames;
  Point3D::VecConstShPtr _points;
  const unsigned int _maxWidth;
  const unsigned int _nRows;
  static std::map<uint64_t, cv::Scalar> _colorMap;
};
}  // namespace pd::vslam
#endif  //VSLAM_OVERLAY_FEATURE_DISPLACEMENT_H__