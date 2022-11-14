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

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv4/opencv2/core/eigen.hpp>

#include "OverlayMatchCandidates.h"
namespace pd::vslam
{
cv::Mat OverlayMatchCandidates::draw() const
{
  cv::Mat mat0;
  cv::eigen2cv(_f0->intensity(), mat0);
  cv::cvtColor(mat0, mat0, cv::COLOR_GRAY2BGR);
  auto ft = _f0->features()[_idx];
  drawFeature(mat0, ft, std::to_string(ft->id()));
  cv::Mat mat1;
  cv::eigen2cv(_f1->intensity(), mat1);
  cv::cvtColor(mat1, mat1, cv::COLOR_GRAY2BGR);
  for (size_t j = 0U; j < _f1->features().size(); j++) {
    if (_mask(_idx, j) < _maxMask) {
      drawFeature(mat1, _f1->features()[j], std::to_string(_mask(_idx, j)));
    }
  }
  cv::Mat mat;
  cv::hconcat(std::vector<cv::Mat>({mat0, mat1}), mat);
  return mat;
}
void OverlayMatchCandidates::drawFeature(
  cv::Mat & mat, Feature2D::ConstShPtr ft, const std::string & annotation, double radius) const
{
  cv::Point center(ft->position().x(), ft->position().y());
  cv::putText(mat, annotation, center, cv::FONT_HERSHEY_COMPLEX, 1.0, cv::Scalar(255, 255, 255));

  cv::rectangle(
    mat, cv::Rect(center - cv::Point(radius, radius), center + cv::Point(radius, radius)),
    cv::Scalar(0, 0, 255), 2);
}

}  // namespace pd::vslam