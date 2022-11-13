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

#include "FeaturePlot.h"
namespace pd::vslam
{
cv::Mat FeaturePlot::draw() const
{
  cv::Mat mat;
  cv::eigen2cv(_frame->intensity(), mat);
  cv::cvtColor(mat, mat, cv::COLOR_GRAY2BGR);

  for (size_t r = 0; r < _frame->height(0); r += _gridCellSize) {
    cv::line(mat, cv::Point(0, r), cv::Point(_frame->width(0), r), cv::Scalar(128, 128, 128));
  }
  for (size_t c = 0; c < _frame->width(0); c += _gridCellSize) {
    cv::line(mat, cv::Point(c, 0), cv::Point(c, _frame->height(0)), cv::Scalar(128, 128, 128));
  }
  for (auto ft : _frame->features()) {
    cv::Point center(ft->position().x(), ft->position().y());
    const double radius = 5;
    if (ft->point()) {
      std::stringstream ss;
      ss << ft->point()->id();
      cv::putText(mat, ss.str(), center, cv::FONT_HERSHEY_COMPLEX, 1.0, cv::Scalar(255, 255, 255));

      cv::circle(mat, center, radius, cv::Scalar(255, 0, 0), 2);
    } else {
      std::stringstream ss;
      ss << ft->id();
      cv::putText(mat, ss.str(), center, cv::FONT_HERSHEY_COMPLEX, 1.0, cv::Scalar(255, 255, 255));

      cv::rectangle(
        mat, cv::Rect(center - cv::Point(radius, radius), center + cv::Point(radius, radius)),
        cv::Scalar(0, 0, 255), 2);
    }
  }
  return mat;
}

}  // namespace pd::vslam