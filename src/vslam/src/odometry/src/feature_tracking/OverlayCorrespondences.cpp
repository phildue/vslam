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

#include "OverlayCorrespondences.h"
namespace pd::vslam
{
cv::Mat OverlayCorrespondences::draw() const
{
  std::vector<cv::Mat> mats;
  for (const auto f : _frames) {
    cv::Mat mat;
    cv::eigen2cv(f->intensity(), mat);
    cv::cvtColor(mat, mat, cv::COLOR_GRAY2BGR);
    cv::rectangle(mat, cv::Point(0, 0), cv::Point(20, 20), cv::Scalar(0, 0, 0), -1);
    cv::putText(
      mat, std::to_string(f->id()), cv::Point(5, 12), cv::FONT_HERSHEY_COMPLEX, 0.5,
      cv::Scalar(255, 255, 255));
    mats.push_back(mat);
  }

  std::set<uint64_t> points;
  for (size_t i = 0U; i < _frames.size(); i++) {
    for (auto ftRef : _frames[i]->featuresWithPoints()) {
      auto p = ftRef->point();
      if (points.find(p->id()) != points.end()) {
        continue;
      }
      points.insert(p->id());
      cv::Scalar color(
        (double)std::rand() / RAND_MAX * 255, (double)std::rand() / RAND_MAX * 255,
        (double)std::rand() / RAND_MAX * 255);
      for (size_t j = 0U; j < _frames.size(); j++) {
        auto ft = _frames[j]->observationOf(p->id());
        if (ft) {
          cv::Point center(ft->position().x(), ft->position().y());
          const double radius = 5;
          cv::circle(mats[j], center, radius, color, 2);
          std::stringstream ss;
          ss << ft->point()->id();
          cv::putText(
            mats[j], ss.str(), center, cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255, 255, 255));
        }
      }
    }
  }
  cv::Mat mat;
  cv::hconcat(mats, mat);
  return mat;
}

}  // namespace pd::vslam