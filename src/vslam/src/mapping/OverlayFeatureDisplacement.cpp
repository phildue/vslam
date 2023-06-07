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

#include <fmt/core.h>

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv4/opencv2/core/eigen.hpp>

#include "OverlayFeatureDisplacement.h"
namespace pd::vslam
{
std::map<uint64_t, cv::Scalar> OverlayFeatureDisplacement::_colorMap = {};

cv::Mat OverlayFeatureDisplacement::draw() const
{
  std::vector<cv::Mat> mats;
  for (const auto f : _frames) {
    cv::Mat mat;
    cv::eigen2cv(f->intensity(), mat);
    cv::cvtColor(mat, mat, cv::COLOR_GRAY2BGR);
    cv::rectangle(mat, cv::Point(0, 0), cv::Point(40, 20), cv::Scalar(0, 0, 0), -1);

    mats.push_back(mat);
  }
  const double radius = 5;
  std::map<uint64_t, double> reprojectionError;
  std::map<uint64_t, int> nPoints;

  for (auto p : _points) {
    auto colorIt = _colorMap.find(p->id());
    if (colorIt == _colorMap.end()) {
      _colorMap[p->id()] = cv::Scalar(
        (double)std::rand() / RAND_MAX * 255, (double)std::rand() / RAND_MAX * 255,
        (double)std::rand() / RAND_MAX * 255);
    }
    const cv::Scalar & color = _colorMap[p->id()];
    for (size_t i = 0U; i < _frames.size(); i++) {
      auto f = _frames[i];
      auto pInF = f->observationOf(p->id());

      if (pInF) {
        auto mat = mats[i];

        cv::circle(mat, cv::Point(pInF->position().x(), pInF->position().y()), radius, color);
        auto reprojection = f->world2image(p->position());
        cv::rectangle(
          mat, cv::Point(reprojection.x() - radius, reprojection.y() - radius),
          cv::Point(reprojection.x() + radius, reprojection.y() + radius), color);
        reprojectionError[f->id()] += (pInF->position() - reprojection).norm();
        nPoints[f->id()]++;
      }
    }
  }
  const int nMatsPerRow = std::ceil(static_cast<double>(mats.size()) / static_cast<double>(_nRows));
  const double s = std::min(
    1.0, (static_cast<double>(_maxWidth) / static_cast<double>(nMatsPerRow)) /
           static_cast<double>(mats[0].cols));
  int i = 0U;
  const Timestamp t0 = (*_frames.begin())->t();
  for (auto f : _frames) {
    auto & mat = mats[i++];
    cv::putText(
      mat,
      fmt::format(
        "#{0:d}, {1:.3f}, #Points: {2:d}, err: {3:.3f}", f->id(),
        static_cast<double>(f->t() - t0) / static_cast<double>(1e9), nPoints[f->id()],
        reprojectionError[f->id()] / nPoints[f->id()]),
      cv::Point(5, 12), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255, 255, 255));
    cv::resize(mat, mat, cv::Size(0, 0), s, s, cv::INTER_AREA);
  }
  std::vector<cv::Mat> rows;
  for (unsigned int r = 0; r < _nRows; r++) {
    std::vector<cv::Mat> matsInRow;
    for (int c = 0; c < nMatsPerRow; c++) {
      if (r * nMatsPerRow + c >= mats.size()) {
        matsInRow.push_back(cv::Mat(mats[0].size(), mats[0].type(), cv::Scalar(0, 0, 0)));
      } else {
        matsInRow.push_back(mats[r * nMatsPerRow + c]);
      }
    }
    cv::Mat row;
    cv::hconcat(matsInRow, row);
    rows.push_back(row);
  }
  cv::Mat mat;
  cv::vconcat(rows, mat);
  return mat;
}

}  // namespace pd::vslam