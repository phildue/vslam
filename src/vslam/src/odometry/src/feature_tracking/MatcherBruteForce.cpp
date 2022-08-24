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

//
// Created by phil on 30.06.21.
//

#include "MatcherBruteForce.h"
#include "utils/utils.h"
#define LOG_TRACKING(level) CLOG(level, "tracking")

namespace pd::vslam
{
MatcherBruteForce::MatcherBruteForce(
  std::function<double(Feature2D::ConstShPtr ref, Feature2D::ConstShPtr target)> distanceFunction,
  double minDistanceRatio)
: _computeDistance(distanceFunction), _minDistanceRatio(minDistanceRatio)
{
  Log::get("tracking");
}

std::vector<MatcherBruteForce::Match> MatcherBruteForce::match(
  const std::vector<Feature2D::ConstShPtr> & featuresRef,
  const std::vector<Feature2D::ConstShPtr> & featuresTarget)
{
  std::vector<Match> matches;
  matches.reserve(featuresRef.size());
  for (size_t i = 0U; i < featuresRef.size(); ++i) {
    std::vector<Match> distances(featuresTarget.size());
    for (size_t j = 0U; j < featuresTarget.size(); ++j) {
      distances[j] = {i, j, _computeDistance(featuresRef[i], featuresTarget[j])};
    }
    std::sort(distances.begin(), distances.end(), [&](auto m0, auto m1) {
      return m0.distance > m1.distance;
    });
    if (distances[0].distance > _minDistanceRatio * distances[1].distance) {
      matches.push_back(distances[0]);
    }
  }
  return matches;
}
double MatcherBruteForce::epipolarError(Feature2D::ConstShPtr ftRef, Feature2D::ConstShPtr ftCur)
{
  // TODO(phil): min baseline?
  const Mat3d F = algorithm::computeF(ftRef->frame(), ftCur->frame());
  const Vec3d xCur = Vec3d(ftRef->position().x(), ftRef->position().y(), 1).transpose();
  const Vec3d xRef = Vec3d(ftCur->position().x(), ftCur->position().y(), 1);
  const Vec3d l = F * xRef;
  const double xFx = std::abs(xCur.transpose() * (l / std::sqrt(l.x() * l.x() + l.y() * l.y())));

  LOG_TRACKING(INFO) << "(" << ftRef->id() << ") --> (" << ftCur->id() << ") xFx = " << xFx
                     << " F = " << F;

  return xFx;
}
double MatcherBruteForce::reprojectionError(
  Feature2D::ConstShPtr ftRef, Feature2D::ConstShPtr ftCur)
{
  // TODO(phil): min baseline?
  // const SE3d Rt = algorithm::computeRelativeTransform(
  //  ftRef->frame()->pose().pose(), ftCur->frame()->pose().pose());
  // const Vec3d p3dRef = ftRef->frame()->p3d(ftRef->position().y(), ftRef->position().x());
  // const double err = (ftCur->position() - ftCur->frame()->camera2image(Rt * p3dRef)).norm();
  const double err =
    (ftCur->position() -
     ftCur->frame()->world2image(ftCur->frame()->image2world(
       ftRef->position(), ftRef->frame()->depth()(ftRef->position().y(), ftRef->position().x()))))
      .norm();
  //LOG_TRACKING(INFO) << "(" << ftRef->id() << ") --> (" << ftCur->id()
  //                   << ") reprojection error: " << err;

  // TODO(phil): whats a good way to way of compute trade off? Compute mean + std offline and normalize..
  return err;
}

}  // namespace pd::vslam
