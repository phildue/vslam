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
}  // namespace pd::vslam
