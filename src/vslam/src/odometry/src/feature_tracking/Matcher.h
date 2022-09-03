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

#ifndef VSLAM_MATCHER_BRUTE_FORCE_H
#define VSLAM_MATCHER_BRUTE_FORCE_H

#include <functional>
#include <vector>

#include "core/core.h"
namespace pd::vslam
{
class Matcher
{
public:
  typedef std::shared_ptr<Matcher> ShPtr;
  typedef std::unique_ptr<Matcher> UnPtr;
  typedef std::shared_ptr<const Matcher> ConstShPtr;
  typedef std::unique_ptr<const Matcher> ConstUnPtr;

  struct Match
  {
    size_t idxRef;
    size_t idxCur;
    double distance;
  };
  virtual std::vector<Match> match(
    const std::vector<Feature2D::ConstShPtr> & descriptorsTarget,
    const std::vector<Feature2D::ConstShPtr> & descriptorsRef) const = 0;

  static double epipolarError(Feature2D::ConstShPtr ftRef, Feature2D::ConstShPtr ftCur);
  static double reprojectionError(Feature2D::ConstShPtr ftRef, Feature2D::ConstShPtr ftCur);
  static double descriptorL1(Feature2D::ConstShPtr ftRef, Feature2D::ConstShPtr ftCur);
};

class MatcherBruteForce : public Matcher
{
public:
  MatcherBruteForce(
    std::function<double(Feature2D::ConstShPtr ref, Feature2D::ConstShPtr target)>
      distanceFunction = [](auto f1, auto f2) { return descriptorL1(f1, f2); },
    double maxDistance = std::numeric_limits<double>::max(), double minDistanceRatio = 0.8);
  std::vector<Match> match(
    const std::vector<Feature2D::ConstShPtr> & descriptorsTarget,
    const std::vector<Feature2D::ConstShPtr> & descriptorsRef) const override;

private:
  const std::function<double(Feature2D::ConstShPtr ref, Feature2D::ConstShPtr target)>
    _computeDistance;
  const double _maxDistance;
  const double _minDistanceRatio;
};

}  // namespace pd::vslam

#endif  // VSLAM_MATCHER_BRUTE_FORCE_H