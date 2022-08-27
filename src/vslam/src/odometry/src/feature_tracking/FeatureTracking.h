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

#ifndef VSLAM_FEATURE_TRACKING_H__
#define VSLAM_FEATURE_TRACKING_H__
#include "Matcher.h"
#include "core/core.h"
namespace pd::vslam
{
class FeatureTracking
{
public:
  typedef std::shared_ptr<FeatureTracking> ShPtr;
  typedef std::unique_ptr<FeatureTracking> UnPtr;
  typedef std::shared_ptr<const FeatureTracking> ConstShPtr;
  typedef std::unique_ptr<const FeatureTracking> ConstUnPtr;

  FeatureTracking(Matcher::ConstShPtr matcher = std::make_shared<MatcherBruteForce>());

  std::vector<Point3D::ShPtr> track(
    Frame::ShPtr frameCur, const std::vector<Frame::ShPtr> & framesRef);

  void extractFeatures(Frame::ShPtr frame) const;

  std::vector<Point3D::ShPtr> match(
    Frame::ShPtr frameCur, const std::vector<Feature2D::ShPtr> & featuresRef) const;

  std::vector<Point3D::ShPtr> match(
    const std::vector<Feature2D::ShPtr> & featuresCur,
    const std::vector<Feature2D::ShPtr> & featuresRef) const;

  std::vector<Feature2D::ShPtr> selectCandidates(
    Frame::ConstShPtr frameCur, const std::vector<Frame::ShPtr> & framesRef) const;

private:
  const size_t _gridCellSize = 30;
  const Matcher::ConstShPtr _matcher;
};
}  // namespace pd::vslam

#endif  //VSLAM_FEATURE_TRACKING_H__
