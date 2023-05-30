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

#ifndef VSLAM_FEATURE_TRACKING_OCV_H__
#define VSLAM_FEATURE_TRACKING_OCV_H__
#include "FeatureTracking.h"
namespace pd::vslam
{
class FeatureTrackingOcv : public FeatureTracking
{
public:
  typedef std::shared_ptr<FeatureTrackingOcv> ShPtr;
  typedef std::unique_ptr<FeatureTrackingOcv> UnPtr;
  typedef std::shared_ptr<const FeatureTrackingOcv> ConstShPtr;
  typedef std::unique_ptr<const FeatureTrackingOcv> ConstUnPtr;

  FeatureTrackingOcv() : FeatureTracking(nullptr){};

  std::vector<Point3D::ShPtr> track(
    Frame::ShPtr frameCur, const std::vector<Frame::ShPtr> & framesRef) const override;

  static std::vector<cv::DMatch> removeNonDistinct(
    const std::vector<std::vector<cv::DMatch>> & matchesKnn,
    const std::vector<Feature2D::ConstShPtr> & featuresCur,
    const std::vector<Feature2D::ConstShPtr> & featuresRef, double maxNextBestRatio = 0.8);

  static std::vector<cv::DMatch> removeModelOutliers(
    const std::vector<cv::DMatch> & matches, const std::vector<Feature2D::ConstShPtr> & featuresCur,
    const std::vector<Feature2D::ConstShPtr> & featuresRef);

private:
  const double _nextBestMatchRatio = 0.8;
  const double _maxReprojectionError = 50.0;
  const size_t _gridCellSize = 15;
  const std::string _descriptorMatcherType = "BruteForce-Hamming";
  void extractFeatures(
    Frame::ShPtr frame, bool applyGrid, std::vector<cv::KeyPoint> & kpts, cv::Mat & desc) const;
};
}  // namespace pd::vslam

#endif  //VSLAM_FEATURE_TRACKING_OCV_H__
