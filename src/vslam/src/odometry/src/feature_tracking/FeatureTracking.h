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
#include <opencv2/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

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

  FeatureTracking(Matcher::ConstShPtr matcher = std::make_shared<Matcher>());

  virtual std::vector<Point3D::ShPtr> track(
    Frame::ShPtr frameCur, const std::vector<Frame::ShPtr> & framesRef) const;

  void extractFeatures(
    Frame::ShPtr frame, bool applyGrid = false,
    size_t nMax = std::numeric_limits<size_t>::max()) const;

  std::vector<Point3D::ShPtr> match(
    Frame::ShPtr frameCur, const std::vector<Feature2D::ShPtr> & featuresRef) const;

  std::vector<Point3D::ShPtr> match(
    const std::vector<Feature2D::ShPtr> & featuresRef,
    const std::vector<Feature2D::ShPtr> & featuresCur) const;

  std::vector<Feature2D::ShPtr> selectCandidates(
    Frame::ConstShPtr frameCur, const std::vector<Frame::ShPtr> & framesRef) const;

  static std::vector<cv::KeyPoint> gridSubsampling(
    const std::vector<cv::KeyPoint> & keypoints, Frame::ConstShPtr frame, double cellSize);

  static std::vector<Feature2D::ShPtr> gridSubsampling(
    const std::vector<Feature2D::ShPtr> & features, Frame::ConstShPtr frame, double cellSize);

  static std::vector<Feature2D::ShPtr> createFeatures(
    const std::vector<cv::KeyPoint> & keypoints, Frame::ShPtr frame = nullptr);
  static std::vector<Feature2D::ShPtr> createFeatures(
    const std::vector<cv::KeyPoint> & keypoints, const cv::Mat & desc,
    const DescriptorType & descType, Frame::ShPtr frame = nullptr);

  static cv::Mat createDescriptorMatrix(
    const std::vector<Feature2D::ConstShPtr> & features, int dtype);

private:
  const size_t _gridCellSize = 30;
  const Matcher::ConstShPtr _matcher;
};
}  // namespace pd::vslam

#endif  //VSLAM_FEATURE_TRACKING_H__
