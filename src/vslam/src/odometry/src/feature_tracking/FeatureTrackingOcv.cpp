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
#include <algorithm>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/opencv.hpp>
#include <opencv4/opencv2/core/eigen.hpp>
#include <set>

#include "FeatureTrackingOcv.h"
#include "MatcherOcv.h"
#include "utils/utils.h"
#define LOG_TRACKING(level) CLOG(level, "tracking")

namespace pd::vslam
{
std::vector<cv::DMatch> FeatureTrackingOcv::removeNonDistinct(
  const std::vector<std::vector<cv::DMatch>> & matchesKnn,
  const std::vector<Feature2D::ConstShPtr> & featuresCur,
  const std::vector<Feature2D::ConstShPtr> & featuresRef, double maxNextBestRatio)
{
  std::vector<cv::DMatch> matchesDistinct;

  for (size_t i = 0; i < matchesKnn.size(); i++) {
    const auto & m = matchesKnn[i][0];
    if (m.distance < maxNextBestRatio * matchesKnn[i][1].distance) {
      if (
        m.queryIdx < 0 || m.trainIdx < 0 || m.queryIdx >= static_cast<int>(featuresRef.size()) ||
        m.trainIdx >= static_cast<int>(featuresCur.size())) {
        break;
      }
      matchesDistinct.push_back(m);
      //LOG_TRACKING(DEBUG) << "Distinct match: [" << featuresRef[m.queryIdx]->id() << "] to ["
      //                    << featuresCur[m.trainIdx]->id() << "]";
    }
  }
  LOG_TRACKING(DEBUG) << "Kept [" << matchesDistinct.size() << "/" << matchesKnn.size()
                      << "] distinct matches.";
  return matchesDistinct;
}
std::vector<cv::DMatch> FeatureTrackingOcv::removeModelOutliers(
  const std::vector<cv::DMatch> & matches, const std::vector<Feature2D::ConstShPtr> & featuresCur,
  const std::vector<Feature2D::ConstShPtr> & featuresRef)
{
  std::vector<cv::Point> pRef;
  std::vector<cv::Point> pTarget;
  for (const auto & m : matches) {
    auto fRef = featuresRef[m.queryIdx];
    pRef.push_back(cv::Point(fRef->position().x(), fRef->position().y()));
    auto fCur = featuresCur[m.trainIdx];
    pTarget.push_back(cv::Point(fCur->position().x(), fCur->position().y()));
  }

  cv::Mat matchMask;
  auto F = cv::findFundamentalMat(pRef, pTarget, cv::FM_RANSAC, 3, 0.99, matchMask);

  std::vector<cv::DMatch> matchesInliers;
  matchesInliers.reserve(matches.size());
  for (size_t i = 0U; i < matches.size(); i++) {
    if (matchMask.at<uchar>(i, 0) > 0U) {
      matchesInliers.push_back(matches[i]);
    } else {
      //LOG_TRACKING(DEBUG) << "Dropped: [" << featuresRef[matches[i].queryIdx]->id() << "] to ["
      //                    << featuresCur[matches[i].trainIdx]->id() << "] as model outlier.";
    }
  }
  LOG_TRACKING(DEBUG) << "Kept [" << matchesInliers.size() << "/" << matches.size() << "] inliers.";
  return matchesInliers;
}
std::vector<Point3D::ShPtr> FeatureTrackingOcv::track(
  Frame::ShPtr frameCur, const std::vector<Frame::ShPtr> & framesRef) const
{
  cv::Mat descCur;
  std::vector<cv::KeyPoint> kptsCur;
  extractFeatures(frameCur, framesRef.empty(), kptsCur, descCur);
  if (framesRef.empty()) {
    LOG_TRACKING(WARNING) << "No reference frames passed.";
    FeatureTracking::createFeatures(kptsCur, descCur, DescriptorType::ORB, frameCur);
    return std::vector<Point3D::ShPtr>();
  }
  auto featuresRef = selectCandidates(frameCur, framesRef);
  auto featuresCur = FeatureTracking::createFeatures(kptsCur, descCur, DescriptorType::ORB);
  for (auto ft : featuresCur) {
    ft->frame() = frameCur;
  }
  auto descRef = FeatureTracking::createDescriptorMatrix(
    std::vector<Feature2D::ConstShPtr>(featuresRef.begin(), featuresRef.end()), descCur.type());

  std::vector<std::vector<cv::DMatch>> matchesKnn;
  cv::Ptr<cv::DescriptorMatcher> matcherOcv =
    cv::DescriptorMatcher::create(_descriptorMatcherType.c_str());
  /*cv::Mat mask = MatcherOcv::computeMaskReprojection(
    std::vector<Feature2D::ConstShPtr>(featuresRef.begin(), featuresRef.end()),
    std::vector<Feature2D::ConstShPtr>(featuresCur.begin(), featuresCur.end()),
    _maxReprojectionError);*/
  matcherOcv->knnMatch(descRef, descCur, matchesKnn, 2);
  LOG_TRACKING(DEBUG) << "Found [" << matchesKnn.size() << "] matches.";
  //TODO: first check what happened to the features that already have points and should be visible

  auto matches = FeatureTrackingOcv::removeNonDistinct(
    matchesKnn, std::vector<Feature2D::ConstShPtr>(featuresCur.begin(), featuresCur.end()),
    std::vector<Feature2D::ConstShPtr>(featuresRef.begin(), featuresRef.end()),
    _nextBestMatchRatio);

  matches = FeatureTrackingOcv::removeModelOutliers(
    matches, std::vector<Feature2D::ConstShPtr>(featuresCur.begin(), featuresCur.end()),
    std::vector<Feature2D::ConstShPtr>(featuresRef.begin(), featuresRef.end()));

  size_t matchedPoints = 0;
  for (const auto & m : matches) {
    auto fRef = featuresRef[m.queryIdx];
    auto fCur = featuresCur[m.trainIdx];
    auto z = frameCur->depth()(fCur->position().y(), fCur->position().x());
    if (fCur->point()) {
      LOG_TRACKING(DEBUG) << "Feature [" << fRef->id() << "] cannot be matched with Feature ["
                          << fCur->id() << "] because it was already matched to Point ["
                          << fCur->point()->id() << "]";
    } else if (fRef->point()) {
      if (!frameCur->observationOf(fRef->point()->id())) {
        fCur->point() = fRef->point();
        fRef->point()->addFeature(fCur);
        //LOG_TRACKING(DEBUG) << "Feature [" << fCur->id() << "] was matched to point: ["
        //                    << fRef->point()->id() << "]";
        matchedPoints++;
      } else {
        LOG_TRACKING(DEBUG) << "Point: [" << fRef->point()->id() << "] was already matched on ["
                            << frameCur->id() << "] with lower distance. Skipping..";
      }
    } else if (z > 0) {
      std::vector<Feature2D::ShPtr> features = {fRef, fCur};
      const Vec3d p3d = frameCur->image2world(fCur->position(), z);
      fCur->point() = std::make_shared<Point3D>(p3d, features);
      fRef->point() = fCur->point();
      //LOG_TRACKING(DEBUG) << "New Point between [" << fRef->id() << "] and [" << fCur->id()
      //                    << "]: [" << fCur->point()->id() << "] with distance: [" << m.distance
      //                    << "]";
    }
  }

  // Apply the grid only on the unmatched features.

  featuresCur = FeatureTracking::gridSubsampling(featuresCur, frameCur, _gridCellSize);
  frameCur->addFeatures(featuresCur);
  std::vector<Point3D::ShPtr> points;
  points.reserve(featuresCur.size());
  std::for_each(featuresCur.begin(), featuresCur.end(), [&](auto ft) {
    if (ft->point() && ft->point()->features().size() == 2) {
      points.push_back(ft->point());
    }
  });

  LOG_TRACKING(DEBUG) << "Frame [" << frameCur->id() << "]: #New Points: [" << points.size()
                      << "] #Matched Points: [" << matchedPoints << "] #Unmatched Features: ["
                      << frameCur->features().size() - frameCur->featuresWithPoints().size() << "]";

  return points;
  //LOG_IMG("Tracking") << std::make_shared<FeaturePlot>(frameCur, _gridCellSize);
}

void FeatureTrackingOcv::extractFeatures(
  Frame::ShPtr frame, bool applyGrid, std::vector<cv::KeyPoint> & kpts, cv::Mat & desc) const
{
  cv::Mat image;
  cv::eigen2cv(frame->intensity(), image);
  cv::Mat mask = cv::Mat::zeros(image.size(), CV_8U);

  forEachPixel(frame->depth(), [&](auto u, auto v, auto d) {
    if (std::isfinite(d) && d > 0.1) {
      mask.at<std::uint8_t>(v, u) = 255U;
    }
  });
  cv::Ptr<cv::DescriptorExtractor> detector = cv::FastFeatureDetector::create();

  detector->detect(image, kpts, mask);
  LOG_TRACKING(DEBUG) << "Frame [" << frame->id() << "]: Detected Keypoints: " << kpts.size();

  if (applyGrid) {
    kpts = FeatureTracking::gridSubsampling(kpts, frame, _gridCellSize);
  }

  LOG_TRACKING(DEBUG) << "Frame [" << frame->id()
                      << "]: Kept keypoints after grid: " << kpts.size();
  cv::Ptr<cv::DescriptorExtractor> extractor = cv::ORB::create();
  extractor->compute(image, kpts, desc);
  LOG_TRACKING(DEBUG) << "Frame [" << frame->id() << "]: Computed descriptors: " << desc.rows << "x"
                      << desc.cols;
}

}  // namespace pd::vslam
