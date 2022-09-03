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

#include "FeatureTracking.h"
#include "utils/utils.h"
#define LOG_TRACKING(level) CLOG(level, "tracking")

namespace pd::vslam
{
class FeaturePlot : public vis::Drawable
{
public:
  FeaturePlot(Frame::ConstShPtr frame) : _frame(frame) {}
  cv::Mat draw() const
  {
    cv::Mat mat;
    cv::eigen2cv(_frame->intensity(), mat);
    cv::cvtColor(mat, mat, cv::COLOR_GRAY2BGR);
    for (auto ft : _frame->features()) {
      cv::Point center(ft->position().x(), ft->position().y());
      double radius = 7;
      if (ft->point()) {
        cv::circle(mat, center, radius, cv::Scalar(255, 0, 0), 2);
      } else {
        cv::rectangle(
          mat,
          cv::Rect(
            center - cv::Point(radius / 2, radius / 2), center + cv::Point(radius / 2, radius / 2)),
          cv::Scalar(0, 0, 255), 2);
      }
    }
    return mat;
  }

private:
  const Frame::ConstShPtr _frame;
};

std::vector<cv::KeyPoint> gridSubsampling(
  const std::vector<cv::KeyPoint> & keypoints, Frame::ConstShPtr frame, double cellSize)
{
  const size_t nRows =
    static_cast<size_t>(static_cast<float>(frame->height(0)) / static_cast<float>(cellSize));
  const size_t nCols =
    static_cast<size_t>(static_cast<float>(frame->width(0)) / static_cast<float>(cellSize));

  /* Create grid for subsampling where each cell contains index of keypoint with the highest response
  *  or the total amount of keypoints if empty */
  std::vector<size_t> grid(nRows * nCols, keypoints.size());
  for (size_t idx = 0U; idx < keypoints.size(); idx++) {
    const auto & kp = keypoints[idx];
    const size_t r =
      static_cast<size_t>(static_cast<float>(kp.pt.y) / static_cast<float>(cellSize));
    const size_t c =
      static_cast<size_t>(static_cast<float>(kp.pt.x) / static_cast<float>(cellSize));
    if (
      grid[r * nCols + c] >= keypoints.size() ||
      kp.response > keypoints[grid[r * nCols + c]].response) {
      grid[r * nCols + c] = idx;
    }
  }

  std::vector<cv::KeyPoint> kptsGrid;
  kptsGrid.reserve(nRows * nCols);
  std::for_each(grid.begin(), grid.end(), [&](auto idx) {
    if (idx < keypoints.size()) {
      kptsGrid.push_back(keypoints[idx]);
    }
  });
  return kptsGrid;
}

FeatureTracking::FeatureTracking(Matcher::ConstShPtr matcher) : _matcher(matcher)
{
  LOG_IMG("Tracking");
  Log::get("tracking");
}

std::vector<Point3D::ShPtr> FeatureTracking::track(
  Frame::ShPtr frameCur, const std::vector<Frame::ShPtr> & framesRef)
{
  extractFeatures(frameCur);
  auto points = match(frameCur, selectCandidates(frameCur, framesRef));
  LOG_IMG("Tracking") << std::make_shared<FeaturePlot>(frameCur);

  return points;
}

void FeatureTracking::extractFeatures(Frame::ShPtr frame) const
{
  cv::Mat image;
  cv::eigen2cv(frame->intensity(), image);
  cv::Mat mask = cv::Mat::zeros(image.size(), CV_8U);

  forEachPixel(frame->depth(), [&](auto u, auto v, auto d) {
    if (std::isfinite(d) && d > 0.1) {
      mask.at<std::uint8_t>(v, u) = 255U;
    }
  });
  cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create();
  std::vector<cv::KeyPoint> kpts;
  detector->detect(image, kpts, mask);
  LOG_TRACKING(DEBUG) << "Detected Keypoints: " << kpts.size();

  kpts = gridSubsampling(kpts, frame, _gridCellSize);

  LOG_TRACKING(DEBUG) << "Remaining keypoints after grid: " << kpts.size();
  cv::Ptr<cv::DescriptorExtractor> extractor = cv::ORB::create();
  cv::Mat desc;
  extractor->compute(image, kpts, desc);
  LOG_TRACKING(DEBUG) << "Computed descriptors: " << desc.rows << "x" << desc.cols;

  std::vector<Feature2D::ShPtr> features;
  features.reserve(desc.rows);
  MatXd descriptor;
  cv::cv2eigen(desc, descriptor);
  for (int i = 0; i < desc.rows; ++i) {
    const auto & kp = kpts[i];
    frame->addFeature(std::make_shared<Feature2D>(
      Vec2d(kp.pt.x, kp.pt.y), frame, kp.octave, kp.response, descriptor.row(i)));
  }
}

std::vector<Point3D::ShPtr> FeatureTracking::match(
  Frame::ShPtr frameCur, const std::vector<Feature2D::ShPtr> & featuresRef) const
{
  return match(frameCur->features(), featuresRef);
}

std::vector<Point3D::ShPtr> FeatureTracking::match(
  const std::vector<Feature2D::ShPtr> & featuresCur,
  const std::vector<Feature2D::ShPtr> & featuresRef) const
{
  const std::vector<Matcher::Match> matches = _matcher->match(
    std::vector<Feature2D::ConstShPtr>(featuresCur.begin(), featuresCur.end()),
    std::vector<Feature2D::ConstShPtr>(featuresRef.begin(), featuresRef.end()));

  LOG_TRACKING(DEBUG) << "#Matches: " << matches.size();

  std::vector<Point3D::ShPtr> points;
  points.reserve(matches.size());
  for (const auto & m : matches) {
    auto fRef = featuresRef[m.idxRef];
    auto fCur = featuresCur[m.idxCur];
    auto frameCur = fCur->frame();
    auto z = frameCur->depth()(fCur->position().y(), fCur->position().x());
    if (fCur->point()) {
      LOG_TRACKING(DEBUG) << fCur->id() << " was already matched. Skipping..";
    } else if (fRef->point()) {
      if (!frameCur->observationOf(fRef->point()->id())) {
        fCur->point() = fRef->point();
        fRef->point()->addFeature(fCur);
        LOG_TRACKING(DEBUG) << "Feature was matched to point: " << fRef->point()->id();

      } else {
        LOG_TRACKING(DEBUG) << "Point: " << fRef->point()->id() << " was already matched on "
                            << frameCur->id() << " with lower distance. Skipping..";
      }
    } else if (z > 0) {
      std::vector<Feature2D::ShPtr> features = {fRef, fCur};
      const Vec3d p3d = frameCur->image2world(fCur->position(), z);
      fCur->point() = std::make_shared<Point3D>(p3d, features);
      fRef->point() = fCur->point();
      points.push_back(fCur->point());
    }
  }

  LOG_TRACKING(DEBUG) << "#New Points: " << points.size();

  return points;
}

std::vector<Feature2D::ShPtr> FeatureTracking::selectCandidates(
  Frame::ConstShPtr frameCur, const std::vector<Frame::ShPtr> & framesRef) const
{
  /* Prefer features from newer frames as they should have closer appearance*/
  std::vector<Frame::ShPtr> framesSorted(framesRef.begin(), framesRef.end());
  std::sort(
    framesSorted.begin(), framesSorted.end(), [](auto f0, auto f1) { return f0->t() > f1->t(); });

  std::set<std::uint64_t> pointIds;
  std::vector<Feature2D::ShPtr> candidates;
  candidates.reserve(framesRef.size() * framesRef.at(0)->features().size());
  for (auto & f : framesSorted) {
    if (f->id() == frameCur->id()) {
      continue;
    }
    for (auto ft : f->features()) {
      if (!ft->point()) {
        candidates.push_back(ft);
      } else if (
        pointIds.find(ft->point()->id()) == pointIds.end() &&
        frameCur->withinImage(frameCur->world2image(ft->point()->position()))) {
        candidates.push_back(ft);
        pointIds.insert(ft->point()->id());
      }
    }
  }
  LOG_TRACKING(DEBUG) << "#Candidate Features: " << candidates.size()
                      << " #Visible Points: " << pointIds.size()
                      << " #Unmatched Features: " << candidates.size() - pointIds.size();

  return candidates;
}
}  // namespace pd::vslam
