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
#include "MatcherBruteForce.h"
#include "utils/utils.h"
#define LOG_TRACKING(level) CLOG(level, "tracking")

namespace pd::vslam
{
FeatureTracking::FeatureTracking() { Log::get("tracking"); }

std::vector<Point3D::ShPtr> FeatureTracking::track(
  Frame::ShPtr frameCur, const std::vector<Frame::ShPtr> & framesRef)
{
  extractFeatures(frameCur);
  return match(frameCur, selectCandidates(frameCur, framesRef));
}

void FeatureTracking::extractFeatures(Frame::ShPtr frame) const
{
  cv::Mat image;
  cv::eigen2cv(frame->intensity(), image);
  cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create();
  std::vector<cv::KeyPoint> kpts;
  detector->detect(image, kpts);
  const size_t nRows =
    static_cast<size_t>(static_cast<float>(frame->height(0)) / static_cast<float>(_gridCellSize));
  const size_t nCols =
    static_cast<size_t>(static_cast<float>(frame->width(0)) / static_cast<float>(_gridCellSize));

  std::vector<size_t> grid(nRows * nCols, kpts.size());
  for (size_t idx = 0U; idx < kpts.size(); idx++) {
    const auto & kp = kpts[idx];
    const size_t r =
      static_cast<size_t>(static_cast<float>(kp.pt.y) / static_cast<float>(_gridCellSize));
    const size_t c =
      static_cast<size_t>(static_cast<float>(kp.pt.x) / static_cast<float>(_gridCellSize));
    if (grid[r * nCols + c] >= kpts.size() || kp.response > kpts[grid[r * nCols + c]].response) {
      grid[r * nCols + c] = idx;
    }
  }
  LOG_TRACKING(INFO) << "Keypoints: " << kpts.size();

  std::vector<cv::KeyPoint> kptsGrid;
  kptsGrid.reserve(nRows * nCols);
  std::for_each(grid.begin(), grid.end(), [&](auto idx) {
    if (idx < kpts.size()) {
      kptsGrid.push_back(kpts[idx]);
    }
  });
  LOG_TRACKING(INFO) << "Remaining keypoints: " << kptsGrid.size();
  cv::Ptr<cv::DescriptorExtractor> extractor = cv::ORB::create();
  cv::Mat desc;
  extractor->compute(image, kptsGrid, desc);
  LOG_TRACKING(INFO) << "Remaining keypoints: " << kptsGrid.size();
  LOG_TRACKING(INFO) << "Computed descriptors: " << desc.rows << "x" << desc.cols;

  std::vector<Feature2D::ShPtr> features;
  features.reserve(kptsGrid.size());
  MatXd descriptor;
  cv::cv2eigen(desc, descriptor);
  for (size_t i = 0U; i < kptsGrid.size(); ++i) {
    const auto & kp = kptsGrid[i];
    frame->addFeature(std::make_shared<Feature2D>(
      Vec2d(kp.pt.x, kp.pt.y), frame, kp.octave, kp.response, descriptor.row(i)));
  }
}

std::vector<Point3D::ShPtr> FeatureTracking::match(
  Frame::ShPtr frameCur, const std::vector<Feature2D::ShPtr> & featuresRef) const
{
  MatcherBruteForce matcher([&](Feature2D::ConstShPtr ftRef, Feature2D::ConstShPtr ftCur) {
    const double d = (ftRef->descriptor() - ftCur->descriptor()).cwiseAbs().sum();
    const double r = MatcherBruteForce::reprojectionError(ftRef, ftCur);

    // LOG_TRACKING(INFO) << "(" << ftRef->id() << ") --> (" << ftCur->id()
    //                    << ") reprojection error: " << d << " r: " << r;
    return std::isfinite(r) ? d + r : d;
  });
  const std::vector<MatcherBruteForce::Match> matches = matcher.match(
    Frame::ConstShPtr(frameCur)->features(),
    std::vector<Feature2D::ConstShPtr>(featuresRef.begin(), featuresRef.end()));

  std::set<Point3D::ShPtr> points;
  for (const auto & m : matches) {
    auto fRef = featuresRef[m.idxRef];
    auto fCur = frameCur->features()[m.idxCur];
    if (fRef->point()) {
      fCur->point() = fRef->point();
      fRef->point()->addFeature(fCur);
    } else if (frameCur->depth()(fCur->position().y(), fCur->position().x()) > 0) {
      std::vector<Feature2D::ShPtr> features = {fRef, fCur};
      const Vec3d p3d = frameCur->p3dWorld(fCur->position().y(), fCur->position().x());
      fRef->point() = fCur->point() = std::make_shared<Point3D>(p3d, features);
      points.insert(fCur->point());
    }
  }
  return std::vector<Point3D::ShPtr>(points.begin(), points.end());
}
std::vector<Feature2D::ShPtr> FeatureTracking::selectCandidates(
  Frame::ConstShPtr frameCur, const std::vector<Frame::ShPtr> & framesRef) const
{
  std::set<std::uint64_t> pointIds;

  std::vector<Feature2D::ShPtr> candidates;
  candidates.reserve(framesRef.size() * framesRef.at(0)->features().size());
  // TODO(unknown): sort frames from new to old
  const double border = 5.0;
  for (auto & f : framesRef) {
    for (auto ft : f->features()) {
      if (!ft->point()) {
        candidates.push_back(ft);
      } else if (std::find(pointIds.begin(), pointIds.end(), ft->point()->id()) == pointIds.end()) {
        Vec2d pIcs = frameCur->world2image(ft->point()->position());
        if (
          border < pIcs.x() && pIcs.x() < f->width() - border && border < pIcs.y() &&
          pIcs.y() < f->height() - border) {
          candidates.push_back(ft);
          pointIds.insert(ft->point()->id());
        }
      }
    }
  }
  return candidates;
}

}  // namespace pd::vslam
