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
FeatureTracking::FeatureTracking(size_t nFeatures) : _nFeatures(nFeatures) { Log::get("tracking"); }

std::vector<Point3D::ShPtr> FeatureTracking::track(
  FrameRgbd::ShPtr frameCur, const std::vector<FrameRgbd::ShPtr> & framesRef)
{
  extractFeatures(frameCur);
  return match(frameCur, selectCandidates(frameCur, framesRef));
}

void FeatureTracking::extractFeatures(FrameRgbd::ShPtr frame) const
{
  cv::Mat image;
  cv::eigen2cv(frame->intensity(), image);
  cv::Ptr<cv::DescriptorExtractor> extractor = cv::ORB::create(_nFeatures);
  cv::Mat desc;
  std::vector<cv::KeyPoint> kpts;
  extractor->detectAndCompute(image, cv::Mat(), kpts, desc);
  MatXd descriptors;
  cv::cv2eigen(desc, descriptors);
  std::vector<Feature2D::ShPtr> features;
  features.reserve(kpts.size());
  for (size_t i = 0U; i < kpts.size(); ++i) {
    const auto & kp = kpts[i];
    frame->addFeature(std::make_shared<Feature2D>(
      Vec2d(kp.pt.x, kp.pt.y), frame, kp.octave, kp.response, descriptors.row(i)));
  }
}

std::vector<Point3D::ShPtr> FeatureTracking::match(
  FrameRgbd::ShPtr frameCur, const std::vector<Feature2D::ShPtr> & featuresRef) const
{
  MatcherBruteForce matcher([&](Feature2D::ConstShPtr ftRef, Feature2D::ConstShPtr ftCur) {
    // TODO(phil): min baseline?
    const Mat3d F = algorithm::computeF(ftRef->frame(), ftCur->frame());
    const Vec3d xCur = Vec3d(ftRef->position().x(), ftRef->position().y(), 1).transpose();
    const Vec3d xRef = Vec3d(ftCur->position().x(), ftCur->position().y(), 1);
    const Vec3d l = F * xRef;
    const double xFx = std::abs(xCur.transpose() * (l / std::sqrt(l.x() * l.x() + l.y() * l.y())));

    const double d = (ftRef->descriptor() - ftCur->descriptor()).cwiseAbs().sum();

    LOG_TRACKING(INFO) << "(" << ftRef->id() << ") --> (" << ftCur->id() << ") xFx = " << xFx
                       << " d = " << d << " F = " << F;

    // TODO(phil): whats a good way to way of compute trade off? Compute mean + std offline and normalize..
    return std::isfinite(xFx) ? d + xFx : d;
  });
  const std::vector<MatcherBruteForce::Match> matches = matcher.match(
    FrameRgbd::ConstShPtr(frameCur)->features(),
    std::vector<Feature2D::ConstShPtr>(featuresRef.begin(), featuresRef.end()));

  std::vector<Point3D::ShPtr> points;
  points.reserve(matches.size());
  for (const auto & m : matches) {
    auto fRef = featuresRef[m.idxRef];
    auto fCur = frameCur->features()[m.idxCur];
    if (fRef->point()) {
      fCur->point() = fRef->point();
      fRef->point()->addFeature(fCur);
    } else {
      std::vector<Feature2D::ShPtr> features = {fRef, fCur};
      const Vec3d p3d = frameCur->image2world(
        fCur->position(), frameCur->depth()(fCur->position().y(), fCur->position().x()));

      fRef->point() = fCur->point() = std::make_shared<Point3D>(p3d, features);
    }
    points.push_back(fCur->point());
  }
  return points;
}
std::vector<Feature2D::ShPtr> FeatureTracking::selectCandidates(
  FrameRgbd::ConstShPtr frameCur, const std::vector<FrameRgbd::ShPtr> & framesRef) const
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
