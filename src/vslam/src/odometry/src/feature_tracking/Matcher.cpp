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

#include "Matcher.h"
#include "utils/utils.h"
#define LOG_TRACKING(level) CLOG(level, "tracking")

namespace pd::vslam
{
Matcher::Matcher(
  std::function<double(Feature2D::ConstShPtr, Feature2D::ConstShPtr)> distanceFunction,
  double maxDistance, double minDistanceRatio)
: Matcher(
    [&](const Feature2D::VecConstShPtr & ftsRef, const Feature2D::VecConstShPtr & ftsTarget) {
      return Matcher::computeDistanceMat(ftsRef, ftsTarget, [&](auto ftRef, auto ftTarget) {
        return distanceFunction(ftRef, ftTarget);
      });
    },
    maxDistance, minDistanceRatio)
{
}
Matcher::Matcher(
  std::function<MatXd(
    const std::vector<Feature2D::ConstShPtr> & target,
    const std::vector<Feature2D::ConstShPtr> & ref)>
    distanceFunction,
  double maxDistance, double minDistanceRatio)
: _computeDistanceMat(distanceFunction),
  _maxDistance(maxDistance),
  _minDistanceRatio(minDistanceRatio)
{
}
std::vector<std::vector<Matcher::Match>> Matcher::knn(
  const std::vector<Feature2D::ConstShPtr> & featuresRef,
  const std::vector<Feature2D::ConstShPtr> & featuresTarget, int k) const
{
  LOG_TRACKING(DEBUG) << "Computing distance of: [" << featuresRef.size()
                      << "] reference features against [" << featuresTarget.size()
                      << "] target features.";

  const MatXd distanceMat = _computeDistanceMat(featuresRef, featuresTarget);
  std::vector<std::vector<Matcher::Match>> neighbors(
    featuresRef.size(), std::vector<Matcher::Match>(k));

  LOG_TRACKING(DEBUG) << "Computed distance mat: " << distanceMat.rows() << "x"
                      << distanceMat.cols();
  for (size_t i = 0U; i < featuresRef.size(); ++i) {
    LOG_TRACKING(DEBUG) << i;
    std::vector<Match> candidates(featuresTarget.size());
    for (size_t j = 0U; j < featuresTarget.size(); ++j) {
      candidates[j] = {i, j, distanceMat(i, j)};
    }
    std::sort(candidates.begin(), candidates.end(), [&](auto m0, auto m1) {
      return m0.distance < m1.distance;
    });
    for (int n = 0; n < k && static_cast<size_t>(n) < candidates.size(); n++) {
      std::cout << "Neighbor [" << i << "][" << n << "]: " << candidates[n].distance << std::endl;
      neighbors[i][n] = candidates[n];
    }
  }
  return neighbors;
}

std::vector<Matcher::Match> Matcher::match(
  const std::vector<Feature2D::ConstShPtr> & featuresRef,
  const std::vector<Feature2D::ConstShPtr> & featuresTarget) const
{
  LOG_TRACKING(DEBUG) << "Matching: [" << featuresRef.size() << "] reference features against ["
                      << featuresTarget.size() << "] target features.";
  auto candidates = knn(featuresRef, featuresTarget, 2);
  std::vector<Match> matches;
  matches.reserve(featuresRef.size());
  for (size_t i = 0U; i < featuresRef.size(); i++) {
    if (
      candidates[i][0].distance < _maxDistance &&
      candidates[i][0].distance < _minDistanceRatio * candidates[i][1].distance) {
      LOG_TRACKING(DEBUG) << "Found match between [" << featuresRef[candidates[i][0].idxRef]->id()
                          << "] and [" << featuresTarget[candidates[i][0].idxCur]->id()
                          << "] with distance [" << candidates[i][0].distance
                          << "] and distance ratio [" << candidates[i][0].distance << "/"
                          << candidates[i][1].distance << "="
                          << candidates[i][0].distance / candidates[i][1].distance
                          << "] to next best match ["
                          << featuresTarget[candidates[i][1].idxCur]->id() << "]";
      matches.push_back(candidates[i][0]);
    }
  }
  std::sort(
    matches.begin(), matches.end(), [](auto m0, auto m1) { return m0.distance < m1.distance; });
  LOG_TRACKING(DEBUG) << "#Matches: " << matches.size();

  return matches;
}
double Matcher::epipolarError(Feature2D::ConstShPtr ftRef, Feature2D::ConstShPtr ftCur)
{
  // TODO(phil): min baseline?
  const Mat3d F = algorithm::computeF(ftRef->frame(), ftCur->frame());
  const Vec3d xCur = Vec3d(ftRef->position().x(), ftRef->position().y(), 1).transpose();
  const Vec3d xRef = Vec3d(ftCur->position().x(), ftCur->position().y(), 1);
  const Vec3d l = F * xRef;
  const double xFx = std::abs(xCur.transpose() * (l / std::sqrt(l.x() * l.x() + l.y() * l.y())));

  //LOG_TRACKING(INFO) << "(" << ftRef->id() << ") --> (" << ftCur->id() << ") xFx = " << xFx
  //                   << " F = " << F;

  return xFx;
}
double Matcher::reprojectionError(Feature2D::ConstShPtr ftRef, Feature2D::ConstShPtr ftCur)
{
  // TODO(phil): min baseline?
  // const SE3d Rt = algorithm::computeRelativeTransform(
  //  ftRef->frame()->pose().pose(), ftCur->frame()->pose().pose());
  // const Vec3d p3dRef = ftRef->frame()->p3d(ftRef->position().y(), ftRef->position().x());
  // const double err = (ftCur->position() - ftCur->frame()->camera2image(Rt * p3dRef)).norm();
  auto frameRef = ftRef->frame();
  auto frameCur = ftCur->frame();
  auto zRef = frameRef->depth()(ftRef->position().y(), ftRef->position().x());
  auto p3d = frameRef->image2world(ftRef->position(), zRef);
  const double err = (ftCur->position() - frameCur->world2image(p3d)).norm();
  //LOG_TRACKING(INFO) << "(" << ftRef->id() << ") --> (" << ftCur->id()
  //                   << ") reprojection error: " << err;

  // TODO(phil): whats a good way to way of compute trade off? Compute mean + std offline and normalize..
  return err;
}

double Matcher::descriptorL1(Feature2D::ConstShPtr ftRef, Feature2D::ConstShPtr ftCur)
{
  return (ftRef->descriptor() - ftCur->descriptor()).cwiseAbs().sum();
}
double Matcher::descriptorL2(Feature2D::ConstShPtr ftRef, Feature2D::ConstShPtr ftCur)
{
  return (ftRef->descriptor() - ftCur->descriptor()).norm();
}
double Matcher::descriptorHamming(Feature2D::ConstShPtr ftRef, Feature2D::ConstShPtr ftCur)
{
  const int dim = ftRef->descriptor().rows();
  int distance = 0U;
  for (int i = 0; i < dim; i++) {
    const auto byteRef = static_cast<uchar>(ftRef->descriptor()(i));
    const auto byteCur = static_cast<uchar>(ftCur->descriptor()(i));
    const uchar byteXor = byteRef ^ byteCur;
    /* Count number of 1 by shifting and Bitwise AND with 1 to check if last bit is 1*/
    distance += (byteXor >> 0 & 1);
    distance += (byteXor >> 1 & 1);
    distance += (byteXor >> 2 & 1);
    distance += (byteXor >> 3 & 1);
    distance += (byteXor >> 4 & 1);
    distance += (byteXor >> 5 & 1);
    distance += (byteXor >> 6 & 1);
    distance += (byteXor >> 7 & 1);
  }
  return distance;
}

MatXd Matcher::computeDistanceMat(
  const std::vector<Feature2D::ConstShPtr> & featuresRef,
  const std::vector<Feature2D::ConstShPtr> & featuresTarget,
  std::function<double(Feature2D::ConstShPtr ref, Feature2D::ConstShPtr target)> distanceFunction)
{
  MatXd mask = MatXd::Zero(featuresRef.size(), featuresTarget.size());
  forEach(mask, [&](int u, int v, double UNUSED(e)) {
    mask(v, u) = distanceFunction(featuresRef[v], featuresTarget[u]);
  });
  return mask;
}

MatXd Matcher::reprojectionHamming(
  const Feature2D::VecConstShPtr & featuresRef, const Feature2D::VecConstShPtr & featuresTarget)
{
  LOG_TRACKING(DEBUG) << "Computing reprojection error..";
  MatXd reprojection = computeDistanceMat(featuresRef, featuresTarget, Matcher::reprojectionError);
  LOG_TRACKING(DEBUG) << "Computing descriptor error..";
  MatXd descriptor = computeDistanceMat(featuresRef, featuresTarget, Matcher::descriptorHamming);
  const VecXd reprojectionMin = reprojection.rowwise().minCoeff();
  const VecXd descriptorMin = descriptor.rowwise().minCoeff();
  for (size_t i = 0U; i < featuresRef.size(); i++) {
    reprojection.row(i) /= reprojectionMin(i);
    descriptor.row(i) /= descriptorMin(i);
  }
  return reprojection + descriptor;
}

}  // namespace pd::vslam
