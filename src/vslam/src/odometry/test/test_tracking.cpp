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
// Created by phil on 10.10.20.
//

#include <core/core.h>
#include <gtest/gtest.h>
#include <utils/utils.h>

#include <Eigen/Dense>
#include <opencv2/highgui.hpp>
#include <opencv4/opencv2/core/eigen.hpp>

#include "odometry/odometry.h"
using namespace testing;
using namespace pd;
using namespace pd::vslam;

TEST(TrackingTest, SelectVisible)
{
  DepthMap depth = utils::loadDepth(TEST_RESOURCE "/depth.png") / 5000.0;
  Image img = utils::loadImage(TEST_RESOURCE "/rgb.png");

  auto cam = std::make_shared<Camera>(525.0, 525.0, 319.5, 239.5);
  auto f0 = std::make_shared<Frame>(img, depth, cam, 0);
  auto f1 = std::make_shared<Frame>(img, depth, cam, 1);
  std::vector<Frame::ShPtr> frames;
  frames.push_back(f0);
  auto tracking = std::make_shared<FeatureTracking>();
  f0->addFeatures(tracking->extractFeatures(f0));
  f1->addFeatures(tracking->extractFeatures(f1));

  auto featuresCandidate = tracking->selectCandidates(f1, frames);
  EXPECT_EQ(f0->features().size(), featuresCandidate.size());
}

TEST(TrackingTest, GridSubsampling)
{
  Image img = utils::loadImage(TEST_RESOURCE "/rgb.png");
  DepthMap depth = utils::loadDepth(TEST_RESOURCE "/depth.png") / 5000.0;
  auto cam = std::make_shared<Camera>(525.0, 525.0, 319.5, 239.5);

  auto f0 = std::make_shared<Frame>(img, depth, cam, 0);

  cv::Mat image;
  cv::eigen2cv(img, image);

  cv::Ptr<cv::DescriptorExtractor> detector = cv::ORB::create();
  std::vector<cv::KeyPoint> kpts;
  detector->detect(image, kpts);
  LOG_IMG("BeforeGridSubsampling")->show() = TEST_VISUALIZE;
  LOG_IMG("BeforeGridSubsampling")->block() = TEST_VISUALIZE;
  LOG_IMG("AfterGridSubsampling")->show() = TEST_VISUALIZE;
  LOG_IMG("AfterGridSubsampling")->block() = TEST_VISUALIZE;
  FeatureTracking::createFeatures(kpts, f0);
  LOG_IMG("BeforeGridSubsampling")
    << std::make_shared<FeaturePlot>(Frame::ConstShPtr(f0), f0->width() * 200);
  auto kptsFiltered = FeatureTracking::gridSubsampling(kpts, f0, 30);
  f0->removeFeatures();
  FeatureTracking::createFeatures(kptsFiltered, f0);
  LOG_IMG("AfterGridSubsampling") << std::make_shared<FeaturePlot>(Frame::ConstShPtr(f0), 30);
}

TEST(TrackingTest, FeatureConversion)
{
  Image img = utils::loadImage(TEST_RESOURCE "/rgb.png");
  DepthMap depth = utils::loadDepth(TEST_RESOURCE "/depth.png") / 5000.0;
  auto cam = std::make_shared<Camera>(525.0, 525.0, 319.5, 239.5);

  auto f0 = std::make_shared<Frame>(img, depth, cam, 0);

  cv::Mat image;
  cv::eigen2cv(img, image);

  cv::Ptr<cv::DescriptorExtractor> detector = cv::ORB::create();
  std::vector<cv::KeyPoint> kpts;
  cv::Mat desc;
  detector->detectAndCompute(image, cv::Mat(), kpts, desc);
  auto features = FeatureTracking::createFeatures(kpts, desc, DescriptorType::ORB);

  for (size_t i = 0U; i < kpts.size(); i++) {
    EXPECT_NEAR(kpts[i].pt.x, features[i]->position().x(), 0.001);
    EXPECT_NEAR(kpts[i].pt.y, features[i]->position().y(), 0.001);
    EXPECT_NEAR(kpts[i].response, features[i]->response(), 0.001);
    EXPECT_NEAR(kpts[i].octave, features[i]->level(), 0.001);
    MatXd descriptor;
    cv::cv2eigen(desc.row(i), descriptor);
    EXPECT_NEAR((descriptor - features[i]->descriptor()).norm(), 0.0, 0.001);
  }

  auto descBack = FeatureTracking::createDescriptorMatrix(
    std::vector<Feature2D::ConstShPtr>(features.begin(), features.end()), desc.type());

  cv::Mat diff;
  cv::subtract(desc, descBack, diff);
  EXPECT_NEAR(cv::norm(diff), 0, 0.001);
}
class PlotMatchCandidates : public vis::Drawable
{
public:
  PlotMatchCandidates(
    Frame::ConstShPtr f0, Frame::ConstShPtr f1, const MatXd & mask, double maxMask, int idx)
  : _f0(f0), _f1(f1), _mask(mask), _maxMask(maxMask), _idx(idx)
  {
  }

  void drawFeature(
    cv::Mat & mat, Feature2D::ConstShPtr ft, const std::string & annotation = "",
    double radius = 5) const
  {
    cv::Point center(ft->position().x(), ft->position().y());
    cv::putText(mat, annotation, center, cv::FONT_HERSHEY_COMPLEX, 1.0, cv::Scalar(255, 255, 255));

    cv::rectangle(
      mat, cv::Rect(center - cv::Point(radius, radius), center + cv::Point(radius, radius)),
      cv::Scalar(0, 0, 255), 2);
  }
  cv::Mat draw() const override
  {
    cv::Mat mat0;
    cv::eigen2cv(_f0->intensity(), mat0);
    cv::cvtColor(mat0, mat0, cv::COLOR_GRAY2BGR);
    auto ft = _f0->features()[_idx];
    drawFeature(mat0, ft, std::to_string(ft->id()));
    cv::Mat mat1;
    cv::eigen2cv(_f1->intensity(), mat1);
    cv::cvtColor(mat1, mat1, cv::COLOR_GRAY2BGR);
    for (size_t j = 0U; j < _f1->features().size(); j++) {
      if (_mask(_idx, j) < _maxMask) {
        drawFeature(mat1, _f1->features()[j], std::to_string(_mask(_idx, j)));
      }
    }
    cv::Mat mat;
    cv::hconcat(std::vector<cv::Mat>({mat0, mat1}), mat);
    return mat;
  }

private:
  const Frame::ConstShPtr _f0, _f1;
  const MatXd _mask;
  const double _maxMask;
  const int _idx;
};

TEST(TrackingTest, MatcherWithCombinedError)
{
  const size_t nFeatures = 5;
  auto cam = std::make_shared<Camera>(525.0, 525.0, 319.5, 239.5);

  auto f0 = std::make_shared<Frame>(
    utils::loadImage(TEST_RESOURCE "/1311868164.363181.png"),
    utils::loadDepth(TEST_RESOURCE "/1311868164.338541.png") / 5000.0, cam, 1311868164363181000U);

  auto f1 = std::make_shared<Frame>(
    utils::loadImage(TEST_RESOURCE "/1311868165.499179.png"),
    utils::loadDepth(TEST_RESOURCE "/1311868165.409979.png") / 5000.0, cam, 1311868165499179000U);

  auto trajectoryGt =
    std::make_shared<Trajectory>(utils::loadTrajectory(TEST_RESOURCE "/trajectory.txt"));
  f0->set(trajectoryGt->poseAt(f0->t())->inverse());
  f1->set(trajectoryGt->poseAt(f1->t())->inverse());

  auto tracking = std::make_shared<FeatureTracking>();
  f0->addFeatures(tracking->extractFeatures(f0, true));
  f1->addFeatures(tracking->extractFeatures(f1));

  MatXd reprojectionError = vslam::Matcher::computeDistanceMat(
    Frame::ConstShPtr(f0)->features(), Frame::ConstShPtr(f1)->features(),
    [](auto ft1, auto ft2) { return vslam::Matcher::reprojectionError(ft1, ft2); });

  EXPECT_NE(reprojectionError.norm(), 0.0);

  MatXd descriptorDistance = vslam::Matcher::computeDistanceMat(
    Frame::ConstShPtr(f0)->features(), Frame::ConstShPtr(f1)->features(),
    [](auto ft1, auto ft2) { return vslam::Matcher::descriptorHamming(ft1, ft2); });

  EXPECT_NE(descriptorDistance.norm(), 0.0);

  const VecXd reprojectionErrorMin = reprojectionError.rowwise().minCoeff();
  const VecXd descriptorDistanceMin = descriptorDistance.rowwise().minCoeff();
  for (size_t i = 0; i < f0->features().size(); i++) {
    reprojectionError.row(i) /= std::max<double>(reprojectionErrorMin(i), 1);
    descriptorDistance.row(i) /= std::max<double>(descriptorDistanceMin(i), 1);
  }
  MatXd combinedDistance = reprojectionError + descriptorDistance;
  const VecXd combinedDistanceMin = combinedDistance.rowwise().minCoeff();
  for (size_t i = 0; i < f0->features().size(); i++) {
    combinedDistance.row(i) /= std::max<double>(combinedDistanceMin(i), 1);
  }
  std::vector<Feature2D::ConstShPtr> featureSubset;
  for (size_t i = 0; i < nFeatures; i++) {
    const int idx0 = static_cast<int>(random::U(0, f0->features().size() - 1));
    featureSubset.push_back(f0->features()[idx0]);
    std::cout << "Reprojection Error Min:" << reprojectionError.row(idx0).minCoeff()
              << " Reprojection Error Max:" << reprojectionError.row(idx0).maxCoeff() << std::endl;

    LOG_IMG("FeatureCandidatesReprojection")->set(TEST_VISUALIZE, TEST_VISUALIZE);
    LOG_IMG("FeatureCandidatesReprojection") << std::make_shared<PlotMatchCandidates>(
      f0, f1, reprojectionError / reprojectionError.row(idx0).minCoeff(), 1.5, idx0);

    std::cout << "Descriptor Distance Min:" << descriptorDistance.row(idx0).minCoeff()
              << " Descriptor Distance Max:" << descriptorDistance.row(idx0).maxCoeff()
              << std::endl;

    LOG_IMG("FeatureCandidatesDescriptorDistance")->set(TEST_VISUALIZE, TEST_VISUALIZE);
    LOG_IMG("FeatureCandidatesDescriptorDistance") << std::make_shared<PlotMatchCandidates>(
      f0, f1, descriptorDistance / descriptorDistance.row(idx0).minCoeff(), 1.5, idx0);

    std::cout << "Combined Error Min:" << combinedDistance.row(idx0).minCoeff()
              << " Combined Error Max:" << combinedDistance.row(idx0).maxCoeff() << std::endl;

    LOG_IMG("FeatureCandidatesCombined")->set(TEST_VISUALIZE, TEST_VISUALIZE);
    LOG_IMG("FeatureCandidatesCombined")
      << std::make_shared<PlotMatchCandidates>(f0, f1, combinedDistance, 1.5, idx0);
  }
  auto matcher = std::make_shared<vslam::Matcher>(vslam::Matcher::reprojectionHamming, 4.0, 0.8);
  const std::vector<vslam::Matcher::Match> matches =
    matcher->match(featureSubset, Frame::ConstShPtr(f1)->features());
  EXPECT_EQ(matches.size(), 3);
}

class PlotCorrespondences : public vis::Drawable
{
public:
  PlotCorrespondences(const std::vector<Frame::ConstShPtr> & frames) : _frames(frames) {}

  cv::Mat draw() const override
  {
    std::vector<cv::Mat> mats;
    for (const auto f : _frames) {
      cv::Mat mat;
      cv::eigen2cv(f->intensity(), mat);
      cv::cvtColor(mat, mat, cv::COLOR_GRAY2BGR);
      mats.push_back(mat);
    }

    std::set<uint64_t> points;
    for (size_t i = 0U; i < _frames.size(); i++) {
      for (auto ftRef : _frames[i]->featuresWithPoints()) {
        auto p = ftRef->point();
        if (points.find(p->id()) != points.end()) {
          continue;
        }
        points.insert(p->id());
        cv::Scalar color(
          (double)std::rand() / RAND_MAX * 255, (double)std::rand() / RAND_MAX * 255,
          (double)std::rand() / RAND_MAX * 255);
        for (size_t j = 0U; j < _frames.size(); j++) {
          auto ft = _frames[j]->observationOf(p->id());
          if (ft) {
            cv::Point center(ft->position().x(), ft->position().y());
            const double radius = 5;
            cv::circle(mats[j], center, radius, color, 2);
            std::stringstream ss;
            ss << ft->point()->id();
            cv::putText(
              mats[j], ss.str(), center, cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255, 255, 255));
          }
        }
      }
    }
    cv::Mat mat;
    cv::hconcat(mats, mat);
    return mat;
  }

private:
  std::vector<Frame::ConstShPtr> _frames;
};

TEST(TrackingTest, TrackThreeFrames)
{
  auto cam = std::make_shared<Camera>(525.0, 525.0, 319.5, 239.5);

  auto f0 = std::make_shared<Frame>(
    utils::loadImage(TEST_RESOURCE "/1311868164.363181.png"),
    utils::loadDepth(TEST_RESOURCE "/1311868164.338541.png") / 5000.0, cam, 1311868164363181000U);

  auto f1 = std::make_shared<Frame>(
    utils::loadImage(TEST_RESOURCE "/1311868165.499179.png"),
    utils::loadDepth(TEST_RESOURCE "/1311868165.409979.png") / 5000.0, cam, 1311868165499179000U);

  auto f2 = std::make_shared<Frame>(
    utils::loadImage(TEST_RESOURCE "/1311868166.763333.png"),
    utils::loadDepth(TEST_RESOURCE "/1311868166.715787.png") / 5000.0, cam, 1311868166763333000U);

  auto trajectoryGt =
    std::make_shared<Trajectory>(utils::loadTrajectory(TEST_RESOURCE "/trajectory.txt"));
  f0->set(trajectoryGt->poseAt(f0->t())->inverse());
  f1->set(trajectoryGt->poseAt(f1->t())->inverse());
  f2->set(trajectoryGt->poseAt(f2->t())->inverse());
  auto tracking = std::make_shared<FeatureTracking>(
    std::make_shared<vslam::Matcher>(vslam::Matcher::reprojectionHamming, 10, 0.8));
  LOG_IMG("Tracking")->set(TEST_VISUALIZE, TEST_VISUALIZE);

  f0->addFeatures(tracking->extractFeatures(f0, true));

  auto points1 = tracking->track(f1, {f0});
  for (auto p : points1) {
    EXPECT_NE(f0->observationOf(p->id()), nullptr);
    EXPECT_NE(f1->observationOf(p->id()), nullptr);
  }
  LOG_IMG("Tracking") << std::make_shared<PlotCorrespondences>(
    std::vector<Frame::ConstShPtr>({f0, f1, f2}));

  auto points2 = tracking->track(f2, {f0, f1});
  size_t nCommonPoints = 0U;
  for (auto p : points1) {
    if (f2->observationOf(p->id())) {
      nCommonPoints++;
    }
  }
  LOG(INFO) << "#Common Points across three frames: " << nCommonPoints;
  EXPECT_GE(nCommonPoints, 20);
  LOG_IMG("Tracking") << std::make_shared<PlotCorrespondences>(
    std::vector<Frame::ConstShPtr>({f0, f1, f2}));
}

TEST(TrackingTest, DISABLED_TrackAndOptimize)
{
  DepthMap depth = utils::loadDepth(TEST_RESOURCE "/depth.png") / 5000.0;
  Image img = utils::loadImage(TEST_RESOURCE "/rgb.png");

  auto cam = std::make_shared<Camera>(525.0, 525.0, 319.5, 239.5);
  auto f0 = std::make_shared<Frame>(img, depth, cam, 0);
  auto f1 = std::make_shared<Frame>(img, depth, cam, 1);
  f0->computePcl();
  f1->computePcl();

  auto tracking = std::make_shared<FeatureTracking>();
  f0->addFeatures(tracking->extractFeatures(f0));
  f1->addFeatures(tracking->extractFeatures(f1));

  auto pose = f1->pose().pose();
  pose.translation().x() += 0.1;
  pose.translation().y() -= 0.1;
  f1->set(PoseWithCovariance(pose, f1->pose().cov()));
  std::vector<Feature2D::ShPtr> f0Features = f0->features();
  std::vector<Feature2D::ShPtr> f1Features = f1->features();

  auto points = tracking->match(f0->features(), f1->features());
  for (auto p : points) {
    EXPECT_TRUE(f0->observationOf(p->id()));
    EXPECT_TRUE(f1->observationOf(p->id()));
  }
  LOG(INFO) << "#Matches: " << points.size();

  cv::Mat mat0, mat1;
  cv::eigen2cv(img, mat0);
  cv::eigen2cv(img, mat1);
  cv::cvtColor(mat0, mat0, cv::COLOR_GRAY2BGR);
  cv::cvtColor(mat1, mat1, cv::COLOR_GRAY2BGR);

  for (auto p : points) {
    auto pIn0 = f0->observationOf(p->id());
    auto pIn1 = f1->observationOf(p->id());

    cv::circle(
      mat0, cv::Point(pIn0->position().x(), pIn0->position().y()), 7, cv::Scalar(0, 255, 0));
    cv::circle(
      mat1, cv::Point(pIn1->position().x(), pIn1->position().y()), 7, cv::Scalar(0, 255, 0));

    auto ft0Noisy = f0->world2image(p->position());
    auto ft1Noisy = f1->world2image(p->position());
    cv::circle(mat0, cv::Point(ft0Noisy.x(), ft0Noisy.y()), 7, cv::Scalar(0, 0, 255));
    cv::circle(mat1, cv::Point(ft1Noisy.x(), ft1Noisy.y()), 7, cv::Scalar(0, 0, 255));
  }

  std::vector<Frame::ShPtr> frames = {f0, f1};

  mapping::BundleAdjustment ba;
  auto results = ba.optimize(std::vector<Frame::ConstShPtr>(frames.begin(), frames.end()));

  EXPECT_LT(results->errorAfter, results->errorBefore);

  f0->set(results->poses.find(f0->id())->second);
  f1->set(results->poses.find(f1->id())->second);

  LOG(INFO) << "Pose: " << f1->pose().pose().matrix();

  for (auto p : points) {
    auto ft0 = f0->world2image(p->position());
    auto ft1 = f1->world2image(p->position());
    cv::circle(mat0, cv::Point(ft0.x(), ft0.y()), 7, cv::Scalar(255, 0, 0));
    cv::circle(mat1, cv::Point(ft1.x(), ft1.y()), 7, cv::Scalar(255, 0, 0));
  }

  if (TEST_VISUALIZE) {
    //cv::imshow("Frame0", mat0);
    //cv::imshow("Frame1", mat1);
    //cv::waitKey(0);
  }
}
