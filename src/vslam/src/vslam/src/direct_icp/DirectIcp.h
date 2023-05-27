#ifndef VSLAM_DIRECT_ICP_H__
#define VSLAM_DIRECT_ICP_H__
#include <map>
#include <memory>
#include <opencv4/opencv2/imgproc.hpp>
#include <string>

#include "core/Camera.h"
#include "core/types.h"

namespace vslam
{
struct Feature
{
  typedef std::shared_ptr<Feature> ShPtr;
  size_t idx;
  Vec2d uv0;
  Vec2d iz0;
  Vec3d p0;
  Vec6d JIJw;
  Vec6d JZJw;
  Vec6d JZJw_Jtz;
  Matd<2, 6> J;
  Mat2d weight;
  Vec2d residual;
  Vec2d uv0t;
  Vec2d iz1w;
  Vec3d p0t;
  Vec3d p1t;
  double error;
  bool valid;
};

class TDistributionBivariate
{
public:
  typedef std::shared_ptr<TDistributionBivariate> ShPtr;
  TDistributionBivariate(double dof);
  void computeWeights(
    const std::vector<Feature::ShPtr> & r, double precision = 1e-3, int maxIterations = 50);
  double computeWeight(const Vec2d & r) const;

private:
  const double _dof;
  Mat2d _scale;
};

class DirectIcp
{
public:
  typedef std::shared_ptr<DirectIcp> ShPtr;

  DirectIcp(Camera::ConstShPtr cam, const std::map<std::string, double> params);
  DirectIcp(
    Camera::ConstShPtr cam, int nLevels = 4, double weightPrior = 0.0,
    double minGradientIntensity = 10 * 8, double minGradientDepth = INFd,
    double maxGradientDepth = 0.5, double maxZ = 5.0, double maxIterations = 100,
    double minParameterUpdate = 1e-6, double maxErrorIncrease = 1.1);
  SE3d computeEgomotion(const cv::Mat & intensity, const cv::Mat & depth, const SE3d & guess);

private:
  std::vector<Camera::ConstShPtr> _cam;
  TDistributionBivariate::ShPtr _weightFunction;
  std::vector<cv::Mat> _I0, _Z0;
  int _nLevels;
  double _weightPrior, _minGradientIntensity, _minGradientDepth, _maxGradientDepth, _maxDepth,
    _maxIterations, _minParameterUpdate, _maxErrorIncrease;

  std::vector<cv::Mat> computePyramidIntensity(const cv::Mat & intensity) const;
  std::vector<cv::Mat> computePyramidDepth(const cv::Mat & depth) const;
  cv::Mat computeJacobianImage(const cv::Mat & image) const;
  cv::Mat computeJacobianDepth(const cv::Mat & depth) const;
  std::vector<Feature::ShPtr> extractFeatures(
    const cv::Mat & intensity, const cv::Mat & depth, Camera::ConstShPtr cam,
    const SE3d & motion) const;
  Matd<2, 6> computeJacobianWarp(const Vec3d & p, Camera::ConstShPtr cam) const;

  Vec6d computeJacobianSE3z(const Vec3d & p) const;
  Vec2d interpolate(const cv::Mat & intensity, const cv::Mat & depth, const Vec2d & uv);
  std::vector<Feature::ShPtr> uniformSubselection(
    Camera::ConstShPtr cam, const std::vector<Feature::ShPtr> & features) const;
};

}  // namespace vslam
#endif
