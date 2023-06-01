#ifndef VSLAM_DIRECT_ICP_H__
#define VSLAM_DIRECT_ICP_H__
#include <map>
#include <memory>
#include <opencv4/opencv2/imgproc.hpp>
#include <string>

#include "core/Camera.h"
#include "core/Frame.h"
#include "core/Pose.h"
#include "core/types.h"

namespace vslam
{
class DirectIcpOverlay;
class DirectIcp
{
public:
  typedef std::shared_ptr<DirectIcp> ShPtr;

  struct Feature
  {
    typedef std::shared_ptr<Feature> ShPtr;
    size_t idx;
    Vec2d uv0;
    Vec2d iz0;
    Vec3d p0;
    Vec6d JZJw;
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
    TDistributionBivariate(double dof, double precision = 1e-3, int maxIterations = 50);
    void computeWeights(const std::vector<Feature::ShPtr> & r);
    double computeWeight(const Vec2d & r) const;

  private:
    const double _dof, _precision;
    const int _maxIterations;
    Mat2d _scale;
  };

  static std::map<std::string, double> defaultParameters();

  DirectIcp(const std::map<std::string, double> params);
  DirectIcp(
    int nLevels = 4, double weightPrior = 0.0, double minGradientIntensity = 5,
    double minGradientDepth = INFd, double maxGradientDepth = 0.3, double maxZ = 5.0,
    double maxIterations = 100, double minParameterUpdate = 1e-4, double maxErrorIncrease = 1.1,
    int maxPoints = INFi);
  Pose computeEgomotion(const Frame & frame0, const Frame & frame1, const Pose & guess);

  Pose computeEgomotion(
    Camera::ConstShPtr cam, const cv::Mat & intensity0, const cv::Mat & depth0,
    const cv::Mat & intensity1, const cv::Mat & depth1, const Pose & guess);

  int nLevels() { return _nLevels; }

private:
  const std::shared_ptr<DirectIcpOverlay> _log;
  const TDistributionBivariate::ShPtr _weightFunction;
  const int _nLevels, _maxPoints;
  const double _weightPrior, _minGradientIntensity, _minGradientDepth, _maxGradientDepth, _maxDepth,
    _maxIterations, _minParameterUpdate, _maxErrorIncrease;

  int _level, _iteration;

  std::vector<Feature::ShPtr> computeResidualsAndJacobian(
    const std::vector<Feature::ShPtr> & features, const Frame & f1, const SE3d & motion);

  std::vector<Feature::ShPtr> extractFeatures(const Frame & frame, const SE3d & motion) const;
  Matd<2, 6> computeJacobianWarp(const Vec3d & p, Camera::ConstShPtr cam) const;

  Vec6d computeJacobianSE3z(const Vec3d & p) const;
  Vec2d interpolate(const cv::Mat & intensity, const cv::Mat & depth, const Vec2d & uv);
  std::vector<Feature::ShPtr> uniformSubselection(
    Camera::ConstShPtr cam, const std::vector<Feature::ShPtr> & features) const;
};

}  // namespace vslam
#endif
