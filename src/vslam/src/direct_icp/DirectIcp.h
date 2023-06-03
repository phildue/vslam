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

  struct Constraint
  {
    typedef std::shared_ptr<Constraint> ShPtr;
    size_t idx;
    Vec2f uv0;
    Vec3f p0;
    Vec2f iz0;
    Vec6f JZJw;
    Matf<2, 6> J;
    Mat2f weight;
    Vec2f residual;
    bool valid;
  };

  struct NormalEquations
  {
    Mat6f A;
    Vec6f b;
    float error;
    NormalEquations operator+(const NormalEquations & that) const
    {
      return NormalEquations({A + that.A, b + that.b, error + that.error});
    }
  };

  class TDistributionBivariate
  {
  public:
    typedef std::shared_ptr<TDistributionBivariate> ShPtr;
    TDistributionBivariate(double dof, double precision = 1e-3, int maxIterations = 50);
    void computeWeights(const std::vector<Constraint::ShPtr> & r);
    double computeWeight(const Vec2f & r) const;

  private:
    const double _dof, _precision;
    const int _maxIterations;
    Mat2f _scale;
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

  std::vector<Constraint::ShPtr> selectConstraintsAndPrecompute(
    const Frame & frame, const SE3f & motion) const;

  std::vector<Constraint::ShPtr> computeResidualsAndJacobian(
    const std::vector<Constraint::ShPtr> & features, const Frame & f1, const SE3f & motion) const;

  NormalEquations computeNormalEquations(
    const std::vector<Constraint::ShPtr> & constraints, const SE3f & motion,
    const SE3f & prior) const;

  Matf<2, 6> computeJacobianWarp(const Vec3f & p, Camera::ConstShPtr cam) const;

  Vec6f computeJacobianSE3z(const Vec3f & p) const;
  Vec2f interpolate(const cv::Mat & intensity, const cv::Mat & depth, const Vec2f & uv) const;
  std::vector<Constraint::ShPtr> uniformSubselection(
    Camera::ConstShPtr cam, const std::vector<Constraint::ShPtr> & features) const;
};

}  // namespace vslam
#endif
