

#include <execution>

#include "DirectIcp.h"
#include "DirectIcpOverlay.h"
#include "core/random.h"
#include "utils/log.h"
#define DETAILED_SCOPES true
namespace vslam
{
DirectIcp::TDistributionBivariate::TDistributionBivariate(
  double dof, double precision, int maxIterations)
: _dof(dof), _precision(precision), _maxIterations(maxIterations)
{
}

void DirectIcp::TDistributionBivariate::computeWeights(const std::vector<Feature::ShPtr> & features)
{
  VecXd weights = VecXd::Ones(features.size());
  std::vector<Mat2d> rrT(features.size());
  for (size_t n = 0; n < features.size(); n++) {
    rrT[n] = features[n]->residual * features[n]->residual.transpose();
  }

  for (int i = 0; i < _maxIterations; i++) {
    TIMED_SCOPE_IF(timerLevel, format("computeWeightsIteration"), DETAILED_SCOPES);
    std::vector<Mat2d> wrrT(features.size());
    for (size_t n = 0; n < features.size(); n++) {
      wrrT[n] = weights(n) * rrT[n];
    }
    Mat2d sum = std::accumulate(rrT.begin(), rrT.end(), Mat2d::Zero().eval());

    const Mat2d scale_i = (sum / features.size()).inverse();

    const double diff = (_scale - scale_i).norm();
    _scale = scale_i;
    for (size_t n = 0; n < features.size(); n++) {
      weights(n) = computeWeight(features[n]->residual);
      features[n]->weight = weights(n) * _scale;
    }

    if (diff < _precision) {
      break;
    }
  }
}
double DirectIcp::TDistributionBivariate::computeWeight(const Vec2d & r) const
{
  return (_dof + 2.0) / (_dof + r.transpose() * _scale * r);
}

std::map<std::string, double> DirectIcp::defaultParameters()
{
  return {{"nLevels", 4.0},           {"weightPrior", 0.0},         {"minGradientIntensity", 5},
          {"minGradientDepth", 0.01}, {"maxGradientDepth", 0.3},    {"maxDepth", 5.0},
          {"maxIterations", 100},     {"minParameterUpdate", 1e-4}, {"maxErrorIncrease", 1.1},
          {"maxPoints", 640 * 480}};
}

DirectIcp::DirectIcp(const std::map<std::string, double> params)
: DirectIcp(
    params.at("nLevels"), params.at("weightPrior"), params.at("minGradientIntensity"),
    params.at("minGradientDepth"), params.at("maxGradientDepth"), params.at("maxDepth"),
    params.at("maxIterations"), params.at("minParameterUpdate"), params.at("maxErrorIncrease"),
    params.at("maxPoints"))
{
}
DirectIcp::DirectIcp(
  int nLevels, double weightPrior, double minGradientIntensity, double minGradientDepth,
  double maxGradientDepth, double maxZ, double maxIterations, double minParameterUpdate,
  double maxErrorIncrease, int maxPoints)
: _log(std::make_shared<DirectIcpOverlay>()),
  _weightFunction(std::make_shared<TDistributionBivariate>(5.0, 1e-3, 10)),
  _nLevels(nLevels),
  _maxPoints(maxPoints),
  _weightPrior(weightPrior),
  _minGradientIntensity(minGradientIntensity),
  _minGradientDepth(minGradientDepth),
  _maxGradientDepth(maxGradientDepth),
  _maxDepth(maxZ),
  _maxIterations(maxIterations),
  _minParameterUpdate(minParameterUpdate),
  _maxErrorIncrease(maxErrorIncrease)
{
}

Pose DirectIcp::computeEgomotion(
  Camera::ConstShPtr cam, const cv::Mat & intensity0, const cv::Mat & depth0,
  const cv::Mat & intensity1, const cv::Mat & depth1, const Pose & guess)
{
  Frame f0(intensity0, depth0, cam);
  f0.computePyramid(_nLevels);
  f0.computeDerivatives();
  f0.computePcl();
  Frame f1(intensity1, depth1, cam);
  f1.computePyramid(_nLevels);

  return computeEgomotion(f0, f1, guess);
}

Pose DirectIcp::computeEgomotion(const Frame & frame0, const Frame & frame1, const Pose & guess)
{
  TIMED_SCOPE(timer, "computeEgomotion");

  SE3d prior = guess.SE3();
  Pose motion = Pose(prior);
  for (_level = _nLevels - 1; _level >= 0; _level--) {
    TIMED_SCOPE_IF(timerLevel, format("computeLevel{}", _level), DETAILED_SCOPES);
    const Frame f0 = frame0.level(_level);
    const Frame f1 = frame1.level(_level);
    std::vector<Feature::ShPtr> features = extractFeatures(f0, motion.SE3());
    features = uniformSubselection(f0.camera(), features);

    std::string reason = "Max iterations exceeded";
    double error = INFd;
    Vec6d dx = Vec6d::Zero();
    for (_iteration = 0; _iteration < _maxIterations; _iteration++) {
      TIMED_SCOPE_IF(timerIter, format("computeIteration{}", _level), DETAILED_SCOPES);

      auto constraints = computeResidualsAndJacobian(features, f1, motion.SE3());

      if (constraints.size() < 6) {
        reason = format("Not enough constraints: {}", constraints.size());
        motion = SE3();
        break;
      }
      {
        TIMED_SCOPE_IF(timer3, format("computeWeights{}", _level), DETAILED_SCOPES);
        _weightFunction->computeWeights(constraints);
      }
      {
        TIMED_SCOPE_IF(timer2, format("computeNormalEquations{}", _level), DETAILED_SCOPES);

        std::vector<Mat6d> As(constraints.size());
        std::vector<Vec6d> bs(constraints.size());
        VecXd errors = VecXd::Zero(constraints.size());
        std::for_each(
          std::execution::par_unseq, constraints.begin(), constraints.end(), [&](auto c) {
            c->error = c->residual.transpose() * c->weight * c->residual;
            errors(c->idx) = c->error;
            As[c->idx] = c->J.transpose() * c->weight * c->J;
            bs[c->idx] = c->J.transpose() * c->weight * c->residual;
          });
        const Mat6d A =
          std::accumulate(As.begin(), As.end(), (_weightPrior * Mat6d::Identity()).eval());
        const Vec6d b = std::accumulate(
          bs.begin(), bs.end(), (_weightPrior * (motion * prior.inverse()).log()).eval());
        const double error_i = errors.sum();
        if (error_i / error > _maxErrorIncrease) {
          reason = format("Error increased: {:.2f}/{:.2f}", error_i, error);
          motion = SE3d::exp(dx) * motion;
          break;
        }
        error = error_i;

        dx = A.ldlt().solve(b);
        //https://stats.stackexchange.com/questions/482985/non-linear-least-squares-covariance-estimate

        motion.cov() = error / (constraints.size() - 6) * A.inverse();
        motion.SE3() = SE3d::exp(-dx) * motion.SE3();
      }
      //_log->update(DirectIcpOverlay::Entry({frame0, frame1, constraints}));
      if (dx.norm() < _minParameterUpdate) {
        reason = format("Minimum step size reached: {:5.f}/{:5.f}", dx.norm(), _minParameterUpdate);
        break;
      }
    }
  }
  return motion;
}

std::vector<DirectIcp::Feature::ShPtr> DirectIcp::extractFeatures(
  const Frame & frame, const SE3d & motion) const
{
  TIMED_SCOPE_IF(timer2, format("extractFeatures{}", _level), DETAILED_SCOPES);
  const cv::Mat & intensity = frame.intensity();
  const cv::Mat & depth = frame.depth();
  const cv::Mat & dI = frame.dI();
  const cv::Mat & dZ = frame.dZ();

  std::vector<Feature::ShPtr> constraints;
  constraints.reserve(intensity.cols * intensity.rows);
  for (int u = 0; u < intensity.cols; u++) {
    for (int v = 0; v < intensity.rows; v++) {
      const double z = depth.at<float>(v, u);
      const double i = intensity.at<uint8_t>(v, u);
      const cv::Vec2f dIvu = dI.at<cv::Vec2f>(v, u);
      const cv::Vec2f dZvu = dZ.at<cv::Vec2f>(v, u);

      if (
        std::isfinite(z) && std::isfinite(dZvu[0]) && std::isfinite(dZvu[1]) && 0 < z &&
        z < _maxDepth && std::abs(dZvu[0]) < _maxGradientDepth &&
        std::abs(dZvu[1]) < _maxGradientDepth &&
        (std::abs(dIvu[0]) > _minGradientIntensity || std::abs(dIvu[1]) > _minGradientIntensity ||
         std::abs(dZvu[0]) > _minGradientDepth || std::abs(dZvu[1]) > _minGradientDepth)) {
        auto c = std::make_shared<Feature>();
        c->idx = constraints.size();
        c->uv0 = Vec2d(u, v);
        c->iz0 = Vec2d(i, z);

        c->p0 = frame.p3d(v, u);
        Mat<double, 2, 6> Jw = computeJacobianWarp(motion * c->p0, frame.camera());
        c->J.row(0) = dIvu[0] * Jw.row(0) + dIvu[1] * Jw.row(1);
        c->JZJw = dZvu[0] * Jw.row(0) + dZvu[1] * Jw.row(1);
        constraints.push_back(c);
      }
    }
  }
  return constraints;
}
Matd<2, 6> DirectIcp::computeJacobianWarp(const Vec3d & p, Camera::ConstShPtr cam) const
{
  const double & x = p.x();
  const double & y = p.y();
  const double z_inv = 1. / p.z();
  const double z_inv_2 = z_inv * z_inv;

  Matd<2, 6> J;
  J(0, 0) = z_inv;
  J(0, 1) = 0.0;
  J(0, 2) = -x * z_inv_2;
  J(0, 3) = y * J(0, 2);
  J(0, 4) = 1.0 - x * J(0, 2);
  J(0, 5) = -y * z_inv;
  J.row(0) *= cam->fx();
  J(1, 0) = 0.0;
  J(1, 1) = z_inv;
  J(1, 2) = -y * z_inv_2;
  J(1, 3) = -1.0 + y * J(1, 2);
  J(1, 4) = -J(1, 3);
  J(1, 5) = x * z_inv;
  J.row(1) *= cam->fy();

  return J;
}

std::vector<DirectIcp::Feature::ShPtr> DirectIcp::computeResidualsAndJacobian(
  const std::vector<DirectIcp::Feature::ShPtr> & features, const Frame & f1, const SE3d & motion)
{
  TIMED_SCOPE_IF(timer1, format("computeResidualAndJacobian{}", _level), DETAILED_SCOPES);
  SE3d motionInv = motion.inverse();
  std::for_each(std::execution::par_unseq, features.begin(), features.end(), [&](auto c) {
    c->valid = false;
    c->p0t = motion * c->p0;

    c->uv0t = f1.project(c->p0t);

    Vec2d iz1w = interpolate(f1.I(), f1.Z(), c->uv0t);

    c->p1t = motionInv * f1.reconstruct(c->uv0t, iz1w(1));

    c->iz1w = Vec2d(iz1w(0), c->p1t.z());

    c->residual = c->iz1w - c->iz0;

    c->J.row(1) = c->JZJw - computeJacobianSE3z(c->p1t);

    c->valid = !(
      c->p0t.z() <= 0 || !f1.camera()->withinImage(c->uv0t, 0.02) || !std::isfinite(iz1w(0)) ||
      !std::isfinite(iz1w(1)) || !std::isfinite(c->residual.norm()) || !std::isfinite(c->J.norm()));
  });
  std::vector<Feature::ShPtr> constraints;
  std::copy_if(features.begin(), features.end(), std::back_inserter(constraints), [](auto c) {
    return c->valid;
  });
  int idx = 0;
  std::for_each(constraints.begin(), constraints.end(), [&idx](auto c) { c->idx = idx++; });
  return constraints;
}

Vec6d DirectIcp::computeJacobianSE3z(const Vec3d & p) const
{
  Vec6d J;
  J(0) = 0.0;
  J(1) = 0.0;
  J(2) = 1.0;
  J(3) = p(1);
  J(4) = -p(0);
  J(5) = 0.0;

  return J;
}

Vec2d DirectIcp::interpolate(const cv::Mat & intensity, const cv::Mat & depth, const Vec2d & uv)
{
  auto sample = [&](int v, int u) -> Vec2d {
    const double z = depth.at<float>(v, u);
    return Vec2d(intensity.at<uint8_t>(v, u), std::isfinite(z) ? z : 0);
  };
  const double u = uv(0);
  const double v = uv(1);
  const double u0 = std::floor(u);
  const double u1 = std::ceil(u);
  const double v0 = std::floor(v);
  const double v1 = std::ceil(v);
  const double w_u1 = u - u0;
  const double w_u0 = 1.0 - w_u1;
  const double w_v1 = v - v0;
  const double w_v0 = 1.0 - w_v1;
  const Vec2d iz00 = sample(v0, u0);
  const Vec2d iz01 = sample(v0, u1);
  const Vec2d iz10 = sample(v1, u0);
  const Vec2d iz11 = sample(v1, u1);

  const double w00 = iz00(1) > 0 ? w_v0 * w_u0 : 0;
  const double w01 = iz01(1) > 0 ? w_v0 * w_u1 : 0;
  const double w10 = iz10(1) > 0 ? w_v1 * w_u0 : 0;
  const double w11 = iz11(1) > 0 ? w_v1 * w_u1 : 0;

  Vec2d izw = w00 * iz00 + w01 * iz01 + w10 * iz10 + w11 * iz11;
  izw /= w00 + w01 + w10 + w11;
  return izw;
}

std::vector<DirectIcp::Feature::ShPtr> DirectIcp::uniformSubselection(
  Camera::ConstShPtr cam, const std::vector<DirectIcp::Feature::ShPtr> & interestPoints) const
{
  TIMED_SCOPE_IF(timer, format("uniformSubselection{}", _level), DETAILED_SCOPES);
  const size_t nNeeded = std::max<size_t>(20, _maxPoints);
  std::vector<bool> mask(cam->width() * cam->height(), false);
  std::vector<Feature::ShPtr> subset;
  subset.reserve(interestPoints.size());
  if (nNeeded < interestPoints.size()) {
    while (subset.size() < nNeeded) {
      auto ip = interestPoints[random::U(0, interestPoints.size() - 1)];
      const size_t idx = ip->uv0(1) * cam->width() + ip->uv0(0);
      if (!mask[idx]) {
        subset.push_back(ip);
        mask[idx] = true;
      }
    }
    return subset;
  }
  return interestPoints;
}

}  // namespace vslam