

#include <execution>
#include <numeric>

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

void DirectIcp::TDistributionBivariate::computeWeights(
  const std::vector<Constraint::ShPtr> & features)
{
  VecXf weights = VecXf::Ones(features.size());
  std::vector<Mat2f> rrT(features.size());
  for (size_t n = 0; n < features.size(); n++) {
    rrT[n] = features[n]->residual * features[n]->residual.transpose();
  }

  for (int i = 0; i < _maxIterations; i++) {
    TIMED_SCOPE_IF(timerLevel, format("computeWeightsIteration"), DETAILED_SCOPES);
    std::vector<Mat2f> wrrT(features.size());
    for (size_t n = 0; n < features.size(); n++) {
      wrrT[n] = weights(n) * rrT[n];
    }
    Mat2f sum = std::accumulate(rrT.begin(), rrT.end(), Mat2f::Zero().eval());

    const Mat2f scale_i = (sum / features.size()).inverse();

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
double DirectIcp::TDistributionBivariate::computeWeight(const Vec2f & r) const
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

  SE3f prior = guess.SE3().cast<float>();
  SE3f motion = prior;
  Mat6f covariance;
  for (_level = _nLevels - 1; _level >= 0; _level--) {
    TIMED_SCOPE_IF(timerLevel, format("computeLevel{}", _level), DETAILED_SCOPES);
    const Frame f0 = frame0.level(_level);
    const Frame f1 = frame1.level(_level);
    const auto constraintsAll = selectConstraintsAndPrecompute(f0, motion);

    std::string reason = "Max iterations exceeded";
    double error = INFd;
    Vec6f dx = Vec6f::Zero();
    for (_iteration = 0; _iteration < _maxIterations; _iteration++) {
      TIMED_SCOPE_IF(timerIter, format("computeIteration{}", _level), DETAILED_SCOPES);

      auto constraintsValid = computeResidualsAndJacobian(constraintsAll, f1, motion);

      if (constraintsValid.size() < 6) {
        reason = format("Not enough constraints: {}", constraintsValid.size());
        motion = SE3f();
        break;
      }
      {
        TIMED_SCOPE_IF(timer3, format("computeWeights{}", _level), DETAILED_SCOPES);
        _weightFunction->computeWeights(constraintsValid);
      }

      const NormalEquations ne = computeNormalEquations(constraintsValid, motion, prior);

      if (ne.error / error > _maxErrorIncrease) {
        reason = format("Error increased: {:.2f}/{:.2f}", ne.error, error);
        motion = SE3f::exp(dx) * motion;
        break;
      }
      error = ne.error;

      dx = ne.A.ldlt().solve(ne.b);

      motion = SE3f::exp(-dx) * motion;
      //https://stats.stackexchange.com/questions/482985/non-linear-least-squares-covariance-estimate
      covariance = error / (constraintsValid.size() - 6) * ne.A.inverse();
      //_log->update(DirectIcpOverlay::Entry({frame0, frame1, constraints}));

      if (dx.norm() < _minParameterUpdate) {
        reason = format("Minimum step size reached: {:5.f}/{:5.f}", dx.norm(), _minParameterUpdate);
        break;
      }
    }
  }
  return Pose(motion.cast<double>(), covariance.cast<double>());
}

std::vector<DirectIcp::Constraint::ShPtr> DirectIcp::selectConstraintsAndPrecompute(
  const Frame & frame, const SE3f & motion) const
{
  TIMED_SCOPE_IF(timer2, format("selectConstraintsAndPrecompute{}", _level), DETAILED_SCOPES);
  const cv::Mat & intensity = frame.intensity();
  const cv::Mat & depth = frame.depth();
  const cv::Mat & dI = frame.dI();
  const cv::Mat & dZ = frame.dZ();

  std::vector<int> vs(intensity.rows);
  for (int v = 0; v < intensity.rows; v++) {
    vs[v] = v;
  }
  //TODO this could be a transform_reduce:
  // ( transform: (uv) -> (vector<vector<Constraint>>) | accumulate: vector<vector<Constraint>> -> (vector<Constraint>))
  std::vector<std::vector<Constraint::ShPtr>> cs(intensity.rows);
  std::transform(std::execution::par_unseq, vs.begin(), vs.end(), cs.begin(), [&](int v) {
    const float * zv = depth.ptr<float>(v);
    const uint8_t * iv = intensity.ptr<uint8_t>(v);
    const cv::Vec2f * dIv = dI.ptr<cv::Vec2f>(v);
    const cv::Vec2f * dZv = dZ.ptr<cv::Vec2f>(v);
    std::vector<Constraint::ShPtr> constraints;
    constraints.reserve(intensity.cols);
    for (int u = 0; u < intensity.cols; u++) {
      if (
        std::isfinite(zv[u]) && std::isfinite(dZv[u][0]) && std::isfinite(dZv[u][1]) && 0 < zv[u] &&
        zv[u] < _maxDepth && std::abs(dZv[u][0]) < _maxGradientDepth &&
        std::abs(dZv[u][1]) < _maxGradientDepth &&
        (std::abs(dIv[u][0]) > _minGradientIntensity ||
         std::abs(dIv[u][1]) > _minGradientIntensity || std::abs(dZv[u][0]) > _minGradientDepth ||
         std::abs(dZv[u][1]) > _minGradientDepth)) {
        auto c = std::make_shared<Constraint>();
        c->idx = constraints.size();
        c->uv0 = Vec2f(u, v);
        c->iz0 = Vec2f(iv[u], zv[u]);

        c->p0 = frame.p3d(v, u).cast<float>();
        Mat<float, 2, 6> Jw = computeJacobianWarp(motion * c->p0, frame.camera());
        c->J.row(0) = dIv[u][0] * Jw.row(0) + dIv[u][1] * Jw.row(1);
        c->JZJw = dZv[u][0] * Jw.row(0) + dZv[u][1] * Jw.row(1);
        constraints.push_back(c);
      }
    }
    return constraints;
  });
  std::vector<Constraint::ShPtr> constraints;
  std::for_each(cs.begin(), cs.end(), [&](auto c) {
    constraints.insert(constraints.end(), c.begin(), c.end());
  });
  return uniformSubselection(frame.camera(), constraints);
}  // namespace vslam
Matf<2, 6> DirectIcp::computeJacobianWarp(const Vec3f & p, Camera::ConstShPtr cam) const
{
  const double & x = p.x();
  const double & y = p.y();
  const double z_inv = 1. / p.z();
  const double z_inv_2 = z_inv * z_inv;

  Matf<2, 6> J;
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

std::vector<DirectIcp::Constraint::ShPtr> DirectIcp::computeResidualsAndJacobian(
  const std::vector<DirectIcp::Constraint::ShPtr> & constraints, const Frame & f1,
  const SE3f & motion) const
{
  TIMED_SCOPE_IF(timer1, format("computeResidualAndJacobian{}", _level), DETAILED_SCOPES);

  /*Cache some constants for faster loop*/
  const SE3f motionInv = motion.inverse();
  const Camera::ConstShPtr cam = f1.camera();
  const Mat3f K = cam->K().cast<float>();
  const Mat3f R = motion.rotationMatrix();
  const Vec3f t = motion.translation();
  const Mat3f Kinv = cam->Kinv().cast<float>();
  const Mat3f Rinv = motionInv.rotationMatrix();
  const Vec3f tinv = motionInv.translation();
  const cv::Mat & I1 = f1.I();
  const cv::Mat & Z1 = f1.Z();
  const float h = f1.height();
  const float w = f1.width();
  const int bh = std::max<int>(0, (int)(0.01f * h));
  const int bw = std::max<int>(0, (int)(0.01f * w));

  auto withinImage = [&](const Vec2f & uv) -> bool {
    return (bw < uv(0) && uv(0) < w - bw && bh < uv(1) && uv(1) < h - bh);
  };

  std::for_each(std::execution::par_unseq, constraints.begin(), constraints.end(), [&](auto c) {
    const Vec3f p0t = K * ((R * c->p0) + t);
    const Vec2f uv0t = Vec2f(p0t(0), p0t(1)) / p0t(2);
    const Vec2f iz1w = interpolate(I1, Z1, uv0t);

    const Vec3f p1t = (Rinv * (iz1w(1) * (Kinv * Vec3f(uv0t(0), uv0t(1), 1.0)))) + tinv;

    c->residual = Vec2f(iz1w(0), p1t.z()) - c->iz0;

    c->J.row(1) = c->JZJw - computeJacobianSE3z(p1t);

    c->valid = p0t.z() > 0 && withinImage(uv0t) && std::isfinite(iz1w.norm()) &&
               std::isfinite(c->residual.norm()) && std::isfinite(c->J.norm());
  });
  std::vector<Constraint::ShPtr> constraintsValid;
  std::copy_if(
    constraints.begin(), constraints.end(), std::back_inserter(constraintsValid),
    [](auto c) { return c->valid; });
  int idx = 0;
  std::for_each(
    constraintsValid.begin(), constraintsValid.end(), [&idx](auto c) { c->idx = idx++; });
  return constraintsValid;
}

DirectIcp::NormalEquations DirectIcp::computeNormalEquations(
  const std::vector<DirectIcp::Constraint::ShPtr> & constraints, const SE3f & motion,
  const SE3f & prior) const
{
  TIMED_SCOPE_IF(timer2, format("computeNormalEquations{}", _level), DETAILED_SCOPES);
  const Vec6f diffPrior = (motion * prior.inverse()).log().cast<float>();
  return std::transform_reduce(
    std::execution::par_unseq, constraints.begin(), constraints.end(),
    NormalEquations({_weightPrior * Mat6f::Identity(), _weightPrior * diffPrior, diffPrior.norm()}),
    std::plus<NormalEquations>{}, [](auto c) {
      return NormalEquations(
        {c->J.transpose() * c->weight * c->J, c->J.transpose() * c->weight * c->residual,
         c->residual.transpose() * c->weight * c->residual});
    });
}

Vec6f DirectIcp::computeJacobianSE3z(const Vec3f & p) const
{
  Vec6f J;
  J(0) = 0.0;
  J(1) = 0.0;
  J(2) = 1.0;
  J(3) = p(1);
  J(4) = -p(0);
  J(5) = 0.0;

  return J;
}

Vec2f DirectIcp::interpolate(
  const cv::Mat & intensity, const cv::Mat & depth, const Vec2f & uv) const
{
  auto sample = [&](int v, int u) -> Vec2f {
    const double z = depth.at<float>(v, u);
    return Vec2f(intensity.at<uint8_t>(v, u), std::isfinite(z) ? z : 0);
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
  const Vec2f iz00 = sample(v0, u0);
  const Vec2f iz01 = sample(v0, u1);
  const Vec2f iz10 = sample(v1, u0);
  const Vec2f iz11 = sample(v1, u1);

  const double w00 = iz00(1) > 0 ? w_v0 * w_u0 : 0;
  const double w01 = iz01(1) > 0 ? w_v0 * w_u1 : 0;
  const double w10 = iz10(1) > 0 ? w_v1 * w_u0 : 0;
  const double w11 = iz11(1) > 0 ? w_v1 * w_u1 : 0;

  Vec2f izw = w00 * iz00 + w01 * iz01 + w10 * iz10 + w11 * iz11;
  izw /= w00 + w01 + w10 + w11;
  return izw;
}

std::vector<DirectIcp::Constraint::ShPtr> DirectIcp::uniformSubselection(
  Camera::ConstShPtr cam, const std::vector<DirectIcp::Constraint::ShPtr> & interestPoints) const
{
  TIMED_SCOPE_IF(timer, format("uniformSubselection{}", _level), DETAILED_SCOPES);
  const size_t nNeeded = std::max<size_t>(20, _maxPoints);
  std::vector<bool> mask(cam->width() * cam->height(), false);
  std::vector<Constraint::ShPtr> subset;
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