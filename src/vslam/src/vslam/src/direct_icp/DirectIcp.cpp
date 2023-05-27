

#include "DirectIcp.h"
#include "core/random.h"
#include "utils/log.h"

#define DETAILED_SCOPES false
namespace vslam
{
TDistributionBivariate::TDistributionBivariate(double dof) : _dof(dof) {}

void TDistributionBivariate::computeWeights(
  const std::vector<Feature::ShPtr> & features, double precision, int maxIterations)
{
  TIMED_SCOPE_IF(timer3, "fit", DETAILED_SCOPES);
  VecXd weights = VecXd::Ones(features.size());
  for (int i = 0; i < maxIterations; i++) {
    Mat2d sum = Mat2d::Zero();
    for (size_t n = 0; n < features.size(); n++) {
      sum += weights(n) * features[n]->residual * features[n]->residual.transpose();
    }
    const Mat2d scale_i = (sum / features.size()).inverse();

    const double diff = (_scale - scale_i).norm();
    _scale = scale_i;
    for (size_t n = 0; n < features.size(); n++) {
      weights(n) = computeWeight(features[n]->residual);
      features[n]->weight = weights(n) * _scale;
    }

    if (diff < precision) {
      break;
    }
  }
}
double TDistributionBivariate::computeWeight(const Vec2d & r) const
{
  return (_dof + 2.0) / (_dof + r.transpose() * _scale * r);
}

DirectIcp::DirectIcp(Camera::ConstShPtr cam, const std::map<std::string, double> params)
: DirectIcp(
    cam, params.at("nLevels"), params.at("weightPrior"), params.at("minGradientIntensity"),
    params.at("minGradientDepth"), params.at("maxGradientDepth"), params.at("maxDepth"),
    params.at("maxIterations"), params.at("minParameterUpdate"), params.at("maxErrorIncrease"))
{
}
DirectIcp::DirectIcp(
  Camera::ConstShPtr cam, int nLevels, double weightPrior, double minGradientIntensity,
  double minGradientDepth, double maxGradientDepth, double maxZ, double maxIterations,
  double minParameterUpdate, double maxErrorIncrease)
: _weightFunction(std::make_shared<TDistributionBivariate>(5.0)),
  _nLevels(nLevels),
  _weightPrior(weightPrior),
  _minGradientIntensity(minGradientIntensity),
  _minGradientDepth(minGradientDepth),
  _maxGradientDepth(maxGradientDepth),
  _maxDepth(maxZ),
  _maxIterations(maxIterations),
  _minParameterUpdate(minParameterUpdate),
  _maxErrorIncrease(maxErrorIncrease)
{
  _cam.resize(_nLevels);
  _cam[0] = cam;
  for (int i = 1; i < _nLevels; i++) {
    _cam[i] = Camera::resize(_cam[i - 1], 0.5);
  }
}
SE3d DirectIcp::computeEgomotion(
  const cv::Mat & intensity, const cv::Mat & depth, const SE3d & guess)
{
  if (_I0.empty()) {
    _I0 = computePyramidIntensity(intensity);
    _Z0 = computePyramidDepth(depth);
    return guess;
  }
  TIMED_SCOPE(timer, "computeEgomotion");
  const std::vector<cv::Mat> I1 = computePyramidIntensity(intensity);
  const std::vector<cv::Mat> Z1 = computePyramidDepth(depth);

  SE3d prior = guess;
  SE3d motion = prior;
  for (int level = _nLevels - 1; level >= 0; level--) {
    TIMED_SCOPE_IF(timerLevel, format("computeLevel{}", level), DETAILED_SCOPES);
    const std::vector<Feature::ShPtr> features =
      extractFeatures(_I0[level], _Z0[level], _cam[level], motion);

    std::string reason = "Max iterations exceeded";
    double error = INFd;
    Vec6d dx = Vec6d::Zero();
    for (int iteration = 0; iteration < _maxIterations; iteration++) {
      TIMED_SCOPE_IF(timerIter, format("computeIteration{}", level), DETAILED_SCOPES);
      //auto subset = uniformSubselection(_cam[level], features);
      {
        TIMED_SCOPE_IF(timer1, "computeResidualAndJacobian", DETAILED_SCOPES);

        std::for_each(features.begin(), features.end(), [&](auto c) {
          c->valid = false;
          c->p0t = motion * c->p0;

          if (c->p0t.z() <= 0) return;

          c->uv0t = _cam[level]->project(c->p0t);

          if (!_cam[level]->withinImage(c->uv0t, 0.02)) return;

          Vec2d iz1w = interpolate(I1[level], Z1[level], c->uv0t);

          if (!std::isfinite(iz1w(0)) || !std::isfinite(iz1w(1))) return;

          c->p1t = motion.inverse() * _cam[level]->reconstruct(c->uv0t, iz1w(1));

          c->iz1w = Vec2d(iz1w(0), c->p1t.z());

          c->residual = c->iz1w - c->iz0;
          if (!std::isfinite(c->residual.norm())) return;

          c->JZJw_Jtz = c->JZJw - computeJacobianSE3z(c->p1t);

          c->J.row(0) = c->JIJw;
          c->J.row(1) = c->JZJw_Jtz;

          if (!std::isfinite(c->J.norm())) return;

          c->valid = true;
        });
      }
      std::vector<Feature::ShPtr> constraints;
      std::copy_if(features.begin(), features.end(), std::back_inserter(constraints), [](auto c) {
        return c->valid;
      });

      if (constraints.size() < 6) {
        reason = format("Not enough constraints: {}", constraints.size());
        motion = SE3();
        break;
      }
      _weightFunction->computeWeights(constraints);

      Mat6d A = _weightPrior * Mat6d::Identity();
      Vec6d b = _weightPrior * (motion * prior.inverse()).log();
      double error_i = b.norm();
      {
        TIMED_SCOPE_IF(timer2, "computeNormalEquations", DETAILED_SCOPES);
        for (size_t i = 0; i < constraints.size(); i++) {
          auto c = constraints[i];
          c->error = c->residual.transpose() * c->weight * c->residual;
          error_i += c->error;
          Matd<6, 2> Jw = c->J.transpose() * c->weight;
          A.noalias() += Jw * c->J;
          b.noalias() += Jw * c->residual;
        }
        if (error_i / error > _maxErrorIncrease) {
          reason = format("Error increased: {:.2f}/{:.2f}", error_i, error);
          motion = SE3d::exp(dx) * motion;
          break;
        }
        error = error_i;

        dx = A.ldlt().solve(b);
      }
      /*
        print(
          "Solving dx = b/A:\n A = \n{}\nb = \n{}\ndx = {}\nscale = \n{}\nerror = {:.4f} #: {} "
          "|A|={:.4f} "
          "|b|={:.4f} "
          "|dx|={:.4f}\nrSum={}\n",
          A, b.transpose(), dx.transpose(), _weightFunction->scale(), error, constraints.size(),
          A.norm(), b.norm(), dx.norm(), rSum.transpose());
        */
      motion = SE3d::exp(-dx) * motion;

      if (dx.norm() < _minParameterUpdate) {
        reason = format("Minimum step size reached: {:5.f}/{:5.f}", dx.norm(), _minParameterUpdate);
        break;
      }
    }
  }
  _I0 = I1;
  _Z0 = Z1;
  return motion;
}

std::vector<cv::Mat> DirectIcp::computePyramidIntensity(const cv::Mat & intensity) const
{
  std::vector<cv::Mat> pyramid(_nLevels);
  pyramid[0] = intensity;
  for (int i = 1; i < _nLevels; i++) {
    cv::pyrDown(pyramid[i - 1], pyramid[i]);
  }
  return pyramid;
}
std::vector<cv::Mat> DirectIcp::computePyramidDepth(const cv::Mat & depth) const
{
  std::vector<cv::Mat> pyramid(_nLevels);
  pyramid[0] = depth;
  for (int i = 1; i < _nLevels; i++) {
    cv::Mat out(cv::Size(pyramid[i - 1].cols / 2, pyramid[i - 1].rows / 2), CV_32F);
    for (int y = 0; y < out.rows; ++y) {
      for (int x = 0; x < out.cols; ++x) {
        out.at<float>(y, x) = pyramid[i - 1].at<float>(y * 2, x * 2);
      }
      pyramid[i] = out;
    }
    //cv::resize(pyramid[i - 1], pyramid[i], cv::Size(), 0.5, 0.5, cv::INTER_NEAREST);
  }
  return pyramid;
}
cv::Mat DirectIcp::computeJacobianImage(const cv::Mat & image) const
{
  cv::Mat dIdx, dIdy;
  cv::Sobel(image, dIdx, CV_32F, 1, 0, 3, 1. / 8.);
  cv::Sobel(image, dIdy, CV_32F, 0, 1, 3, 1. / 8.);

  cv::Mat dIdxy;
  cv::merge(std::vector<cv::Mat>({dIdx, dIdy}), dIdxy);
  return dIdxy;
}

cv::Mat DirectIcp::computeJacobianDepth(const cv::Mat & depth) const
{
  cv::Mat dZdx(cv::Size(depth.cols, depth.rows), CV_32F),
    dZdy(cv::Size(depth.cols, depth.rows), CV_32F);

  auto validZ = [](float z0, float z1) {
    return std::isfinite(z0) && z0 > 0 && std::isfinite(z1) && z1 > 0;
  };
  for (int y = 0; y < depth.rows; ++y) {
    for (int x = 0; x < depth.cols; ++x) {
      const int y0 = std::max(y - 1, 0);
      const int y1 = std::min(y + 1, depth.rows - 1);
      const int x0 = std::max(x - 1, 0);
      const int x1 = std::min(x + 1, depth.cols - 1);
      const float zyx0 = depth.at<float>(y, x0);
      const float zyx1 = depth.at<float>(y, x1);
      const float zy0x = depth.at<float>(y0, x);
      const float zy1x = depth.at<float>(y1, x);

      dZdx.at<float>(y, x) =
        validZ(zyx0, zyx1) ? (zyx1 - zyx0) * 0.5f : std::numeric_limits<float>::quiet_NaN();
      dZdy.at<float>(y, x) =
        validZ(zy0x, zy1x) ? (zy1x - zy0x) * 0.5f : std::numeric_limits<float>::quiet_NaN();
    }
  }
  cv::Mat dZdxy;
  cv::merge(std::vector<cv::Mat>({dZdx, dZdy}), dZdxy);
  return dZdxy;
}
std::vector<Feature::ShPtr> DirectIcp::extractFeatures(
  const cv::Mat & intensity, const cv::Mat & depth, Camera::ConstShPtr cam,
  const SE3d & motion) const
{
  const cv::Mat dI = computeJacobianImage(intensity);
  const cv::Mat dZ = computeJacobianDepth(depth);

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

        c->p0 = cam->reconstruct(c->uv0, c->iz0(1));
        Mat<double, 2, 6> Jw = computeJacobianWarp(motion * c->p0, cam);
        c->JIJw = dIvu[0] * Jw.row(0) + dIvu[1] * Jw.row(1);
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

std::vector<Feature::ShPtr> DirectIcp::uniformSubselection(
  Camera::ConstShPtr cam, const std::vector<Feature::ShPtr> & interestPoints) const
{
  TIMED_SCOPE_IF(timer, "uniformSubselection", DETAILED_SCOPES);
  const size_t nNeeded = std::max<size_t>(20, size_t(1000));
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