
#include <fmt/chrono.h>
#include <fmt/core.h>

#include "vslam/vslam.h"

using fmt::format;
using fmt::print;
template <typename T>
struct fmt::formatter<T, std::enable_if_t<std::is_base_of_v<Eigen::DenseBase<T>, T>, char>>
: ostream_formatter
{
};
#include <algorithm>
#include <execution>

using namespace pd;
using namespace pd::vslam;

#include <opencv4/opencv2/calib3d.hpp>
#include <opencv4/opencv2/core/eigen.hpp>
#include <opencv4/opencv2/core/utility.hpp>
#include <opencv4/opencv2/imgproc.hpp>

cv::Mat createIntensityDepthOverlay(const cv::Mat & intensity, const cv::Mat & depth)
{
  cv::Mat depthNorm, depthBgr, intensityBgr, joint;
  double Min, zMax;
  cv::minMaxLoc(depth, &Min, &zMax);

  depth.convertTo(depthBgr, CV_8UC3, 255.0 / zMax);
  cv::applyColorMap(depthBgr, depthBgr, cv::COLORMAP_JET);

  intensity.convertTo(intensityBgr, CV_8UC3, 255);
  cv::cvtColor(intensityBgr, intensityBgr, cv::COLOR_GRAY2BGR);

  cv::Mat weightsI(cv::Size(intensity.cols, intensity.rows), CV_32FC1, 0.7);
  cv::Mat weightsZ(cv::Size(intensity.cols, intensity.rows), CV_32FC1, 0.3);
  cv::blendLinear(intensityBgr, depthBgr, weightsI, weightsZ, joint);
  return joint;
}
struct Feature
{
  typedef std::shared_ptr<Feature> ShPtr;
  Vec2d uv0;
  Vec2d iz0;
  Vec2d JI;
  Vec2d JZ;
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
  TDistributionBivariate(double dof, const Mat2d & scale) : _dof(dof), _scale(scale) {}

  void fit(
    const std::vector<Vec2d> & r, const VecXd & w, double precision = 1e-3, int maxIterations = 50)
  {
    _weights = w;
    int i = 0;
    for (; i < maxIterations; i++) {
      Mat2d sum = Mat2d::Zero();
      for (size_t n = 0; n < r.size(); n++) {
        Mat2d outer = r[n] * r[n].transpose();
        sum.noalias() += _weights(n) * outer;
      }

      const Mat2d scale_i = (sum / r.size()).inverse();
      const double diff = (_scale - scale_i).norm();
      _scale = scale_i;
      for (size_t n = 0; n < r.size(); n++) {
        _weights(n) = computeWeight(r[n]);
      }

      if (diff < precision) {
        break;
      }
    }
  }
  double computeWeight(const Vec2d & r) const
  {
    return (_dof + 2.0) / (_dof + r.transpose() * _scale * r);
  }
  VecXd weights() const { return _weights; }
  Mat2d & scale() { return _scale; }

private:
  const double _dof;
  Mat2d _scale;
  VecXd _weights;
};

class LogNull
{
public:
  typedef std::shared_ptr<LogNull> ShPtr;
  void logLevel(
    int UNUSED(level), const cv::Mat & UNUSED(intensity0), const cv::Mat & UNUSED(depth0),
    const cv::Mat & UNUSED(intensity1), const cv::Mat & UNUSED(depth1))
  {
  }
  void logIteration(int UNUSED(iteration), const std::vector<Feature::ShPtr> & UNUSED(constraints))
  {
  }
  void logConverged(const std::string & UNUSED(reason), const SE3d & UNUSED(motion)) {}
};

class LogShow
{
public:
  typedef std::shared_ptr<LogShow> ShPtr;
  LogShow(int waitTime) : _waitTime(waitTime) { _s = 2; }
  void logLevel(
    int level, const cv::Mat & intensity0, const cv::Mat & depth0, const cv::Mat & intensity1,
    const cv::Mat & depth1)
  {
    _level = level;
    cv::Mat ZI0 = createIntensityDepthOverlay(intensity0, depth0);
    cv::Mat ZI1 = createIntensityDepthOverlay(intensity1, depth1);
    _size = cv::Size(intensity0.cols, intensity0.rows);
    cv::vconcat(std::vector<cv::Mat>({ZI0, ZI1}), _ZI01);
    _ZI01.convertTo(_ZI01, CV_8UC3);
  }
  void logIteration(int iteration, const std::vector<Feature::ShPtr> & constraints)
  {
    _iteration = iteration;
    cv::Mat JI(_size, CV_32FC1, cv::Scalar(0)), JZ(_size, CV_32FC1, cv::Scalar(0));
    double JImax = 0, JZmax = 0;
    cv::Mat error(_size, CV_32FC1, cv::Scalar(0));
    cv::Mat I1w(_size, CV_32FC1, cv::Scalar(0));
    cv::Mat Z1w(_size, CV_32FC1, cv::Scalar(0));

    cv::Mat rI(_size, CV_32FC1, cv::Scalar(0));
    cv::Mat rZ(_size, CV_32FC1, cv::Scalar(0));
    double errorMax = 0, zMax = 0, rImax = 0, rZmax = 0;
    for (const auto & c : constraints) {
      JI.at<float>(c->uv0(1), c->uv0(0)) = c->JI.norm();
      JZ.at<float>(c->uv0(1), c->uv0(0)) = c->JZ.norm();
      JImax = std::max({JImax, c->JI.norm()});
      JZmax = std::max({JZmax, c->JZ.norm()});

      error.at<float>(c->uv0(1), c->uv0(0)) = c->error;
      errorMax = std::max({errorMax, c->error});

      I1w.at<float>(c->uv0(1), c->uv0(0)) = c->iz1w(0) * 255.0;
      Z1w.at<float>(c->uv0(1), c->uv0(0)) = c->iz1w(1);
      zMax = std::max({zMax, c->iz1w(1)});

      rI.at<float>(c->uv0(1), c->uv0(0)) = c->residual(0);
      rZ.at<float>(c->uv0(1), c->uv0(0)) = c->residual(1);
      rZmax = std::max({rZmax, c->residual(1)});
      rImax = std::max({rImax, c->residual(0)});
    }
    JI.convertTo(JI, CV_8UC3, 255.0 / JImax);
    JZ.convertTo(JZ, CV_8UC3, 255.0 / JZmax);
    cv::Mat JIZ;
    cv::vconcat(std::vector<cv::Mat>({JI, JZ}), JIZ);
    JIZ.convertTo(JIZ, CV_8UC3);
    cv::resize(JIZ, JIZ, cv::Size(640 / _s, 2 * 480 / _s));
    cv::resize(_ZI01, _ZI01, cv::Size(640 / _s, 2 * 480 / _s));

    cv::imshow("JIZ", JIZ);
    cv::imshow("_ZI01", _ZI01);

    I1w.convertTo(I1w, CV_8UC3);
    Z1w.convertTo(Z1w, CV_8UC3, 255.0 / zMax);
    cv::resize(I1w, I1w, cv::Size(640 / _s, 480 / _s));
    cv::imshow("I1w", I1w);
    cv::resize(Z1w, Z1w, cv::Size(640 / _s, 480 / _s));
    cv::imshow("Z1w", Z1w);
    cv::Mat IZ1w = createIntensityDepthOverlay(I1w, Z1w);
    cv::resize(IZ1w, IZ1w, cv::Size(640 / _s, 480 / _s));
    cv::imshow("IZ1w", IZ1w);

    rI.convertTo(rI, CV_8UC3, 255.0 / rImax);
    rZ.convertTo(rZ, CV_8UC3, 255.0 / rZmax);
    cv::resize(rI, rI, cv::Size(640 / _s, 480 / _s));
    cv::imshow("rI", rI);
    cv::resize(rZ, rZ, cv::Size(640 / _s, 480 / _s));
    cv::imshow("rZ", rZ);
    error.convertTo(error, CV_8UC3, 255.0 / errorMax);
    cv::resize(error, error, cv::Size(640 / _s, 480 / _s));
    cv::imshow("error", error);
    /*
    cv::Mat overlay;
    cv::hconcat(std::vector<cv::Mat>({_ZI01, JIZ}), overlay);
    cv::resize(overlay, overlay, cv::Size(640 * 2 / _s, 480 / _s));
    cv::imshow("Overlay", overlay);
    */
    cv::waitKey(_waitTime);
  }
  void logConverged(const std::string & reason, const SE3d & motion)
  {
    print(
      "Converged on level [{}] after [{}] iterations because: {}\nt={}m, {:.3}°\n", _level,
      _iteration, reason, motion.translation().transpose(),
      motion.log().block(3, 0, 3, 1).norm() * 180.0 / M_PI);
    cv::waitKey(_waitTime);
  }

private:
  int _waitTime;
  double _s;
  int _level;
  int _iteration;
  cv::Mat _ZI01;
  cv::Size _size;
};

template <class WeightFunction, class Log = LogNull>
class DirectIcp_
{
public:
#define INFd std::numeric_limits<double>::infinity()
  typedef std::shared_ptr<DirectIcp_> ShPtr;

  DirectIcp_(
    Camera::ConstShPtr cam, WeightFunction::ShPtr weightFunction,
    const std::map<std::string, double> params, Log::ShPtr log = std::make_shared<LogNull>())
  : DirectIcp_(
      cam, weightFunction, params.at("nLevels"), params.at("weightPrior"),
      params.at("minGradientIntensity"), params.at("minGradientDepth"),
      params.at("maxGradientDepth"), params.at("maxDepth"), params.at("maxIterations"),
      params.at("minParameterUpdate"), params.at("maxErrorIncrease"), log)
  {
  }
  DirectIcp_(
    Camera::ConstShPtr cam, WeightFunction::ShPtr weightFunction, int nLevels = 4,
    double weightPrior = 0.0, double minGradientIntensity = 10 * 8, double minGradientDepth = INFd,
    double maxGradientDepth = 0.5, double maxZ = 5.0, double maxIterations = 100,
    double minParameterUpdate = 1e-6, double maxErrorIncrease = 1.1,
    Log::ShPtr log = std::make_shared<LogNull>())
  : _weightFunction(weightFunction),
    _nLevels(nLevels),
    _weightPrior(weightPrior),
    _minGradientIntensity(minGradientIntensity),
    _minGradientDepth(minGradientDepth),
    _maxGradientDepth(maxGradientDepth),
    _maxDepth(maxZ),
    _maxIterations(maxIterations),
    _minParameterUpdate(minParameterUpdate),
    _maxErrorIncrease(maxErrorIncrease),
    _log(log)
  {
    _cam.resize(_nLevels);
    _cam[0] = cam;
    for (int i = 1; i < _nLevels; i++) {
      _cam[i] = Camera::resize(_cam[i - 1], 0.5);
    }
  }
  SE3d computeEgomotion(const cv::Mat & intensity, const cv::Mat & depth, const SE3d & guess)
  {
    TIMED_SCOPE(timerF, "computeEgomotion");

    if (_I0.empty()) {
      _I0 = computePyramidIntensity(intensity);
      _Z0 = computePyramidDepth(depth);
      return guess;
    }
    const std::vector<cv::Mat> I1 = computePyramidIntensity(intensity);
    const std::vector<cv::Mat> Z1 = computePyramidDepth(depth);

    SE3d prior = guess;
    SE3d motion = prior;
    for (int level = _nLevels - 1; level >= 0; level--) {
      _log->logLevel(level, _I0[level], _Z0[level], I1[level], Z1[level]);

      const std::vector<Feature::ShPtr> features =
        extractFeatures(_I0[level], _Z0[level], _cam[level], motion);

      Matd<-1, 4> pcl0 = Matd<-1, 4>::Zero(features.size(), 4);

      std::string reason = "Max iterations exceeded";
      double error = INFd;
      Vec6d dx = Vec6d::Zero();
      int idx = 0;
      for (int iteration = 0; iteration < _maxIterations; iteration++) {
        std::for_each(features.begin(), features.end(), [&](auto c) {
          c->valid = false;
          c->p0t = motion * c->p0;

          if (c->p0t.z() <= 0) return;

          c->uv0t = _cam[level]->camera2image(c->p0t);

          if (!withinImage(c->uv0t, I1[level].rows, I1[level].cols)) return;

          Vec2d iz1w = interpolate(I1[level], Z1[level], c->uv0t);

          if (!std::isfinite(iz1w(0)) || !std::isfinite(iz1w(1))) return;

          c->p1t = motion.inverse() * _cam[level]->image2camera(c->uv0t, iz1w(1));

          c->iz1w = Vec2d(iz1w(0), c->p1t.z());

          c->residual = c->iz1w - c->iz0;
          if (!std::isfinite(c->residual.norm())) return;

          c->JZJw_Jtz = c->JZJw - computeJacobianSE3z(c->p1t);

          c->J.row(0) = c->JIJw;
          c->J.row(1) = c->JZJw_Jtz;

          if (!std::isfinite(c->J.norm())) return;

          if (false) {
            print(
              "{}: {} {} {} {} {}\n", idx, c->uv0.transpose(), c->uv0t.transpose(),
              c->iz0.transpose(), c->iz1w.transpose(), c->residual.transpose());
          }
          c->valid = true;
        });
        std::vector<Feature::ShPtr> constraints;
        std::copy_if(features.begin(), features.end(), std::back_inserter(constraints), [](auto c) {
          return c->valid;
        });

        if (constraints.size() < 6) {
          reason = format("Not enough constraints: {}", constraints.size());
          motion = SE3();
          break;
        }
        std::vector<Vec2d> residual;
        std::transform(
          constraints.begin(), constraints.end(), std::back_inserter(residual),
          [](auto c) { return c->residual; });
        _weightFunction->fit(residual, VecXd::Ones(residual.size()));

        Mat6d A = _weightPrior * Mat6d::Identity();
        Vec6d b = _weightPrior * (motion * prior.inverse()).log();
        double error_i = b.norm();
        Vec2d rSum = Vec2d::Zero();
        for (size_t i = 0; i < constraints.size(); i++) {
          auto c = constraints[i];
          rSum += c->residual;
          c->weight = _weightFunction->computeWeight(c->residual) * _weightFunction->scale();
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
        /*
        print(
          "Solving dx = b/A:\n A = \n{}\nb = \n{}\ndx = {}\nscale = \n{}\nerror = {:.4f} #: {} "
          "|A|={:.4f} "
          "|b|={:.4f} "
          "|dx|={:.4f}\nrSum={}\n",
          A, b.transpose(), dx.transpose(), _weightFunction->scale(), error, constraints.size(),
          A.norm(), b.norm(), dx.norm(), rSum.transpose());
        */
        _log->logIteration(iteration, constraints);
        motion = SE3d::exp(-dx) * motion;

        if (dx.norm() < _minParameterUpdate) {
          reason =
            format("Minimum step size reached: {:5.f}/{:5.f}", dx.norm(), _minParameterUpdate);
          break;
        }
      }
      _log->logConverged(reason, motion);
    }
    _I0 = I1;
    _Z0 = Z1;
    return motion;
  }

private:
  std::vector<Camera::ConstShPtr> _cam;
  WeightFunction::ShPtr _weightFunction;
  std::vector<cv::Mat> _I0, _Z0;
  int _nLevels;
  double _weightPrior, _minGradientIntensity, _minGradientDepth, _maxGradientDepth, _maxDepth,
    _maxIterations, _minParameterUpdate, _maxErrorIncrease;

  Log::ShPtr _log;

  std::vector<cv::Mat> computePyramidIntensity(const cv::Mat & intensity) const
  {
    std::vector<cv::Mat> pyramid(_nLevels);
    pyramid[0] = intensity;
    for (int i = 1; i < _nLevels; i++) {
      cv::pyrDown(pyramid[i - 1], pyramid[i]);
    }
    return pyramid;
  }
  std::vector<cv::Mat> computePyramidDepth(const cv::Mat & depth) const
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
  cv::Mat computeJacobianImage(const cv::Mat & image) const
  {
    cv::Mat dIdx, dIdy;
    cv::Sobel(image, dIdx, CV_32F, 1, 0, 3, 1. / 8.);
    cv::Sobel(image, dIdy, CV_32F, 0, 1, 3, 1. / 8.);

    cv::Mat dIdxy;
    cv::merge(std::vector<cv::Mat>({dIdx, dIdy}), dIdxy);
    return dIdxy;
  }

  cv::Mat computeJacobianDepth(const cv::Mat & depth) const
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
  std::vector<Feature::ShPtr> extractFeatures(
    const cv::Mat & intensity, const cv::Mat & depth, Camera::ConstShPtr cam,
    const SE3d & motion) const
  {
    const cv::Mat dI = computeJacobianImage(intensity);
    const cv::Mat dZ = computeJacobianDepth(depth);

    std::vector<Feature::ShPtr> constraints;
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
          c->uv0 = Vec2d(u, v);
          c->iz0 = Vec2d(i, z);

          c->p0 = cam->image2camera(c->uv0, c->iz0(1));
          Mat<double, 2, 6> Jw = computeJacobianWarp(motion * c->p0, cam);
          c->JIJw = dIvu[0] * Jw.row(0) + dIvu[1] * Jw.row(1);
          c->JZJw = dZvu[0] * Jw.row(0) + dZvu[1] * Jw.row(1);
          constraints.push_back(c);
        }
      }
    }
    return constraints;
  }
  Matd<2, 6> computeJacobianWarp(const Vec3d & p, Camera::ConstShPtr cam) const
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

  Vec6d computeJacobianSE3z(const Vec3d & p) const
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
  bool withinImage(const Vec2d & uv, int h, int w)
  {
    int border = std::max<int>(1, (int)(0.05 * (double)w));
    return (border < uv(0) && uv(0) < w - border && border < uv(1) && uv(1) < h - border);
  }

  Vec2d interpolate(const cv::Mat & intensity, const cv::Mat & depth, const Vec2d & uv)
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
};
void runEvaluationScript(const std::string & pathAlgo, const std::string & pathGt);

typedef DirectIcp_<TDistributionBivariate, LogShow> DirectIcp;
class Main
{
public:
  Main(int UNUSED(argc), char ** UNUSED(argv))
  {
    _dl =
      std::make_unique<tum::DataLoader>("/mnt/dataset/tum_rgbd/", "rgbd_dataset_freiburg2_desk");

    const std::map<std::string, double> params = {
      {"nLevels", 4.0},           {"weightPrior", 0.0},         {"minGradientIntensity", 5},
      {"minGradientDepth", 0.01}, {"maxGradientDepth", 0.3},    {"maxDepth", 5.0},
      {"maxIterations", 100},     {"minParameterUpdate", 1e-4}, {"maxErrorIncrease", 5.0}};

    auto weightFunction = std::make_shared<TDistributionBivariate>(5.0, Mat2d::Identity());
    //auto log = std::make_shared<LogNull>();
    auto log = std::make_shared<LogShow>(1);

    _directIcp = std::make_shared<DirectIcp>(_dl->cam(), weightFunction, params, log);
  }

  Frame::UnPtr loadFrame(size_t fNo)
  {
    // tum depth format: https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats
    auto f = _dl->loadFrame(fNo);
    return f;
  }

  void run()
  {
    const std::string experimentId = "test_c++";
    const std::string trajectoryAlgoPath = format(
      "{}/algorithm_results/{}/{}-algo.txt", _dl->datasetPath(), experimentId, _dl->sequenceId());

    Trajectory::ShPtr traj = std::make_shared<Trajectory>();
    size_t fEnd = _dl->timestamps().size();
    SE3d motion;
    SE3d pose;
    for (size_t fId = 0; fId < fEnd; fId++) {
      try {
        print(
          "{}/{}: {} m, {:.3f}°\n", fId, fEnd, pose.translation().transpose(),
          (std::abs(pose.angleX()) + std::abs(pose.angleY()) + std::abs(pose.angleZ())) * 180.0 /
            M_PI);

        cv::Mat img =
          cv::imread(_dl->datasetPath() + "/" + _dl->pathsImage()[fId], cv::IMREAD_GRAYSCALE);
        cv::Mat depth_ =
          cv::imread(_dl->datasetPath() + "/" + _dl->pathsDepth()[fId], cv::IMREAD_ANYDEPTH);

        if (depth_.empty() || img.empty()) {
          throw std::runtime_error("Could not load images.");
        }

        if (depth_.type() != CV_16U) {
          throw std::runtime_error("Depth image loaded incorrectly.");
        }

        cv::Mat depth(cv::Size(depth_.cols, depth_.rows), CV_32FC1);
        for (int u = 0; u < depth_.cols; u++) {
          for (int v = 0; v < depth_.rows; v++) {
            const ushort d = depth_.at<ushort>(v, u);
            depth.at<float>(v, u) =
              0.0002f * static_cast<float>(d > 0 ? d : std::numeric_limits<ushort>::quiet_NaN());
          }
        }

        auto overlay = createIntensityDepthOverlay(img, depth);
        cv::imshow("Frame", overlay);
        cv::waitKey(1);

        motion = _directIcp->computeEgomotion(img, depth, motion);
        pose = motion * pose;
        traj->append(_dl->timestamps()[fId], Pose(pose.inverse(), Mat6d::Identity()));
      } catch (const pd::Exception & e) {
        std::cerr << e.what() << std::endl;
      }
      if (fId > 25 && fId % 25 == 0) {
        utils::writeTrajectory(*traj, trajectoryAlgoPath);
        runEvaluationScript(trajectoryAlgoPath, _dl->pathGt());
      };
    }
    utils::writeTrajectory(*traj, trajectoryAlgoPath);
    runEvaluationScript(trajectoryAlgoPath, _dl->pathGt());
    for (auto t_pose : traj->poses()) {
      print("{} -> {}\n", t_pose.first, t_pose.second->twist().transpose());
    }
  }

protected:
  tum::DataLoader::ConstUnPtr _dl;
  DirectIcp::ShPtr _directIcp;
};

int main(int argc, char ** argv)
{
  Main m(argc, argv);
  m.run();
  return 0;
}

void runEvaluationScript(const std::string & pathAlgo, const std::string & pathGt)
{
  const int ret =
    system(format(
             "python3 /home/ros/vslam_ros/src/vslam/script/vslampy/evaluation/_tum/evaluate_rpe.py "
             "--verbose "
             "--fixed_delta "
             "{} {}",
             pathGt, pathAlgo)
             .c_str());
  if (ret != 0) {
    throw std::runtime_error("Running evaluation script failed!");
  }
}