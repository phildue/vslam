#include <fmt/chrono.h>
#include <fmt/core.h>
using fmt::format;
using fmt::print;

#include "RelativePoseError.h"
namespace pd::vslam::evaluation
{
RelativePoseError::RelativePoseError(
  Trajectory::ConstShPtr algo, Trajectory::ConstShPtr gt, double dT)
: _dT(dT * 1e9), _algo(algo), _gt(gt)
{
}

void RelativePoseError::compute()
{
  if (_algo->poses().empty() || _gt->poses().empty()) {
    throw pd::Exception(format(
      "Can't compute, at least one trajectory is empty. Ln Algo: [{}], Ln Gt: [{}]",
      _algo->poses().size(), _gt->poses().size()));
  }
  const size_t nSamples =
    _dT <= 0 ? _algo->poses().size()
             : static_cast<size_t>(static_cast<double>(_algo->tEnd() - _algo->tStart()) / (_dT));
  _errorsTranslation.reserve(nSamples);
  _errorsAngles.reserve(nSamples);
  _timestamps.reserve(nSamples);

  for (auto t_p : _algo->poses()) {
    const auto t = t_p.first;
    if (t + _dT > _algo->tEnd()) {
      break;
    }
    try {
      auto refAlgo = t_p.second;
      auto curAlgo = _algo->nearestPoseAt(t + _dT);
      auto refGt = _gt->nearestPoseAt(t);
      auto curGt = _gt->nearestPoseAt(t + _dT);
      if (
        std::abs(static_cast<double>(t) - static_cast<double>(refGt.first)) > _dT ||
        std::abs(static_cast<double>(t + _dT) - static_cast<double>(curGt.first)) > _dT ||
        std::abs(static_cast<double>(t) - static_cast<double>(curAlgo.first)) > 2 * _dT) {
        continue;
      }

      auto motionGt = algorithm::computeRelativeTransform(refGt.second->SE3(), curGt.second->SE3());
      auto motionAlgo = algorithm::computeRelativeTransform(refAlgo->SE3(), curAlgo.second->SE3());

      auto error = (motionAlgo.inverse() * motionGt).log();
      _errorsTranslation.push_back(error.head(3).norm());
      _errorsAngles.push_back(error.tail(3).norm() / M_PI * 180.0);
      _timestamps.push_back(t);
    } catch (const pd::Exception & e) {  //TODO put to LOG
      print("{}\n", e.what());
    }
  }
  if (_errorsTranslation.empty() || _errorsAngles.empty()) {
    throw pd::Exception(format("Can't compute, Don't have enough valid samples"));
  }
  Eigen::Map<VecXd> rmseT(_errorsTranslation.data(), _errorsTranslation.size());
  _statTranslation.rmse = std::sqrt(rmseT.dot(rmseT) / _errorsTranslation.size());
  _statTranslation.mean = rmseT.mean();
  _statTranslation.min = rmseT.minCoeff();
  _statTranslation.max = rmseT.maxCoeff();
  _statTranslation.median = algorithm::median(_errorsTranslation);
  _statTranslation.stddev = linalg::stddev(_errorsTranslation, _statTranslation.mean);

  Eigen::Map<VecXd> rmseR(_errorsAngles.data(), _errorsAngles.size());
  _statAngles.rmse = std::sqrt(rmseR.dot(rmseR) / _errorsAngles.size());
  _statAngles.mean = rmseR.mean();
  _statAngles.min = rmseR.minCoeff();
  _statAngles.max = rmseR.maxCoeff();
  _statAngles.median = algorithm::median(_errorsAngles);
  _statAngles.stddev = linalg::stddev(_errorsAngles, _statAngles.mean);
}

const RelativePoseError::Statistics & RelativePoseError::angle() const { return _statAngles; }
const RelativePoseError::Statistics & RelativePoseError::translation() const
{
  return _statTranslation;
}
const std::vector<double> & RelativePoseError::errorsTranslation() const
{
  return _errorsTranslation;
}
const std::vector<double> & RelativePoseError::errorsAngles() const { return _errorsAngles; }
const std::vector<Timestamp> & RelativePoseError::timestamps() const { return _timestamps; }
std::string RelativePoseError::toString() const
{
  return format(
    "dT: {12:.6f}, n: {13:d}\n"
    "|Translation [m/dt]  | Rotation [°/dt] |\n"
    "|RMSE:   {0:00.3f}   |      {6:000.3f}  |\n"
    "|Max:    {1:00.3f}   |      {7:000.3f}  |\n"
    "|Mean:   {2:00.3f}   |      {8:000.3f}  |\n"
    "|Std:    {3:00.3f}   |      {9:000.3f}  |\n"
    "|Median: {4:00.3f}   |      {10:000.3f}  |\n"
    "|Min:    {5:00.3f}   |      {11:000.3f}  |\n",
    _statTranslation.rmse, _statTranslation.max, _statTranslation.mean, _statTranslation.stddev,
    _statTranslation.median, _statTranslation.min, _statAngles.rmse, _statAngles.max,
    _statAngles.mean, _statAngles.stddev, _statAngles.median, _statAngles.min, _dT / 1e9,
    _timestamps.size());
}
RelativePoseError::ConstUnPtr RelativePoseError::compute(
  Trajectory::ConstShPtr algo, Trajectory::ConstShPtr gt, double dT)
{
  RelativePoseError::UnPtr rpe = std::make_unique<RelativePoseError>(algo, gt, dT);
  rpe->compute();
  return rpe;
}
PlotRPE::PlotRPE(const std::map<std::string, RelativePoseError::ConstShPtr> & errors)
: _errors(errors)
{
}
void PlotRPE::plot() const
{
  const Timestamp t0 = _errors.begin()->second->timestamps()[0];

  vis::plt::figure();
  vis::plt::subplot(2, 1, 1);
  vis::plt::title("$Translational Error$");
  vis::plt::ylabel("$|t|_2 [m]$");
  vis::plt::xlabel("$t-t_0 [s]$");
  vis::plt::grid(true);

  for (const auto n_e : _errors) {
    std::vector<double> tAx;
    std::transform(
      n_e.second->timestamps().begin(), n_e.second->timestamps().end(), std::back_inserter(tAx),
      [&](auto t) { return (t - t0) / 1e9; });
    vis::plt::named_plot(n_e.first, tAx, n_e.second->errorsTranslation(), ".--");
  }
  vis::plt::legend();
  vis::plt::subplot(2, 1, 2);
  vis::plt::title("$Rotational Error$");
  vis::plt::ylabel("$|\\theta|_2   [°]$");
  vis::plt::xlabel("$t-t_0 [s]$");
  vis::plt::grid(true);
  for (const auto n_e : _errors) {
    std::vector<double> tAx;
    std::transform(
      n_e.second->timestamps().begin(), n_e.second->timestamps().end(), std::back_inserter(tAx),
      [&](auto t) { return (t - t0) / 1e9; });
    vis::plt::named_plot(n_e.first, tAx, n_e.second->errorsAngles(), ".--");
  }
  vis::plt::legend();

  std::vector<std::string> names;
  std::vector<std::vector<double>> errT;
  std::vector<std::vector<double>> errR;
  for (const auto n_e : _errors) {
    errT.push_back(n_e.second->errorsTranslation());
    errR.push_back(n_e.second->errorsAngles());
    names.push_back(n_e.first);
  }

  vis::plt::figure();
  vis::plt::subplot(1, 2, 1);
  vis::plt::grid(true);
  vis::plt::title("Translation");
  vis::plt::boxplot(errT, names);

  vis::plt::subplot(1, 2, 2);
  vis::plt::grid(true);
  vis::plt::title("Rotation");
  vis::plt::boxplot(errT, names);
}
}  // namespace pd::vslam::evaluation
