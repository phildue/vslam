// Copyright 2022 Philipp.Duernry
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
#include <fmt/chrono.h>
#include <fmt/core.h>
using fmt::format;
#include "PlotKalman.h"
#include "utils/utils.h"

namespace pd::vslam
{
PlotKalman::PlotKalman() : _plot(std::make_shared<PlotKalman::Plot>()) { LOG_IMG("Kalman"); }
void PlotKalman::append(const Entry & e)
{
  _plot->append(e);
  if (_plot->timestamps().size() > 10) {
    LOG_IMG("Kalman") << _plot;
  }
}

void PlotKalman::Plot::append(const Entry & e)
{
  //TODO avoid memory leak;
  _entries.push_back(e);
  _timestamps.push_back(e.t);
  _t = e.t;
}
void PlotKalman::Plot::plot() const
{
  if (_timestamps.empty()) {
    vis::plt::figure();
    return;  //TODO warn? except?
  }
  const Timestamp t0 = _timestamps[0];
  std::vector<double> t, tz;
  std::transform(_timestamps.begin(), _timestamps.end(), std::back_inserter(t), [&](auto tt) {
    return static_cast<double>(tt - t0) / 1e9;
  });
  std::vector<std::string> names = {"tx", "ty", "tz", "rx", "ry", "rz"};

  std::map<std::string, std::vector<double>> x, ex, m, c, u, cx, cex, cm, k;
  for (const auto & e : _entries) {
    x["tx"].push_back(e.state(6));
    x["ty"].push_back(e.state(7));
    x["tz"].push_back(e.state(8));
    auto xSE3 = SE3d::exp(e.state.block(6, 0, 6, 1));
    x["rx"].push_back(xSE3.angleX() / M_PI * 180.0);
    x["ry"].push_back(xSE3.angleY() / M_PI * 180.0);
    x["rz"].push_back(xSE3.angleZ() / M_PI * 180.0);

    ex["tx"].push_back(e.expectation(0));
    ex["ty"].push_back(e.expectation(1));
    ex["tz"].push_back(e.expectation(2));
    xSE3 = SE3d::exp(e.expectation);
    ex["rx"].push_back(xSE3.angleX() / M_PI * 180.0);
    ex["ry"].push_back(xSE3.angleY() / M_PI * 180.0);
    ex["rz"].push_back(xSE3.angleZ() / M_PI * 180.0);

    m["tx"].push_back(e.measurement(0));
    m["ty"].push_back(e.measurement(1));
    m["tz"].push_back(e.measurement(2));
    xSE3 = SE3d::exp(e.measurement);
    m["rx"].push_back(xSE3.angleX() / M_PI * 180.0);
    m["ry"].push_back(xSE3.angleY() / M_PI * 180.0);
    m["rz"].push_back(xSE3.angleZ() / M_PI * 180.0);

    c["tx"].push_back(e.correction(0));
    c["ty"].push_back(e.correction(1));
    c["tz"].push_back(e.correction(2));
    xSE3 = SE3d::exp(e.correction);
    c["rx"].push_back(xSE3.angleX() / M_PI * 180.0);
    c["ry"].push_back(xSE3.angleY() / M_PI * 180.0);
    c["rz"].push_back(xSE3.angleZ() / M_PI * 180.0);

    u["tx"].push_back(e.update(0));
    u["ty"].push_back(e.update(1));
    u["tz"].push_back(e.update(2));
    xSE3 = SE3d::exp(e.update.block(6, 0, 6, 1));
    u["rx"].push_back(xSE3.angleX() / M_PI * 180.0);
    u["ry"].push_back(xSE3.angleY() / M_PI * 180.0);
    u["rz"].push_back(xSE3.angleZ() / M_PI * 180.0);

    for (size_t i = 0U; i < names.size(); i++) {
      cx[names[i]].push_back(e.covState(6 + i, 6 + i));
      cex[names[i]].push_back(e.covExpectation(i, i));
      cm[names[i]].push_back(e.covMeasurement(i, i));
      k[names[i]].push_back(e.kalmanGain.col(i).sum());
    }
  }
  for (auto n : names) {
    vis::plt::figure();
    vis::plt::subplot(4, 2, 1);
    createVelocityPlot(t, x[n], n);
    vis::plt::subplot(4, 2, 3);
    createExpMeasPlot(t, ex[n], m[n], n);
    vis::plt::subplot(4, 2, 5);
    createCorrectionPlot(t, c[n], n);
    vis::plt::subplot(4, 2, 7);
    createUpdatePlot(t, u[n], n);
    vis::plt::subplot(4, 2, 2);
    plotStateCov(t, cx[n], n);
    vis::plt::subplot(4, 2, 4);
    plotExpectationCov(t, cex[n], n);
    vis::plt::subplot(4, 2, 6);
    plotMeasurementCov(t, cm[n], n);
    vis::plt::subplot(4, 2, 8);
    plotKalmanGain(t, k[n], n);
  }
}
void PlotKalman::Plot::createExpMeasPlot(
  const std::vector<double> & t, const std::vector<double> & e, const std::vector<double> & m,
  const std::string & name) const
{
  vis::plt::title("Expectation / Measurement");
  vis::plt::ylabel(format("${}$", name));
  vis::plt::xlabel("$t-t_0 [s]$");
  vis::plt::named_plot("Expectation", t, e);
  vis::plt::named_plot("Measurement", t, m);
  vis::plt::legend();
  vis::plt::grid(true);
}
void PlotKalman::Plot::createVelocityPlot(
  const std::vector<double> & t, const std::vector<double> & x, const std::string & name) const
{
  vis::plt::title("State");
  vis::plt::ylabel(format("${}$", name));
  vis::plt::xlabel("$t-t_0 [s]$");
  vis::plt::named_plot("State", t, x);
  vis::plt::legend();
  vis::plt::grid(true);
}
void PlotKalman::Plot::createCorrectionPlot(
  const std::vector<double> & t, const std::vector<double> & c, const std::string & name) const
{
  vis::plt::title("Innovation");
  vis::plt::ylabel(format("{}", name));
  vis::plt::xlabel("$t-t_0 [s]$");
  vis::plt::grid(true);
  vis::plt::plot(t, c);
}
void PlotKalman::Plot::createUpdatePlot(
  const std::vector<double> & t, const std::vector<double> & u, const std::string & name) const
{
  vis::plt::title("Update");
  vis::plt::ylabel(format("{}", name));
  vis::plt::xlabel("$t-t_0 [s]$");
  vis::plt::grid(true);
  vis::plt::plot(t, u);
}

void PlotKalman::Plot::plotStateCov(
  const std::vector<double> & t, const std::vector<double> & cx, const std::string & name) const
{
  vis::plt::ylabel(format("$| \\Sigma_x |$ {}", name));
  vis::plt::xlabel("$t-t_0 [s]$");
  vis::plt::grid(true);
  vis::plt::plot(t, cx);
}
void PlotKalman::Plot::plotExpectationCov(
  const std::vector<double> & t, const std::vector<double> & ce, const std::string & name) const
{
  vis::plt::ylabel(format("$| \\Sigma_e |$ {}", name));
  vis::plt::xlabel("$t-t_0 [s]$");
  vis::plt::grid(true);
  vis::plt::plot(t, ce);
}
void PlotKalman::Plot::plotMeasurementCov(
  const std::vector<double> & t, const std::vector<double> & ce, const std::string & name) const
{
  vis::plt::ylabel(format("$| \\Sigma_m |$ {}", name));
  vis::plt::xlabel("$t-t_0 [s]$");
  vis::plt::grid(true);
  vis::plt::plot(t, ce);
}
void PlotKalman::Plot::plotKalmanGain(
  const std::vector<double> & t, const std::vector<double> & k, const std::string & name) const
{
  vis::plt::ylabel(format("$| K |$ {}", name));
  vis::plt::xlabel("$t-t_0 [s]$");
  vis::plt::grid(true);
  vis::plt::plot(t, k);
}
void operator<<(PlotKalman::ShPtr log, const PlotKalman::Entry & e) { log->append(e); }

PlotKalman::ShPtr PlotKalman::make()
{
  return Log::DISABLED ? std::make_shared<PlotKalmanNull>() : std::make_shared<PlotKalman>();
}

std::string PlotKalman::Plot::csv() const
{
  std::stringstream ss;
  ss << "Timestamp,px,py,pz,rx,ry,rz,vtx,vty,vtz,vrx,vry,vrz";
  for (int i = 0; i < 12; i++) {
    for (int j = 0; j < 12; j++) {
      ss << ",cov" << i << j;
    }
  }
  ss << ",";
  ss << "evtx,evty,evtz,evrx,evry,evrz";
  for (int i = 0; i < 6; i++) {
    for (int j = 0; j < 6; j++) {
      ss << ",ecov" << i << j;
    }
  }
  ss << ",mvtx,mvty,mvtz,mvrx,mvry,mvrz";
  for (int i = 0; i < 6; i++) {
    for (int j = 0; j < 6; j++) {
      ss << ",mcov" << i << j;
    }
  }
  ss << "\r\n";

  for (const auto & e : _entries) {
    ss << e.t << ",";
    ss << utils::toCsv(e.state, ",") << ",";
    ss << utils::toCsv(e.covState, ",") << ",";
    ss << utils::toCsv(e.expectation, ",") << ",";
    ss << utils::toCsv(e.covExpectation, ",") << ",";
    ss << utils::toCsv(e.measurement, ",") << ",";
    ss << utils::toCsv(e.covMeasurement, ",");

    ss << "\r\n";
  }
  return ss.str();
}
std::string PlotKalman::Plot::id() const { return format("{}", _t); }

}  // namespace pd::vslam
