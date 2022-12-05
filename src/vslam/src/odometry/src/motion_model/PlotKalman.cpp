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
#include <fmt/chrono.h>
#include <fmt/core.h>

#include "PlotKalman.h"
using fmt::format;

namespace pd::vslam
{
PlotKalman::ShPtr PlotKalman::_instance = nullptr;

void PlotKalman::append(Timestamp t, const Entry & e)
{
  //TODO avoid memory leak;
  _entries.push_back(e);
  _timestamps.push_back(t);
}
void PlotKalman::plot() const
{
  if (_timestamps.empty()) {
    return;  //TODO warn? except?
  }
  const Timestamp t0 = _timestamps[0];
  std::vector<double> t, tz;
  std::transform(_timestamps.begin(), _timestamps.end(), std::back_inserter(t), [&](auto tt) {
    return static_cast<double>(tt - t0) / 1e9;
  });
  std::vector<std::string> names = {"vx", "vy", "vz", "ax", "ay", "az"};

  std::map<std::string, std::vector<double>> x, ex, m, c, u, cx, cex, cm, k;
  for (const auto & e : _entries) {
    x["vx"].push_back(e.state(6));
    x["vy"].push_back(e.state(7));
    x["vz"].push_back(e.state(8));
    auto xSE3 = SE3d::exp(e.state.block(6, 0, 6, 1));
    x["ax"].push_back(xSE3.angleX() / M_PI * 180.0);
    x["ay"].push_back(xSE3.angleY() / M_PI * 180.0);
    x["az"].push_back(xSE3.angleZ() / M_PI * 180.0);

    ex["vx"].push_back(e.expectation(0));
    ex["vy"].push_back(e.expectation(1));
    ex["vz"].push_back(e.expectation(2));
    xSE3 = SE3d::exp(e.expectation);
    ex["ax"].push_back(xSE3.angleX() / M_PI * 180.0);
    ex["ay"].push_back(xSE3.angleY() / M_PI * 180.0);
    ex["az"].push_back(xSE3.angleZ() / M_PI * 180.0);

    m["vx"].push_back(e.measurement(0));
    m["vy"].push_back(e.measurement(1));
    m["vz"].push_back(e.measurement(2));
    xSE3 = SE3d::exp(e.measurement);
    m["ax"].push_back(xSE3.angleX() / M_PI * 180.0);
    m["ay"].push_back(xSE3.angleY() / M_PI * 180.0);
    m["az"].push_back(xSE3.angleZ() / M_PI * 180.0);

    c["vx"].push_back(e.correction(0));
    c["vy"].push_back(e.correction(1));
    c["vz"].push_back(e.correction(2));
    xSE3 = SE3d::exp(e.correction);
    c["ax"].push_back(xSE3.angleX() / M_PI * 180.0);
    c["ay"].push_back(xSE3.angleY() / M_PI * 180.0);
    c["az"].push_back(xSE3.angleZ() / M_PI * 180.0);

    u["vx"].push_back(e.update(0));
    u["vy"].push_back(e.update(1));
    u["vz"].push_back(e.update(2));
    xSE3 = SE3d::exp(e.update.block(6, 0, 6, 1));
    u["ax"].push_back(xSE3.angleX() / M_PI * 180.0);
    u["ay"].push_back(xSE3.angleY() / M_PI * 180.0);
    u["az"].push_back(xSE3.angleZ() / M_PI * 180.0);

    for (size_t i = 0U; i < names.size(); i++) {
      cx[names[i]].push_back(e.covState(6 + i, 6 + i));
      cex[names[i]].push_back(e.covExpectation(i, i));
      cm[names[i]].push_back(e.covMeasurement(i, i));
      k[names[i]].push_back(e.kalmanGain.col(i).sum());
    }
  }
  for (auto n : names) {
    vis::plt::figure();
    vis::plt::subplot(3, 2, 1);
    createExpMeasPlot(t, ex[n], m[n], n);
    vis::plt::subplot(3, 2, 3);
    createCorrectionPlot(t, c[n], n);
    vis::plt::subplot(3, 2, 5);
    createUpdatePlot(t, u[n], n);
    vis::plt::subplot(3, 2, 2);
    plotStateCov(t, cx[n], n);
    vis::plt::subplot(3, 2, 4);
    plotExpectationCov(t, cex[n], n);
    vis::plt::subplot(3, 2, 6);
    plotKalmanGain(t, k[n], n);
  }
}
void PlotKalman::createExpMeasPlot(
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
void PlotKalman::createCorrectionPlot(
  const std::vector<double> & t, const std::vector<double> & c, const std::string & name) const
{
  vis::plt::title("Correction");
  vis::plt::ylabel(format("{}", name));
  vis::plt::xlabel("$t-t_0 [s]$");
  vis::plt::grid(true);
  vis::plt::plot(t, c);
}
void PlotKalman::createUpdatePlot(
  const std::vector<double> & t, const std::vector<double> & u, const std::string & name) const
{
  vis::plt::title("Update");
  vis::plt::ylabel(format("{}", name));
  vis::plt::xlabel("$t-t_0 [s]$");
  vis::plt::grid(true);
  vis::plt::plot(t, u);
}

void PlotKalman::plotStateCov(
  const std::vector<double> & t, const std::vector<double> & cx, const std::string & name) const
{
  vis::plt::ylabel(format("$| \\Sigma_x |$ {}", name));
  vis::plt::xlabel("$t-t_0 [s]$");
  vis::plt::grid(true);
  vis::plt::plot(t, cx);
}
void PlotKalman::plotExpectationCov(
  const std::vector<double> & t, const std::vector<double> & ce, const std::string & name) const
{
  vis::plt::ylabel(format("$| \\Sigma_e |$ {}", name));
  vis::plt::xlabel("$t-t_0 [s]$");
  vis::plt::grid(true);
  vis::plt::plot(t, ce);
}
void PlotKalman::plotKalmanGain(
  const std::vector<double> & t, const std::vector<double> & k, const std::string & name) const
{
  vis::plt::ylabel(format("$| K |$ {}", name));
  vis::plt::xlabel("$t-t_0 [s]$");
  vis::plt::grid(true);
  vis::plt::plot(t, k);
}
PlotKalman::UnPtr PlotKalman::make()
{
  return LOG_PLT("Kalman")->show() || LOG_PLT("Kalman")->save()
           ? std::make_unique<PlotKalman>()
           : std::make_unique<PlotKalmanNull>();
}
PlotKalman::ShPtr PlotKalman::get()
{
  if (!_instance) {
    _instance = make();
  }
  return _instance;
}
}  // namespace pd::vslam
