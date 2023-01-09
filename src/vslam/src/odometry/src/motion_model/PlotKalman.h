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
#include "core/core.h"
#include "utils/utils.h"
namespace pd::vslam
{
class PlotKalman
{
public:
  typedef std::shared_ptr<PlotKalman> ShPtr;
  typedef std::unique_ptr<PlotKalman> UnPtr;
  typedef std::shared_ptr<const PlotKalman> ConstShPtr;
  typedef std::unique_ptr<const PlotKalman> ConstUnPtr;
  struct Entry
  {
    Timestamp t;
    VecXd state, expectation, measurement, correction, update;
    MatXd covState, covExpectation, covMeasurement, kalmanGain;
  };

  PlotKalman();

  class Plot : public vis::Plot
  {
  public:
    void plot() const override;
    std::string csv() const override;
    void append(const Entry & e);
    const std::vector<Timestamp> & timestamps() { return _timestamps; }
    std::string id() const override;

  private:
    std::vector<Entry> _entries;
    std::vector<Timestamp> _timestamps;
    Timestamp _t;
    void createExpMeasPlot(
      const std::vector<double> & t, const std::vector<double> & e, const std::vector<double> & m,
      const std::string & name) const;
    void createCorrectionPlot(
      const std::vector<double> & t, const std::vector<double> & c, const std::string & name) const;
    void createVelocityPlot(
      const std::vector<double> & t, const std::vector<double> & x, const std::string & name) const;
    void createUpdatePlot(
      const std::vector<double> & t, const std::vector<double> & u, const std::string & name) const;
    void plotStateCov(
      const std::vector<double> & t, const std::vector<double> & cx,
      const std::string & name) const;
    void plotExpectationCov(
      const std::vector<double> & t, const std::vector<double> & ce,
      const std::string & name) const;
    void plotKalmanGain(
      const std::vector<double> & t, const std::vector<double> & k, const std::string & name) const;
    void plotMeasurementCov(
      const std::vector<double> & t, const std::vector<double> & ce,
      const std::string & name) const;
  };

  virtual void append(const Entry & e);
  static ShPtr make();
  Trajectory::ConstShPtr & trajectoryGt() { return _trajGt; }
  const Trajectory::ConstShPtr & trajectoryGt() const { return _trajGt; }

private:
  Trajectory::ConstShPtr _trajGt;
  //TODO remove singleton and provide instances via Log:: interface
  static ShPtr _instance;
  std::shared_ptr<PlotKalman::Plot> _plot;
};
void operator<<(PlotKalman::ShPtr log, const PlotKalman::Entry & e);

class PlotKalmanNull : public PlotKalman
{
  void append(const Entry & UNUSED(e)) override {}
};
}  // namespace pd::vslam
