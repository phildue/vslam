#ifndef PLOT_TRAJECTORY_H__
#define PLOT_TRAJECTORY_H__
#include "core/core.h"
#include "utils/utils.h"
namespace pd::vslam::evaluation
{
class PlotTrajectory : public vis::Plot
{
public:
  PlotTrajectory(const std::map<std::string, Trajectory::ConstShPtr> & trajectories);
  PlotTrajectory(const std::map<std::string, Trajectory::ShPtr> & trajectories);
  void plot() const override;
  std::string csv() const override { return ""; }

private:
  std::map<std::string, Trajectory::ConstShPtr> _trajectories;
};
class PlotTrajectoryCovariance : public vis::Plot
{
public:
  PlotTrajectoryCovariance(const std::map<std::string, Trajectory::ConstShPtr> & trajectories);
  PlotTrajectoryCovariance(const std::map<std::string, Trajectory::ShPtr> & trajectories);
  void plot() const override;
  std::string csv() const override { return ""; }

private:
  std::map<std::string, Trajectory::ConstShPtr> _trajectories;
};
class PlotTrajectoryMotion : public vis::Plot
{
public:
  PlotTrajectoryMotion(const std::map<std::string, Trajectory::ConstShPtr> & trajectories);
  PlotTrajectoryMotion(const std::map<std::string, Trajectory::ShPtr> & trajectories);
  void plot() const override;
  std::string csv() const override { return ""; }

private:
  std::map<std::string, Trajectory::ConstShPtr> _trajectories;
};
}  // namespace pd::vslam::evaluation
#endif