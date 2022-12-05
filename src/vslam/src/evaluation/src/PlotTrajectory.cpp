#include "PlotTrajectory.h"
namespace pd::vslam::evaluation
{
PlotTrajectory::PlotTrajectory(const std::map<std::string, Trajectory::ConstShPtr> & trajectories)
: _trajectories(trajectories)
{
}
PlotTrajectory::PlotTrajectory(const std::map<std::string, Trajectory::ShPtr> & trajectories)
{
  for (auto n_t : trajectories) {
    _trajectories[n_t.first] = n_t.second;
  }
}
void PlotTrajectory::plot() const
{
  vis::plt::figure();
  vis::plt::subplot(1, 3, 1);
  vis::plt::ylabel("$t_y   [m]$");
  vis::plt::xlabel("$t_x   [m]$");
  vis::plt::grid(true);

  for (auto traj : _trajectories) {
    std::vector<double> x, y;
    for (auto p : traj.second->poses()) {
      x.push_back(p.second->SE3().translation().x());
      y.push_back(p.second->SE3().translation().y());
    }
    vis::plt::named_plot(traj.first, x, y);
  }
  vis::plt::axis("equal");
  vis::plt::legend();

  vis::plt::subplot(1, 3, 2);
  vis::plt::ylabel("$t_z   [m]$");
  vis::plt::xlabel("$t-t_0 [s]$");
  vis::plt::grid(true);
  for (auto n_traj : _trajectories) {
    std::vector<double> t, tz;
    Timestamp t0 = n_traj.second->tStart();
    for (auto p : n_traj.second->poses()) {
      t.push_back((p.first - t0) / 1e9);
      tz.push_back(p.second->SE3().translation().z());
    }
    vis::plt::named_plot(n_traj.first, t, tz);
  }
  vis::plt::legend();

  vis::plt::subplot(1, 3, 3);
  vis::plt::ylabel("$v [m/s]$");
  vis::plt::xlabel("$t-t_0 [s]$");
  vis::plt::grid(true);

  for (auto n_traj : _trajectories) {
    std::vector<double> t, v;
    Timestamp t0 = n_traj.second->tStart();
    Timestamp tRef = n_traj.second->tStart();
    for (auto p : n_traj.second->poses()) {
      t.push_back((p.first - t0) / 1e9);
      v.push_back((n_traj.second->motionBetween(tRef, p.first)->twist() / ((p.first - tRef) / 1e9))
                    .head(3)
                    .norm());
      tRef = p.first;
    }
    vis::plt::named_plot(n_traj.first, t, v);
  }
  vis::plt::legend();
}
PlotTrajectoryCovariance::PlotTrajectoryCovariance(
  const std::map<std::string, Trajectory::ConstShPtr> & trajectories)
: _trajectories(trajectories)
{
}
PlotTrajectoryCovariance::PlotTrajectoryCovariance(
  const std::map<std::string, Trajectory::ShPtr> & trajectories)
{
  for (auto n_t : trajectories) {
    _trajectories[n_t.first] = n_t.second;
  }
}
void PlotTrajectoryCovariance::plot() const
{
  vis::plt::figure();
  vis::plt::title("Covariance");
  vis::plt::ylabel("$| \\Sigma |$");
  vis::plt::xlabel("$t-t_0 [s]$");
  vis::plt::grid(true);
  for (auto n_traj : _trajectories) {
    std::vector<double> t, tz;
    Timestamp t0 = n_traj.second->tStart();
    for (auto p : n_traj.second->poses()) {
      t.push_back((p.first - t0) / 1e9);
      tz.push_back(p.second->twistCov().determinant());
    }
    vis::plt::named_plot(n_traj.first, t, tz);
  }
  vis::plt::legend();
}

PlotTrajectoryMotion::PlotTrajectoryMotion(
  const std::map<std::string, Trajectory::ConstShPtr> & trajectories)
: _trajectories(trajectories)
{
}
PlotTrajectoryMotion::PlotTrajectoryMotion(
  const std::map<std::string, Trajectory::ShPtr> & trajectories)
{
  for (auto n_t : trajectories) {
    _trajectories[n_t.first] = n_t.second;
  }
}
void PlotTrajectoryMotion::plot() const
{
  vis::plt::figure();

  vis::plt::subplot(3, 2, 1);
  vis::plt::ylabel("$\\Delta t_x   [m]$");
  vis::plt::xlabel("$t-t_0 [s]$");
  vis::plt::grid(true);
  for (auto n_traj : _trajectories) {
    std::vector<double> t, tx;
    Timestamp t0 = n_traj.second->tStart();
    SE3d posePrev = n_traj.second->poses().begin()->second->SE3();
    for (auto p : n_traj.second->poses()) {
      t.push_back((p.first - t0) / 1e9);
      auto motion = p.second->SE3() * posePrev.inverse();
      tx.push_back(motion.translation().x());
    }
    vis::plt::named_plot(n_traj.first, t, tx);
  }
  vis::plt::legend();
  vis::plt::subplot(3, 2, 3);
  vis::plt::ylabel("$\\Delta t_y   [m]$");
  vis::plt::xlabel("$t-t_0 [s]$");
  vis::plt::grid(true);
  for (auto n_traj : _trajectories) {
    std::vector<double> t, ty;
    Timestamp t0 = n_traj.second->tStart();
    SE3d posePrev = n_traj.second->poses().begin()->second->SE3();
    for (auto p : n_traj.second->poses()) {
      t.push_back((p.first - t0) / 1e9);
      auto motion = p.second->SE3() * posePrev.inverse();
      ty.push_back(motion.translation().y());
    }
    vis::plt::named_plot(n_traj.first, t, ty);
  }
  vis::plt::legend();

  vis::plt::subplot(3, 2, 5);
  vis::plt::ylabel("$\\Delta t_z   [m]$");
  vis::plt::xlabel("$t-t_0 [s]$");
  vis::plt::grid(true);
  for (auto n_traj : _trajectories) {
    std::vector<double> t, tz;
    Timestamp t0 = n_traj.second->tStart();
    SE3d posePrev = n_traj.second->poses().begin()->second->SE3();
    for (auto p : n_traj.second->poses()) {
      t.push_back((p.first - t0) / 1e9);
      auto motion = p.second->SE3() * posePrev.inverse();
      tz.push_back(motion.translation().z());
    }
    vis::plt::named_plot(n_traj.first, t, tz);
  }
  vis::plt::legend();

  vis::plt::subplot(3, 2, 2);
  vis::plt::ylabel("$\\Delta r_x   [\\circ]$");
  vis::plt::xlabel("$t-t_0 [s]$");
  vis::plt::grid(true);
  for (auto n_traj : _trajectories) {
    std::vector<double> t, rx;
    Timestamp t0 = n_traj.second->tStart();
    SE3d posePrev = n_traj.second->poses().begin()->second->SE3();
    for (auto p : n_traj.second->poses()) {
      t.push_back((p.first - t0) / 1e9);
      auto motion = p.second->SE3() * posePrev.inverse();
      rx.push_back(motion.angleX() / M_PI * 180.0);
    }
    vis::plt::named_plot(n_traj.first, t, rx);
  }
  vis::plt::legend();

  vis::plt::subplot(3, 2, 4);
  vis::plt::ylabel("$\\Delta r_y   [\\circ]$");
  vis::plt::xlabel("$t-t_0 [s]$");
  vis::plt::grid(true);
  for (auto n_traj : _trajectories) {
    std::vector<double> t, ry;
    Timestamp t0 = n_traj.second->tStart();
    SE3d posePrev = n_traj.second->poses().begin()->second->SE3();
    for (auto p : n_traj.second->poses()) {
      t.push_back((p.first - t0) / 1e9);
      auto motion = p.second->SE3() * posePrev.inverse();
      ry.push_back(motion.angleY() / M_PI * 180.0);
    }
    vis::plt::named_plot(n_traj.first, t, ry);
  }
  vis::plt::legend();

  vis::plt::subplot(3, 2, 6);
  vis::plt::ylabel("$\\Delta r_z   [\\circ]$");
  vis::plt::xlabel("$t-t_0 [s]$");
  vis::plt::grid(true);
  for (auto n_traj : _trajectories) {
    std::vector<double> t, rz;
    Timestamp t0 = n_traj.second->tStart();
    SE3d posePrev = n_traj.second->poses().begin()->second->SE3();
    for (auto p : n_traj.second->poses()) {
      t.push_back((p.first - t0) / 1e9);
      auto motion = p.second->SE3() * posePrev.inverse();
      rz.push_back(motion.angleZ() / M_PI * 180.0);
    }
    vis::plt::named_plot(n_traj.first, t, rz);
  }
  vis::plt::legend();
}
}  // namespace pd::vslam::evaluation