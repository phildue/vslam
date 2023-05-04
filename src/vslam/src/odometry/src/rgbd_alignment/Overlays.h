
#ifndef VSLAM_ODOMETRY_OVERLAYS_H__
#define VSLAM_ODOMETRY_OVERLAYS_H__
#include <utils/utils.h>

#include "DirectIcp.h"
namespace pd::vslam
{
class OverlayWeightedResidual : public vis::Drawable
{
public:
  OverlayWeightedResidual(
    int height, int width, const std::vector<DirectIcp::Constraint> & constraints, const VecXd & r,
    const VecXd & w);
  cv::Mat draw() const override;

private:
  const int _height, _width;
  const VecXd & _r;
  const VecXd & _w;
  const std::vector<DirectIcp::Constraint> & _constraints;
};

class OverlaySteepestDescent : public vis::Drawable
{
public:
  OverlaySteepestDescent(
    int height, int width, const std::vector<DirectIcp::Constraint> & constraints, const VecXd & w,
    const Matd<-1, 6> & JIJpJt, const Matd<-1, 6> & JZJpJt_Jtz, double wRi, double wRz);
  cv::Mat draw() const override;

private:
  const int _height, _width;
  const VecXd & _w;
  const Matd<-1, 6> & _JIJpJt;
  const Matd<-1, 6> & _JZJpJt_Jtz;
  const std::vector<DirectIcp::Constraint> & _constraints;
  const double _wRi, _wRz;
};

class OverlayResidualGradient : public vis::Drawable
{
public:
  OverlayResidualGradient(
    int height, int width, const std::vector<DirectIcp::Constraint> & constraints, const VecXd & w,
    const VecXd & r, const Matd<-1, 6> & JIJpJt, const Matd<-1, 6> & JZJpJt_Jtz, double wRi,
    double wRz);
  cv::Mat draw() const override;

private:
  const int _height, _width;
  const VecXd & _w;
  const VecXd & _r;
  const Matd<-1, 6> & _JIJpJt;
  const Matd<-1, 6> & _JZJpJt_Jtz;
  const std::vector<DirectIcp::Constraint> & _constraints;
  const double _wRi, _wRz;
};

class PlotResiduals : public vis::Plot
{
public:
  PlotResiduals(
    Timestamp t, size_t iteration, const std::vector<DirectIcp::Constraint> & constraints,
    const MatXd & r, const VecXd & w);
  void plot(matplot::figure_handle f) override;
  std::string csv() const override;
  std::string id() const;

private:
  const Timestamp _t;
  const size_t _iteration;
  const std::vector<DirectIcp::Constraint> _constraints;
  const MatXd _r;
  const VecXd _w;
};
}  // namespace pd::vslam
#endif