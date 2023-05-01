
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