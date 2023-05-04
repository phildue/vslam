
#include <fmt/core.h>
#include <fmt/ostream.h>
#include <matplot/matplot.h>

#include <execution>
using fmt::format;
using fmt::print;
#include "Overlays.h"
namespace pd::vslam
{
OverlayWeightedResidual::OverlayWeightedResidual(
  int height, int width, const std::vector<DirectIcp::Constraint> & constraints, const VecXd & r,
  const VecXd & w)
: _height(height), _width(width), _r(r), _w(w), _constraints(constraints)
{
}
cv::Mat OverlayWeightedResidual::draw() const
{
  MatXd R = MatXd::Zero(_height, _width);
  std::for_each(std::execution::par_unseq, _constraints.begin(), _constraints.end(), [&](auto c) {
    R(c.v, c.u) = _w(c.idx) * std::abs(_r(c.idx));
  });
  R.noalias() = 255.0 * (R / R.maxCoeff());
  return vis::drawMat(R.cast<image_value_t>());
}

OverlaySteepestDescent::OverlaySteepestDescent(
  int height, int width, const std::vector<DirectIcp::Constraint> & constraints, const VecXd & w,
  const Matd<-1, 6> & JIJpJt, const Matd<-1, 6> & JZJpJt_Jtz, double wRi, double wRz)
: _height(height),
  _width(width),
  _w(w),
  _JIJpJt(JIJpJt),
  _JZJpJt_Jtz(JZJpJt_Jtz),
  _constraints(constraints),
  _wRi(wRi),
  _wRz(wRz)
{
}
cv::Mat OverlaySteepestDescent::draw() const
{
  MatXd J = MatXd::Zero(_height, _width);
  std::for_each(std::execution::par_unseq, _constraints.begin(), _constraints.end(), [&](auto c) {
    J(c.v, c.u) = _w(c.idx) * (_wRi * _JIJpJt.row(c.idx) + _wRz * _JZJpJt_Jtz.row(c.idx)).norm();
  });
  J.noalias() = 255.0 * (J / J.maxCoeff());
  return vis::drawMat(J.cast<image_value_t>());
}

OverlayResidualGradient::OverlayResidualGradient(
  int height, int width, const std::vector<DirectIcp::Constraint> & constraints, const VecXd & w,
  const VecXd & r, const Matd<-1, 6> & JIJpJt, const Matd<-1, 6> & JZJpJt_Jtz, double wRi,
  double wRz)
: _height(height),
  _width(width),
  _w(w),
  _r(r),
  _JIJpJt(JIJpJt),
  _JZJpJt_Jtz(JZJpJt_Jtz),
  _constraints(constraints),
  _wRi(wRi),
  _wRz(wRz)
{
}
cv::Mat OverlayResidualGradient::draw() const
{
  MatXd O = MatXd::Zero(_height, _width);
  std::for_each(std::execution::par_unseq, _constraints.begin(), _constraints.end(), [&](auto c) {
    O(c.v, c.u) =
      (_r(c.idx) * _w(c.idx) * _wRi * _JIJpJt.row(c.idx) + _wRz * _JZJpJt_Jtz.row(c.idx)).norm();
  });
  O.noalias() = 255.0 * (O / O.maxCoeff());
  return vis::drawMat(O.cast<image_value_t>());
}

PlotResiduals::PlotResiduals(
  Timestamp t, size_t iteration, const std::vector<DirectIcp::Constraint> & constraints,
  const MatXd & r, const VecXd & w)
: _t(t), _iteration(iteration), _constraints(constraints), _r(r), _w(w)
{
}
void PlotResiduals::plot(matplot::figure_handle f)
{
  using namespace matplot;
  if (_r.cols() == 2) {
    VecXd rI_ = _r.col(0);
    VecXd rZ_ = _r.col(1);

    std::vector<double> weights(_w.data(), _w.data() + _w.rows()),
      rI(rI_.data(), rI_.data() + rI_.rows()), rZ(rZ_.data(), rZ_.data() + rZ_.rows());

    auto ax0 = f->add_subplot(1, 2, 1, true);
    ax0->scatter(rI, weights);
    ax0->xlabel("Residual Intensity");
    ax0->ylabel("Weight");
    figure();
    auto ax1 = f->add_subplot(1, 2, 2, true);
    ax1->scatter(rZ, weights);
    ax1->xlabel("Residual Depth");
    ax1->ylabel("Weight");
  } else {
    throw pd::Exception("Not implemented for case r.cols() != 2");
  }
}

std::string PlotResiduals::csv() const
{
  std::stringstream ss;
  ss << "Id,rI,rZ,weight\r\n";

  for (const auto & c : _constraints) {
    ss << c.idx << ",";
    ss << _r(c.idx, 0) << ",";
    ss << _r(c.idx, 1) << ",";
    ss << _w(c.idx);
    ss << "\r\n";
  }
  return ss.str();
}
std::string PlotResiduals::id() const { return format("{}_{}", _t, _iteration); }

}  // namespace pd::vslam