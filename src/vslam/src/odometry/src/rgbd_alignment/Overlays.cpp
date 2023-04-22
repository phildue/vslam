
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
    R(c.v, c.u) = _w(c.idx) * _r(c.idx);
  });
  R.noalias() = MatXd((R.array() - R.minCoeff()) / (R.maxCoeff() - R.minCoeff()));
  return vis::drawMat(R.cast<image_value_t>());
}

PlotResiduals::PlotResiduals(
  Timestamp t, size_t iteration, const std::vector<DirectIcp::Constraint> & constraints,
  const MatXd & r, const VecXd & w)
: _t(t), _iteration(iteration), _constraints(constraints), _r(r), _w(w)
{
}
void PlotResiduals::plot() const
{
  using namespace matplot;
  if (_r.cols() == 2) {
    VecXd rI_ = _r.col(0);
    VecXd rZ_ = _r.col(1);

    std::vector<double> weights(_w.data(), _w.data() + _w.size()),
      rI(rI_.data(), rI_.data() + rI_.size()), rZ(rZ_.data(), rZ_.data() + rZ_.size());
    scatter3(rI, rZ, weights);
    xlabel("Residual Intensity");
    ylabel("Residual Depth");
    zlabel("Weight");
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