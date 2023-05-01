#include <fmt/core.h>
using fmt::format;
#include <matplot/matplot.h>

#include <algorithm>

#include "PlotAlignment.h"
#include "utils/utils.h"
namespace pd::vslam
{
PlotAlignment::UnPtr PlotAlignment::make(Timestamp t)
{
  return Log::DISABLED ? std::make_unique<PlotAlignmentNull>() : std::make_unique<PlotAlignment>(t);
}
void PlotAlignment::append(const PlotAlignment::Entry & e) { _results[e.level] = e.results; }
std::string PlotAlignment::csv() const
{
  std::stringstream ss;
  ss << "Timestamp,Level,Iteration,Squared Error,Error Reduction,Step "
        "Size,nConstraints,tx,ty,tz,rx,ry,rz,H11,H12,H13,H14,H15,H16,H21,H22,H23,H24,H25,H26,H31,"
        "H32,H33,H34,H35,H36,"
        "H41,H42,H43,"
        "H44,H45,H46,H51,H52,H53,H54,H55,H56,H61,H62,H63,H64,H65,H66\r\n";

  for (auto level_r = _results.rbegin(); level_r != _results.rend(); ++level_r) {
    int _level = level_r->first;
    auto _r = level_r->second;

    if (_r->iteration <= 1) continue;

    std::vector<double> dChi2(_r->iteration - 1);
    dChi2[0] = 0;
    for (size_t i = 1; i < _r->iteration - 1; i++) {
      dChi2[i] = _r->chi2(i) - _r->chi2(i - 1);
    }
    for (size_t i = 0; i < _r->iteration - 1; i++) {
      ss << _t << "," << _level << "," << i << "," << _r->chi2(i) << "," << dChi2[i] << ","
         << _r->stepSize(i) << "," << _r->normalEquations[i]->nConstraints();

      ss << "," << utils::toCsv(_r->solution(i));
      ss << "," << utils::toCsv(_r->covariance(i));
      ss << "\r\n";
    }
  }

  return ss.str();
}
void PlotAlignment::plot(matplot::figure_handle f)
{
  int plotId = 0;
  f->title(format("Alignment"));
  for (auto level_r = _results.rbegin(); level_r != _results.rend(); ++level_r) {
    std::vector<double> err, dx;
    std::vector<int> iterations;
    for (size_t i = 1; i < level_r->second->iteration - 1; i++) {
      iterations.push_back(i);
      dx.push_back(level_r->second->stepSize(i));
      err.push_back(level_r->second->chi2(i));
    }
    auto ax0 = f->add_subplot(_results.size(), 2, plotId++, true);
    ax0->plot(iterations, err, "-o");
    ax0->xlabel("Iteration");
    ax0->ylabel("Chi2");

    auto ax1 = f->add_subplot(_results.size(), 2, plotId++, true);
    ax1->plot(iterations, dx, "-o");
    ax1->xlabel("Iteration");
    ax1->ylabel("|dx|");
  }
  f->size(640 * 2, 480 * 2);
}

std::string PlotAlignment::id() const { return format("{}", _t); }
void operator<<(PlotAlignment::ShPtr log, const PlotAlignment::Entry & e) { log->append(e); }
}  // namespace pd::vslam