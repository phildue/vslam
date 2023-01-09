#include <fmt/core.h>
using fmt::format;
#include "PlotAlignment.h"
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
        "Size,nConstraints,H11,H12,H13,H14,H15,H16,H21,H22,H23,H24,H25,H26,H31,H32,H33,H34,H35,H36,"
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

      Mat6d cov = (_r->normalEquations[i]->chi2() * _r->normalEquations[i]->nConstraints()) /
                  (_r->normalEquations[i]->nConstraints() - _r->normalEquations[i]->nParameters()) *
                  _r->normalEquations[i]->A().inverse();
      for (int j = 0; j < 6; j++) {
        for (int k = 0; k < 6; k++) {
          ss << "," << cov(j, k);
        }
      }
      ss << "\r\n";
    }
  }

  return ss.str();
}
std::string PlotAlignment::id() const { return format("{}", _t); }
void operator<<(PlotAlignment::ShPtr log, const PlotAlignment::Entry & e) { log->append(e); }
}  // namespace pd::vslam