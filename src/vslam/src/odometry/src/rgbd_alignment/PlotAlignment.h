#ifndef VSLAM_PLOT_ALIGNMENT_H__
#define VSLAM_PLOT_ALIGNMENT_H__

#include "core/core.h"
#include "least_squares/least_squares.h"
#include "utils/utils.h"
namespace pd::vslam
{
class PlotAlignment : public vis::Plot
{
public:
  typedef std::shared_ptr<PlotAlignment> ShPtr;
  typedef std::unique_ptr<PlotAlignment> UnPtr;
  typedef std::shared_ptr<const PlotAlignment> ConstShPtr;
  typedef std::unique_ptr<const PlotAlignment> ConstUnPtr;

  struct Entry
  {
    int level;
    least_squares::Solver::Results::ConstShPtr results;
  };
  PlotAlignment(Timestamp t) : _t(t) {}
  void plot() const {};
  virtual void append(const Entry & e);

  std::string csv() const override;
  std::string id() const override;
  static UnPtr make(Timestamp t);

private:
  const Timestamp _t;
  std::map<int, least_squares::Solver::Results::ConstShPtr> _results;
};
void operator<<(PlotAlignment::ShPtr log, const PlotAlignment::Entry & e);

class PlotAlignmentNull : public PlotAlignment
{
public:
  PlotAlignmentNull() : PlotAlignment(0UL) {}

  void append(const Entry & UNUSED(e)) override {}
};

}  // namespace pd::vslam
#endif