#ifndef VSLAM_RELATIVE_POSE_ERROR_H__
#define VSLAM_RELATIVE_POSE_ERROR_H__
#include "core/core.h"
#include "utils/utils.h"
namespace pd::vslam::evaluation
{
class RelativePoseError
{
public:
  typedef std::shared_ptr<RelativePoseError> ShPtr;
  typedef std::unique_ptr<RelativePoseError> UnPtr;
  typedef std::shared_ptr<const RelativePoseError> ConstShPtr;
  typedef std::unique_ptr<const RelativePoseError> ConstUnPtr;

  struct Statistics
  {
    double rmse = 0.0;
    double mean = 0.0;
    double median = 0.0;
    double stddev = 0.0;
    double min = 0.0;
    double max = 0.0;
  };

  RelativePoseError(Trajectory::ConstShPtr algo, Trajectory::ConstShPtr gt, double dT);
  void compute();

  const Statistics & angle() const;
  const Statistics & translation() const;
  const std::vector<double> & errorsTranslation() const;
  const std::vector<double> & errorsAngles() const;
  const std::vector<Timestamp> & timestamps() const;
  std::string toString() const;
  static RelativePoseError::ConstUnPtr compute(
    Trajectory::ConstShPtr algo, Trajectory::ConstShPtr gt, double dT);

private:
  const double _dT;
  const Trajectory::ConstShPtr _algo, _gt;
  std::vector<double> _errorsTranslation;
  std::vector<double> _errorsAngles;
  std::vector<Timestamp> _timestamps;
  Statistics _statTranslation, _statAngles;
};

class PlotRPE : public vis::Plot
{
public:
  PlotRPE(const std::map<std::string, RelativePoseError::ConstShPtr> & errors);
  void plot() const override;
  std::string csv() const override { return ""; }

private:
  std::map<std::string, RelativePoseError::ConstShPtr> _errors;
};
}  // namespace pd::vslam::evaluation

#endif  //VSLAM_RELATIVE_POSE_ERROR_H__