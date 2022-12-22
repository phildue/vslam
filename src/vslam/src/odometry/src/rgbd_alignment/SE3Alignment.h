// Copyright 2022 Philipp.Duernay
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

#ifndef VSLAM_SE3_ALIGNMENT
#define VSLAM_SE3_ALIGNMENT

#include "RgbdAlignment.h"
#include "core/core.h"
#include "least_squares/least_squares.h"
#include "lukas_kanade/lukas_kanade.h"
namespace pd::vslam
{
class SE3Alignment : public RgbdAlignment
{
public:
  typedef std::shared_ptr<SE3Alignment> ShPtr;
  typedef std::unique_ptr<SE3Alignment> UnPtr;
  typedef std::shared_ptr<const SE3Alignment> ConstShPtr;
  typedef std::unique_ptr<const SE3Alignment> ConstUnPtr;

  class InterestPointSelection
  {
  public:
    typedef std::shared_ptr<InterestPointSelection> ShPtr;
    typedef std::unique_ptr<InterestPointSelection> UnPtr;
    typedef std::shared_ptr<const InterestPointSelection> ConstShPtr;
    typedef std::unique_ptr<const InterestPointSelection> ConstUnPtr;

    InterestPointSelection(
      const std::vector<double> & minGradient, double minDepth = 0.1, double maxDepth = 50,
      double maxPointsPart = 0.09);
    std::vector<Eigen::Vector2i> select(
      Frame::ConstShPtr frameRef, Frame::ConstShPtr frameCur, int level) const;
    std::vector<Eigen::Vector2i> selectRandomSubset(
      const std::vector<Eigen::Vector2i> & interestPoints, int height, int width,
      double part) const;
    std::vector<double> _minGradient2;
    double _minDepth;
    double _maxDepth;
    double _maxDepthDiff;
    double _maxPointsPart;
  };

  SE3Alignment(
    double minGradient, vslam::least_squares::Solver::ShPtr solver,
    vslam::least_squares::Loss::ShPtr loss, bool includePrior = false,
    bool initializeOnPrediction = true);

  PoseWithCovariance::UnPtr align(Frame::ConstShPtr from, Frame::ConstShPtr to) const override;
  PoseWithCovariance::UnPtr align(
    const Frame::VecConstShPtr & from, Frame::ConstShPtr to) const override;

  //TODO do we really need this modifiable? should simple be set at start
  bool & includePrior() { return _includePrior; }
  bool & initializeOnPrediction() { return _initializeOnPrediction; }
  const bool & includePrior() const { return _includePrior; }
  const bool & initializeOnPrediction() const { return _initializeOnPrediction; }

protected:
  const double _minGradient;
  const vslam::least_squares::Loss::ShPtr _loss;
  const vslam::least_squares::Solver::ShPtr _solver;
  bool _includePrior, _initializeOnPrediction;
  InterestPointSelection::ConstUnPtr _interestPointSelection;
};
}  // namespace pd::vslam
#endif  // VSLAM_SE3_ALIGNMENT
