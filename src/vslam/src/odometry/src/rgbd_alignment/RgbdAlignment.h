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

#ifndef VSLAM_ALIGNER_H__
#define VSLAM_ALIGNER_H__

#include <core/core.h>
#include <least_squares/least_squares.h>
namespace pd::vslam
{
class RgbdAlignment
{
public:
  typedef std::shared_ptr<RgbdAlignment> ShPtr;
  typedef std::unique_ptr<RgbdAlignment> UnPtr;
  typedef std::shared_ptr<const RgbdAlignment> ConstShPtr;
  typedef std::unique_ptr<const RgbdAlignment> ConstUnPtr;

  RgbdAlignment(
    vslam::least_squares::Solver::ShPtr solver, vslam::least_squares::Loss::ShPtr loss,
    bool includePrior = false, bool initializeOnPrediction = true, int nLevels = 4,
    const std::vector<double> & minGradient = {0, 0, 0, 0, 0}, double minDepth = 0.1,
    double maxDepth = 50, double minDepthDiff = 0.1, double maxDepthDiff = 0.1,
    const std::vector<double> & maxPointsPart = {1.0, 1.0, 1.0, 1.0, 1.0},
    int distanceToBorder = 7.0);

  virtual Pose align(Frame::ShPtr from, Frame::ShPtr to) const;
  virtual Pose align(const Frame::VecShPtr & from, Frame::ShPtr to) const;

  virtual Pose align(Frame::ConstShPtr from, Frame::ConstShPtr to) const;
  virtual Pose align(const Frame::VecConstShPtr & from, Frame::ConstShPtr to) const;
  virtual least_squares::Problem::UnPtr setupProblem(
    const Vec6d & twist, Frame::ConstShPtr from, Frame::ConstShPtr to, int level) const;
  virtual void preprocessReference(Frame::ShPtr frame) const;
  virtual void preprocessReference(Frame::VecShPtr frames) const;
  virtual void preprocessTarget(Frame::ShPtr frame) const;

protected:
  const vslam::least_squares::Loss::ShPtr _loss;
  const vslam::least_squares::Solver::ShPtr _solver;
  const bool _includePrior, _initializeOnPrediction;
  const int _nLevels;
  std::vector<double> _minGradient2;
  const double _minDepth;
  const double _maxDepth;
  const double _minDepthDiff, _maxDepthDiff;
  const std::vector<double> _maxPoints;
  const int _distanceToBorder;
  virtual std::vector<Vec2i> selectInterestPoints(Frame::ConstShPtr frame, int level) const;
  virtual std::vector<Vec2i> uniformSubselection(
    Frame::ConstShPtr frame, const std::vector<Vec2i> & interestPoints, int level) const;
};

}  // namespace pd::vslam

#endif  //VSLAM_ALIGNER_H__
