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

#ifndef VSLAM_FEATURE_SELECTION_H__
#define VSLAM_FEATURE_SELECTION_H__

#include "Matcher.h"
#include "core/core.h"
namespace pd::vslam
{
class FeatureSelection
{
public:
  typedef std::shared_ptr<FeatureSelection> ShPtr;
  typedef std::unique_ptr<FeatureSelection> UnPtr;
  typedef std::shared_ptr<const FeatureSelection> ConstShPtr;
  typedef std::unique_ptr<const FeatureSelection> ConstUnPtr;

  virtual Feature2D::VecShPtr select(Frame::ConstShPtr frame) = 0;
};

class FeatureSubsampling
{
public:
  typedef std::shared_ptr<FeatureSubsampling> ShPtr;
  typedef std::unique_ptr<FeatureSubsampling> UnPtr;
  typedef std::shared_ptr<const FeatureSubsampling> ConstShPtr;
  typedef std::unique_ptr<const FeatureSubsampling> ConstUnPtr;

  virtual Feature2D::VecShPtr select(Feature2D::VecShPtr features) = 0;
};

class GradientMagnitudeDepth : public FeatureSelection
{
public:
  typedef std::shared_ptr<GradientMagnitudeDepth> ShPtr;
  typedef std::unique_ptr<GradientMagnitudeDepth> UnPtr;
  typedef std::shared_ptr<const GradientMagnitudeDepth> ConstShPtr;
  typedef std::unique_ptr<const GradientMagnitudeDepth> ConstUnPtr;

  GradientMagnitudeDepth(
    const std::vector<int> & minGradient, double minDepth = 0.1, double maxDepth = 50.0);

  Feature2D::VecShPtr select(Frame::ConstShPtr frame) override;

  std::vector<double> _minGradient2;
  double _minDepth;
  double _maxDepth;
};

class UniformSubsampling
{
public:
  typedef std::shared_ptr<FeatureSubsampling> ShPtr;
  typedef std::unique_ptr<FeatureSubsampling> UnPtr;
  typedef std::shared_ptr<const FeatureSubsampling> ConstShPtr;
  typedef std::unique_ptr<const FeatureSubsampling> ConstUnPtr;

  UniformSubsampling(const std::vector<size_t> & maxPoints);

  Feature2D::VecShPtr select(Feature2D::VecShPtr features) override;

private:
  std::vector<size_t> _maxPoints;
};

class GridSubsampling
{
public:
  typedef std::shared_ptr<FeatureSubsampling> ShPtr;
  typedef std::unique_ptr<FeatureSubsampling> UnPtr;
  typedef std::shared_ptr<const FeatureSubsampling> ConstShPtr;
  typedef std::unique_ptr<const FeatureSubsampling> ConstUnPtr;

  GridSubsampling(const std::vector<double> & cellSize);

  Feature2D::VecShPtr select(Feature2D::VecShPtr features) override;

private:
  std::vector<double> _cellSize;
};


}  // namespace pd::vslam