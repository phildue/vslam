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

#ifndef VSLAM_DIRECT_ICP_H__
#define VSLAM_DIRECT_ICP_H__
#include <memory>

#include "Warp.h"
#include "core/core.h"
#include "least_squares/least_squares.h"
namespace pd::vslam
{
class DirectIcp : public least_squares::Problem
{
public:
  DirectIcp(
    const SE3d & se3, Frame::ConstShPtr fRef, Frame::ConstShPtr fTo,
    const std::vector<Eigen::Vector2i> & interestPoints, int level,
    least_squares::Loss::ShPtr = std::make_shared<least_squares::QuadraticLoss>());

  virtual ~DirectIcp() = default;

  void updateX(const Eigen::VectorXd & dx) override;
  void setX(const Eigen::VectorXd & x) override { _se3 = SE3d::exp(x); }

  Eigen::VectorXd x() const override { return _se3.log(); }

  least_squares::NormalEquations::UnPtr computeNormalEquations() override;

protected:
  const int _level;
  const Frame::ConstShPtr _f0;
  const Frame::ConstShPtr _f1;
  const std::shared_ptr<least_squares::Loss> _loss;
  const Image & _I1;
  const DepthMap & _Z1;
  struct Constraint
  {
    size_t idx;
    int u;
    int v;
    Vec3d p;
    double I0, I1Wxp;
    double Z0, Z1Wxp;
    Vec6d J, J_I, J_Z;
    double r_I, r_Z;
    double residual;
    double weight;
  };
  std::vector<Constraint> _constraints;
  SE3d _se3;

  Vec6d J_T_x(const Vec3d & p);
  Vec6d J_T_y(const Vec3d & p);
  Vec6d J_T_z(const Vec3d & p);
  void interpolate(double u, double v, Constraint & c) const;
};

}  // namespace pd::vslam
#endif
