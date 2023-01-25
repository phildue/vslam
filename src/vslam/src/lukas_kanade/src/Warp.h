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

#ifndef VSLAM_WARP_H__
#define VSLAM_WARP_H__
#include <memory>
#include <vector>

#include "core/core.h"
namespace pd::vslam::lukas_kanade
{
class Warp
{
public:
  typedef std::shared_ptr<Warp> ShPtr;
  typedef std::unique_ptr<Warp> UnPtr;
  typedef std::shared_ptr<const Warp> ConstShPtr;
  typedef std::unique_ptr<const Warp> ConstUnPtr;

  virtual size_t nParameters() { return _nParameters; }
  Warp(size_t nParameters) : _nParameters(nParameters), _x(Eigen::VectorXd::Zero(nParameters)) {}
  Warp(size_t nParameters, const Eigen::VectorXd & x) : _nParameters(nParameters), _x(x) {}
  virtual ~Warp() = default;
  virtual void updateAdditive(const Eigen::VectorXd & dx) = 0;
  virtual void updateCompositional(const Eigen::VectorXd & dx) = 0;
  virtual Eigen::Vector2d apply(int u, int v) const = 0;
  virtual double apply(const Image & img, int u, int v) const;
  virtual Image apply(const Image & img) const;

  virtual Eigen::MatrixXd J(int u, int v) const = 0;

  virtual void setX(const Eigen::VectorXd & x) { _x = x; }
  virtual Eigen::VectorXd x() const { return _x; }

protected:
  const size_t _nParameters;
  Eigen::VectorXd _x;
};

class WarpAffine : public Warp
{
public:
  WarpAffine(const Eigen::VectorXd & x, double cx, double cy);
  void updateAdditive(const Eigen::VectorXd & dx);
  void updateCompositional(const Eigen::VectorXd & dx);
  Eigen::Vector2d apply(int u, int v) const;
  Eigen::MatrixXd J(int u, int v) const;
  void setX(const Eigen::VectorXd & x);

private:
  Eigen::Matrix3d toMat(const Eigen::VectorXd & x) const;
  Eigen::Matrix3d _w;
  const double _cx, _cy;
};

class WarpOpticalFlow : public Warp
{
public:
  WarpOpticalFlow(const Eigen::VectorXd & x);
  void updateAdditive(const Eigen::VectorXd & dx);
  void updateCompositional(const Eigen::VectorXd & dx);
  Eigen::Vector2d apply(int u, int v) const;
  Eigen::MatrixXd J(int UNUSED(u), int UNUSED(v)) const;

  void setX(const Eigen::VectorXd & x);

private:
  Eigen::Matrix3d toMat(const Eigen::VectorXd & x) const;
  Eigen::Matrix3d _w;
};

/*
Warp based reprojection with SE3 (T) transformation:
uv_1 = p( T * p^-1( uv_0, Z(uv_0) ) )
*/
class WarpSE3 : public Warp
{
public:
  typedef std::shared_ptr<WarpSE3> ShPtr;
  typedef std::unique_ptr<WarpSE3> UnPtr;
  typedef std::shared_ptr<const WarpSE3> ConstShPtr;
  typedef std::unique_ptr<const WarpSE3> ConstUnPtr;

  WarpSE3(
    const SE3d & poseCur, const Eigen::MatrixXd & depth, Camera::ConstShPtr camRef,
    Camera::ConstShPtr camCur, const SE3d & poseRef = {});
  WarpSE3(
    const SE3d & poseCur, const std::vector<Vec3d> & pcl, size_t width, Camera::ConstShPtr camRef,
    Camera::ConstShPtr camCur, const SE3d & poseRef = {});
  virtual ~WarpSE3() = default;
  void updateAdditive(const Eigen::VectorXd & dx);
  void updateCompositional(const Eigen::VectorXd & dx);

  Eigen::Vector2d apply(int u, int v) const override;
  double apply(const Image & img, int u, int v) const override;
  Eigen::MatrixXd J(int u, int v) const;

  DepthMap apply(const DepthMap & img) const;

  void setX(const Eigen::VectorXd & x);
  SE3d poseCur() const;

private:
  SE3d _se3, _poseRef;
  const int _width;
  const std::shared_ptr<const Camera> _camCur, _camRef;
  std::vector<Eigen::Vector3d> _pcl;
  double interpolate(const Image & img, const Eigen::Vector2d & uv, double z) const;
};

}  // namespace pd::vslam::lukas_kanade
#endif
