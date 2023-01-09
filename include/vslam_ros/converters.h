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

#ifndef VSLAM_ROS2_CONVERTER_H__
#define VSLAM_ROS2_CONVERTER_H__
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <vslam/vslam.h>

#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_with_covariance.hpp>
#include <geometry_msgs/msg/twist_with_covariance.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sophus/se3.hpp>

namespace vslam_ros
{
void convert(pd::vslam::Timestamp t, builtin_interfaces::msg::Time & tRos);
void convert(const builtin_interfaces::msg::Time & tRos, pd::vslam::Timestamp & t);

pd::vslam::Camera::ShPtr convert(const sensor_msgs::msg::CameraInfo & msg);
geometry_msgs::msg::Pose convert(const Sophus::SE3d & se3);
void convert(const Sophus::SE3d & se3, geometry_msgs::msg::Twist & twist);
Sophus::SE3d convert(const geometry_msgs::msg::Pose & ros);

Sophus::SE3d convert(const geometry_msgs::msg::TransformStamped & tf);
void convert(const Sophus::SE3d & sophus, geometry_msgs::msg::TransformStamped & tf);
void convert(
  const pd::vslam::PoseWithCovariance & p, geometry_msgs::msg::PoseWithCovariance & pRos);
void convert(
  const pd::vslam::PoseWithCovariance & p, geometry_msgs::msg::TwistWithCovariance & pRos);
void convert(
  const geometry_msgs::msg::PoseWithCovariance & pRos, pd::vslam::PoseWithCovariance & p);
void convert(
  const geometry_msgs::msg::TwistWithCovariance & pRos, pd::vslam::PoseWithCovariance & p);

void convert(
  const std::vector<pd::vslam::Point3D::ConstShPtr> & pcl, sensor_msgs::msg::PointCloud2 & pclRos);

}  // namespace vslam_ros
#endif
