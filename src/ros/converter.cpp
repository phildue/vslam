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

#include "vslam_ros/converters.h"
namespace vslam_ros
{
pd::vslam::Camera::ShPtr convert(const sensor_msgs::msg::CameraInfo & msg)
{
  const double fx = msg.k[0 * 3 + 0];
  const double fy = msg.k[1 * 3 + 1];
  const double cx = msg.k[0 * 3 + 2];
  const double cy = msg.k[1 * 3 + 2];

  return std::make_shared<pd::vslam::Camera>(fx, fy, cx, cy);
}

geometry_msgs::msg::Pose convert(const Sophus::SE3d & se3)
{
  geometry_msgs::msg::Pose pose;

  const auto t = se3.translation();
  const auto q = se3.unit_quaternion();
  pose.position.x = t.x();
  pose.position.y = t.y();
  pose.position.z = t.z();
  pose.orientation.w = q.w();
  pose.orientation.x = q.x();
  pose.orientation.y = q.y();
  pose.orientation.z = q.z();
  return pose;
}
Sophus::SE3d convert(const geometry_msgs::msg::Pose & ros)
{
  return Sophus::SE3d(
    Eigen::Quaterniond(ros.orientation.w, ros.orientation.x, ros.orientation.y, ros.orientation.z),
    Eigen::Vector3d(ros.position.x, ros.position.y, ros.position.z));
}
void convert(const Sophus::SE3d & se3, geometry_msgs::msg::Twist & twist)
{
  twist.angular.x = se3.log().tail(3).x();
  twist.angular.y = se3.log().tail(3).y();
  twist.angular.z = se3.log().tail(3).z();

  twist.linear.x = se3.log().head(3).x();
  twist.linear.y = se3.log().head(3).y();
  twist.linear.z = se3.log().head(3).z();
}
Sophus::SE3d convert(const geometry_msgs::msg::TransformStamped & tf)
{
  return Sophus::SE3d(
    Eigen::Quaterniond(
      tf.transform.rotation.w, tf.transform.rotation.x, tf.transform.rotation.y,
      tf.transform.rotation.z),
    Eigen::Vector3d(
      tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z));
}
void convert(const Sophus::SE3d & sophus, geometry_msgs::msg::TransformStamped & tf)
{
  const auto t = sophus.translation();
  const auto q = sophus.unit_quaternion();
  tf.transform.translation.x = t.x();
  tf.transform.translation.y = t.y();
  tf.transform.translation.z = t.z();
  tf.transform.rotation.w = q.w();
  tf.transform.rotation.x = q.x();
  tf.transform.rotation.y = q.y();
  tf.transform.rotation.z = q.z();
}

void convert(const pd::vslam::PoseWithCovariance & p, geometry_msgs::msg::PoseWithCovariance & pRos)
{
  pRos.pose = convert(p.pose());
  for (int i = 0; i < 6; i++) {
    for (int j = 0; j < 6; j++) {
      pRos.covariance[i * 6 + j] = p.cov()(i, j);
    }
  }
}
void convert(
  const pd::vslam::PoseWithCovariance & p, geometry_msgs::msg::TwistWithCovariance & pRos)
{
  convert(p.pose(), pRos.twist);
  for (int i = 0; i < 6; i++) {
    for (int j = 0; j < 6; j++) {
      pRos.covariance[i * 6 + j] = p.cov()(i, j);
    }
  }
}
}  // namespace vslam_ros
