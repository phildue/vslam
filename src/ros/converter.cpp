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
void convert(vslam::Timestamp t, builtin_interfaces::msg::Time & tRos)
{
  tRos.sec = static_cast<int32_t>(t / 1e9);
  tRos.nanosec = t - tRos.sec * 1e9;
}
void convert(const builtin_interfaces::msg::Time & tRos, vslam::Timestamp & t)
{
  t = tRos.sec * 1e9 + tRos.nanosec;
}
vslam::Camera::ShPtr convert(const sensor_msgs::msg::CameraInfo & msg)
{
  double fx = msg.k[0 * 3 + 0];
  double fy = msg.k[1 * 3 + 1];
  double cx = msg.k[0 * 3 + 2];
  double cy = msg.k[1 * 3 + 2];

  if ((fx == 0 || cx == 0) && (msg.p[0 * 4 + 0] > 0 && msg.p[1 * 4 + 3] <= 1e-3)) {
    //seems like k is not set but instead P TODO: its a bit hacky or risky
    /*
    P(i)rect = [[fu 0  cx  -fu*bx],
               [0  fv  cy -fv*by],
               [0   0   1  0]]
    */
    fx = msg.p[0 * 4 + 0];
    fy = msg.p[1 * 4 + 1];
    cx = msg.p[0 * 4 + 2];
    cy = msg.p[1 * 4 + 2];
  }

  return std::make_shared<vslam::Camera>(fx, fy, cx, cy, std::ceil(cx * 2.0), std::ceil(cy * 2.0));
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

void convert(const vslam::Pose & p, geometry_msgs::msg::PoseWithCovariance & pRos)
{
  pRos.pose = convert(p.pose());
  for (int i = 0; i < 6; i++) {
    for (int j = 0; j < 6; j++) {
      pRos.covariance[i * 6 + j] = p.cov()(i, j);
    }
  }
  //TODO check ros2 conventions for covariance ordering
}
void convert(const vslam::Pose & p, geometry_msgs::msg::TwistWithCovariance & pRos)
{
  convert(p.pose(), pRos.twist);
  for (int i = 0; i < 6; i++) {
    for (int j = 0; j < 6; j++) {
      pRos.covariance[i * 6 + j] = p.cov()(i, j);
    }
  }
  //TODO check ros2 conventions for covariance ordering
}
#if false
void convert(
  const std::vector<vslam::Point3D::ConstShPtr> & pcl, sensor_msgs::msg::PointCloud2 & pclRos)
{
  //TODO(phil) is this copying the pcl two times?
  pcl::PointCloud<pcl::PointXYZRGB> pclPcl;
  for (auto p : pcl) {
    auto ft0 = p->features()[0];
    pcl::PointXYZRGB pPcl;
    pPcl.x = p->position().x();
    pPcl.y = p->position().y();
    pPcl.z = p->position().z();
    pPcl.r = ft0->frame()->intensity()(ft0->position().y(), ft0->position().x());
    pPcl.g = ft0->frame()->intensity()(ft0->position().y(), ft0->position().x());
    pPcl.b = ft0->frame()->intensity()(ft0->position().y(), ft0->position().x());

    pclPcl.push_back(pPcl);
  }
  pcl::toROSMsg(pclPcl, pclRos);
}
#endif
void convert(const geometry_msgs::msg::PoseWithCovariance & pRos, vslam::Pose & p)
{
  p.pose() = convert(pRos.pose);
  for (int i = 0; i < 6; i++) {
    for (int j = 0; j < 6; j++) {
      p.cov()(i, j) = pRos.covariance[i * 6 + j];
    }
  }
  //TODO check ros2 conventions for covariance ordering
}
void convert(const geometry_msgs::msg::TwistWithCovariance & twistRos, vslam::Pose & p)
{
  vslam::Vec6d twist;
  twist(0) = twistRos.twist.linear.x;
  twist(1) = twistRos.twist.linear.y;
  twist(2) = twistRos.twist.linear.z;
  twist(3) = twistRos.twist.angular.x;
  twist(4) = twistRos.twist.angular.y;
  twist(5) = twistRos.twist.angular.z;
  p.pose() = vslam::SE3d::exp(twist);
  for (int i = 0; i < 6; i++) {
    for (int j = 0; j < 6; j++) {
      p.cov()(i, j) = twistRos.covariance[i * 6 + j];
    }
  }
  //TODO check ros2 conventions for covariance ordering
}
void convert(const vslam::Trajectory & traj, nav_msgs::msg::Path & trajRos)
{
  trajRos.poses.reserve(traj.poses().size());
  for (const auto t_p : traj.poses()) {
    auto pose = t_p.second->SE3();
    auto t = t_p.first;

    geometry_msgs::msg::PoseStamped poseRos;
    convert(t, poseRos.header.stamp);
    poseRos.pose = vslam_ros::convert(pose);
    poseRos.header.frame_id =
      "world";  //TODO introduces coordinate frames t trajectory and pose class
    trajRos.poses.push_back(poseRos);
  }
}

}  // namespace vslam_ros
