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

#ifndef VSLAM_ROS2_NODE_MAPPING_H__
#define VSLAM_ROS2_NODE_MAPPING_H__
//#define USE_ROS2_SYNC
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/synchronizer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>

#include <Eigen/Dense>
#include <chrono>
#include <image_transport/image_transport.hpp>
#include <image_transport/subscriber_filter.hpp>
#include <memory>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <opencv4/opencv2/core/eigen.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <stereo_msgs/msg/disparity_image.hpp>

#include "vslam/vslam.h"
#include "vslam_ros/Queue.h"
#include "vslam_ros/visibility_control.h"

namespace vslam_ros
{
class NodeRgbdAlignment : public rclcpp::Node
{
public:
  COMPOSITION_PUBLIC
  NodeRgbdAlignment(const rclcpp::NodeOptions & options);

private:
  int _fNo;

  // Publications
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr _pubOdom;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr _pubPath;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr _pubPoseGraph;
  std::shared_ptr<tf2_ros::TransformBroadcaster> _pubTf;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr _pubPclMap;

  // Subscriptions
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr _subCamInfo;
#ifdef USE_ROS2_SYNC
  image_transport::SubscriberFilter _subImage, _subDepth;
#else
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr _subImage, _subDepth;
  std::shared_ptr<vslam_ros::Queue> _queue;
#endif
  // Tf
  bool _tfAvailable;
  std::string _frameId;
  std::string _fixedFrameId;
  std::string _cameraFrameId;
  std::unique_ptr<tf2_ros::Buffer> _tfBuffer;
  std::shared_ptr<tf2_ros::TransformListener> _subTf;

  // Clients
  rclcpp::Client<std_srvs::srv::SetBool>::SharedPtr _cliReplayer;

  // Timers
  rclcpp::TimerBase::SharedPtr _periodicTimer;

  // Synchronization
  using ExactPolicy =
    message_filters::sync_policies::ExactTime<sensor_msgs::msg::Image, sensor_msgs::msg::Image>;
  using ApproximatePolicy = message_filters::sync_policies::ApproximateTime<
    sensor_msgs::msg::Image, sensor_msgs::msg::Image>;
  using ExactSync = message_filters::Synchronizer<ExactPolicy>;
  using ApproximateSync = message_filters::Synchronizer<ApproximatePolicy>;
  std::shared_ptr<ExactSync> _exactSync;
  std::shared_ptr<ApproximateSync> _approximateSync;

  // Algorithm
  vslam::DirectIcp::ShPtr _directIcp;
  vslam::Camera::ShPtr _camera;
  vslam::Pose _motion, _pose;
  vslam::Trajectory _trajectory;
  vslam::Frame::ShPtr _frame0;
  std::map<std::string, double> _paramsIcp;

  // Buffers
  geometry_msgs::msg::TransformStamped
    _world2origin;  //transforms from fixed frame to initial pose of optical frame

  void cameraCallback(sensor_msgs::msg::CameraInfo::ConstSharedPtr msg);
  void imageCallback(
    sensor_msgs::msg::Image::ConstSharedPtr msgImg,
    sensor_msgs::msg::Image::ConstSharedPtr msgDepth);

  void timerCallback();
  void publish(const rclcpp::Time & t);
  void lookupTf();
  void triggerReplayer();
};
}  // namespace vslam_ros

#endif  //VSLAM_ROS2_NODE_MAPPING_H__
