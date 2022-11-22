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

#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>

#include <Eigen/Dense>
#include <chrono>
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
class NodeMapping : public rclcpp::Node
{
public:
  COMPOSITION_PUBLIC
  NodeMapping(const rclcpp::NodeOptions & options);

  bool ready();
  void processFrame(
    sensor_msgs::msg::Image::ConstSharedPtr msgImg,
    sensor_msgs::msg::Image::ConstSharedPtr msgDepth);

  void depthCallback(sensor_msgs::msg::Image::ConstSharedPtr msgDepth);

  void imageCallback(sensor_msgs::msg::Image::ConstSharedPtr msgImg);
  void dropCallback(
    sensor_msgs::msg::Image::ConstSharedPtr msgImg,
    sensor_msgs::msg::Image::ConstSharedPtr msgDepth);

  void cameraCallback(sensor_msgs::msg::CameraInfo::ConstSharedPtr msg);
  pd::vslam::Frame::ShPtr createFrame(
    sensor_msgs::msg::Image::ConstSharedPtr msgImg,
    sensor_msgs::msg::Image::ConstSharedPtr msgDepth) const;

private:
  const bool _includeKeyFrame;
  bool _camInfoReceived;
  bool _tfAvailable;
  int _fNo;
  std::string _frameId;
  std::string _fixedFrameId;

  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr _pubOdom;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr _pubPath;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr _pubPoseGraph;
  std::shared_ptr<tf2_ros::TransformBroadcaster> _pubTf;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr _pubPclMap;

  std::unique_ptr<tf2_ros::Buffer> _tfBuffer;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr _subCamInfo;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr _subImage;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr _subDepth;
  std::shared_ptr<tf2_ros::TransformListener> _subTf;
  rclcpp::Client<std_srvs::srv::SetBool>::SharedPtr _cliReplayer;

  const std::shared_ptr<vslam_ros::Queue> _queue;

  pd::vslam::Odometry::ShPtr _odometry;
  pd::vslam::KeyFrameSelection::ShPtr _keyFrameSelection;
  pd::vslam::MotionPrediction::ShPtr _prediction;
  pd::vslam::Map::ShPtr _map;
  pd::vslam::FeatureTracking::ShPtr _tracking;
  pd::vslam::Matcher::ShPtr _matcher;
  pd::vslam::mapping::BundleAdjustment::ShPtr _ba;

  pd::vslam::Camera::ShPtr _camera;
  geometry_msgs::msg::TransformStamped
    _world2origin;  //transforms from fixed frame to initial pose of optical frame
  nav_msgs::msg::Path _path;

  void publish(sensor_msgs::msg::Image::ConstSharedPtr msgImg, pd::vslam::Frame::ConstShPtr frame);
  void lookupTf(sensor_msgs::msg::Image::ConstSharedPtr msgImg);
  void signalReplayer();
};
}  // namespace vslam_ros

#endif  //VSLAM_ROS2_NODE_MAPPING_H__
   