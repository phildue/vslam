#ifndef NODE_GT_LOADER_H__
#define NODE_GT_LOADER_H__
#include <iostream>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <stereo_msgs/msg/disparity_image.hpp>
#include <string>

#include "vslam_ros/visibility_control.h"

namespace vslam_ros
{
class NodeGtLoader : public rclcpp::Node
{
public:
  COMPOSITION_PUBLIC
  NodeGtLoader(const rclcpp::NodeOptions & options);

private:
  void callback(const nav_msgs::msg::Odometry::ConstSharedPtr msg);
  const rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr _sub;
  const rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr _pub;
  std::string _gtFileName;
  nav_msgs::msg::Path _pathImu;
};
}  // namespace vslam_ros
#endif