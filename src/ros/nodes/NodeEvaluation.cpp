#include <vslam_ros/converters.h>

#include <Eigen/Dense>
#include <iostream>

#include "NodeEvaluation.h"
#include "vslam_ros/visibility_control.h"
using namespace vslam;
namespace vslam_ros
{
NodeEvaluation::NodeEvaluation(const rclcpp::NodeOptions & options)
: rclcpp::Node("NodeEvaluation", options),
  _sub(create_subscription<nav_msgs::msg::Odometry>(
    "/odom", 10, std::bind(&NodeEvaluation::callback, this, std::placeholders::_1))),
  _pub(create_publisher<nav_msgs::msg::Path>("/path/gt", 10))
{
  _outputDirectory = declare_parameter(
    "outputDirectory", "/media/data/dataset/rgbd_dataset_freiburg2_desk/test.txt");
  _algoFileName =
    declare_parameter("algoOutputFile", "/media/data/dataset/rgbd_dataset_freiburg2_desk/test.txt");
  log::initialize(_outputDirectory);
  RCLCPP_INFO(get_logger(), "Creating output at [%s]", _outputDirectory.c_str());
  _trajAlgo = std::make_shared<Trajectory>();

  declare_parameter(
    "gtTrajectoryFile",
    "/media/data/dataset/rgbd_dataset_freiburg2_desk/rgbd_dataset_freiburg2_desk-groundtruth.txt");

  _pathGt.header.frame_id = "world";
  try {
    _trajGt =
      vslam::evaluation::tum::loadTrajectory(get_parameter("gtTrajectoryFile").as_string(), true);
    convert(*_trajGt, _pathGt);
  } catch (const std::exception & e) {
    RCLCPP_ERROR(
      get_logger(),
      "Error during loading ground truth trajectory: [%s]. Will not compute KPIs or show GT.",
      e.what());
  }
}
void NodeEvaluation::callback(const nav_msgs::msg::Odometry::ConstSharedPtr msg)
{
  vslam::Pose pose;
  vslam::Timestamp t;
  vslam_ros::convert(msg->pose, pose);
  vslam_ros::convert(msg->header.stamp, t);
  _trajAlgo->append(t, pose);

  vslam::evaluation::tum::writeTrajectory(*_trajAlgo, _algoFileName, true);
  if (_trajGt) {
    try {
      //auto rpe =
      //  std::make_shared<pd::vslam::evaluation::RelativePoseError>(_trajAlgo, _trajGt, 1.0);
    } catch (const std::exception & e) {
      RCLCPP_ERROR(get_logger(), e.what());
    }
    _pathGt.header.stamp = msg->header.stamp;
    _pub->publish(_pathGt);
  }
}
}  // namespace vslam_ros
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(vslam_ros::NodeEvaluation)
