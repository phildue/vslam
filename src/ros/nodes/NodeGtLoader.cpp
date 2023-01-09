#include <vslam_ros/converters.h>

#include <Eigen/Dense>
#include <iostream>

#include "NodeGtLoader.h"
#include "vslam_ros/visibility_control.h"
using namespace pd::vslam;
namespace vslam_ros
{
NodeGtLoader::NodeGtLoader(const rclcpp::NodeOptions & options)
: rclcpp::Node("NodeGtLoader", options),
  _sub(create_subscription<nav_msgs::msg::Odometry>(
    "/odom", 10, std::bind(&NodeGtLoader::callback, this, std::placeholders::_1))),
  _pub(create_publisher<nav_msgs::msg::Path>("/path/gt", 10))
{
  declare_parameter(
    "gtTrajectoryFile",
    "/media/data/dataset/rgbd_dataset_freiburg2_desk/rgbd_dataset_freiburg2_desk-groundtruth.txt");
  _gtFileName = get_parameter("gtTrajectoryFile").as_string();
  _pathGt.header.frame_id = "world";
  _trajGt = pd::vslam::utils::loadTrajectory(_gtFileName);
  _trajAlgo = std::make_shared<pd::vslam::Trajectory>();
  //TODO just convert the trajectory type to a ros path
  std::ifstream gtFile;
  gtFile.open(_gtFileName);
  if (!gtFile.is_open()) {
    std::runtime_error("Could not open file at: " + _gtFileName);
  }
  Sophus::SE3d pose0;
  int nPoses = 0;
  std::string line;
  while (getline(gtFile, line)) {
    std::vector<std::string> elements;
    std::string s;
    std::istringstream lines(line);
    while (getline(lines, s, ' ')) {
      elements.push_back(s);
    }
    if (elements[0] == "#") {  //skip comments
      continue;
    }
    Eigen::Vector3d t;
    t << std::stod(elements[1]), std::stod(elements[2]), std::stod(elements[3]);
    Eigen::Quaterniond q(
      std::stod(elements[7]), std::stod(elements[4]), std::stod(elements[5]),
      std::stod(elements[6]));
    Sophus::SE3d pose(q, t);

    if (nPoses == 0) {
      pose0 = pose;
    }
    //pose = pose0.inverse() * pose;

    std::vector<std::string> tElements;
    std::string st;
    std::istringstream tLine(elements[0]);
    while (getline(tLine, st, '.')) {
      tElements.push_back(st);
    }
    geometry_msgs::msg::PoseStamped pStamped;
    pStamped.header.frame_id = "world";
    pStamped.header.stamp.sec = std::stoull(tElements[0]);
    pStamped.header.stamp.nanosec = std::stoull(tElements[1]) * 100000;

    pStamped.pose = vslam_ros::convert(pose);
    _pathGt.poses.push_back(pStamped);
    nPoses++;
  }
}
void NodeGtLoader::callback(const nav_msgs::msg::Odometry::ConstSharedPtr msg)
{
  pd::vslam::Pose pose;
  pd::vslam::Timestamp t;
  vslam_ros::convert(msg->pose, pose);
  vslam_ros::convert(msg->header.stamp, t);
  _trajAlgo->append(t, pose);

  try {
    auto rpe = std::make_shared<pd::vslam::evaluation::RelativePoseError>(_trajAlgo, _trajGt, 1.0);
  } catch (const std::exception & e) {
    RCLCPP_ERROR(get_logger(), e.what());
  }

  _pathGt.header.stamp = msg->header.stamp;
  _pub->publish(_pathGt);
}
}  // namespace vslam_ros
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(vslam_ros::NodeGtLoader)
