#include "NodeResultWriter.h"

namespace vslam_ros
{
NodeResultWriter::NodeResultWriter(const rclcpp::NodeOptions & options)
: rclcpp::Node("NodeResultWriter", options),
  _sub(this->create_subscription<nav_msgs::msg::Odometry>(
    "/odom", 10, std::bind(&NodeResultWriter::callback, this, std::placeholders::_1)))
{
  declare_parameter("algoOutputFile", "/media/data/dataset/rgbd_dataset_freiburg2_desk/test.txt");
  _outputFile = get_parameter("algoOutputFile").as_string();
  RCLCPP_INFO(get_logger(), "Creating output at [%s]", _outputFile.c_str());
  _algoFile.open(_outputFile, std::ios_base::out);
  _algoFile << "# Algorithm Trajectory\n";
  _algoFile << "# file: " << _outputFile << "\n";
  _algoFile << "# timestamp tx ty tz qx qy qz qw\n";
}
void NodeResultWriter::callback(const nav_msgs::msg::Odometry::ConstSharedPtr msg)
{
  if (!_algoFile.is_open()) {
    std::runtime_error("Could not open file at: " + _outputFile);
  }

  _algoFile << msg->header.stamp.sec << "." << msg->header.stamp.nanosec << " "
            << msg->pose.pose.position.x << " " << msg->pose.pose.position.y << " "
            << msg->pose.pose.position.z << " " << msg->pose.pose.orientation.x << " "
            << msg->pose.pose.orientation.y << " " << msg->pose.pose.orientation.z << " "
            << msg->pose.pose.orientation.w;
  for (int i = 0; i < 36; i++) {
    _algoFile << " " << msg->pose.covariance[i];
  }
  _algoFile << "\n";
}
}  // namespace vslam_ros
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(vslam_ros::NodeResultWriter)