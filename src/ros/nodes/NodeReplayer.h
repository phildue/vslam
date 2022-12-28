#include <cv_bridge/cv_bridge.h>

#include <Eigen/Dense>
#include <iostream>
#include <nav_msgs/msg/path.hpp>
#include <opencv4/opencv2/core/eigen.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/opencv.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sstream>
#include <std_srvs/srv/set_bool.hpp>
#include <stereo_msgs/msg/disparity_image.hpp>
#include <tf2_msgs/msg/tf_message.hpp>
#include <thread>

#include "NodeGtLoader.h"
#include "NodeResultWriter.h"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp/serialization.hpp"
#include "rosbag2_cpp/reader.hpp"
#include "rosbag2_cpp/readers/sequential_reader.hpp"
#include "vslam_ros/visibility_control.h"
#include "vslam_ros/vslam_ros.h"

namespace vslam_ros
{
class NodeReplayer : public rclcpp::Node
{
public:
  COMPOSITION_PUBLIC

  NodeReplayer(const rclcpp::NodeOptions & options);
  virtual ~NodeReplayer();

  void publish(rosbag2_storage::SerializedBagMessageSharedPtr msg);
  void publishNext();
  void play(rcl_time_point_value_t tStart, rcl_time_point_value_t tEnd);
  bool hasNext() { return _reader->has_next(); }
  rcl_time_point_value_t seek(rcl_time_point_value_t t);

private:
  std::unique_ptr<rosbag2_cpp::readers::SequentialReader> open(const std::string & bagName) const;
  void srvSetReady(
    const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
    std::shared_ptr<std_srvs::srv::SetBool::Response> response);
  std::unique_ptr<rosbag2_cpp::readers::SequentialReader> _reader;
  bool _visualize = true;
  int _idx = 0;
  int _fNoOut = 0;
  std::atomic<bool> _running;
  std::thread _thread;
  rclcpp::Publisher<tf2_msgs::msg::TFMessage>::SharedPtr _pubTf;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr _pubImg;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr _pubDepth;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr _pubCamInfo;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr _subOdom;
  rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr _srvReady;
  std::condition_variable _cond;
  std::mutex _mutex;
  bool _nodeReady;
  std::string _bagName;
  std::string _syncTopic;
  double _duration;
  std::map<std::string, unsigned long int> _msgCtr;
  std::map<std::string, unsigned long int> _nMessages;
  rcl_time_point_value_t getStartingTime(const rosbag2_storage::BagMetadata & meta) const;
  rcl_time_point_value_t getEndTime(
    rcl_time_point_value_t tStart, const rosbag2_storage::BagMetadata & meta) const;
};
}  // namespace vslam_ros