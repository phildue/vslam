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
  void play();
  bool hasNext() { return _reader->has_next(); }

private:
  void srvSetReady(
    const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
    std::shared_ptr<std_srvs::srv::SetBool::Response> response);
  std::unique_ptr<rosbag2_cpp::readers::SequentialReader> _reader;
  int _nFrames;
  rcutils_time_point_value_t _tStart = 0;
  bool _visualize = true;
  int _idx = 0;
  int _fNo = 0;
  double _period = 0.05;
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
  std::atomic<bool> _nodeReady;
  std::string _bagName;
  std::string _syncTopic;
  double _duration;
};
}  // namespace vslam_ros