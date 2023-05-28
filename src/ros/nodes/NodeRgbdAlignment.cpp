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

//
// Created by phil on 07.08.21.
//

#include <cv_bridge/cv_bridge.h>

#include "NodeRgbdAlignment.h"
#include "vslam_ros/converters.h"

using namespace vslam;
using namespace std::chrono_literals;

namespace vslam_ros
{
NodeRgbdAlignment::NodeRgbdAlignment(const rclcpp::NodeOptions & options)
: rclcpp::Node("NodeRgbdAlignment", options),
  _fNo(0),
  _pubOdom(create_publisher<nav_msgs::msg::Odometry>("/odom", 10)),
  _pubPath(create_publisher<nav_msgs::msg::Path>("/path", 10)),
  _pubPoseGraph(create_publisher<nav_msgs::msg::Path>("/path/pose_graph", 10)),
  _pubTf(std::make_shared<tf2_ros::TransformBroadcaster>(this)),
  _pubPclMap(create_publisher<sensor_msgs::msg::PointCloud2>("/pcl/map", 10)),
  _subCamInfo(create_subscription<sensor_msgs::msg::CameraInfo>(
    "/camera/rgb/camera_info", 10,
    std::bind(&NodeRgbdAlignment::cameraCallback, this, std::placeholders::_1))),
  _tfAvailable(false),
  _tfBuffer(std::make_unique<tf2_ros::Buffer>(get_clock())),
  _subTf(std::make_shared<tf2_ros::TransformListener>(*_tfBuffer))
{
  declare_parameter("replayMode", true);
  declare_parameter("tf.base_link_id", _fixedFrameId);
  declare_parameter("tf.frame_id", _frameId);
  _fixedFrameId = get_parameter("tf.base_link_id").as_string();
  _frameId = get_parameter("tf.frame_id").as_string();
  if (get_parameter("replayMode").as_bool()) {
    _cliReplayer = create_client<std_srvs::srv::SetBool>("set_ready");
  }

  for (auto name_value : DirectIcp::defaultParameters()) {
    _paramsIcp[name_value.first] = declare_parameter(name_value.first, name_value.second);
  }

  int queue_size = declare_parameter("sync.queue_size", 5);
  double max_interval = declare_parameter("sync.max_interval", 0.2);

#ifdef USE_ROS2_SYNC

  image_transport::TransportHints hints(this, "raw");
  rclcpp::QoS image_sub_qos = rclcpp::SensorDataQoS();
  const auto image_sub_rmw_qos = image_sub_qos.get_rmw_qos_profile();
  _subImage.subscribe(this, "/camera/rgb/image_color", hints.getTransport(), image_sub_rmw_qos);
  _subDepth.subscribe(this, "/camera/depth/image", hints.getTransport(), image_sub_rmw_qos);

  if (max_interval > 0.0) {
    _approximateSync.reset(
      new ApproximateSync(ApproximatePolicy(queue_size), _subImage, _subDepth));
    _approximateSync->setMaxIntervalDuration(rclcpp::Duration::from_seconds(max_interval));
    _approximateSync->registerCallback(&NodeRgbdAlignment::imageCallback, this);
  } else {
    _exactSync.reset(new ExactSync(ExactPolicy(queue_size), _subImage, _subDepth));
    _exactSync->registerCallback(&NodeRgbdAlignment::imageCallback, this);
  }
#else
  _queue = std::make_unique<vslam_ros::Queue>(queue_size, max_interval * 1e9);
  _subImage = create_subscription<sensor_msgs::msg::Image>(
    "/camera/rgb/image_color", 10,
    [this](sensor_msgs::msg::Image::ConstSharedPtr msg) { _queue->pushImage(msg); });
  _subDepth = create_subscription<sensor_msgs::msg::Image>(
    "/camera/depth/image", 10, [this](sensor_msgs::msg::Image::ConstSharedPtr msg) {
      _queue->pushDepth(msg);

      if (_queue->size() >= 1 && _camera) {
        try {
          auto img = _queue->popClosestImg();
          auto depth = _queue->popClosestDepth(
            rclcpp::Time(img->header.stamp.sec, img->header.stamp.nanosec).nanoseconds());
          this->imageCallback(img, depth);

        } catch (const std::runtime_error & e) {
          RCLCPP_WARN(get_logger(), "%s", e.what());
        }
      }
    });
#endif

  RCLCPP_INFO(get_logger(), "Ready.");
}

void NodeRgbdAlignment::cameraCallback(sensor_msgs::msg::CameraInfo::ConstSharedPtr msg)
{
  auto camera = vslam_ros::convert(*msg);

  if (camera->fx() > 0.0 && camera->cx() > 0.0) {
    _subCamInfo.reset();
    _camera = camera;
    _directIcp = std::make_shared<DirectIcp>(_camera, _paramsIcp);

    _periodicTimer =
      create_wall_timer(std::chrono::seconds(1), [this, msg]() { this->timerCallback(); });

    _cameraFrameId = msg->header.frame_id;
    RCLCPP_INFO(
      get_logger(), "Valid camera calibration received: %s \n. Node Ready.",
      camera->toString().c_str());

  } else {
    RCLCPP_ERROR(
      get_logger(), "Invalid camera calibration received: %s \n.", camera->toString().c_str());
  }
  triggerReplayer();
}

void NodeRgbdAlignment::imageCallback(
  sensor_msgs::msg::Image::ConstSharedPtr msgImg, sensor_msgs::msg::Image::ConstSharedPtr msgDepth)
{
  if (!_camera) {
    RCLCPP_WARN(get_logger(), "Camera parameters are not available yet");
    triggerReplayer();
    return;
  }

  try {
    TIMED_SCOPE(timerF, "processFrame");

    namespace enc = sensor_msgs::image_encodings;
    auto cv_ptr = cv_bridge::toCvShare(msgImg);
    if (enc::isColor(msgImg->encoding)) {
      cv_ptr = cv_bridge::cvtColor(cv_ptr, "mono8");
    }
    Timestamp t;
    convert(msgImg->header.stamp, t);
    cv::Mat depth = cv_bridge::toCvShare(msgDepth)->image;
    if (depth.type() == CV_16UC1) {
      depth = evaluation::tum::convertDepthMat(depth, 0.001);
    }
    _motion = _directIcp->computeEgomotion(cv_ptr->image, depth, _motion);

    _pose = _motion * _pose;
    _trajectory.append(t, _pose);

    publish(msgImg->header.stamp);

    _fNo++;
  } catch (const std::runtime_error & e) {
    RCLCPP_WARN(get_logger(), "%s", e.what());
  }
  triggerReplayer();
}

void NodeRgbdAlignment::timerCallback()
{
  if (!_tfAvailable) {
    lookupTf();
  }
  RCLCPP_INFO(
    get_logger(),
    format("Egomotion: {} m {}°", _motion.translation().transpose(), _motion.totalRotationDegrees())
      .c_str());
}

void NodeRgbdAlignment::lookupTf()
{
  try {
    _world2origin =
      _tfBuffer->lookupTransform(_fixedFrameId, _cameraFrameId.substr(1), tf2::TimePointZero);
    _tfAvailable = true;

  } catch (tf2::TransformException & ex) {
    RCLCPP_WARN(get_logger(), "%s", ex.what());
  }
}
void NodeRgbdAlignment::triggerReplayer()
{
  if (!get_parameter("replayMode").as_bool()) {
    return;
  }
  while (!_cliReplayer->wait_for_service(1s)) {
    if (!rclcpp::ok()) {
      throw std::runtime_error("Interrupted while waiting for the service. Exiting.");
    }
    RCLCPP_INFO(get_logger(), "Replayer Service not available, waiting again...");
  }
  auto request = std::make_shared<std_srvs::srv::SetBool::Request>();
  request->data = true;
  using ServiceResponseFuture = rclcpp::Client<std_srvs::srv::SetBool>::SharedFuture;
  auto response_received_callback = [this](ServiceResponseFuture future) {
    if (!future.get()->success) {
      RCLCPP_WARN(get_logger(), "Last replayer signal result was not valid.");
    }
  };
  _cliReplayer->async_send_request(request, response_received_callback);
}

void NodeRgbdAlignment::publish(const rclcpp::Time & t)
{
  // tf from origin (odom) to another fixed frame (world)
  geometry_msgs::msg::TransformStamped tfOrigin = _world2origin;
  tfOrigin.header.stamp = t;
  tfOrigin.header.frame_id = _fixedFrameId;
  tfOrigin.child_frame_id = _frameId;

  // Send current camera pose as estimate for pose of optical frame
  geometry_msgs::msg::TransformStamped tf;
  tf.header.stamp = t;
  tf.header.frame_id = _frameId;
  tf.child_frame_id = "camera";  //camera name?
  vslam_ros::convert(_pose.SE3().inverse(), tf);
  _pubTf->sendTransform({tf, tfOrigin});

  // Send pose, twist and path in optical frame
  nav_msgs::msg::Odometry odom;
  odom.header.stamp = t;
  odom.header.frame_id = _frameId;
  vslam_ros::convert(_pose.inverse(), odom.pose);
  vslam_ros::convert(_motion.inverse(), odom.twist);
  _pubOdom->publish(odom);

  nav_msgs::msg::Path path;
  path.header = odom.header;
  vslam_ros::convert(_trajectory.inverse(), path);
  _pubPath->publish(path);
}

}  // namespace vslam_ros

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(vslam_ros::NodeRgbdAlignment)
