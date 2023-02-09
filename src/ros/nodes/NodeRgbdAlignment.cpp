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

using namespace pd::vslam;
using namespace std::chrono_literals;
#include <fmt/core.h>
#include <fmt/ostream.h>

using fmt::format;
using fmt::print;
template <typename T>
struct fmt::formatter<T, std::enable_if_t<std::is_base_of_v<Eigen::DenseBase<T>, T>, char>>
: ostream_formatter
{
};
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
  declare_parameter("frame_alignment.method", "rgbd");
  declare_parameter("frame_alignment.trackKeyFrame", false);
  declare_parameter("frame_alignment.includeKeyFrame", false);
  declare_parameter("frame_alignment.includePrior", false);
  declare_parameter("frame_alignment.initOnPrior", false);
  declare_parameter(
    "frame_alignment.features.min_gradients", std::vector<double>({10.0, 10.0, 10.0, 10.0}));
  declare_parameter("frame_alignment.features.min_depth", 0.0);
  declare_parameter("frame_alignment.features.max_depth", 4.0);
  declare_parameter("frame_alignment.features.min_depth_diff", 0.05);
  declare_parameter("frame_alignment.features.max_depth_diff", 1.0);
  declare_parameter(
    "frame_alignment.features.max_points_part", std::vector<double>({1.0, 1.0, 1.0, 0.25}));
  declare_parameter("frame_alignment.pyramid.levels", std::vector<double>({0.25, 0.5, 1.0}));
  declare_parameter("frame_alignment.solver.max_iterations", 100);
  declare_parameter("frame_alignment.solver.min_step_size", 1e-7);
  declare_parameter("frame_alignment.solver.min_gradient", 1e-7);
  declare_parameter("frame_alignment.solver.max_increase", 1e-7);
  declare_parameter("frame_alignment.solver.min_reduction", 0.0);
  declare_parameter("frame_alignment.loss.function", "None");
  declare_parameter("frame_alignment.loss.huber.c", 10.0);
  declare_parameter("frame_alignment.loss.tdistribution.v", 5.0);
  declare_parameter("keyframe_selection.method", "idx");
  declare_parameter("keyframe_selection.idx.period", 5);
  declare_parameter("keyframe_selection.custom.min_visible_points", 50);
  declare_parameter("keyframe_selection.custom.max_translation", 0.2);
  declare_parameter("keyframe_selection.custom.max_rotation", 1.0);
  declare_parameter("prediction.model", "NoMotion");
  declare_parameter("prediction.kalman.process_noise.pose.translation", 1e-15);
  declare_parameter("prediction.kalman.process_noise.pose.rotation", 1e-15);
  declare_parameter("prediction.kalman.process_noise.velocity.translation", 1e-15);
  declare_parameter("prediction.kalman.process_noise.velocity.rotation", 1e-15);
  declare_parameter("prediction.kalman.initial_uncertainty.pose.translation", 1e-15);
  declare_parameter("prediction.kalman.initial_uncertainty.pose.rotation", 1e-15);
  declare_parameter("prediction.kalman.initial_uncertainty.velocity.translation", 1e-15);
  declare_parameter("prediction.kalman.initial_uncertainty.velocity.rotation", 1e-15);
  declare_parameter("map.n_keyframes", 7);
  declare_parameter("map.n_frames", 7);

  RCLCPP_INFO(get_logger(), "Setting up..");

  if (get_parameter("replayMode").as_bool()) {
    _cliReplayer = create_client<std_srvs::srv::SetBool>("set_ready");
  }

  _map = std::make_shared<Map>(
    get_parameter("map.n_keyframes").as_int(), get_parameter("map.n_frames").as_int());

  _includeKeyFrame = get_parameter("frame_alignment.includeKeyFrame").as_bool();
  if (_trackKeyFrame && _includeKeyFrame) {
    throw pd::Exception("Should be either trackKeyFrame OR includeKeyFrame");
  }

  least_squares::Loss::ShPtr loss =
    std::make_shared<least_squares::QuadraticLoss>(std::make_shared<least_squares::Scaler>());
  auto paramLoss = get_parameter("frame_alignment.loss.function").as_string();
  if (paramLoss == "Tukey") {
    loss =
      std::make_shared<least_squares::TukeyLoss>(std::make_shared<least_squares::MedianScaler>());
  } else if (paramLoss == "Huber") {
    loss = std::make_shared<least_squares::HuberLoss>(
      std::make_shared<least_squares::MedianScaler>(),
      get_parameter("frame_alignment.loss.huber.c").as_double());
  } else if (paramLoss == "tdistribution") {
    loss = std::make_shared<least_squares::LossTDistribution>(
      std::make_shared<least_squares::ScalerTDistribution>(
        get_parameter("frame_alignment.loss.tdistribution.v").as_double()),
      get_parameter("frame_alignment.loss.tdistribution.v").as_double());
  } else {
    RCLCPP_WARN(get_logger(), "Unknown loss selected. Assuming None");
  }
  auto solver = std::make_shared<least_squares::GaussNewton>(
    get_parameter("frame_alignment.solver.min_step_size").as_double(),
    get_parameter("frame_alignment.solver.max_iterations").as_int(),
    get_parameter("frame_alignment.solver.min_gradient").as_double(),
    get_parameter("frame_alignment.solver.min_reduction").as_double(),
    get_parameter("frame_alignment.solver.max_increase").as_double());

  if (get_parameter("frame_alignment.method").as_string() == "icp") {
    _rgbdAlignment = std::make_shared<RgbdAlignmentIcp>(
      solver, loss, get_parameter("frame_alignment.includePrior").as_bool(),
      get_parameter("frame_alignment.initOnPrior").as_bool(),
      get_parameter("frame_alignment.features.min_gradients").as_double_array(),
      get_parameter("frame_alignment.features.min_depth").as_double(),
      get_parameter("frame_alignment.features.max_depth").as_double(),
      get_parameter("frame_alignment.features.min_depth_diff").as_double(),
      get_parameter("frame_alignment.features.max_depth_diff").as_double(),
      get_parameter("frame_alignment.features.max_points_part").as_double_array());
  } else if (get_parameter("frame_alignment.method").as_string() == "rgbd") {
    _rgbdAlignment = std::make_shared<RgbdAlignment>(
      solver, loss, get_parameter("frame_alignment.includePrior").as_bool(),
      get_parameter("frame_alignment.initOnPrior").as_bool(),
      get_parameter("frame_alignment.features.min_gradients").as_double_array(),
      get_parameter("frame_alignment.features.min_depth").as_double(),
      get_parameter("frame_alignment.features.max_depth").as_double(),
      get_parameter("frame_alignment.features.min_depth_diff").as_double(),
      get_parameter("frame_alignment.features.max_depth_diff").as_double(),
      get_parameter("frame_alignment.features.max_points_part").as_double_array());
  } else if (get_parameter("frame_alignment.method").as_string() == "rgb") {
    _rgbdAlignment = std::make_shared<RgbdAlignmentRgb>(
      solver, loss, get_parameter("frame_alignment.includePrior").as_bool(),
      get_parameter("frame_alignment.initOnPrior").as_bool(),
      get_parameter("frame_alignment.features.min_gradients").as_double_array(),
      get_parameter("frame_alignment.features.min_depth").as_double(),
      get_parameter("frame_alignment.features.max_depth").as_double(),
      get_parameter("frame_alignment.features.min_depth_diff").as_double(),
      get_parameter("frame_alignment.features.max_depth_diff").as_double(),
      get_parameter("frame_alignment.features.max_points_part").as_double_array());
  } else {
    throw pd::Exception(format(
      "Unknown frame_alignment method {} available are: [rgbd, depth, rgb]",
      get_parameter("prediction.model").as_string().c_str()));
  }

  if (get_parameter("prediction.model").as_string() == "NoMotion") {
    _motionModel = std::make_shared<MotionModelNoMotion>();
  } else if (get_parameter("prediction.model").as_string() == "ConstantMotion") {
    auto covarianceVector = get_parameter("prediction.constant_speed.covariance").as_double_array();
    Eigen::Map<Vec6d> covDiag(covarianceVector.data());
    _motionModel = std::make_shared<MotionModelConstantSpeed>(covDiag.asDiagonal());
  } else if (get_parameter("prediction.model").as_string() == "Kalman") {
    const double processVarRot = std::pow(
      get_parameter("prediction.kalman.process_noise.pose.rotation").as_double() / 180.0 * M_PI, 2);
    const double processVarTrans =
      std::pow(get_parameter("prediction.kalman.process_noise.pose.translation").as_double(), 2);
    const double processVarRotVel = std::pow(
      (get_parameter("prediction.kalman.process_noise.velocity.rotation").as_double() / 180.0 *
       M_PI),
      2);
    const double processVarTransVel = std::pow(
      get_parameter("prediction.kalman.process_noise.velocity.translation").as_double(), 2);
    Vec12d processVarDiag;
    processVarDiag << processVarTrans, processVarTrans, processVarTrans, processVarRot,
      processVarRot, processVarRot, processVarTransVel, processVarTransVel, processVarTransVel,
      processVarRotVel, processVarRotVel, processVarRotVel;

    const double stateVarRot = std::pow(
      get_parameter("prediction.kalman.initial_uncertainty.pose.rotation").as_double() / 180.0 *
        M_PI,
      2);
    const double stateVarTrans = std::pow(
      get_parameter("prediction.kalman.initial_uncertainty.pose.translation").as_double(), 2);
    const double stateVarRotVel = std::pow(
      (get_parameter("prediction.kalman.initial_uncertainty.velocity.rotation").as_double() /
       180.0 * M_PI),
      2);
    const double stateVarTransVel = std::pow(
      get_parameter("prediction.kalman.initial_uncertainty.velocity.translation").as_double(), 2);
    Vec12d stateVarDiag;
    stateVarDiag << stateVarTrans, stateVarTrans, stateVarTrans, stateVarRot, stateVarRot,
      stateVarRot, stateVarTransVel, stateVarTransVel, stateVarTransVel, stateVarRotVel,
      stateVarRotVel, stateVarRotVel;

    _motionModel = std::make_shared<MotionModelConstantSpeedKalman>(
      processVarDiag.asDiagonal(), stateVarDiag.asDiagonal());
  } else {
    throw pd::Exception(format(
      "Unknown prediction model {} available are: [NoMotion, ConstantMotion, Kalman]",
      get_parameter("prediction.model").as_string().c_str()));
  }

  if (get_parameter("keyframe_selection.method").as_string() == "idx") {
    _keyFrameSelection = std::make_shared<KeyFrameSelectionIdx>(
      get_parameter("keyframe_selection.idx.period").as_int());
  } else if (get_parameter("keyframe_selection.method").as_string() == "custom") {
    _keyFrameSelection = std::make_shared<KeyFrameSelectionCustom>(
      _map, get_parameter("keyframe_selection.custom.min_visible_points").as_int(),
      get_parameter("keyframe_selection.custom.max_translation").as_double(),
      get_parameter("keyframe_selection.custom.max_rotation").as_double() / 180.0 * M_PI);
  } else if (get_parameter("keyframe_selection.method").as_string() == "never") {
    _keyFrameSelection = std::make_shared<KeyFrameSelectionNever>();
  } else {
    throw pd::Exception("Unknown method for key frame selection.");
  }
  declare_parameter("bundle_adjustment.huber_constant", 1.43);
  declare_parameter("bundle_adjustment.max_iterations", 50);

  _ba = std::make_shared<mapping::BundleAdjustment>(
    get_parameter("bundle_adjustment.max_iterations").as_int(),
    get_parameter("bundle_adjustment.huber_constant").as_double());

  _matcher = std::make_shared<Matcher>(Matcher::reprojectionHamming, 5.0, 0.8);
  _tracking = std::make_shared<FeatureTracking>(_matcher);
  _trackKeyFrame = get_parameter("frame_alignment.trackKeyFrame").as_bool();

  // _cameraName = this->declare_parameter<std::string>("camera","/camera/rgb");
  //sync.registerDropCallback(std::bind(&StereoAlignmentROS::dropCallback, this,std::placeholders::_1, std::placeholders::_2));
  LOG_IMG("KeyFrames");
  declare_parameter("log.config_dir", "/share/cfg/log/");
  declare_parameter("log.root_dir", "/tmp/log/vslam");
  LogImage::rootFolder() = get_parameter("log.root_dir").as_string();
  for (const auto & name : Log::registeredLogs()) {
    RCLCPP_INFO(get_logger(), "Found logger: %s", name.c_str());
    Log::get(name)->configure(get_parameter("log.config_dir").as_string() + "/" + name + ".conf");
  }
  for (const auto & name : Log::registeredLogsImage()) {
    RCLCPP_INFO(get_logger(), "Found image logger: %s", name.c_str());

    declare_parameter("log.image." + name + ".show", false);
    declare_parameter("log.image." + name + ".block", false);
    declare_parameter("log.image." + name + ".save", false);
    declare_parameter("log.image." + name + ".rate", 1);
    LOG_IMG(name)->set(
      get_parameter("log.image." + name + ".show").as_bool(),
      get_parameter("log.image." + name + ".block").as_bool(),
      get_parameter("log.image." + name + ".save").as_bool(),
      get_parameter("log.image." + name + ".rate").as_int());
    RCLCPP_INFO(get_logger(), "Found image logger:\n%s", LOG_IMG(name)->toString().c_str());
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

  if (camera->fx() > 0.0 && camera->principalPoint().norm() > 0.0) {
    _subCamInfo.reset();
    _camera = camera;

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
    TIMED_FUNC(timerF);
    Frame::ShPtr frame = createFrame(msgImg, msgDepth);

    frame->set(_motionModel->predictPose(frame->t()));

    Pose pose;
    if (_trackKeyFrame && _map->lastKf()) {
      pose = _rgbdAlignment->align(_map->lastKf(), frame);

    } else if (
      _includeKeyFrame && _map->lastKf() && _map->lastFrame() &&
      _map->lastKf() != _map->lastFrame()) {
      pose = _rgbdAlignment->align({_map->lastFrame(), _map->lastKf()}, frame);

    } else if (_map->lastFrame()) {
      pose = _rgbdAlignment->align(_map->lastFrame(), frame);
    }
    //TODO
    //_motionModel->update(pose, frame->t());

    frame->set(pose);

    _keyFrameSelection->update(frame);

    _map->insert(frame, _keyFrameSelection->isKeyFrame() || _map->keyFrames().empty());

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
  if (_map->lastFrame()) {
    auto x = _map->lastFrame()->pose().inverse().twist();
    auto cx = _map->lastFrame()->pose().inverse().twistCov();

    RCLCPP_INFO(
      get_logger(), "Pose: %.3f, %.3f, %.3f, %.3f, %.3f, %.3f | Cov | %.3f\n", x(0), x(1), x(2),
      x(3), x(4), x(5), cx.norm());
  }

  RCLCPP_INFO(
    get_logger(), format(
                    "Translational Speed: {} [m/s] | Cov | {}",
                    _motionModel->speed().SE3().translation().transpose() * 1e9,
                    _motionModel->speed().twistCov().block(0, 0, 3, 3).diagonal().transpose() * 1e9)
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

Frame::UnPtr NodeRgbdAlignment::createFrame(
  sensor_msgs::msg::Image::ConstSharedPtr msgImg,
  sensor_msgs::msg::Image::ConstSharedPtr msgDepth) const
{
  //TODO(me) try to avoid copying here
  namespace enc = sensor_msgs::image_encodings;
  auto cv_ptr = cv_bridge::toCvShare(msgImg);
  if (enc::isColor(msgImg->encoding)) {
    cv_ptr = cv_bridge::cvtColor(cv_ptr, "mono8");
  }
  Image img;
  cv::cv2eigen(cv_ptr->image, img);

  Eigen::MatrixXd depth;
  cv::cv2eigen(cv_bridge::toCvShare(msgDepth)->image, depth);

  depth = depth.array().isNaN().select(0, depth);
  const Timestamp t =
    rclcpp::Time(msgImg->header.stamp.sec, msgImg->header.stamp.nanosec).nanoseconds();

  auto f = std::make_unique<Frame>(img, depth, _camera, t);
  f->computePyramid(get_parameter("frame_alignment.pyramid.levels").as_double_array().size());
  f->computeIntensityDerivatives();
  f->computePcl();
  f->computeNormals();
  return f;
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
  vslam_ros::convert(_map->lastFrame()->pose().SE3().inverse(), tf);
  _pubTf->sendTransform({tf, tfOrigin});

  // Send pose, twist and path in optical frame
  nav_msgs::msg::Odometry odom;
  odom.header.stamp = t;
  odom.header.frame_id = _frameId;
  vslam_ros::convert(_map->lastFrame()->pose().inverse(), odom.pose);
  vslam_ros::convert(_motionModel->speed().inverse(), odom.twist);
  _pubOdom->publish(odom);

  geometry_msgs::msg::PoseStamped poseStamped;
  poseStamped.header = odom.header;
  poseStamped.pose = vslam_ros::convert(_map->lastFrame()->pose().pose().inverse());
  _path.header = odom.header;
  _path.poses.push_back(poseStamped);
  _pubPath->publish(_path);

  if (!_map->points().empty()) {
    sensor_msgs::msg::PointCloud2 pcl;
    vslam_ros::convert(Map::ConstShPtr(_map)->points(), pcl);
    pcl.header = odom.header;
    _pubPclMap->publish(pcl);
  }

  if (!_map->keyFrames().empty()) {
    nav_msgs::msg::Path poseGraph;
    poseGraph.header = odom.header;
    for (auto kf : _map->keyFrames()) {
      geometry_msgs::msg::PoseStamped kfPoseStamped;
      kfPoseStamped.header = odom.header;
      vslam_ros::convert(kf->t(), kfPoseStamped.header.stamp);
      kfPoseStamped.pose = vslam_ros::convert(kf->pose().pose().inverse());
      poseGraph.poses.push_back(kfPoseStamped);
    }
    _pubPoseGraph->publish(poseGraph);
  }
}

}  // namespace vslam_ros

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(vslam_ros::NodeRgbdAlignment)
