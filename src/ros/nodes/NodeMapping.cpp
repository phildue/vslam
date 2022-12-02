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

#include "NodeMapping.h"
#include "vslam_ros/converters.h"
using namespace pd::vslam;
using namespace std::chrono_literals;

namespace vslam_ros
{
NodeMapping::NodeMapping(const rclcpp::NodeOptions & options)
: rclcpp::Node("NodeMapping", options),
  _includeKeyFrame(false),
  _camInfoReceived(false),
  _tfAvailable(false),
  _fNo(0),
  _frameId("odom"),
  _fixedFrameId("world"),
  _pubOdom(create_publisher<nav_msgs::msg::Odometry>("/odom", 10)),
  _pubPath(create_publisher<nav_msgs::msg::Path>("/path", 10)),
  _pubPoseGraph(create_publisher<nav_msgs::msg::Path>("/path/pose_graph", 10)),
  _pubTf(std::make_shared<tf2_ros::TransformBroadcaster>(this)),
  _pubPclMap(create_publisher<sensor_msgs::msg::PointCloud2>("/pcl/map", 10)),
  _tfBuffer(std::make_unique<tf2_ros::Buffer>(get_clock())),
  _subCamInfo(create_subscription<sensor_msgs::msg::CameraInfo>(
    "/camera/rgb/camera_info", 10,
    std::bind(&NodeMapping::cameraCallback, this, std::placeholders::_1))),
  _subImage(create_subscription<sensor_msgs::msg::Image>(
    "/camera/rgb/image_color", 10,
    std::bind(&NodeMapping::imageCallback, this, std::placeholders::_1))),
  _subDepth(create_subscription<sensor_msgs::msg::Image>(
    "/camera/depth/image", 10,
    std::bind(&NodeMapping::depthCallback, this, std::placeholders::_1))),
  _subTf(std::make_shared<tf2_ros::TransformListener>(*_tfBuffer)),
  _queue(std::make_shared<vslam_ros::Queue>(10, 0.20 * 1e9))
{
  /*
  _cliReplayer = create_client<std_srvs::srv::SetBool>(
    "set_ready", rmw_qos_profile_services_default,
    create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive));*/
  _cliReplayer = create_client<std_srvs::srv::SetBool>("set_ready");

  declare_parameter("frame.base_link_id", _fixedFrameId);
  declare_parameter("frame.frame_id", _frameId);
  declare_parameter("odometry.method", "rgbd");
  declare_parameter("odometry.trackKeyFrame", false);
  declare_parameter("odometry.includeKeyFrame", false);
  declare_parameter("odometry.rgbd.includePrior", false);
  declare_parameter("odometry.rgbd.features.min_gradient", 1);
  declare_parameter("odometry.rgbd.pyramid.levels", std::vector<double>({0.25, 0.5, 1.0}));
  declare_parameter("odometry.rgbd.solver.max_iterations", 100);
  declare_parameter("odometry.rgbd.solver.min_step_size", 1e-7);
  declare_parameter("odometry.rgbd.loss.function", "None");
  declare_parameter("odometry.rgbd.loss.huber.c", 10.0);
  declare_parameter("odometry.rgbd.loss.tdistribution.v", 5.0);
  declare_parameter("keyframe_selection.method", "idx");
  declare_parameter("keyframe_selection.idx.period", 5);
  declare_parameter("keyframe_selection.custom.min_visible_points", 50);
  declare_parameter("keyframe_selection.custom.max_translation", 0.2);
  declare_parameter("prediction.model", "NoMotion");
  declare_parameter("map.n_keyframes", 7);
  declare_parameter("map.n_frames", 7);

  Log::_blockLevel = Level::Unknown;
  Log::_showLevel = Level::Unknown;

  RCLCPP_INFO(get_logger(), "Setting up..");
  _map = std::make_shared<Map>(
    get_parameter("map.n_keyframes").as_int(), get_parameter("map.n_frames").as_int());
  _trackKeyFrame = get_parameter("odometry.trackKeyFrame").as_bool();
  _includeKeyFrame = get_parameter("odometry.includeKeyFrame").as_bool();
  if (_trackKeyFrame && _includeKeyFrame) {
    throw pd::Exception("Should be either trackKeyFrame OR includeKeyFrame");
  }
  if (get_parameter("odometry.method").as_string() == "rgbd") {
    least_squares::Loss::ShPtr loss = nullptr;
    least_squares::Scaler::ShPtr scaler;
    auto paramLoss = get_parameter("odometry.rgbd.loss.function").as_string();
    if (paramLoss == "Tukey") {
      loss =
        std::make_shared<least_squares::TukeyLoss>(std::make_shared<least_squares::MedianScaler>());
    } else if (paramLoss == "Huber") {
      loss = std::make_shared<least_squares::HuberLoss>(
        std::make_shared<least_squares::MedianScaler>(),
        get_parameter("odometry.rgbd.loss.huber.c").as_double());
    } else if (paramLoss == "tdistribution") {
      loss = std::make_shared<least_squares::LossTDistribution>(
        std::make_shared<least_squares::ScalerTDistribution>(
          get_parameter("odometry.rgbd.loss.tdistribution.v").as_double()),
        get_parameter("odometry.rgbd.loss.tdistribution.v").as_double());
    }

    auto solver = std::make_shared<least_squares::GaussNewton>(
      get_parameter("odometry.rgbd.solver.min_step_size").as_double(),
      get_parameter("odometry.rgbd.solver.max_iterations").as_int());

    _rgbdAlignment = std::make_shared<SE3Alignment>(
      get_parameter("odometry.rgbd.features.min_gradient").as_int(), solver, loss,
      get_parameter("odometry.rgbd.includePrior").as_bool());
  } else if (get_parameter("odometry.method").as_string() == "rgbd_opencv") {
    _rgbdAlignment = std::make_shared<RgbdAlignmentOpenCv>();
  } else {
    RCLCPP_ERROR(get_logger(), "Unknown odometry method available are: [rgbd, rgbd_opencv]");
  }

  if (get_parameter("prediction.model").as_string() == "NoMotion") {
    _motionModel = std::make_shared<MotionModelNoMotion>();
  } else if (get_parameter("prediction.model").as_string() == "ConstantMotion") {
    _motionModel = std::make_shared<MotionModelConstantSpeed>();
  } else if (get_parameter("prediction.model").as_string() == "Kalman") {
    Matd<6, 6> covConstantVel;
    covConstantVel << 4.43799e-36, 8.26448e-37, -1.67565e-36, 5.60168e-38, -1.48752e-36,
      -7.90401e-37, 8.26448e-37, 4.20937e-36, 1.35359e-36, 1.45756e-36, -4.93781e-38, -7.6944e-37,
      -1.67565e-36, 1.35359e-36, 4.58383e-36, 7.77025e-37, 1.06878e-36, 2.86771e-39, 5.60168e-38,
      1.45756e-36, 7.77025e-37, 9.08575e-37, -5.95007e-39, 3.22266e-38, -1.48752e-36, -4.93781e-38,
      1.06878e-36, -5.95007e-39, 8.99202e-37, 7.02648e-38, -7.90401e-37, -7.6944e-37, 2.86771e-39,
      3.22266e-38, 7.02648e-38, 8.21283e-37;
    Matd<12, 12> covProcess = Matd<12, 12>::Identity();
    covProcess.block(6, 6, 6, 6) = covConstantVel;
    _motionModel = std::make_shared<MotionModelConstantSpeedKalman>(covProcess);
  } else {
    RCLCPP_ERROR(
      get_logger(), "Unknown odometry method %s available are: [NoMotion, ConstantMotion, Kalman]",
      get_parameter("prediction.model").as_string().c_str());
  }

  if (get_parameter("keyframe_selection.method").as_string() == "idx") {
    _keyFrameSelection = std::make_shared<KeyFrameSelectionIdx>(
      get_parameter("keyframe_selection.idx.period").as_int());
  } else if (get_parameter("keyframe_selection.method").as_string() == "custom") {
    _keyFrameSelection = std::make_shared<KeyFrameSelectionCustom>(
      _map, get_parameter("keyframe_selection.custom.min_visible_points").as_int(),
      get_parameter("keyframe_selection.custom.max_translation").as_double());
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
    LOG_IMG(name)->set(
      get_parameter("log.image." + name + ".show").as_bool(),
      get_parameter("log.image." + name + ".block").as_bool(),
      get_parameter("log.image." + name + ".save").as_bool());
    RCLCPP_INFO(get_logger(), "Found image logger:\n%s", LOG_IMG(name)->toString().c_str());
  }
  for (const auto & name : Log::registeredLogsPlot()) {
    declare_parameter("log.plot." + name + ".show", false);
    declare_parameter("log.plot." + name + ".block", false);
    LOG_PLT(name)->show() = get_parameter("log.plot." + name + ".show").as_bool();
    LOG_PLT(name)->block() = get_parameter("log.plot." + name + ".block").as_bool();
    RCLCPP_INFO(get_logger(), "Found plot logger:\n%s", LOG_PLT(name)->toString().c_str());
  }
  RCLCPP_INFO(get_logger(), "Ready.");
}

bool NodeMapping::ready() { return _queue->size() >= 1; }

void NodeMapping::processFrame(
  sensor_msgs::msg::Image::ConstSharedPtr msgImg, sensor_msgs::msg::Image::ConstSharedPtr msgDepth)
{
  TIMED_FUNC(timerF);

  try {
    auto frame = createFrame(msgImg, msgDepth);

    auto frameRef = _map->lastKf();

    if (frameRef) {
      frame->set(*_motionModel->predictPose(frame->t()));
      auto pose = _includeKeyFrame && _map->lastFrame()
                    ? _rgbdAlignment->align({_map->lastFrame(), frameRef}, frame)
                    : _rgbdAlignment->align(frameRef, frame);
      frame->set(*pose);
      _motionModel->update(frameRef, frame);
      //frame->set(*_motionModel->pose());
    }

    _keyFrameSelection->update(frame);

    _map->insert(frame, _keyFrameSelection->isKeyFrame());
    /*
    auto outBa = _keyFrameSelection->isKeyFrame()
                   ? _ba->optimize(Map::ConstShPtr(_map)->keyFrames())
                   : _ba->optimize({frame}, Map::ConstShPtr(_map)->keyFrames());*/
    if (_keyFrameSelection->isKeyFrame()) {
      auto points = _tracking->track(frame, _map->keyFrames());
      _map->insert(points);
      if (_map->nKeyFrames() >= 2) {
        LOG_IMG("KeyFrames") << std::make_shared<OverlayFeatureDisplacement>(
          Map::ConstShPtr(_map)->keyFrames());

        auto outBa = _ba->optimize(
          {Map::ConstShPtr(_map)->keyFrame(0)},
          Map::ConstShPtr(_map)->keyFrames(1, _map->nKeyFrames()));

        _map->updatePointsAndPoses(outBa->poses, outBa->positions);
        LOG_IMG("KeyFrames") << std::make_shared<OverlayFeatureDisplacement>(
          Map::ConstShPtr(_map)->keyFrames());
      }
    }

    publish(msgImg, frame);

    _fNo++;
  } catch (const std::runtime_error & e) {
    RCLCPP_WARN(this->get_logger(), "%s", e.what());
  }
}

void NodeMapping::lookupTf(sensor_msgs::msg::Image::ConstSharedPtr msgImg)
{
  try {
    _world2origin = _tfBuffer->lookupTransform(
      _fixedFrameId, msgImg->header.frame_id.substr(1), tf2::TimePointZero);
    _tfAvailable = true;

  } catch (tf2::TransformException & ex) {
    RCLCPP_INFO(get_logger(), "%s", ex.what());
  }
}
void NodeMapping::signalReplayer()
{
  if (get_parameter("use_sim_time").as_bool()) {
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
}  // namespace vslam_ros

Frame::ShPtr NodeMapping::createFrame(
  sensor_msgs::msg::Image::ConstSharedPtr msgImg,
  sensor_msgs::msg::Image::ConstSharedPtr msgDepth) const
{
  auto cvImage = cv_bridge::toCvShare(msgImg);
  cv::Mat mat = cvImage->image;
  cv::cvtColor(mat, mat, cv::COLOR_RGB2GRAY);
  Image img;
  cv::cv2eigen(mat, img);
  auto cvDepth = cv_bridge::toCvShare(msgDepth);

  Eigen::MatrixXd depth;
  cv::cv2eigen(cvDepth->image, depth);
  depth = depth.array().isNaN().select(0, depth);
  const Timestamp t =
    rclcpp::Time(msgImg->header.stamp.sec, msgImg->header.stamp.nanosec).nanoseconds();

  auto f = std::make_shared<Frame>(img, depth, _camera, t);
  f->computePyramid(get_parameter("odometry.rgbd.pyramid.levels").as_double_array().size());
  f->computeDerivatives();
  f->computePcl();
  return f;
}
void NodeMapping::publish(sensor_msgs::msg::Image::ConstSharedPtr msgImg, Frame::ConstShPtr frame)
{
  if (!_tfAvailable) {
    lookupTf(msgImg);
    return;
  }

  auto x = frame->pose().pose().inverse().log();
  RCLCPP_INFO(
    get_logger(), "Pose: %.3f, %.3f, %.3f, %.3f, %.3f, %.3f", x(0), x(1), x(2), x(3), x(4), x(5));

  // Send the transformation from fixed frame to origin of optical frame
  // TODO(unknown): possibly only needs to be sent once
  geometry_msgs::msg::TransformStamped tfOrigin = _world2origin;
  tfOrigin.header.stamp = msgImg->header.stamp;
  tfOrigin.header.frame_id = _fixedFrameId;
  tfOrigin.child_frame_id = _frameId;
  _pubTf->sendTransform(tfOrigin);

  // Send current camera pose as estimate for pose of optical frame
  geometry_msgs::msg::TransformStamped tf;
  tf.header.stamp = msgImg->header.stamp;
  tf.header.frame_id = _frameId;
  tf.child_frame_id = "camera";  //camera name?
  vslam_ros::convert(frame->pose().pose().inverse(), tf);
  _pubTf->sendTransform(tf);

  // Send pose, twist and path in optical frame
  nav_msgs::msg::Odometry odom;
  odom.header = msgImg->header;
  odom.header.frame_id = _frameId;
  vslam_ros::convert(frame->pose().inverse(), odom.pose);
  vslam_ros::convert(_motionModel->speed()->inverse(), odom.twist);
  _pubOdom->publish(odom);

  geometry_msgs::msg::PoseStamped poseStamped;
  poseStamped.header = odom.header;
  poseStamped.pose = vslam_ros::convert(frame->pose().pose().inverse());
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

void NodeMapping::depthCallback(sensor_msgs::msg::Image::ConstSharedPtr msgDepth)
{
  _queue->pushDepth(msgDepth);

  if (ready()) {
    try {
      auto img = _queue->popClosestImg();
      processFrame(
        img, _queue->popClosestDepth(
               rclcpp::Time(img->header.stamp.sec, img->header.stamp.nanosec).nanoseconds()));
    } catch (const std::runtime_error & e) {
      RCLCPP_WARN(get_logger(), "%s", e.what());
    }
  }
  signalReplayer();
}

void NodeMapping::imageCallback(sensor_msgs::msg::Image::ConstSharedPtr msgImg)
{
  _queue->pushImage(msgImg);
}
void NodeMapping::dropCallback(
  sensor_msgs::msg::Image::ConstSharedPtr msgImg, sensor_msgs::msg::Image::ConstSharedPtr msgDepth)
{
  RCLCPP_INFO(get_logger(), "Message dropped.");
  if (msgImg) {
    const auto ts = msgImg->header.stamp.nanosec;
    RCLCPP_INFO(get_logger(), "Image: %10.0f", (double)ts);
  }
  if (msgDepth) {
    const auto ts = msgDepth->header.stamp.nanosec;
    RCLCPP_INFO(get_logger(), "Depth: %10.0f", (double)ts);
  }
}

void NodeMapping::cameraCallback(sensor_msgs::msg::CameraInfo::ConstSharedPtr msg)
{
  if (_camInfoReceived) {
    return;
  }

  _camera = vslam_ros::convert(*msg);
  _camInfoReceived = true;

  RCLCPP_INFO(get_logger(), "Camera calibration received. Node ready.");
}

}  // namespace vslam_ros

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(vslam_ros::NodeMapping)
