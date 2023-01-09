#include <fmt/chrono.h>
#include <fmt/core.h>
using fmt::format;
#include <filesystem>

#include "NodeReplayer.h"
namespace fs = std::filesystem;
using namespace pd;
using namespace pd::vslam;
using namespace std::chrono_literals;

namespace vslam_ros
{
NodeReplayer::NodeReplayer(const rclcpp::NodeOptions & options)
: rclcpp::Node("NodeReplayer", options),
  _srvReady(create_service<std_srvs::srv::SetBool>(
    "set_ready",
    std::bind(&NodeReplayer::serviceReadyCb, this, std::placeholders::_1, std::placeholders::_2))),
  _nodeReady(true)
{
  declare_parameter(
    "bag_file",
    "/mnt/dataset/tum_rgbd/rgbd_dataset_freiburg1_desk2/rgbd_dataset_freiburg1_desk2.db3");
  _bagName = get_parameter("bag_file").as_string();
  RCLCPP_INFO(get_logger(), "Opening: %s", _bagName.c_str());
  _reader = open(_bagName);
  _meta = _reader->get_metadata();

  declare_parameter("timeout", 1.0);
  declare_parameter("sync_topic", "/camera/depth/image");
  declare_parameter("start_random", false);
  declare_parameter("delay", 0.0);
  declare_parameter("duration", 100.0);

  RCLCPP_INFO(get_logger(), "Opened: %s", _bagName.c_str());

  _pubTf = create_publisher<tf2_msgs::msg::TFMessage>("/tf", 100);
  _pubImg = create_publisher<sensor_msgs::msg::Image>("/camera/rgb/image_color", 100);
  _pubCamInfo = create_publisher<sensor_msgs::msg::CameraInfo>("/camera/rgb/camera_info", 100);
  _pubDepth = create_publisher<sensor_msgs::msg::Image>("/camera/depth/image", 100);
  _syncTopic = get_parameter("sync_topic").as_string();
  auto meta = _reader->get_metadata();
  _duration = static_cast<double>(meta.duration.count()) / 1e9;
  RCLCPP_INFO(
    get_logger(), "Bag has length of [%.2fs] and contains [%ld] from [%ld] topics", _duration,
    meta.message_count, meta.topics_with_message_count.size());

  for (const auto & mc : _reader->get_metadata().topics_with_message_count) {
    RCLCPP_INFO(
      get_logger(), "Topic: [%s] is type [%s] has [%ld] messages", mc.topic_metadata.name.c_str(),
      mc.topic_metadata.type.c_str(), mc.message_count);
    _nMessages[mc.topic_metadata.name] = mc.message_count;
    _msgCtr[mc.topic_metadata.name] = 0U;
  }
  const auto tStartDesired = getStartingTime(meta);

  _tStart = seek(tStartDesired);
  if (_tStart == 0) {
    RCLCPP_WARN(get_logger(), "Could not start at desired t = [%ld].", tStartDesired);
    _reader->close();
    _reader = open(_bagName);
    _meta = _reader->get_metadata();
    _tStart = meta.starting_time.time_since_epoch().count();
  }
  _tEnd = getEndTime(_tStart, meta);
  _duration = static_cast<double>(_tEnd - _tStart) / 1e9;
  _tLast = _tStart;
  RCLCPP_INFO(
    get_logger(), "%s",
    format(
      "Will play from t0=[{:%Y-%m-%d %H:%M:%S}] to t1=[{:%Y-%m-%d %H:%M:%S}] for [{:.3f}]s.",
      time::to_time_point(_tStart), time::to_time_point(_tEnd), _duration)
      .c_str());
  _period = 100;
  _tWaitingForNode = 0;
  _periodicTimer = create_wall_timer(
    std::chrono::nanoseconds(_period), std::bind(&NodeReplayer::replayNext, this));
}
std::unique_ptr<rosbag2_cpp::readers::SequentialReader> NodeReplayer::open(
  const std::string & bagName) const
{
  if (!fs::exists(bagName)) {
    throw std::runtime_error("Did not find bag file: " + bagName);
  }
  rosbag2_storage::StorageOptions storageOptions;
  storageOptions.uri = bagName;
  storageOptions.storage_id = "sqlite3";

  rosbag2_cpp::ConverterOptions converterOptions;

  auto reader = std::make_unique<rosbag2_cpp::readers::SequentialReader>();
  reader->open(storageOptions, converterOptions);
  return reader;
}
void NodeReplayer::publish(rosbag2_storage::SerializedBagMessageSharedPtr msg)
{
  //std::cout << fNo << "/" << _nFrames << " t: " << msg->time_stamp << " topic: " << msg->topic_name << std::endl;

  if (msg->topic_name == "/camera/rgb/image_color") {
    auto img = std::make_shared<sensor_msgs::msg::Image>();
    rclcpp::SerializedMessage extracted_serialized_msg(*msg->serialized_data);
    rclcpp::Serialization<sensor_msgs::msg::Image> serialization;
    serialization.deserialize_message(&extracted_serialized_msg, img.get());
    _pubImg->publish(*img);
    //RCLCPP_INFO(
    //  get_logger(), "Topic [%s], replayed [%ld/%ld] messages.", msg->topic_name.c_str(),
    //  _msgCtr[msg->topic_name], _nMessages[msg->topic_name]);
  }

  if (msg->topic_name == "/camera/rgb/camera_info") {
    auto camInfo = std::make_shared<sensor_msgs::msg::CameraInfo>();
    rclcpp::SerializedMessage extracted_serialized_msg(*msg->serialized_data);
    rclcpp::Serialization<sensor_msgs::msg::CameraInfo> serialization;
    serialization.deserialize_message(&extracted_serialized_msg, camInfo.get());
    _pubCamInfo->publish(*camInfo);
  }

  if (msg->topic_name == "/camera/depth/image") {
    auto depth = std::make_shared<sensor_msgs::msg::Image>();
    rclcpp::SerializedMessage extracted_serialized_msg(*msg->serialized_data);
    rclcpp::Serialization<sensor_msgs::msg::Image> serialization;
    serialization.deserialize_message(&extracted_serialized_msg, depth.get());
    _pubDepth->publish(*depth);
    //RCLCPP_INFO(
    //  get_logger(), "Topic [%s], replayed [%ld/%ld] messages.", msg->topic_name.c_str(),
    //  _msgCtr[msg->topic_name], _nMessages[msg->topic_name]);
  }
  if (msg->topic_name == "/tf") {
    auto tf = std::make_shared<tf2_msgs::msg::TFMessage>();
    rclcpp::SerializedMessage extracted_serialized_msg(*msg->serialized_data);
    rclcpp::Serialization<tf2_msgs::msg::TFMessage> serialization;
    serialization.deserialize_message(&extracted_serialized_msg, tf.get());
    _pubTf->publish(*tf);
  }
}

void NodeReplayer::replayNext()
{
  if (!_reader->has_next() || _tLast > _tEnd) {
    RCLCPP_INFO(get_logger(), "Replay has ended. Will shutdown in 10 seconds..");
    _reader->close();
    std::this_thread::sleep_for(std::chrono::seconds(10));
    rclcpp::shutdown();
    return;
  }
  if (!_nodeReady) {
    const rcl_time_point_value_t timeout = get_parameter("timeout").as_double() * 1e9;
    _tWaitingForNode += _period;
    if (_tWaitingForNode < timeout) {
      return;
    } else {
      RCLCPP_WARN(get_logger(), "Timed out during waiting for node to be ready. Continuing..");
      _tWaitingForNode = 0;
      std::unique_lock<std::mutex> lk(_mutex);
      _nodeReady = true;
    }
  }

  auto msg = _reader->read_next();
  _msgCtr[msg->topic_name]++;

  if (msg->topic_name == _syncTopic) {
    if (_msgCtr[_syncTopic] % (_nMessages[_syncTopic] / 100) == 0) {
      RCLCPP_INFO(
        get_logger(), "Replayed [%d]%s, [%ld/%ld] msgs, [%.3f/%.3f]s of [%s] t=[%ld]",
        static_cast<int>(
          static_cast<float>(_msgCtr[_syncTopic]) / static_cast<float>(_nMessages[_syncTopic]) *
          100),
        "%", _msgCtr[_syncTopic], _nMessages[_syncTopic],
        static_cast<double>(
          static_cast<double>(msg->time_stamp - _meta.starting_time.time_since_epoch().count()) /
          1e9),
        _meta.duration.count() / 1e9, _bagName.substr(_bagName.find_last_of('/') + 1).c_str(),
        msg->time_stamp);
    }
    std::unique_lock<std::mutex> lk(_mutex);
    _nodeReady = false;
  }
  publish(msg);
  _tLast = msg->time_stamp;
}  // namespace vslam_ros

rcl_time_point_value_t NodeReplayer::seek(rcl_time_point_value_t t)
{
  if (t == _reader->get_metadata().starting_time.time_since_epoch().count()) {
    return t;
  }

  rcl_time_point_value_t tsLast = 0UL;
  while (_reader->has_next()) {
    auto msg = _reader->read_next();
    _msgCtr[msg->topic_name]++;

    if (tsLast > 0L && tsLast <= t && t <= msg->time_stamp) {
      return msg->time_stamp;
    }
    tsLast = msg->time_stamp;
  }
  return 0L;
}

void NodeReplayer::serviceReadyCb(
  const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
  std::shared_ptr<std_srvs::srv::SetBool::Response> response)
{
  {
    std::unique_lock<std::mutex> lk(_mutex);
    _nodeReady = request->data;
  }

  _cond.notify_all();
  response->success = true;
}

rcl_time_point_value_t NodeReplayer::getStartingTime(
  const rosbag2_storage::BagMetadata & meta) const
{
  if (get_parameter("start_random").as_bool()) {
    return pd::vslam::random::U(
      static_cast<uint64_t>(meta.starting_time.time_since_epoch().count()),
      static_cast<uint64_t>((meta.starting_time + meta.duration).time_since_epoch().count()));
  } else
    return meta.starting_time.time_since_epoch().count() +
           static_cast<uint64_t>(get_parameter("delay").as_double() * 1e9);
}
rcl_time_point_value_t NodeReplayer::getEndTime(
  rcl_time_point_value_t tStart, const rosbag2_storage::BagMetadata & meta) const
{
  rcl_time_point_value_t duration =
    get_parameter("duration").as_double() > 0.0
      ? static_cast<rcl_time_point_value_t>(get_parameter("duration").as_double() * 1e9)
      : meta.duration.count() - (tStart - meta.starting_time.time_since_epoch().count());
  return (tStart + duration);
}

}  // namespace vslam_ros
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(vslam_ros::NodeReplayer)