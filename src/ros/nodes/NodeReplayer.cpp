#include <filesystem>

#include "NodeReplayer.h"
namespace fs = std::filesystem;
using namespace pd;
using namespace pd::vslam;

namespace vslam_ros
{
NodeReplayer::NodeReplayer(const rclcpp::NodeOptions & options)
: rclcpp::Node("Replayer", options),
  _running(true),
  _srvReady(create_service<std_srvs::srv::SetBool>(
    "set_ready",
    std::bind(&NodeReplayer::srvSetReady, this, std::placeholders::_1, std::placeholders::_2))),
  _nodeReady(true)
{
  declare_parameter(
    "bag_file",
    "/media/data/dataset/rgbd_dataset_freiburg1_desk2/rgbd_dataset_freiburg1_desk2.db3");
  declare_parameter("period", 0.05);
  declare_parameter("timeout", 10);
  declare_parameter("sync_topic", "/camera/depth/image");
  _period = get_parameter("period").as_double();
  rosbag2_storage::StorageOptions storageOptions;
  storageOptions.uri = get_parameter("bag_file").as_string();
  storageOptions.storage_id = "sqlite3";
  rosbag2_cpp::ConverterOptions converterOptions;
  RCLCPP_INFO(get_logger(), "Opening: %s", get_parameter("bag_file").as_string().c_str());

  if (!fs::exists(storageOptions.uri)) {
    throw std::runtime_error("Did not find bag file: " + storageOptions.uri);
  }
  _reader = std::make_unique<rosbag2_cpp::readers::SequentialReader>();
  _reader->open(storageOptions, converterOptions);
  _bagName = get_parameter("bag_file").as_string();
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

  _thread = std::thread([&]() { play(); });
}

void NodeReplayer::publish(rosbag2_storage::SerializedBagMessageSharedPtr msg)
{
  //std::cout << fNo << "/" << _nFrames << " t: " << msg->time_stamp << " topic: " << msg->topic_name << std::endl;
  _msgCtr[msg->topic_name]++;

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

void NodeReplayer::play()
{
  rcl_time_point_value_t lastTs = 0U;
  rcl_time_point_value_t firstTs = 0U;
  while (_reader->has_next() && _running) {
    auto msg = _reader->read_next();
    if (msg->topic_name == _syncTopic) {
      if (_msgCtr[_syncTopic] % (_nMessages[_syncTopic] / 100) == 0) {
        RCLCPP_INFO(
          get_logger(), "Replayed [%d]%s, [%ld/%ld] msgs, [%.3f/%.3f]s of [%s]",
          static_cast<int>(
            static_cast<float>(_msgCtr[_syncTopic]) / static_cast<float>(_nMessages[_syncTopic]) *
            100),
          "%", _msgCtr[_syncTopic], _nMessages[_syncTopic],
          static_cast<double>(static_cast<double>(msg->time_stamp - firstTs) / 1e9), _duration,
          _bagName.substr(_bagName.find_last_of('/') + 1).c_str());
      }
      if (!_nodeReady) {
        //RCLCPP_INFO(get_logger(), "Waiting for reply from node..");

        std::unique_lock<std::mutex> lk(_mutex);
        if (!_cond.wait_for(lk, std::chrono::seconds(get_parameter("timeout").as_int()), [&]() {
              return _nodeReady;
            })) {
          RCLCPP_WARN(get_logger(), "Timed out during waiting for node to be ready. Continuing..");
        } else {
          //RCLCPP_INFO(get_logger(), "Confirmation received. Continuing..");
        }
      }
      _nodeReady = false;
    }
    publish(msg);
    if (lastTs > 0U) {
      std::this_thread::sleep_for(std::chrono::nanoseconds(msg->time_stamp - lastTs));
    } else {
      firstTs = msg->time_stamp;
    }
    lastTs = msg->time_stamp;
  }
  rclcpp::shutdown();
}

void NodeReplayer::srvSetReady(
  const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
  std::shared_ptr<std_srvs::srv::SetBool::Response> response)
{
  //RCLCPP_INFO(get_logger(), "Received setSrvReady()");
  {
    std::unique_lock<std::mutex> lk(_mutex);
    _nodeReady = request->data;
  }
  //RCLCPP_INFO(get_logger(), "Notifying");

  _cond.notify_all();
  response->success = true;
  //RCLCPP_INFO(get_logger(), "Returning request..");
}
void NodeReplayer::publishNext() { publish(_reader->read_next()); }

NodeReplayer::~NodeReplayer()
{
  _running = false;
  if (_thread.joinable()) {
    _thread.join();
  }
  _reader->close();
}
}  // namespace vslam_ros
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(vslam_ros::NodeReplayer)