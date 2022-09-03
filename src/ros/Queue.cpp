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

#include "vslam_ros/Queue.h"

#include "rclcpp/rclcpp.hpp"
namespace vslam_ros
{
int Queue::size() const { return std::min<int>(_images.size(), _depths.size()); }
void Queue::pushImage(sensor_msgs::msg::Image::ConstSharedPtr img)
{
  std::lock_guard<std::mutex> g(_mutex);

  if (_images.size() >= _queueSize) {
    popClosestImg();
  }

  _images[rclcpp::Time(img->header.stamp.sec, img->header.stamp.nanosec).nanoseconds()] = img;
}
void Queue::pushDepth(sensor_msgs::msg::Image::ConstSharedPtr depth)
{
  std::lock_guard<std::mutex> g(_mutex);
  if (_depths.size() >= _queueSize) {
    popClosestDepth();
  }
  _depths[rclcpp::Time(depth->header.stamp.sec, depth->header.stamp.nanosec).nanoseconds()] = depth;
}
sensor_msgs::msg::Image::ConstSharedPtr Queue::popClosestImg(std::uint64_t t)
{
  std::lock_guard<std::mutex> g(_mutex);
  if (_images.empty()) {
    throw std::runtime_error("Queue empty!");
  }
  if (t <= 0) {
    auto msg = _images.begin()->second;
    _images.erase(_images.begin());
    return msg;
  } else {
    int minDiff = std::numeric_limits<int>::max();
    sensor_msgs::msg::Image::ConstSharedPtr closestMsg = nullptr;
    std::uint64_t closestT = 0U;
    for (const auto & t_msg : _images) {
      int diff = std::abs(static_cast<int>(t_msg.first - t));
      if (diff < minDiff) {
        minDiff = diff;
        closestMsg = t_msg.second;
        closestT = t_msg.first;
      }
    }
    if (closestMsg == nullptr) {
      throw std::runtime_error(
        "Did not find image which is close enough to: " + std::to_string(t) +
        " closest: " + std::to_string(minDiff));
    }
    _images.erase(closestT);
    return closestMsg;
  }
}
sensor_msgs::msg::Image::ConstSharedPtr Queue::popClosestDepth(std::uint64_t t)
{
  std::lock_guard<std::mutex> g(_mutex);
  if (_depths.empty()) {
    throw std::runtime_error("Queue empty!");
  }
  if (t <= 0) {
    auto msg = _depths.begin()->second;
    _depths.erase(_depths.begin());
    return msg;
  } else {
    int minDiff = std::numeric_limits<int>::max();
    sensor_msgs::msg::Image::ConstSharedPtr closestMsg = nullptr;
    std::uint64_t closestT = 0U;
    for (const auto & t_msg : _depths) {
      int diff = std::abs(static_cast<int>(t_msg.first - t));
      if (diff < minDiff) {
        minDiff = diff;
        closestMsg = t_msg.second;
        closestT = t_msg.first;
      }
    }
    if (minDiff > _maxDiff) {
      throw std::runtime_error(
        "Did not find depth which is close enough to: " + std::to_string(t) +
        " closest: " + std::to_string(minDiff));
    }
    _depths.erase(closestT);

    return closestMsg;
  }
}

}  // namespace vslam_ros
