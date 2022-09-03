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

#ifndef VSLAM_ROS_QUEUE_H__
#define VSLAM_ROS_QUEUE_H__
#include <map>
#include <mutex>
#include <sensor_msgs/msg/image.hpp>

namespace vslam_ros
{
class Queue
{
public:
  Queue(int queueSize, int maxDiff) : _maxDiff(maxDiff), _queueSize(queueSize) {}
  int size() const;
  void pushImage(sensor_msgs::msg::Image::ConstSharedPtr img);
  void pushDepth(sensor_msgs::msg::Image::ConstSharedPtr depth);
  sensor_msgs::msg::Image::ConstSharedPtr popClosestImg(std::uint64_t t = 0U);
  sensor_msgs::msg::Image::ConstSharedPtr popClosestDepth(std::uint64_t t = 0U);

private:
  std::map<std::uint64_t, sensor_msgs::msg::Image::ConstSharedPtr> _images;
  std::map<std::uint64_t, sensor_msgs::msg::Image::ConstSharedPtr> _depths;
  mutable std::mutex _mutex;
  const int _maxDiff;
  const size_t _queueSize;
};
}  // namespace vslam_ros
#endif
