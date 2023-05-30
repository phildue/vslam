#include "utils.h"

namespace vslam
{
void runPerformanceLogParserpy(const std::string & file)
{
  const int ret =
    system(format(
             "python3 /home/ros/vslam_ros/src/vslam/script/vslampy/plot/parse_performance_log.py "
             "--file {}",
             file)
             .c_str());
  if (ret != 0) {
    throw std::runtime_error("Running evaluation script failed!");
  }
}
}  // namespace vslam

namespace vslam::time
{
std::chrono::time_point<std::chrono::high_resolution_clock> to_time_point(Timestamp t)
{
  auto epoch = std::chrono::time_point<std::chrono::high_resolution_clock>();
  auto duration = std::chrono::nanoseconds(t);
  return epoch + duration;
}
}  // namespace vslam::time