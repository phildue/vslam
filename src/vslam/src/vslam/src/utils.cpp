#include "utils.h"

namespace vslam::time
{
std::chrono::time_point<std::chrono::high_resolution_clock> to_time_point(Timestamp t)
{
  auto epoch = std::chrono::time_point<std::chrono::high_resolution_clock>();
  auto duration = std::chrono::nanoseconds(t);
  return epoch + duration;
}
}  // namespace vslam::time