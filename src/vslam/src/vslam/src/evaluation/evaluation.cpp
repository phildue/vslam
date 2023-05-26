#include "core/types.h"
#include "evaluation.h"
#define SCRIPT_DIR "/home/ros/vslam_ros/src/vslam/script/"
namespace vslam::evaluation
{
void runPerformanceLogParserpy(const std::string & file)
{
  const int ret = system(format(
                           "python3 " SCRIPT_DIR "vslampy/plot/parse_performance_log.py "
                           "--file {}",
                           file)
                           .c_str());
  if (ret != 0) {
    throw std::runtime_error("Running evaluation script failed!");
  }
}
}  // namespace vslam::evaluation