#ifndef VSLAM_TIME_H__
#define VSLAM_TIME_H__
#include <chrono>

#include "types.h"
namespace vslam
{
void runPerformanceLogParserpy(const std::string & file);

}

namespace vslam::time
{
std::chrono::time_point<std::chrono::high_resolution_clock> to_time_point(Timestamp t);
}

#endif