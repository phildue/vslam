#ifndef VSLAM_TIME_H__
#define VSLAM_TIME_H__
#include <chrono>

#include "core/types.h"
namespace vslam
{
}

namespace vslam::time
{
std::chrono::time_point<std::chrono::high_resolution_clock> to_time_point(Timestamp t);
}

#endif