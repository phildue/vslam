#include "core/types.h"
#include "log.h"
INITIALIZE_EASYLOGGINGPP

namespace vslam::log
{
void initialize(const std::string & folder)
{
  el::Loggers::reconfigureAllLoggers(
    el::ConfigurationType::Filename, format("{}/vslam.log", folder));
}
}  // namespace vslam::log