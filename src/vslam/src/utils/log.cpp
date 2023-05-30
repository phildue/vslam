#include "core/types.h"
#include "log.h"
INITIALIZE_EASYLOGGINGPP
#include <filesystem>
namespace fs = std::filesystem;
namespace vslam::log
{
void initialize(const std::string & folder)
{
  fs::remove_all(folder);
  fs::create_directories(folder);
  el::Loggers::reconfigureAllLoggers(
    el::ConfigurationType::Filename, format("{}/vslam.log", folder));
}
}  // namespace vslam::log