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

void create(const std::string & name)
{
  el::Loggers::getLogger(name);
  el::Configurations defaultConf;
  defaultConf.setToDefault();
  defaultConf.set(el::Level::Debug, el::ConfigurationType::Format, "%datetime %level %msg");
  el::Loggers::reconfigureLogger(name, defaultConf);
}
}  // namespace vslam::log