#ifndef VSLAM_TUM_H__
#define VSLAM_TUM_H__
#include <core/core.h>

#include <string>
#include <vector>
namespace pd::vslam::tum
{
void readAssocTextfile(
  std::string filename, std::vector<std::string> & inputRGBPaths,
  std::vector<std::string> & inputDepthPaths, std::vector<Timestamp> & timestamps);

Camera::ShPtr Camera();

}  // namespace pd::vslam::tum

#endif