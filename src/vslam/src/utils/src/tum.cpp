#include <experimental/filesystem>
#include <filesystem>
#include <fstream>
#include <memory>

#include "tum.h"
#include "utils/utils.h"
namespace fs = std::experimental::filesystem;

namespace pd::vslam::tum
{
void readAssocTextfile(
  std::string filename, std::vector<std::string> & inputRGBPaths,
  std::vector<std::string> & inputDepthPaths, std::vector<Timestamp> & timestamps)
{
  if (!fs::exists(filename)) {
    throw pd::Exception("Could not find file [" + filename + "]");
  }
  std::string line;
  std::ifstream in_stream(filename.c_str());
  if (!in_stream.is_open()) {
    std::runtime_error("Could not open file at: " + filename);
  }

  while (!in_stream.eof()) {
    std::getline(in_stream, line);
    std::stringstream ss(line);
    std::string buf;
    int c = 0;
    while (ss >> buf) {
      c++;
      if (c == 1) {
        timestamps.push_back(static_cast<Timestamp>(std::stod(ss.str()) * 1e9));
      } else if (c == 2) {
        inputDepthPaths.push_back(buf);
      } else if (c == 4) {
        inputRGBPaths.push_back(buf);
      }
    }
  }
  in_stream.close();
}

Camera::ShPtr Camera() { return std::make_shared<pd::vslam::Camera>(525.0, 525.0, 319.5, 239.5); }

}  // namespace pd::vslam::tum
