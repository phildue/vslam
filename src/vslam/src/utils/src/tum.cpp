#include <fmt/chrono.h>
#include <fmt/core.h>

#include <experimental/filesystem>
#include <filesystem>
#include <fstream>
#include <memory>
using fmt::format;
using fmt::print;
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

DataLoader::DataLoader(const std::string & datasetRoot, const std::string & sequenceId)
: _datasetRoot(datasetRoot),
  _sequenceId(sequenceId),
  _datasetPath(format("{}/{}/{}", datasetRoot, sequenceId, sequenceId)),
  _cam(tum::Camera())
{
  tum::readAssocTextfile(_datasetPath + "/assoc.txt", _imgFilenames, _depthFilenames, _timestamps);
  _trajectoryGt = utils::loadTrajectory(_datasetPath + "/groundtruth.txt", true);
}
Frame::UnPtr DataLoader::loadFrame(std::uint64_t fNo) const
{
  // tum depth format: https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats
  return std::make_unique<Frame>(
    utils::loadImage(_datasetPath + "/" + _imgFilenames.at(fNo)),
    utils::loadDepth(_datasetPath + "/" + _depthFilenames.at(fNo)) / 5000.0, _cam,
    _timestamps.at(fNo));
}

}  // namespace pd::vslam::tum
