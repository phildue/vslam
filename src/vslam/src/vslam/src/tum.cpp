
#include <experimental/filesystem>
#include <filesystem>
#include <fstream>
#include <memory>
#include <opencv2/highgui.hpp>

#include "tum.h"
namespace fs = std::experimental::filesystem;

namespace vslam::tum
{
vslam::Camera::ShPtr Camera()
{
  return std::make_shared<vslam::Camera>(525.0, 525.0, 319.5, 239.5, 640, 480);
}

Trajectory::UnPtr loadTrajectory(const std::string & path, bool invertPoses)
{
  if (!fs::exists(fs::path(path))) {
    throw std::runtime_error(format("Could not find [{}]", path));
  }
  std::ifstream gtFile;
  gtFile.open(path);

  if (!gtFile.is_open()) {
    throw std::runtime_error("Could not open file at: " + path);
  }

  //TODO shouldn't this better be objects or unique ptrs
  std::map<Timestamp, Pose::ConstShPtr> poses;

  std::string line;
  while (getline(gtFile, line)) {
    std::vector<std::string> elements;
    std::string s;
    std::istringstream lines(line);
    while (getline(lines, s, ' ')) {
      elements.push_back(s);
    }

    if (elements[0] == "#") {
      continue;
    }  //Skip comments

    //Pose
    Eigen::Vector3d trans;
    trans << std::stod(elements[1]), std::stod(elements[2]), std::stod(elements[3]);
    Eigen::Quaterniond q(
      std::stod(elements[7]), std::stod(elements[4]), std::stod(elements[5]),
      std::stod(elements[6]));
    auto se3 = invertPoses ? SE3d(q, trans) : SE3d(q, trans).inverse();

    //Covariance
    Eigen::Matrix<double, 6, 6> cov = Eigen::Matrix<double, 6, 6>::Identity();
    if (elements.size() >= 8 + 36) {
      for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
          cov(i, j) = std::stod(elements[8 + i * 6 + j]);
        }
      }
    }

    //Timestamp
    std::vector<std::string> tElements;
    std::string st;
    std::istringstream tLine(elements[0]);
    while (getline(tLine, st, '.')) {
      tElements.push_back(st);
    }
    auto sec = std::stoull(tElements[0]);
    auto nanosec = std::stoull(tElements[1]) * std::pow(10, 9 - tElements[1].size());

    poses.insert({sec * 1e9 + nanosec, std::make_shared<Pose>(se3, cov)});
  }
  return std::make_unique<Trajectory>(poses);
}

void writeTrajectory(const Trajectory & traj, const std::string & path, bool writeCovariance)
{
  if (!fs::is_directory(fs::path(path).parent_path())) {
    fs::create_directories(fs::path(path).parent_path());
  }
  std::fstream algoFile;
  algoFile.open(path, std::ios_base::out);
  algoFile << "# Algorithm Trajectory\n";
  algoFile << "# file: " << path << "\n";
  algoFile << "# timestamp tx ty tz qx qy qz qw\n";
  if (!algoFile.is_open()) {
    std::runtime_error("Could not open file at: " + path);
  }

  for (const auto & pose : traj.poses()) {
    std::string ts = format("{}", pose.first);
    ts = format("{}.{}", ts.substr(0, 10), ts.substr(10));

    ts = ts.substr(0, ts.size() - 3);

    const auto t = pose.second->pose().translation();
    const auto q = pose.second->pose().unit_quaternion();
    algoFile << ts << " " << t.x() << " " << t.y() << " " << t.z() << " " << q.x() << " " << q.y()
             << " " << q.z() << " " << q.w();
    if (writeCovariance) {
      for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
          algoFile << " " << pose.second->cov()(i, j);
        }
      }
    }

    algoFile << "\n";
  }
}

void runEvaluateRPEpy(const std::string & pathAlgo, const std::string & pathGt)
{
  const int ret =
    system(format(
             "python3 /home/ros/vslam_ros/src/vslam/script/vslampy/evaluation/_tum/evaluate_rpe.py "
             "--verbose "
             "--fixed_delta "
             "{} {}",
             pathGt, pathAlgo)
             .c_str());
  if (ret != 0) {
    throw std::runtime_error("Running evaluation script failed!");
  }
}

DataLoader::DataLoader(const std::string & datasetRoot, const std::string & sequenceId)
: _datasetRoot(datasetRoot),
  _sequenceId(sequenceId),
  _datasetPath(format("{}/{}/{}", datasetRoot, sequenceId, sequenceId)),
  _cam(tum::Camera()),
  _pathGt(_datasetPath + "/groundtruth.txt")
{
  readAssocTextfile(_datasetPath + "/assoc.txt");
  _trajectoryGt = loadTrajectory(_pathGt, true);
}
/*
Frame::UnPtr DataLoader::loadFrame(std::uint64_t fNo) const
{
  // tum depth format: https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats
  return std::make_unique<Frame>(
    utils::loadImage(_datasetPath + "/" + _imgFilenames.at(fNo)),
    utils::loadDepth(_datasetPath + "/" + _depthFilenames.at(fNo)) / 5000.0, _cam,
    _timestamps.at(fNo));
}
*/
cv::Mat DataLoader::loadDepth(std::uint64_t fNo) const
{
  // tum depth format: https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats
  const std::string path = format("{}/{}", datasetPath(), pathsDepth()[fNo]);
  cv::Mat depth_ = cv::imread(path, cv::IMREAD_ANYDEPTH);
  if (depth_.empty()) {
    throw std::runtime_error(format("Could not load depth from [{}]", path));
  }
  if (depth_.type() != CV_16U) {
    throw std::runtime_error(format("Depth image loaded incorrectly from [{}].", path));
  }

  cv::Mat depth(cv::Size(depth_.cols, depth_.rows), CV_32FC1);
  for (int u = 0; u < depth_.cols; u++) {
    for (int v = 0; v < depth_.rows; v++) {
      const ushort d = depth_.at<ushort>(v, u);
      depth.at<float>(v, u) =
        0.0002f * static_cast<float>(d > 0 ? d : std::numeric_limits<ushort>::quiet_NaN());
    }
  }

  return depth;
}
cv::Mat DataLoader::loadIntensity(std::uint64_t fNo) const
{
  const std::string path = format("{}/{}", datasetPath(), pathsImage()[fNo]);
  cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);

  if (img.empty()) {
    throw std::runtime_error(format("Could not load image from [{}]", path));
  }
  return img;
}

void DataLoader::readAssocTextfile(std::string filename)
{
  if (!fs::exists(filename)) {
    throw std::runtime_error("Could not find file [" + filename + "]");
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
      if (c == 3) {
        buf.erase(std::remove(buf.begin(), buf.end(), '.'), buf.end());
        buf.erase(std::remove(buf.begin(), buf.end(), ' '), buf.end());
        const long td = std::stol(format("{}000", buf));
        _timestamps.push_back(td);
      } else if (c == 2) {
        _depthFilenames.push_back(buf);
      } else if (c == 4) {
        _imgFilenames.push_back(buf);
      }
    }
  }
  in_stream.close();
}

}  // namespace vslam::tum
