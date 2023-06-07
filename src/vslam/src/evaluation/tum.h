#ifndef VSLAM_TUM_H__
#define VSLAM_TUM_H__
#include <memory>
#include <string>
#include <vector>

#include "core/Camera.h"
#include "core/Trajectory.h"
#include "core/types.h"
namespace vslam::evaluation::tum
{
/**
   * @brief Load trajectory from file (TUM RGBD Format)
   *
   * @param path filepath
   * @return Trajectory
   */
Trajectory::UnPtr loadTrajectory(const std::string & path, bool invertPoses = false);

/**
   * @brief Write trajectory to txt file (TUM RGBD Format)
   *
   * @param traj
   * @param path
   * @param writeCovariance
   */
void writeTrajectory(
  const Trajectory & traj, const std::string & path, bool writeCovariance = false);

void computeRPE(const std::string & pathAlgo, const std::string & pathGt);

cv::Mat convertDepthMat(const cv::Mat & depth, float factor = 0.0002);

vslam::Camera::ShPtr Camera();

class DataLoader
{
public:
  typedef std::shared_ptr<DataLoader> ShPtr;
  typedef std::unique_ptr<DataLoader> UnPtr;
  typedef std::shared_ptr<const DataLoader> ConstShPtr;
  typedef std::unique_ptr<const DataLoader> ConstUnPtr;

  DataLoader(
    const std::string & datasetRoot = "/mnt/dataset/tum_rgbd/",
    const std::string & sequenceId = "rgbd_dataset_freiburg2_desk");

  //Frame::UnPtr loadFrame(std::uint64_t fNo) const;
  cv::Mat loadDepth(std::uint64_t fNo) const;
  cv::Mat loadIntensity(std::uint64_t fNo) const;

  size_t nFrames() const { return _timestamps.size(); }
  Camera::ConstShPtr cam() const { return _cam; }
  const std::string & pathGt() const { return _pathGt; }
  Trajectory::ConstShPtr trajectoryGt() const { return _trajectoryGt; }
  std::string sequencePath() const { return _sequencePath; }
  std::string extracteDataPath() const { return _extractedDataPath; }
  std::string sequenceId() const { return _sequenceId; }
  std::string datasetRoot() const { return _datasetRoot; }

  const std::vector<std::string> & pathsImage() const { return _imgFilenames; }
  const std::vector<std::string> & pathsDepth() const { return _depthFilenames; }
  const std::vector<Timestamp> & timestamps() const { return _timestamps; }

private:
  std::string _datasetRoot;
  std::string _sequenceId;
  std::string _extractedDataPath;
  std::string _sequencePath;
  Camera::ShPtr _cam;
  std::string _pathGt;
  Trajectory::ShPtr _trajectoryGt;
  std::vector<std::string> _imgFilenames, _depthFilenames;
  std::vector<Timestamp> _timestamps;
  void readAssocTextfile(std::string filename);
};

}  // namespace vslam::evaluation::tum

#endif