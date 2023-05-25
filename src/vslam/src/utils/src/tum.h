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

  Frame::UnPtr loadFrame(std::uint64_t fNo) const;
  size_t nFrames() const { return _timestamps.size(); }
  Camera::ConstShPtr cam() const { return _cam; }
  const std::string & pathGt() const { return _pathGt; }
  Trajectory::ConstShPtr trajectoryGt() const { return _trajectoryGt; }
  std::string datasetPath() const { return _datasetPath; }
  std::string sequenceId() const { return _sequenceId; }
  std::string datasetRoot() const { return _datasetRoot; }

  const std::vector<std::string> & pathsImage() const { return _imgFilenames; }
  const std::vector<std::string> & pathsDepth() const { return _depthFilenames; }
  const std::vector<Timestamp> & timestamps() const { return _timestamps; }

private:
  std::string _datasetRoot;
  std::string _sequenceId;
  std::string _datasetPath;
  Camera::ShPtr _cam;
  std::string _pathGt;
  Trajectory::ShPtr _trajectoryGt;
  std::vector<std::string> _imgFilenames, _depthFilenames;
  std::vector<Timestamp> _timestamps;
};

}  // namespace pd::vslam::tum

#endif