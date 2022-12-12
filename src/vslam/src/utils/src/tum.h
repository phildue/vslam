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
    const std::string & path =
      "/mnt/dataset/tum_rgbd/rgbd_dataset_freiburg2_desk/rgbd_dataset_freiburg2_desk");

  Frame::UnPtr loadFrame(std::uint64_t fNo) const;
  size_t nFrames() const { return _timestamps.size(); }
  Camera::ConstShPtr cam() const { return _cam; }
  Trajectory::ConstShPtr trajectoryGt() const { return _trajectoryGt; }
  std::string datasetPath() const { return _datasetPath; }

private:
  std::string _datasetPath;
  Camera::ShPtr _cam;
  Trajectory::ShPtr _trajectoryGt;
  std::vector<std::string> _imgFilenames, _depthFilenames;
  std::vector<Timestamp> _timestamps;
};

}  // namespace pd::vslam::tum

#endif