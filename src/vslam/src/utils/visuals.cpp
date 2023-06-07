#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "visuals.h"

namespace vslam
{
cv::Mat colorizedDepth(const cv::Mat & depth, double zMax)
{
  double Min, zMax_ = zMax;
  if (zMax < 0) {
    cv::minMaxLoc(depth, &Min, &zMax_);
  }

  cv::Mat depthNorm, depthBgr;
  depth.convertTo(depthBgr, CV_8UC3, 255.0 / zMax_);
  cv::applyColorMap(depthBgr, depthBgr, cv::COLORMAP_JET);
  return depthBgr;
}

cv::Mat blend(const cv::Mat & mat1, const cv::Mat & mat2, double weight1)
{
  cv::Mat joint;
  cv::Mat weightsI(cv::Size(mat1.cols, mat1.rows), CV_32FC1, weight1);
  cv::Mat weightsZ(cv::Size(mat1.cols, mat1.rows), CV_32FC1, 1.0 - weight1);
  cv::blendLinear(mat1, mat2, weightsI, weightsZ, joint);
  return joint;
}

cv::Mat colorizedRgbd(const cv::Mat & intensity, const cv::Mat & depth, double zMax)
{
  cv::Mat depthColor = colorizedDepth(depth, zMax);
  cv::Mat intensityColor;
  cv::cvtColor(intensity, intensityColor, cv::COLOR_GRAY2BGR);
  return blend(intensityColor, depthColor, 0.7);
}
}  // namespace vslam