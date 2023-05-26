#ifndef VSLAM_VISUALS_H__
#define VSLAM_VISUALS_H__

#include "types.h"

namespace vslam
{
cv::Mat colorizedDepth(const cv::Mat & depth, double zMax = -1);
cv::Mat blend(const cv::Mat & mat1, const cv::Mat & mat2, double weight1);
cv::Mat colorizedRgbd(const cv::Mat & intensity, const cv::Mat & depth, double zMax = -1);
}  // namespace vslam
#endif  // VSLAM_VISUALS_H__
