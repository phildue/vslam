
#include "DirectIcpOverlay.h"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "utils/visuals.h"
namespace vslam
{
void DirectIcpOverlay::update(const Entry & e)
{
  auto img0 = colorizedRgbd(e.f0.I(), e.f0.Z());
  auto img1 = colorizedRgbd(e.f1.I(), e.f1.Z());
  cv::Mat overlay;
  cv::vconcat(std::vector<cv::Mat>({img0, img1}), overlay);
  cv::imshow("Overlay", overlay);
  cv::waitKey(1);
}
}  // namespace vslam