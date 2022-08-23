// Copyright 2022 Philipp.Duernay
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

#include <utils/utils.h>

#include "IterativeClosestPoint.h"
#define LOG_ODOM(level) CLOG(level, "odometry")
namespace pd::vslam
{
PoseWithCovariance::UnPtr IterativeClosestPoint::align(
  Frame::ConstShPtr from, Frame::ConstShPtr to) const
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloudFrom(new pcl::PointCloud<pcl::PointXYZ>);

  for (size_t v = 0; v < from->height(_level); v++) {
    for (size_t u = 0; u < from->width(_level); u++) {
      if (std::isfinite(from->depth(_level)(v, u)) && from->depth(_level)(v, u) > 0) {
        auto p = from->pose().pose() * from->p3d(v, u);
        cloudFrom->emplace_back(p.x(), p.y(), p.z());
      }
    }
  }
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloudTo(new pcl::PointCloud<pcl::PointXYZ>);

  for (size_t v = 0; v < to->height(_level); v++) {
    for (size_t u = 0; u < to->width(_level); u++) {
      if (std::isfinite(to->depth(_level)(v, u)) && to->depth(_level)(v, u) > 0) {
        auto p = to->pose().pose() * to->p3d(v, u);
        cloudTo->emplace_back(p.x(), p.y(), p.z());
      }
    }
  }

  pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
  icp.setMaximumIterations(_maxIterations);
  icp.setInputSource(cloudFrom);
  icp.setInputTarget(cloudTo);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloudAligned(new pcl::PointCloud<pcl::PointXYZ>);
  const auto guess =
    algorithm::computeRelativeTransform(from->pose().pose(), to->pose().pose()).inverse().matrix();
  icp.align(*cloudAligned, guess.cast<float>());

  /*
  // Visualization
  pcl::visualization::PCLVisualizer viewer ("ICP demo");
  // Create two vertically separated viewports
  int v1 (0);
  int v2 (1);
  viewer.createViewPort (0.0, 0.0, 0.5, 1.0, v1);
  viewer.createViewPort (0.5, 0.0, 1.0, 1.0, v2);

  // The color we will be using
  float bckgr_gray_level = 0.0;  // Black
  float txt_gray_lvl = 1.0 - bckgr_gray_level;

  // Original point cloud is white
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud_in_color_h (cloudFrom, (int) 255 * txt_gray_lvl, (int) 255 * txt_gray_lvl,
                                                                          (int) 255 * txt_gray_lvl);
  viewer.addPointCloud (cloudFrom, cloud_in_color_h, "cloud_in_v1", v1);
  viewer.addPointCloud (cloudFrom, cloud_in_color_h, "cloud_in_v2", v2);

  // ICP aligned point cloud is red
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud_icp_color_h (cloudTo, 180, 20, 20);
  viewer.addPointCloud (cloudTo, cloud_icp_color_h, "cloud_icp_v2", v2);

  // Adding text descriptions in each viewport
  viewer.addText ("White: Original point cloud\nGreen: Matrix transformed point cloud", 10, 15, 16, txt_gray_lvl, txt_gray_lvl, txt_gray_lvl, "icp_info_1", v1);
  viewer.addText ("White: Original point cloud\nRed: ICP aligned point cloud", 10, 15, 16, txt_gray_lvl, txt_gray_lvl, txt_gray_lvl, "icp_info_2", v2);

  // Set background color
  viewer.setBackgroundColor (bckgr_gray_level, bckgr_gray_level, bckgr_gray_level, v1);
  viewer.setBackgroundColor (bckgr_gray_level, bckgr_gray_level, bckgr_gray_level, v2);

  // Set camera position and orientation
  viewer.setCameraPosition (-3.68332, 2.94092, 5.71266, 0.289847, 0.921947, -0.256907, 0);
  viewer.setSize (1280, 1024);  // Visualiser window size

  // Register keyboard callback :

  // Display the visualiser
  while (!viewer.wasStopped ())
  {
  viewer.spinOnce ();

  }
  */
  if (icp.hasConverged()) {
    LOG_ODOM(INFO) << "ICP has converged, score is: " << icp.getFitnessScore();
    return std::make_unique<PoseWithCovariance>(
      SE3d(icp.getFinalTransformation().cast<double>()).inverse() * from->pose().pose(),
      MatXd::Identity(6, 6));
  } else {
    LOG_ODOM(ERROR) << "ICP has not converged";
    return std::make_unique<PoseWithCovariance>(to->pose());
  }
}

}  // namespace pd::vslam
