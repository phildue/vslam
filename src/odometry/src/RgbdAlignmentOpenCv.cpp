
#include <Eigen/Dense>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/core/eigen.hpp>

#include <opencv2/rgbd.hpp>

#include "RgbdAlignmentOpenCv.h"

#define LOG_ODOM(level) CLOG(level,"odometry")
namespace pd::vslam{


        RgbdAlignmentOpenCv::RgbdAlignmentOpenCv()
        {
                Log::get("odometry",ODOMETRY_CFG_DIR"/log/odometry.conf");
        }


        PoseWithCovariance::UnPtr RgbdAlignmentOpenCv::align(FrameRgbd::ConstShPtr from, FrameRgbd::ConstShPtr to) const
        {
                cv::Mat camMat,srcImage,srcDepth,dstImage,dstDepth,guess;
                cv::eigen2cv(from->camera()->K(),camMat);
                auto relativePose = algorithm::computeRelativeTransform(from->pose().pose(),to->pose().pose());
                cv::eigen2cv(relativePose.matrix(),guess);
                
                LOG_ODOM(INFO) << "Aligning from: \n\t"<< from->pose().pose().log().transpose() 
                << "\nto:\t" << to->pose().pose().log().transpose()
                << "\nguess:\t" << relativePose.log().transpose();

                cv::rgbd::RgbdOdometry estimator(camMat,cv::rgbd::Odometry::DEFAULT_MIN_DEPTH(), cv::rgbd::Odometry::DEFAULT_MAX_DEPTH(),
                  cv::rgbd::Odometry::DEFAULT_MAX_DEPTH_DIFF(), std::vector<int>({100,100,100}));

                cv::eigen2cv(from->intensity(),srcImage);
                cv::eigen2cv(from->depth(),srcDepth);
                cv::eigen2cv(to->intensity(),dstImage);
                cv::eigen2cv(to->depth(),dstDepth);
                srcImage.convertTo(srcImage,CV_8UC1);
                srcDepth.convertTo(srcDepth,CV_32FC1);
                dstImage.convertTo(dstImage,CV_8UC1);
                dstDepth.convertTo(dstDepth,CV_32FC1);
                auto odomFrameFrom = cv::rgbd::OdometryFrame::create(srcImage,srcDepth);
                auto odomFrameTo = cv::rgbd::OdometryFrame::create(dstImage,dstDepth);

                cv::Mat RtCv;
                MatXd Rt;
                bool success = estimator.compute(odomFrameFrom,odomFrameTo,RtCv,guess);
                cv::cv2eigen(RtCv,Rt);

                if(success)
                {
                        SE3d se3(Rt);
                        LOG_ODOM(DEBUG) << "Successfully aligned: Rt=\n\t" << se3.log().transpose();
                        return std::make_unique<PoseWithCovariance>( se3 * from->pose().pose(), MatXd::Identity(6,6) );
                }else{
                        LOG_ODOM(DEBUG) << "Alignment failed.";
                        return std::make_unique<PoseWithCovariance>( from->pose() );
                }
        }


}