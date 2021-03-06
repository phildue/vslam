#include <opencv4/opencv2/calib3d.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/core/utility.hpp>
#include <Eigen/Dense>
#include <opencv4/opencv2/core/eigen.hpp>
#include "Frame.h"
#define USE_OPENCV
namespace pd::vslam{

        FrameRgb::FrameRgb(const Image& intensity, Camera::ConstShPtr cam, size_t nLevels, const Timestamp& t, const PoseWithCovariance& pose)
        :_t(t),
        _pose(pose){

                _intensity.resize(nLevels);
                _dIx.resize(nLevels);
                _dIy.resize(nLevels);
                _cam.resize(nLevels);
                const double s = 0.5;

                #ifdef USE_OPENCV
                //TODO replace using custom implementation
                cv::Mat mat;
                cv::eigen2cv(intensity,mat);
                std::vector<cv::Mat> mats;
                cv::buildPyramid(mat,mats,nLevels-1);
                for(size_t i = 0; i < mats.size(); i++){
                        cv::cv2eigen(mats[i],_intensity[i]);
                        cv::Mat mati_blur;
                        cv::GaussianBlur(mats[i],mati_blur,cv::Size(3,3),0,0,cv::BORDER_DEFAULT);
                        cv::Mat dIdx,dIdy;
                        cv::Sobel(mati_blur,dIdx,CV_16S,1,0,3);
                        cv::Sobel(mati_blur,dIdy,CV_16S,0,1,3);
                        cv::cv2eigen(dIdx,_dIx[i]);
                        cv::cv2eigen(dIdy,_dIy[i]);
                        _cam[i] = Camera::resize(cam,std::pow(s,i));

                }
                #else
                //TODO make based on scales
                Mat<double,5,5> gaussianKernel;
                gaussianKernel << 1,4,6,4,1,
                                4,16,24,16,4,
                                6,24,36,24,6,
                                4,16,24,16,4,
                                1,4,6,4,1;
                for(size_t i = 0; i < nLevels; i++)
                {
                        if(i == 0)
                        {
                                _intensity[i] = intensity;
                                _cam[i] = cam;
                                
                        }else{
                                Image imgBlur = algorithm::conv2d(_intensity[i-1].cast<double>(),gaussianKernel).cast<uint8_t>();
                                //TODO move padding to separate function
                                imgBlur.col(0) = imgBlur.col(2);
                                imgBlur.col(1) = imgBlur.col(2);
                                imgBlur.col(imgBlur.cols()-2) = imgBlur.col(imgBlur.cols()-3);
                                imgBlur.col(imgBlur.cols()-1) = imgBlur.col(imgBlur.cols()-3);
                                imgBlur.row(0) = imgBlur.row(2);
                                imgBlur.row(1) = imgBlur.row(2);
                                imgBlur.row(imgBlur.rows()-2) = imgBlur.row(imgBlur.rows()-3);
                                imgBlur.row(imgBlur.rows()-1) = imgBlur.row(imgBlur.rows()-3);

                                _intensity[i] = algorithm::resize(imgBlur,s);
                                _cam[i] = Camera::resize(_cam[i-1],s);

                        } 
                        _dIx[i] = algorithm::conv2d(_intensity[i].cast<double>(),Kernel2d<double>::sobelX()).cast<int>();
                        _dIy[i] = algorithm::conv2d(_intensity[i].cast<double>(),Kernel2d<double>::sobelY()).cast<int>();
                                
                }
                #endif
        }
        Eigen::Vector2d FrameRgb::camera2image(const Eigen::Vector3d &pCamera, size_t level) const
        {
                return _cam.at(level)->camera2image(pCamera);
        }
        Eigen::Vector3d FrameRgb::image2camera(const Eigen::Vector2d &pImage, double depth, size_t level) const
        {
                return _cam.at(level)->image2camera(pImage,depth);
        }
        Eigen::Vector2d FrameRgb::world2image(const Eigen::Vector3d &pWorld, size_t level) const
        {
                return  camera2image(_pose.pose() * pWorld,level);
        }
        Eigen::Vector3d FrameRgb::image2world(const Eigen::Vector2d &pImage, double depth, size_t level) const
        {
                return _pose.pose().inverse() * image2camera(pImage,depth,level);

        }
        


        FrameRgbd::FrameRgbd(const Image& intensity,const MatXd& depth, Camera::ConstShPtr cam, size_t nLevels, const Timestamp& t, const PoseWithCovariance& pose)
        :FrameRgb(intensity,cam,nLevels,t,pose)
        {
                auto depth2pcl = [](const DepthMap& d, Camera::ConstShPtr c)
                {
                        std::vector<Vec3d> pcl(d.rows()*d.cols());
                        for(int v = 0; v < d.rows(); v++)
                        {
                                for(int u = 0; u < d.cols(); u++)
                                {
                                        if ( std::isfinite(d(v,u)) && d(v,u) > 0.0 )
                                        {
                                                pcl[v * d.cols() + u] = c->image2camera({u+0.5,v+0.5},d(v,u));
                                        }else{
                                                pcl[v * d.cols() + u] = Eigen::Vector3d::Zero();
                                        }
                                }
                        }
                        return pcl;
                };
                _depth.resize(nLevels);
                _pcl.resize(nLevels);
                const double s = 0.5;
                #ifdef USE_OPENCV_DEPTH
                cv::Mat mat;
                cv::eigen2cv(depth,mat);
                std::vector<cv::Mat> mats;
                cv::buildPyramid(mat,mats,nLevels-1);
                for(size_t i = 0; i < mats.size(); i++)
                {
                        cv::cv2eigen(mats[i],_depth[i]);
                        _pcl[i] = depth2pcl(_depth[i],camera(i));

                }
                #else
                for(size_t i = 0; i < nLevels; i++)
                {
                        if(i == 0)
                        {
                                _depth[i] = depth;
                                _pcl[i] = depth2pcl(depth,cam);
                                
                        }else{
                                DepthMap depthBlur = algorithm::medianBlur<double>(_depth[i-1],3,3,[](double v){ return v <= 0.0;});
                                _depth[i] = algorithm::resize(depthBlur,s);
                                _pcl[i] = depth2pcl(_depth[i],camera(i));

                        } 
                                
                }
                #endif
        }
        std::vector<Vec3d> FrameRgbd::pcl(size_t level, bool removeInvalid) const
        {
                if(removeInvalid)
                {
                        std::vector<Vec3d> pcl;
                        pcl.reserve(_pcl.at(level).size());
                        std::copy_if(_pcl.at(level).begin(),_pcl.at(level).end(),std::back_inserter(pcl),[](auto p){return p.z() > 0 && std::isfinite(p.z());});
                        return pcl;
                }else{
                        return _pcl.at(level);
                }
        }
        std::vector<Vec3d> FrameRgbd::pclWorld(size_t level, bool removeInvalid) const
        {
                auto points = pcl(level,removeInvalid);
                std::transform(points.begin(),points.end(),points.begin(),[&](auto p){return pose().pose().inverse() * p;});
                return points;
        }



} // namespace pd::vision


