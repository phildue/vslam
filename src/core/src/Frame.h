#ifndef VSLAM_FRAME_H__
#define VSLAM_FRAME_H__
#include <memory>

#include "types.h"
#include "PoseWithCovariance.h"
#include "algorithm.h"
#include "Camera.h"
#include "Kernel2d.h"
namespace pd::vslam{

        
        class FrameRgb{

                public:

                typedef std::shared_ptr<FrameRgb> ShPtr;
                typedef std::shared_ptr<const FrameRgb> ConstShPtr;
                typedef std::unique_ptr<FrameRgb> UnPtr;
                typedef std::unique_ptr<const FrameRgb> ConstUnPtr;
                
                FrameRgb(const Image& intensity, Camera::ConstShPtr cam, size_t nLevels = 1, const Timestamp& t = 0U, const PoseWithCovariance& pose = {});

                const Image& intensity(size_t level = 0) const {return _intensity.at(level);}
                const MatXd& dIx(size_t level = 0) const { return _dIx.at(level);}
                const MatXd& dIy(size_t level = 0) const { return _dIy.at(level);}

                const PoseWithCovariance& pose() const {return _pose;}
                
                const Timestamp& t() const {return _t;}
                Camera::ConstShPtr camera(size_t level = 0) const { return _cam.at(level); }
                size_t width(size_t level = 0) const {return _intensity.at(level).cols();}
                size_t height(size_t level = 0) const {return _intensity.at(level).rows();}
                size_t nLevels() const { return _intensity.size();}
                Eigen::Vector2d camera2image(const Eigen::Vector3d &pCamera, size_t level = 0) const;
                Eigen::Vector3d image2camera(const Eigen::Vector2d &pImage, double depth = 1.0, size_t level = 0) const;
                Eigen::Vector2d world2image(const Eigen::Vector3d &pWorld, size_t level = 0) const;
                Eigen::Vector3d image2world(const Eigen::Vector2d &pImage, double depth = 1.0, size_t level = 0) const;
          
                void set(const PoseWithCovariance& pose){_pose = pose;}


                virtual ~FrameRgb(){};
                private:
                ImageVec _intensity;
                MatXdVec _dIx,_dIy;
                Camera::ConstShPtrVec _cam;
                Timestamp _t;
                PoseWithCovariance _pose; //<< Pf = pose * Pw
        };

        class FrameRgbd : public FrameRgb {

                public:

                typedef std::shared_ptr<FrameRgbd> ShPtr;
                typedef std::shared_ptr<const FrameRgbd> ConstShPtr;
                typedef std::unique_ptr<FrameRgbd> UnPtr;
                typedef std::unique_ptr<const FrameRgbd> ConstUnPtr;
                
                FrameRgbd(const Image& rgb, const DepthMap& depth, Camera::ConstShPtr cam, size_t nLevels = 1, const Timestamp& t = 0U, const PoseWithCovariance& pose = {});

                const DepthMap& depth(size_t level = 0) const {return _depth.at(level);}
                const Vec3d& p3d(int v, int u, size_t level = 0) const {return _pcl.at(level)[v * width(level) + u];}
                Vec3d p3dWorld(int v, int u, size_t level = 0) const {return pose().pose().inverse() * _pcl.at(level)[v * width() + u];}
                std::vector<Vec3d> pcl (size_t level = 0, bool removeInvalid = false) const;
                std::vector<Vec3d> pclWorld (size_t level = 0, bool removeInvalid = false) const;

                virtual ~FrameRgbd(){};
                private:
                DepthMapVec _depth;
                std::vector<std::vector<Vec3d>> _pcl;

        };
} // namespace pd::vision



#endif