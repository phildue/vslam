#ifndef VSLAM_MOTION_PREDICTION
#define VSLAM_MOTION_PREDICTION

#include "core/core.h"
namespace pd::vslam{
class MotionPrediction{
        public:
        typedef std::shared_ptr<MotionPrediction> ShPtr;
        typedef std::unique_ptr<MotionPrediction> UnPtr;
        typedef std::shared_ptr<const MotionPrediction> ConstShPtr;
        typedef std::unique_ptr<const MotionPrediction> ConstUnPtr;

        virtual void update(PoseWithCovariance::ConstShPtr pose, Timestamp timestamp) = 0;
        virtual PoseWithCovariance::UnPtr predict(uint64_t timestamp) const = 0;
        
        static ShPtr make(const std::string& model);

};
class MotionPredictionNoMotion : public MotionPrediction{
        public:
        typedef std::shared_ptr<MotionPredictionNoMotion> ShPtr;
        typedef std::unique_ptr<MotionPredictionNoMotion> UnPtr;
        typedef std::shared_ptr<const MotionPredictionNoMotion> ConstShPtr;
        typedef std::unique_ptr<const MotionPredictionNoMotion> ConstUnPtr;
        MotionPredictionNoMotion()
        : MotionPrediction()
        , _lastPose(std::make_shared<PoseWithCovariance>(SE3d(),MatXd::Identity(6,6)))
        {}
        
        void update(PoseWithCovariance::ConstShPtr pose, Timestamp UNUSED(timestamp)) override { _lastPose = pose;}
        PoseWithCovariance::UnPtr predict(Timestamp UNUSED(timestamp)) const override { return std::make_unique<PoseWithCovariance>(_lastPose->pose(),_lastPose->cov());}
        private:
        PoseWithCovariance::ConstShPtr _lastPose;
};
class MotionPredictionConstant : public MotionPrediction{
        public:
        typedef std::shared_ptr<MotionPredictionConstant> ShPtr;
        typedef std::unique_ptr<MotionPredictionConstant> UnPtr;
        typedef std::shared_ptr<const MotionPredictionConstant> ConstShPtr;
        typedef std::unique_ptr<const MotionPredictionConstant> ConstUnPtr;
        MotionPredictionConstant()
        : MotionPrediction()
        , _lastPose(std::make_shared<PoseWithCovariance>(SE3d(),MatXd::Identity(6,6)))
        {}
        
        void update(PoseWithCovariance::ConstShPtr pose, Timestamp timestamp) override;
        PoseWithCovariance::UnPtr predict(Timestamp timestamp) const override;
        private:
        Vec6d _speed = Vec6d::Zero();
        PoseWithCovariance::ConstShPtr _lastPose;
        Timestamp _lastT;
};
}
#endif// VSLAM_MOTION_PREDICTION

