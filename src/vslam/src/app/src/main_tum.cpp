

#include "vslam/vslam.h"

using namespace pd;
using namespace pd::vslam;
class Main
{
public:
  Main(int UNUSED(argc), char ** UNUSED(argv))
  {
    _datasetPath = "/mnt/dataset/tum_rgbd/rgbd_dataset_freiburg2_desk/rgbd_dataset_freiburg2_desk";
    _cam = tum::Camera();

    tum::readAssocTextfile(
      _datasetPath + "/assoc.txt", _imgFilenames, _depthFilenames, _timestamps);
    _trajectoryGt =
      std::make_shared<Trajectory>(utils::loadTrajectory(_datasetPath + "/groundtruth.txt"));

    auto solver = std::make_shared<least_squares::GaussNewton>(1e-7, 25);
    auto loss =
      std::make_shared<least_squares::HuberLoss>(std::make_shared<least_squares::MeanScaler>());
    _map = std::make_shared<Map>(5, 3);
    _rgbdAlignment = std::make_shared<SE3Alignment>(18, solver, loss, true);
    _motionModel = std::make_shared<MotionModelConstantSpeed>();
    _keyFrameSelection = std::make_shared<KeyFrameSelectionCustom>(_map, 100, 0.1);
    _ba = std::make_shared<mapping::BundleAdjustment>(0, 1.43);

    _matcher = std::make_shared<vslam::Matcher>(vslam::Matcher::reprojectionHamming, 5.0, 0.8);
    _tracking = std::make_shared<FeatureTracking>(_matcher);

    for (auto log : Log::registeredLogsImage()) {
      LOG_IMG(log)->set(false, false);
    }
    for (auto name : {"solver", "odometry"}) {
      el::Loggers::getLogger(name);
      el::Configurations defaultConf;
      defaultConf.setToDefault();
      defaultConf.set(el::Level::Debug, el::ConfigurationType::Enabled, "false");
      defaultConf.set(el::Level::Info, el::ConfigurationType::Enabled, "false");
      el::Loggers::reconfigureLogger(name, defaultConf);
    }
  }

  Frame::ShPtr loadFrame(size_t fNo)
  {
    // tum depth format: https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats
    auto f = std::make_shared<Frame>(
      utils::loadImage(_datasetPath + "/" + _imgFilenames.at(fNo)),
      utils::loadDepth(_datasetPath + "/" + _depthFilenames.at(fNo)) / 5000.0, _cam,
      _timestamps.at(fNo));
    f->computePyramid(3);
    f->computeDerivatives();
    f->computePcl();
    return f;
  }

  void run()
  {
    const int nFrames = _imgFilenames.size();

    Trajectory traj;
    std::vector<double> rmseTranslation, rmseRotation;
    rmseTranslation.reserve(nFrames);
    rmseRotation.reserve(nFrames);
    for (int fId = 0; fId < nFrames; fId++) {
      auto frame = loadFrame(fId);

      auto frameRef = _map->lastKf();

      if (frameRef) {
        frame->set(*_motionModel->predictPose(frame->t()));
        auto pose = _map->lastFrame() ? _rgbdAlignment->align({_map->lastFrame(), frameRef}, frame)
                                      : _rgbdAlignment->align(frameRef, frame);
        frame->set(*pose);
        _motionModel->update(frameRef, frame);
        //frame->set(*_motionModel->pose());
      }

      _keyFrameSelection->update(frame);

      _map->insert(frame, _keyFrameSelection->isKeyFrame());
      /*
    auto outBa = _keyFrameSelection->isKeyFrame()
                   ? _ba->optimize(Map::ConstShPtr(_map)->keyFrames())
                   : _ba->optimize({frame}, Map::ConstShPtr(_map)->keyFrames());*/
      if (_keyFrameSelection->isKeyFrame()) {
        auto points = _tracking->track(frame, _map->keyFrames());
        _map->insert(points);
        if (_map->nKeyFrames() >= 2) {
          LOG_IMG("KeyFrames") << std::make_shared<OverlayFeatureDisplacement>(
            Map::ConstShPtr(_map)->keyFrames());

          auto outBa = _ba->optimize(
            {Map::ConstShPtr(_map)->keyFrame(0)},
            Map::ConstShPtr(_map)->keyFrames(1, _map->nKeyFrames()));

          _map->updatePointsAndPoses(outBa->poses, outBa->positions);
          LOG_IMG("KeyFrames") << std::make_shared<OverlayFeatureDisplacement>(
            Map::ConstShPtr(_map)->keyFrames());
        }
      }
      traj.append(frame->t(), std::make_shared<PoseWithCovariance>(frame->pose().inverse()));
      try {
        auto relativePose = algorithm::computeRelativeTransform(
          traj.poseAt(frame->t() - 1 * 1e9)->pose().inverse(), frame->pose().pose());
        auto relativePoseGt = algorithm::computeRelativeTransform(
          _trajectoryGt->poseAt(frame->t() - 1 * 1e9)->pose().inverse(),
          _trajectoryGt->poseAt(frame->t())->pose().inverse());
        auto error = (relativePose.inverse() * relativePoseGt).log();
        rmseTranslation.push_back(error.head(3).norm());
        rmseRotation.push_back(error.tail(3).norm());
        Eigen::Map<VecXd> rmseT(rmseTranslation.data(), rmseTranslation.size());
        Eigen::Map<VecXd> rmseR(rmseRotation.data(), rmseTranslation.size());
        std::cout << fId << "/" << nFrames << ": " << frame->pose().pose().log().transpose()
                  << "\n |Translation|"
                  << "\n |Current: " << rmseTranslation.at(rmseTranslation.size() - 1)
                  << "\n |RMSE: " << std::sqrt(rmseT.dot(rmseT) / rmseTranslation.size())
                  << "\n |Max: " << rmseT.maxCoeff() << "\n |Mean: " << rmseT.mean()
                  << "\n |Min: " << rmseT.minCoeff() << "\n |Rotation|"
                  << "\n |Current: " << rmseRotation.at(rmseRotation.size() - 1)
                  << "\n |RMSE: " << std::sqrt(rmseR.dot(rmseR) / rmseRotation.size())
                  << "\n |Max: " << rmseR.maxCoeff() << "\n |Mean: " << rmseR.mean()
                  << "\n |Min: " << rmseR.minCoeff() << std::endl;
      } catch (const pd::Exception & e) {
        std::cerr << e.what() << std::endl;
        std::cout << fId << "/" << nFrames << ": " << frame->pose().pose().log().transpose()
                  << std::endl;
      }
      utils::writeTrajectory(traj, "trajectory.txt");
    }
    // TODO(unknown): call evaluation script?
  }

protected:
  std::vector<std::string> _depthFilenames;
  std::vector<std::string> _imgFilenames;
  std::vector<Timestamp> _timestamps;
  Trajectory::ConstShPtr _trajectoryGt;
  Camera::ConstShPtr _cam;
  RgbdAlignment::ShPtr _rgbdAlignment;
  KeyFrameSelection::ShPtr _keyFrameSelection;
  MotionModel::ShPtr _motionModel;
  Map::ShPtr _map;
  FeatureTracking::ShPtr _tracking;
  vslam::Matcher::ShPtr _matcher;
  mapping::BundleAdjustment::ShPtr _ba;

  std::string _datasetPath;
};

int main(int argc, char ** argv)
{
  Main m(argc, argv);
  m.run();
  return 0;
}