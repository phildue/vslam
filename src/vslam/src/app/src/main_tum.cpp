
#include <fmt/chrono.h>
#include <fmt/core.h>
using fmt::format;
using fmt::print;

#include "vslam/vslam.h"

using namespace pd;
using namespace pd::vslam;
using namespace pd::vslam::evaluation;

class Main
{
public:
  Main(int UNUSED(argc), char ** UNUSED(argv))
  {
    _dl = std::make_unique<tum::DataLoader>(
      "/mnt/dataset/tum_rgbd/rgbd_dataset_freiburg2_desk/rgbd_dataset_freiburg2_desk");
    auto solver = std::make_shared<least_squares::GaussNewton>(1e-7, 25);
    auto loss =
      std::make_shared<least_squares::QuadraticLoss>(std::make_shared<least_squares::Scaler>());
    _map = std::make_shared<Map>(5, 3);
    _rgbdAlignment = std::make_shared<RgbdAlignment>(solver, loss);
    Matd<12, 12> covProcess = Matd<12, 12>::Identity();
    for (int i = 0; i < 6; i++) {
      covProcess(i, i) = 1e-9;
    }
    covProcess(6, 6) = 1e-6;
    covProcess(7, 7) = 1e-6;
    covProcess(8, 8) = 1e-6;
    covProcess(9, 9) = 1e-6;
    covProcess(10, 10) = 1e-6;
    covProcess(11, 11) = 1e-6;

    LOG_IMG("Kalman")->set(true, true, true);

    _motionModel =
      std::make_shared<MotionModelConstantSpeedKalman>(covProcess, Matd<12, 12>::Identity() * 100);
    //_motionModel = std::make_shared<MotionModelConstantSpeed>();
    _keyFrameSelection = std::make_shared<KeyFrameSelectionCustom>(_map, 100, 0.1);
    _ba = std::make_shared<mapping::BundleAdjustment>(0, 1.43);

    _matcher = std::make_shared<vslam::Matcher>(vslam::Matcher::reprojectionHamming, 5.0, 0.8);
    _tracking = std::make_shared<FeatureTracking>(_matcher);
    LogImage::rootFolder() = format("{}/algorithm_results/app/log/", _dl->datasetPath());
    for (const auto & name : Log::registeredLogs()) {
      print("Found logger: {}\n", name.c_str());
      Log::get(name)->configure(format("{}/log/{}.conf", CONFIG_DIR, name));
    }

    for (auto log : Log::registeredLogsImage()) {
      LOG_IMG(log)->set(false, false);
    }
    LOG_IMG("Kalman")->set(true, true, true);
  }

  Frame::UnPtr loadFrame(size_t fNo)
  {
    // tum depth format: https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats
    auto f = _dl->loadFrame(fNo);
    f->computePyramid(3);
    f->computeDerivatives();
    f->computePcl();
    return f;
  }

  void run()
  {
    Trajectory::ShPtr traj = std::make_shared<Trajectory>();
    for (size_t fId = 0; fId < 50; fId++) {
      Frame::ShPtr frame = loadFrame(fId);

      auto frameRef = _map->lastKf();

      if (frameRef) {
        frame->set(*_motionModel->predictPose(frame->t()));
        Pose::ConstShPtr pose = _includeLastFrame && _map->lastFrame()
                                  ? _rgbdAlignment->align({_map->lastFrame(), frameRef}, frame)
                                  : _rgbdAlignment->align(frameRef, frame);
        _motionModel->update(pose, frame->t());
        frame->set(*pose);
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
      auto x = frame->pose().pose().inverse().log();
      auto cx = frame->pose().cov();

      print(
        "Pose: {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f} | Cov | {:.3f}\n", x(0), x(1), x(2),
        x(3), x(4), x(5), cx.norm());
      print(
        "Translational Speed: {:.3f} | Cov | {:.3f}\n",
        _motionModel->speed()->SE3().translation().norm(),
        _motionModel->speed()->twistCov().block(0, 0, 3, 3).norm());

      traj->append(frame->t(), std::make_shared<PoseWithCovariance>(frame->pose()));
      try {
        auto rpe = RelativePoseError::compute(traj, _dl->trajectoryGt(), 1.0);
        print("{}\n", rpe->toString());
      } catch (const pd::Exception & e) {
        std::cerr << e.what() << std::endl;
        std::cout << fId << "/" << _dl->nFrames() << ": " << frame->pose().pose().log().transpose()
                  << std::endl;
      }
      utils::writeTrajectory(
        *traj, format("{}/algorithm_results/{}", _dl->datasetPath(), "trajectory.txt"));
    }
  }

protected:
  tum::DataLoader::ConstUnPtr _dl;
  RgbdAlignment::ShPtr _rgbdAlignment;
  KeyFrameSelection::ShPtr _keyFrameSelection;
  MotionModel::ShPtr _motionModel;
  Map::ShPtr _map;
  FeatureTracking::ShPtr _tracking;
  vslam::Matcher::ShPtr _matcher;
  mapping::BundleAdjustment::ShPtr _ba;
  bool _includeLastFrame = true;
};

int main(int argc, char ** argv)
{
  Main m(argc, argv);
  m.run();
  return 0;
}