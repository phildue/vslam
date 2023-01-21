cmake_minimum_required(VERSION 3.8)
project(odometry VERSION 1.0.0)

set( sources
  src/motion_model/MotionModel.cpp
  src/motion_model/EKFConstantVelocitySE3.cpp
  src/motion_model/PlotKalman.cpp
  src/rgbd_alignment/IterativeClosestPoint.cpp
  src/rgbd_alignment/LukasKanadeWithDepth.cpp
  src/rgbd_alignment/RgbdAlignment.cpp
  src/rgbd_alignment/RgbdAlignmentRgb.cpp
  src/rgbd_alignment/RgbdAlignmentIcp.cpp
  src/rgbd_alignment/PlotAlignment.cpp
  #src/rgbd_alignment/RgbdAligner.cpp
  #src/rgbd_alignment/RgbdAlignmentOpenCv.cpp
  src/feature_tracking/OverlayMatchCandidates.cpp
  src/feature_tracking/OverlayFeatures.cpp
  src/feature_tracking/OverlayCorrespondences.cpp
  src/feature_tracking/FeatureTracking.cpp
  src/feature_tracking/FeatureTrackingOcv.cpp
  src/feature_tracking/Matcher.cpp)

set( headers
  src/motion_model/MotionModel.h
  src/motion_model/KalmanFilter.h
  src/motion_model/EKFConstantVelocitySE3.h
  src/motion_model/PlotKalman.h
  src/rgbd_alignment/IterativeClosestPoint.h
  src/rgbd_alignment/LukasKanadeWithDepth.h
  src/rgbd_alignment/RgbdAlignment.h
  src/rgbd_alignment/RgbdAlignmentIcp.h
  src/rgbd_alignment/RgbdAlignmentRgb.h
  src/rgbd_alignment/PlotAlignment.h

  #src/rgbd_alignment/RgbdAligner.h
  #src/rgbd_alignment/SE3Alignment.h
  #src/rgbd_alignment/RgbdAlignmentOpenCv.h
  src/feature_tracking/OverlayMatchCandidates.h
  src/feature_tracking/OverlayFeatures.h
  src/feature_tracking/OverlayCorrespondences.h
  src/feature_tracking/FeatureTracking.h
  src/feature_tracking/FeatureTrackingOcv.h
  src/feature_tracking/Matcher.h
  include/odometry/odometry.h )

pd_setup_lib(${PROJECT_NAME} ${PROJECT_VERSION} "${sources}" "${headers}" pd)

target_link_libraries(${PROJECT_NAME} core utils least_squares lukas_kanade)
if(UNIX)
  target_link_libraries(${PROJECT_NAME} stdc++fs)
endif()

find_package(Eigen3 REQUIRED)
target_link_libraries(${PROJECT_NAME} Eigen3::Eigen)

find_package(Sophus REQUIRED)
target_link_libraries(${PROJECT_NAME} Sophus::Sophus)

find_package(manif REQUIRED)
target_link_libraries(${PROJECT_NAME} MANIF::manif)

find_package(PCL REQUIRED)
target_include_directories(${PROJECT_NAME} PUBLIC ${PCL_INCLUDE_DIRS})
target_compile_definitions(${PROJECT_NAME} PRIVATE ${PCL_DEFINITIONS})
target_link_libraries(${PROJECT_NAME} ${PCL_LIBRARIES})

# Add tests
option(VSLAM_BUILD_TESTS "Enable to build tests" ON)
if(VSLAM_BUILD_TESTS)
  enable_testing()
  find_package(GTest CONFIG REQUIRED)

  option(VSLAM_TEST_VISUALIZE "Enable test visualizations." ON)
  if(VSLAM_TEST_VISUALIZE)
    set(TEST_VISUALIZE true)
  else()
    set(TEST_VISUALIZE false)
  endif()





endif()
