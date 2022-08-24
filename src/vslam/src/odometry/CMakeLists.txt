cmake_minimum_required(VERSION 3.8)
project(odometry VERSION 1.0.0)

set( sources
  src/MotionPrediction.cpp
  src/KeyFrameSelection.cpp
  src/Odometry.cpp
  src/iterative_closest_point/IterativeClosestPoint.cpp
  src/iterative_closest_point/IterativeClosestPointOcv.cpp
  src/direct_image_alignment/SE3Alignment.cpp
  src/direct_image_alignment/RgbdAlignmentOpenCv.cpp
  src/feature_tracking/FeatureTracking.cpp
  src/feature_tracking/MatcherBruteForce.cpp
  src/mapping/BundleAdjustment.cpp
  src/mapping/MapOptimization.cpp
  src/mapping/Map.cpp )

set( headers
  src/MotionPrediction.h
  src/KeyFrameSelection.h
  src/Odometry.h
  src/iterative_closest_point/IterativeClosestPoint.h
  src/iterative_closest_point/IterativeClosestPointOcv.h
  src/direct_image_alignment/SE3Alignment.h
  src/direct_image_alignment/RgbdAlignmentOpenCv.h
  src/feature_tracking/FeatureTracking.h
  src/feature_tracking/MatcherBruteForce.h
  src/mapping/BundleAdjustment.h
  src/mapping/MapOptimization.h
  src/mapping/Map.h
  include/odometry/odometry.h )

pd_setup_lib(${PROJECT_NAME} ${PROJECT_VERSION} "${sources}" "${headers}" pd)

target_link_libraries(${PROJECT_NAME} kalman lukas_kanade utils core)

target_link_libraries(${PROJECT_NAME} core utils least_squares)
if(UNIX)
  target_link_libraries(${PROJECT_NAME} stdc++fs)
endif()

find_package(Eigen3 REQUIRED)
target_link_libraries(${PROJECT_NAME} Eigen3::Eigen)

find_package(Sophus REQUIRED)
target_link_libraries(${PROJECT_NAME} Sophus::Sophus)

find_package(PCL REQUIRED)
target_include_directories(${PROJECT_NAME} PUBLIC ${PCL_INCLUDE_DIRS})
target_compile_definitions(${PROJECT_NAME} PRIVATE ${PCL_DEFINITIONS})
target_link_libraries(${PROJECT_NAME} ${PCL_LIBRARIES})

find_package(Ceres REQUIRED)
target_include_directories(${PROJECT_NAME} PUBLIC ${CERES_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${CERES_LIBRARIES})

# Add tests
option(VSLAM_ODOMETRY_BUILD_TESTS "Enable to build odometry tests" ON)
if(VSLAM_ODOMETRY_BUILD_TESTS)
  enable_testing()
  find_package(GTest CONFIG REQUIRED)

  option(VSLAM_TEST_VISUALIZE "Enable test visualizations." ON)
  if(VSLAM_TEST_VISUALIZE)
    set(TEST_VISUALIZE true)
  else()
    set(TEST_VISUALIZE false)
  endif()



  pd_add_test(alignment_se3 ${PROJECT_NAME} 
  TEST_RESOURCE="${CMAKE_CURRENT_LIST_DIR}/test/resource"
  TEST_VISUALIZE=${TEST_VISUALIZE})

  pd_add_test(alignment_se3_multi_frame ${PROJECT_NAME} 
  TEST_RESOURCE="${CMAKE_CURRENT_LIST_DIR}/test/resource"
  TEST_VISUALIZE=${TEST_VISUALIZE})

  pd_add_test(odometry_icp ${PROJECT_NAME} 
  TEST_RESOURCE="${CMAKE_CURRENT_LIST_DIR}/test/resource"
  TEST_VISUALIZE=${TEST_VISUALIZE})

  pd_add_test(odometry_rgbd_cv ${PROJECT_NAME} 
  TEST_RESOURCE="${CMAKE_CURRENT_LIST_DIR}/test/resource"
  TEST_VISUALIZE=${TEST_VISUALIZE})

  pd_add_test(trajectory ${PROJECT_NAME} 
  TEST_RESOURCE="${CMAKE_CURRENT_LIST_DIR}/test/resource"
  TEST_VISUALIZE=${TEST_VISUALIZE})

  pd_add_test(odometry ${PROJECT_NAME} 
  TEST_RESOURCE="${CMAKE_CURRENT_LIST_DIR}/test/resource"
  TEST_VISUALIZE=${TEST_VISUALIZE})

  pd_add_test(tracking ${PROJECT_NAME} 
  TEST_RESOURCE="${CMAKE_CURRENT_LIST_DIR}/test/resource"
  TEST_VISUALIZE=${TEST_VISUALIZE})

endif()
