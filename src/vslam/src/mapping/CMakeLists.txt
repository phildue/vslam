cmake_minimum_required(VERSION 3.8)
project(mapping VERSION 1.0.0)

set( sources
  src/KeyFrameSelection.cpp
  src/OverlayFeatureDisplacement.cpp
  src/BundleAdjustment.cpp
  src/Map.cpp )

set( headers
  src/KeyFrameSelection.h
  src/OverlayFeatureDisplacement.h
  src/BundleAdjustment.h
  src/Map.h
  include/mapping/mapping.h )

pd_setup_lib(${PROJECT_NAME} ${PROJECT_VERSION} "${sources}" "${headers}" pd)

target_link_libraries(${PROJECT_NAME} lukas_kanade utils core)

if(UNIX)
  target_link_libraries(${PROJECT_NAME} stdc++fs)
endif()

find_package(Eigen3 REQUIRED)
target_link_libraries(${PROJECT_NAME} Eigen3::Eigen)

find_package(Sophus REQUIRED)
target_link_libraries(${PROJECT_NAME} Sophus::Sophus)

find_package(Ceres REQUIRED)
target_include_directories(${PROJECT_NAME} PUBLIC ${CERES_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${CERES_LIBRARIES})

# Add tests
option(VSLAM_BUILD_TESTS "Enable to build odometry tests" ON)
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
