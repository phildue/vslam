cmake_minimum_required(VERSION 3.8)
project(evaluation VERSION 1.0.0)

set( sources
  src/RelativePoseError.cpp
  src/PlotTrajectory.cpp
   )

set( headers
src/RelativePoseError.h
src/PlotTrajectory.h
  include/evaluation/evaluation.h )
pd_setup_lib(${PROJECT_NAME} ${PROJECT_VERSION} "${sources}" "${headers}" pd)
target_link_libraries(${PROJECT_NAME} core utils)

if(UNIX)
  target_link_libraries(${PROJECT_NAME} stdc++fs)
endif()

find_package(Eigen3 REQUIRED)
target_link_libraries(${PROJECT_NAME} Eigen3::Eigen)

find_package(Sophus REQUIRED)
target_link_libraries(${PROJECT_NAME} Sophus::Sophus)

# Add tests
option(VSLAM_BUILD_TESTS "Enable to build tests" ON)
if(VSLAM_BUILD_TESTS)
  enable_testing()
  find_package(GTest CONFIG REQUIRED)
endif()
