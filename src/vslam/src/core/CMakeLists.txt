cmake_minimum_required(VERSION 3.8)
project(core VERSION 1.0.0)

set( sources
  src/Camera.cpp
  src/algorithm.cpp
  src/PoseWithCovariance.cpp
  src/Frame.cpp
  src/Trajectory.cpp
  src/Point3D.cpp
  src/Feature2D.cpp )

set( headers
  src/Camera.h
  src/algorithm.h
  src/Exceptions.h
  src/PoseWithCovariance.h
  src/Frame.h
  src/Trajectory.h
  src/Point3D.h
  src/Feature2D.h
  include/core/core.h )
pd_setup_lib(${PROJECT_NAME} ${PROJECT_VERSION} "${sources}" "${headers}" pd)

if(UNIX)
  target_link_libraries(${PROJECT_NAME} stdc++fs)
endif()

find_package(Eigen3 REQUIRED)
target_link_libraries(${PROJECT_NAME} Eigen3::Eigen)

find_package(Sophus REQUIRED)
target_link_libraries(${PROJECT_NAME} Sophus::Sophus)

# Try for OpenCV 4.X, but settle for whatever is installed
find_package(OpenCV 4 QUIET)
if(NOT OpenCV_FOUND)
  find_package(OpenCV REQUIRED)
endif()
message(STATUS "Found OpenCV version ${OpenCV_VERSION}")
target_include_directories(${PROJECT_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES})

# Add tests
option(VSLAM_BUILD_TESTS "Enable to build tests" ON)
if(VSLAM_BUILD_TESTS)
  enable_testing()
  find_package(GTest CONFIG REQUIRED)

  pd_add_test(algorithm core)
  pd_add_test(camera core)
  pd_add_test(random core)
  pd_add_test(frame ${PROJECT_NAME})
endif()
