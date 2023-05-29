cmake_minimum_required(VERSION 3.8)
project(vslam VERSION 1.0.0)

set( sources
  src/direct_icp/DirectIcp.cpp
  src/evaluation/evaluation.cpp
  src/evaluation/tum.cpp
  src/core/Camera.cpp
  src/core/Frame.cpp
  src/core/Feature2D.cpp
  src/core/Point3D.cpp
  src/core/Trajectory.cpp
  src/core/Pose.cpp
  src/core/random.cpp
  src/utils/log.cpp
  src/utils/visuals.cpp
  src/utils/utils.cpp
  /usr/local/include/easylogging++.cc
   )

set( headers
  src/evaluation/evaluation.h
  src/evaluation/tum.h
  src/core/Trajectory.h
  src/core/Pose.h
  src/core/Camera.h
  src/core/Frame.h
  src/core/Feature2D.h
  src/core/Point3D.h
  src/core/macros.h
  src/core/types.h
  src/core/random.h
  src/direct_icp/DirectIcp.h
  src/utils/utils.h
  src/utils/visuals.h
  src/utils/log.h
  /usr/local/include/easylogging++.h 
  include/vslam/core.h
  include/vslam/direct_icp.h 
  include/vslam/evaluation.h 
  include/vslam/utils.h 

  )

pd_setup_lib(${PROJECT_NAME} ${PROJECT_VERSION} "${sources}" "${headers}" pd)

message(STATUS "Flags: ${CMAKE_CXX_FLAGS}")
if(UNIX)
  target_link_libraries(${PROJECT_NAME} stdc++fs)
endif()

find_package(Eigen3 REQUIRED)
target_link_libraries(${PROJECT_NAME} Eigen3::Eigen)

find_package(Sophus REQUIRED)
target_link_libraries(${PROJECT_NAME} Sophus::Sophus)

# While format is not implemented in gcc we use this thirdparty
find_package(fmt)
target_link_libraries(${PROJECT_NAME} fmt::fmt)


# Try for OpenCV 4.X, but settle for whatever is installed
find_package(OpenCV 4 QUIET)
if(NOT OpenCV_FOUND)
  find_package(OpenCV REQUIRED)
endif()
message(STATUS "Found OpenCV version ${OpenCV_VERSION}")
target_include_directories(${PROJECT_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES})

find_package(Matplot++ REQUIRED)
target_link_libraries(${PROJECT_NAME} Matplot++::matplot)

target_compile_definitions(${PROJECT_NAME} PUBLIC ELPP_FEATURE_PERFORMANCE_TRACKING)
option(VSLAM_LOG_PERFORMANCE_TRACKING ON)
if(NOT VSLAM_LOG_PERFORMANCE_TRACKING)
  target_compile_definitions(${PROJECT_NAME} PUBLIC ELPP_DISABLE_PERFORMANCE_TRACKING)
endif()
option(VSLAM_LOG_MINIMAL Off)
if(VSLAM_LOG_MINIMAL)
    target_compile_definitions(${PROJECT_NAME} PUBLIC ELPP_DISABLE_DEBUG_LOGS)
    target_compile_definitions(${PROJECT_NAME} PUBLIC ELPP_DISABLE_INFO_LOGS)
endif()

# Add tests
option(VSLAM_BUILD_TESTS "Enable to build tests" ON)
if(VSLAM_BUILD_TESTS)
  enable_testing()
  find_package(GTest CONFIG REQUIRED)

  #pd_add_test(algorithm core)
  #pd_add_test(camera core)
  #pd_add_test(random core)
  #pd_add_test(frame ${PROJECT_NAME})
endif()
