cmake_minimum_required(VERSION 3.8)
project(kalman VERSION 1.0.0)

set( sources
        src/EKFConstantVelocitySE3.cpp )

set( headers
  src/EKFConstantVelocitySE3.h
  include/kalman/kalman.h )

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
option(KALMAN_BUILD_TESTS "Enable to build tests" ON)
if(KALMAN_BUILD_TESTS)
  enable_testing()
  find_package(GTest CONFIG REQUIRED)
endif()

# Install
install( DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/cfg/
    DESTINATION share/vslam/cfg/
    FILES_MATCHING # install only matched files
    PATTERN "*" )# select all files