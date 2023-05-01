cmake_minimum_required(VERSION 3.8)
project(least_squares VERSION 1.0.0)

set( sources
  src/Loss.cpp
  src/Scaler.cpp
  src/NormalEquations.cpp
  src/Solver.cpp
  src/GaussNewton.cpp 
  )

set( headers
  src/GaussNewton.h
  src/NormalEquations.h
  src/Scaler.h
  src/Prior.h
  src/Loss.h
  src/Solver.h
  include/least_squares/least_squares.h )

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

  option(VSLAM_TEST_VISUALIZE "Enable test visualizations." ON)
  if(VSLAM_TEST_VISUALIZE)
    set(TEST_VISUALIZE true)
  else()
    set(TEST_VISUALIZE false)
  endif()
  pd_add_test(loss least_squares
  TEST_VISUALIZE=${TEST_VISUALIZE})
endif()
