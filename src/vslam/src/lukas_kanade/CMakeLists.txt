cmake_minimum_required(VERSION 3.8)
project(lukas_kanade VERSION 1.0.0)

set( sources
  src/Warp.cpp
  src/InverseCompositional.cpp
  src/ForwardAdditive.cpp
  src/InverseCompositionalStacked.cpp )

set( headers
  src/ForwardAdditive.h
  src/InverseCompositional.h
  src/Warp.h
  include/lukas_kanade/lukas_kanade.h )

pd_setup_lib(${PROJECT_NAME} ${PROJECT_VERSION} "${sources}" "${headers}" pd)

target_link_libraries(${PROJECT_NAME} core utils least_squares)
if(UNIX)
  target_link_libraries(${PROJECT_NAME} stdc++fs)
endif()

find_package(Eigen3 REQUIRED)
target_link_libraries(${PROJECT_NAME} Eigen3::Eigen)

find_package(Sophus REQUIRED)
target_link_libraries(${PROJECT_NAME} Sophus::Sophus)
find_package(TBB REQUIRED)

target_link_libraries(${PROJECT_NAME} TBB::tbb)

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
  pd_add_test(lukas_kanade_flow ${PROJECT_NAME} 
  TEST_RESOURCE="${CMAKE_CURRENT_LIST_DIR}/test/resource"
  TEST_VISUALIZE=${TEST_VISUALIZE})

  pd_add_test(warp ${PROJECT_NAME} 
  TEST_RESOURCE="${CMAKE_CURRENT_LIST_DIR}/test/resource"
  TEST_VISUALIZE=${TEST_VISUALIZE})

  #pd_add_test(lukas_kanade_affine ${PROJECT_NAME})
    #pd_add_test(lukas_kanade_se3 ${PROJECT_NAME})
    #target_compile_definitions(lukas_kanade_affineTest PUBLIC TEST_RESOURCE="${CMAKE_CURRENT_LIST_DIR}/test/resource")
    #target_compile_definitions(lukas_kanade_se3Test PUBLIC TEST_RESOURCE="${CMAKE_CURRENT_LIST_DIR}/test/resource")

endif()
