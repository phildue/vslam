cmake_minimum_required(VERSION 3.8)
project(vslam VERSION 1.0.0)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/tools ${CMAKE_BINARY_DIR}/)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_BINARY_DIR})
set(VSLAM_CONFIG_DIR "${CMAKE_CURRENT_LIST_DIR}/config")
include(Macros)

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
add_library(pd::vslam ALIAS vslam)
#target_link_libraries(${PROJECT_NAME} INTERFACE core lukas_kanade least_squares utils odometry mapping evaluation)
target_include_directories(${PROJECT_NAME} INTERFACE
$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include/>
$<INSTALL_INTERFACE:include/vslam>)

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


add_executable(evaluate_tum test/main_tum.cpp)
target_compile_definitions(evaluate_tum PUBLIC "CONFIG_DIR=\"${VSLAM_CONFIG_DIR}\"")
target_link_libraries(evaluate_tum vslam)



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

  pd_add_test(frame2 ${PROJECT_NAME} 
  TEST_RESOURCE="${CMAKE_CURRENT_LIST_DIR}/test/resource"
  TEST_VISUALIZE=${TEST_VISUALIZE})
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

  pd_add_test(rgbd_alignment ${PROJECT_NAME} 
    TEST_RESOURCE="${CMAKE_CURRENT_LIST_DIR}/test/resource"
    TEST_VISUALIZE=${TEST_VISUALIZE})
  

  pd_add_test(trajectory ${PROJECT_NAME} 
  TEST_RESOURCE="${CMAKE_CURRENT_LIST_DIR}/test/resource"
  TEST_VISUALIZE=${TEST_VISUALIZE})

  pd_add_test(ekf_se3 ${PROJECT_NAME} 
  TEST_RESOURCE="${CMAKE_CURRENT_LIST_DIR}/test/resource"
  TEST_VISUALIZE=${TEST_VISUALIZE})
  
  pd_add_test(motion_model ${PROJECT_NAME} 
  TEST_RESOURCE="${CMAKE_CURRENT_LIST_DIR}/test/resource"
  TEST_VISUALIZE=${TEST_VISUALIZE})

  pd_add_test(rpe ${PROJECT_NAME} 
  TEST_RESOURCE="${CMAKE_CURRENT_LIST_DIR}/test/resource"
  TEST_VISUALIZE=${TEST_VISUALIZE})

  pd_add_test(tracking ${PROJECT_NAME} 
  TEST_RESOURCE="${CMAKE_CURRENT_LIST_DIR}/test/resource"
  TEST_VISUALIZE=${TEST_VISUALIZE})

  pd_add_test(track_and_optimize ${PROJECT_NAME} 
  TEST_RESOURCE="${CMAKE_CURRENT_LIST_DIR}/test/resource"
  TEST_VISUALIZE=${TEST_VISUALIZE})

  pd_add_test(ba ${PROJECT_NAME} 
  TEST_RESOURCE="${CMAKE_CURRENT_LIST_DIR}/test/resource"
  TEST_VISUALIZE=${TEST_VISUALIZE})

  pd_add_test(ceres_ba ${PROJECT_NAME} 
  TEST_RESOURCE="${CMAKE_CURRENT_LIST_DIR}/test/resource"
  TEST_VISUALIZE=${TEST_VISUALIZE})

  pd_add_test(map ${PROJECT_NAME} 
  TEST_RESOURCE="${CMAKE_CURRENT_LIST_DIR}/test/resource"
  TEST_VISUALIZE=${TEST_VISUALIZE})
endif()

# Installation
install(TARGETS ${PROJECT_NAME}
    EXPORT ${PROJECT_NAME}Targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin)
    
install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/include/
    DESTINATION include/
    FILES_MATCHING # install only matched files
    PATTERN "*.h")# select header files

install(EXPORT ${PROJECT_NAME}Targets
    DESTINATION lib/cmake/${PROJECT_NAME}
    FILE ${PROJECT_NAME}Targets.cmake
    NAMESPACE pd::
    DESTINATION share/${PROJECT_NAME}/cmake)

include(CMakePackageConfigHelpers)

write_basic_package_version_file( ${PROJECT_NAME}ConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)

# install export target and config for find_package
include(CMakePackageConfigHelpers)
configure_package_config_file(
  "${CMAKE_CURRENT_LIST_DIR}/tools/${PROJECT_NAME}Config.cmake.in" "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
  INSTALL_DESTINATION "share/${PROJECT_NAME}/cmake/"
)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake" DESTINATION "share/${PROJECT_NAME}/cmake/")


