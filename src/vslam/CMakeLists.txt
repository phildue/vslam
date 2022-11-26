cmake_minimum_required(VERSION 3.8)
project(vslam VERSION 1.0.0)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/tools ${CMAKE_BINARY_DIR}/)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_BINARY_DIR})

include(Macros)

add_subdirectory(src/core)
add_subdirectory(src/utils)
add_subdirectory(src/least_squares)
add_subdirectory(src/lukas_kanade)
add_subdirectory(src/odometry)
add_subdirectory(src/mapping)

add_library(vslam INTERFACE)
add_library(pd::vslam ALIAS vslam)
target_link_libraries(${PROJECT_NAME} INTERFACE core lukas_kanade least_squares utils odometry mapping)
target_include_directories(${PROJECT_NAME} INTERFACE
$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include/>
$<INSTALL_INTERFACE:include/vslam>)

add_subdirectory(src/app)

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

  pd_add_test(ekf_se3 ${PROJECT_NAME} 
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
install(TARGETS ${PROJECT_NAME} core utils least_squares lukas_kanade odometry mapping
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


