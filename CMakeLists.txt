cmake_minimum_required(VERSION 3.8)
project(vslam_ros)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()
find_package(Eigen3 REQUIRED)
# Try for OpenCV 4.X, but settle for whatever is installed
find_package(OpenCV 4 QUIET)
if(NOT OpenCV_FOUND)
  find_package(OpenCV REQUIRED)
endif()
message(STATUS "Found OpenCV version ${OpenCV_VERSION}")
find_package(cv_bridge REQUIRED)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(stereo_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(rosbag2_cpp REQUIRED)
find_package(rosbag2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(stereo_image_proc REQUIRED)
option(VSLAM_BUILD_TESTS "Enable unit test build" ${BUILD_TESTING})
add_subdirectory(src/vslam)

add_library(${PROJECT_NAME} STATIC
    src/ros/converter.cpp
    src/ros/Queue.cpp)

set_property(TARGET ${PROJECT_NAME} PROPERTY POSITION_INDEPENDENT_CODE ON)

target_link_libraries(${PROJECT_NAME} pd::vslam)
target_include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/src/ros/>
    $<INSTALL_INTERFACE:include/${PROJECT_NAME}/>)
target_include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
    $<INSTALL_INTERFACE:include/>)

add_library(vslam_ros_nodes SHARED
  src/ros/nodes/NodeStereoDepth.cpp
  src/ros/nodes/NodeRgbdAlignment.cpp
  src/ros/nodes/NodeEvaluation.cpp
  src/ros/nodes/NodeReplayer.cpp)
target_link_libraries(vslam_ros_nodes ${PROJECT_NAME})
set_property(TARGET vslam_ros_nodes PROPERTY POSITION_INDEPENDENT_CODE ON)

rclcpp_components_register_nodes(
    vslam_ros_nodes
    PLUGIN "vslam_ros::NodeReplayer"
    EXECUTABLE node_replayer
)
rclcpp_components_register_node(
    vslam_ros_nodes
    PLUGIN "vslam_ros::NodeRgbdAlignment"
    EXECUTABLE node_mapping
)
rclcpp_components_register_node(
    vslam_ros_nodes
    PLUGIN "vslam_ros::NodeEvaluation"
    EXECUTABLE node_evaluation
)
rclcpp_components_register_node(
    vslam_ros_nodes
    PLUGIN "vslam_ros::NodeStereoDepth"
    EXECUTABLE node_stereo_depth
)
add_executable(composition_evaluation_tum src/ros/compositions/evaluation_tum.cpp)
target_link_libraries(composition_evaluation_tum vslam_ros_nodes)

add_executable(composition_evaluation_kitti src/ros/compositions/evaluation_kitti.cpp)
target_link_libraries(composition_evaluation_kitti vslam_ros_nodes)

ament_target_dependencies(${PROJECT_NAME}
  OpenCV
  Eigen3
  rclcpp
  rclcpp_components
  std_msgs
  sensor_msgs
  nav_msgs
  stereo_msgs
  cv_bridge
  message_filters
  geometry_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  tf2_msgs
  std_srvs
  rosbag2_cpp
  pcl_conversions
  stereo_image_proc
  )
ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME})
ament_export_dependencies(vslam)

install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/src/ros/
    DESTINATION include/${PROJECT_NAME}
    FILES_MATCHING # install only matched files
    PATTERN "*.h*") # select header files

install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/include/
    DESTINATION include/)

install(DIRECTORY
    launch
    DESTINATION share/${PROJECT_NAME}/)

install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}/)

install(DIRECTORY
  src/vslam/config
  DESTINATION share/${PROJECT_NAME}/)

install(
  TARGETS ${PROJECT_NAME} vslam_ros_nodes node_mapping node_evaluation composition_evaluation_tum composition_evaluation_kitti
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib
  INCLUDES DESTINATION include)


# if(BUILD_TESTING)
  # find_package(ament_lint_auto REQUIRED)
  # ament_lint_auto_find_test_dependencies()
# endif()

ament_package()