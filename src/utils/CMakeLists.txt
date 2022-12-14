cmake_minimum_required(VERSION 3.8)
project(utils VERSION 1.0.0)

set(sources
        src/Log.cpp
        src/visuals.cpp
        src/utils.cpp
        ${CMAKE_BINARY_DIR}/include/easylogging++.cc
        )

set(headers
        src/Log.h
        src/visuals.h
        include/utils/utils.h
        )

pd_setup_lib(${PROJECT_NAME} ${PROJECT_VERSION} "${sources}" "${headers}" pd)
target_link_libraries(${PROJECT_NAME} core)

if (UNIX)
	target_link_libraries(${PROJECT_NAME} stdc++fs)
endif()

# Try for OpenCV 4.X, but settle for whatever is installed
find_package(OpenCV 4 QUIET)
if (NOT OpenCV_FOUND)
  find_package(OpenCV REQUIRED)
endif ()
message(STATUS "Found OpenCV version ${OpenCV_VERSION}")
target_include_directories(${PROJECT_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES})

find_package(Eigen3 3.4 REQUIRED)
target_link_libraries(${PROJECT_NAME} Eigen3::Eigen3)

find_package(Sophus REQUIRED)
find_package(easyloggingpp REQUIRED)
target_link_libraries(${PROJECT_NAME} Sophus::Sophus)
target_link_libraries(${PROJECT_NAME} easyloggingpp::easyloggingpp)

find_package(matplotlib_cpp)
target_link_libraries(${PROJECT_NAME} matplotlib_cpp::matplotlib_cpp)

target_compile_definitions(${PROJECT_NAME} PUBLIC 
$<BUILD_INTERFACE:LOG_DIR="${CMAKE_BINARY_DIR}/log">
$<INSTALL_INTERFACE:LOG_DIR="/tmp/vslam/log">)



target_compile_definitions(${PROJECT_NAME} PUBLIC
        ELPP_FEATURE_PERFORMANCE_TRACKING
)
option(VSLAM_PERFORMANCE_TRACKING ON)
if(NOT VSLAM_PERFORMANCE_TRACKING)
	target_compile_definitions(${PROJECT_NAME} PUBLIC
		ELPP_DISABLE_PERFORMANCE_TRACKING)	
endif()

# Add tests
option(VSLAM_BUILD_TESTS "Enable to build tests" ON)
if(VSLAM_BUILD_TESTS)
	enable_testing()
        find_package(GTest CONFIG REQUIRED)

        pd_add_test(log ${PROJECT_NAME})
        pd_add_test(plot ${PROJECT_NAME})
        
        target_compile_definitions(logTest PUBLIC TEST_RESOURCE="${CMAKE_CURRENT_LIST_DIR}/test/resource")
        target_compile_definitions(plotTest PUBLIC TEST_RESOURCE="${CMAKE_CURRENT_LIST_DIR}/test/resource")

endif()
