cmake_minimum_required(VERSION 3.8)
project(utils VERSION 1.0.0)

set( sources
        src/Log.cpp
        src/visuals.cpp
        src/utils.cpp
        src/tum.cpp
        /usr/local/include/easylogging++.cc )

set( headers
        src/tum.h
        src/Log.h
        src/visuals.h
        include/utils/utils.h
        /usr/local/include/easylogging++.h )

pd_setup_lib(${PROJECT_NAME} ${PROJECT_VERSION} "${sources}" "${headers}" pd)
target_link_libraries(${PROJECT_NAME} core)

if(UNIX)
      target_link_libraries(${PROJECT_NAME} stdc++fs)
endif()


# Try for OpenCV 4.X, but settle for whatever is installed
find_package(OpenCV 4 QUIET)
if(NOT OpenCV_FOUND)
  find_package(OpenCV REQUIRED)
endif()
message(STATUS "Found OpenCV version ${OpenCV_VERSION}")
target_include_directories(${PROJECT_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES})

find_package(Eigen3 REQUIRED)
target_link_libraries(${PROJECT_NAME} Eigen3::Eigen)

find_package(Sophus REQUIRED)

target_link_libraries(${PROJECT_NAME} Sophus::Sophus)

#find_package(matplotlib_cpp)
#target_link_libraries(${PROJECT_NAME} matplotlib_cpp::matplotlib_cpp)
# Avoid crash with multiple headers see https://github.com/lava/matplotlib-cpp/issues/178
#target_compile_definitions(${PROJECT_NAME} PUBLIC WITHOUT_NUMPY)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(PythonLibs 3.0 REQUIRED)
target_include_directories(${PROJECT_NAME} PUBLIC ${PYTHON3_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${PYTHON_LIBRARIES} ${PYTHON3_LIBRARIES})

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
    option(VSLAM_TEST_VISUALIZE "Enable test visualizations." ON)
    if(VSLAM_TEST_VISUALIZE)
        set(TEST_VISUALIZE true)
    else()
        set(TEST_VISUALIZE false)
    endif()
    pd_add_test(log ${PROJECT_NAME})
    pd_add_test(plot ${PROJECT_NAME}
    TEST_RESOURCE="${CMAKE_CURRENT_LIST_DIR}/test/resource"
    TEST_VISUALIZE=${TEST_VISUALIZE})
endif()
