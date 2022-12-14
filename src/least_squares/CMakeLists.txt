cmake_minimum_required(VERSION 3.8)
project(least_squares VERSION 1.0.0)

set(sources
        src/Loss.cpp
        src/Scaler.cpp

        )

set(headers
        src/GaussNewton.h
        src/GaussNewton.hpp
        src/Scaler.h
        src/Prior.h
        src/Loss.h
        src/Solver.h
        include/least_squares/least_squares.h
        )

pd_setup_lib(${PROJECT_NAME} ${PROJECT_VERSION} "${sources}" "${headers}" pd)
target_link_libraries(${PROJECT_NAME} core utils)

target_compile_definitions(${PROJECT_NAME} PUBLIC 
$<BUILD_INTERFACE:LEAST_SQUARES_CFG_DIR="${CMAKE_CURRENT_LIST_DIR}/cfg">
$<INSTALL_INTERFACE:LEAST_SQUARES_CFG_DIR="/share/vslam/cfg">)


if (UNIX)
	target_link_libraries(${PROJECT_NAME} stdc++fs)
endif()

find_package(Eigen3 3.4 REQUIRED)
target_link_libraries(${PROJECT_NAME} Eigen3::Eigen3)

find_package(Sophus REQUIRED)
target_link_libraries(${PROJECT_NAME} Sophus::Sophus)

# Add tests
option(VSLAM_BUILD_TESTS "Enable to build tests" ON)
if(VSLAM_BUILD_TESTS)
	enable_testing()
        find_package(GTest CONFIG REQUIRED)

        pd_add_test(loss least_squares)
endif()

# Install
install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/cfg/
    DESTINATION share/vslam/cfg/
    FILES_MATCHING # install only matched files
    PATTERN "*" # select all files
    )