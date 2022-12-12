cmake_minimum_required(VERSION 3.8)
project(VSLAM_APP)


add_executable(evaluate_tum src/main_tum.cpp)
target_compile_definitions(evaluate_tum PUBLIC "CONFIG_DIR=\"${VSLAM_CONFIG_DIR}\"")

target_link_libraries(evaluate_tum vslam)