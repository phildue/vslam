cmake_minimum_required(VERSION 3.8)
project(vslam VERSION 1.0.0)


add_executable(evaluate_tum src/main_tum.cpp)
target_compile_definitions(evaluate_tum PUBLIC "CONFIG_DIR=\"${VSLAM_CONFIG_DIR}\"")
target_link_libraries(evaluate_tum vslam)


