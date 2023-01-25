cmake_minimum_required(VERSION 3.8)
project(VSLAM_APP)


add_executable(evaluate_tum src/main_tum.cpp)
target_compile_definitions(evaluate_tum PUBLIC "CONFIG_DIR=\"${VSLAM_CONFIG_DIR}\"")

target_link_libraries(evaluate_tum vslam)

add_executable(run_dvo src/main_tum_dvo.cpp)
target_link_libraries(run_dvo utils)
target_link_libraries(run_dvo dvo_core)
target_compile_features(run_dvo PUBLIC cxx_std_20)


add_executable(compare_dvo src/main_dvo_compare.cpp)
target_link_libraries(compare_dvo vslam)
target_link_libraries(compare_dvo dvo_core)
target_compile_features(compare_dvo PUBLIC cxx_std_20)