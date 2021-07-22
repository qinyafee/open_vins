#cmake_minimum_required(VERSION 2.8.8)
cmake_minimum_required(VERSION 3.1)
project(ov_dyn)

add_subdirectory(camera_models)
add_subdirectory(vins_estimator)

add_subdirectory(ov_core)
#add_subdirectory(ov_eval)
add_subdirectory(ov_msckf)

