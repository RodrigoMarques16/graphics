cmake_minimum_required(VERSION 3.8)
project(graphics LANGUAGES CXX CUDA)
add_compile_options(-arch=sm_30)

include_directories(math)
# include_directories(render)

add_subdirectory(math)
# add_subdirectory(render)

add_executable(cudart main.cu)
set_property(TARGET cudart PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(cudart math)
# target_link_libraries(cudart PUBLIC render)