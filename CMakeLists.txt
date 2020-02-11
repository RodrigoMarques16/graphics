cmake_minimum_required(VERSION 3.0.0)

project(graphics VERSION 0.1.0
                 DESCRIPTION "Pathtracer, probably"
                 LANGUAGES CXX)

enable_testing()
include_directories(${CMAKE_CURRENT_LIST_DIR})
add_compile_options(-std=c++2a -fconcepts -Wall -Werror -Wextra -march=native -g -funsafe-math-optimizations)

add_subdirectory(src)
add_subdirectory(include)