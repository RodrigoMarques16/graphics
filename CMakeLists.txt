cmake_minimum_required(VERSION 3.0.0)

project(graphics VERSION 0.1.0
                 DESCRIPTION "Pathtracer, probably"
                 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
enable_testing()
add_compile_options(-Wall -Wextra -march=native -g)
add_subdirectory(src)