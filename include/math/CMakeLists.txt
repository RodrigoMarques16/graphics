add_library(math vec3.hpp vec3.cpp unit3.hpp unit3.cpp rgb.hpp onb.hpp)
target_include_directories(math PUBLIC .)
target_include_directories(math PUBLIC ..)