add_library(math vec3.hpp vec3.cpp unit3.cpp unit3.hpp onb.hpp onb.cpp ray.hpp ray.cpp hit.hpp vec2.hpp)
target_include_directories(math INTERFACE math)