add_library(shapes triangle.hpp sphere.hpp)
target_link_libraries(shapes PUBLIC math)
target_include_directories(shapes INTERFACE ../math)