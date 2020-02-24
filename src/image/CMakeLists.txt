add_library(image image.hpp image.cpp rgb.hpp rgb.cpp scene.hpp scene.cpp)
target_include_directories(image INTERFACE shapes)
target_link_libraries(image INTERFACE shapes)