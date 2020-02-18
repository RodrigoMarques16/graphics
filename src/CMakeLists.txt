add_subdirectory(math)

add_executable(graphics main.cpp)
target_include_directories(graphics PUBLIC math)
target_link_libraries(graphics PUBLIC math)
