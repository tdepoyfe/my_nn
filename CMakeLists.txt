cmake_minimum_required(VERSION 3.14)

# Project name
project(my_nn VERSION 1.0)

# Require at least C++17
set(CMAKE_CXX_STANDARD 17)

include_directories(src)
include_directories(lib)

# Add source directory
add_subdirectory(src)

# Add test directory
add_subdirectory(tests)
