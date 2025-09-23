cmake_minimum_required(VERSION 3.24)
project(LinkNodeCommunity LANGUAGES CXX)

# Find pybind11 via the Python package declared in pyproject (build-system requires)
find_package(pybind11 CONFIG REQUIRED)

# If you keep C++ in a subdir:
add_subdirectory(cpp)