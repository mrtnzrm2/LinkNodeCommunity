cmake_minimum_required(VERSION 3.24)

project(node_community_hierarchy_cpp)

# Prefer using an existing pybind11 (found by top-level) but allow standalone.
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
  include(FetchContent)
  FetchContent_Declare(
      pybind11
      GIT_REPOSITORY https://github.com/pybind/pybind11
      GIT_TAG        v2.13.6
  )
  FetchContent_MakeAvailable(pybind11)
endif()

# Use shared fastcluster; if not present (standalone), add it from ../libs.
if(NOT TARGET fastcluster)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../libs/hclust-cpp)
endif()

# Use pybind11's dedicated macro for creating a Python module.
pybind11_add_module(${PROJECT_NAME} src/main.cpp)

# Link against shared fastcluster and pybind11
target_link_libraries(${PROJECT_NAME} PRIVATE fastcluster pybind11::pybind11)

# Add the include directory for the mcqd library.
include_directories(src/mcqd)

# Install the built Python extension when invoked from the top level
install(TARGETS ${PROJECT_NAME} DESTINATION .)
