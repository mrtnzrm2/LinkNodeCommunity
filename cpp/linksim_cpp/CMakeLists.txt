cmake_minimum_required(VERSION 3.24)
project(linksim_cpp)

# Prefer an existing pybind11 (provided by top-level) but allow standalone builds.
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

pybind11_add_module(${PROJECT_NAME} src/main.cpp)

# Install the built Python extension when invoked from the top level
install(TARGETS ${PROJECT_NAME} DESTINATION .)
