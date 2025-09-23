# Install script for directory: /Users/jmrtnza/Documents/Documents - Jorge’s MacBook Air/Work/Research/LINKPROJECT/LinkNodeCommunity/cpp/link_hierarchy_statistics_cpp

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/var/folders/b9/p9l8475x7x9_m4v1k1xflxzm0000gn/T/tmp0jkb76wg/wheel/platlib")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set path to fallback-tool for dependency-resolution.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/." TYPE MODULE FILES "/Users/jmrtnza/Documents/Documents - Jorge’s MacBook Air/Work/Research/LINKPROJECT/LinkNodeCommunity/build/cp312-cp312-macosx_15_0_arm64/cpp/link_hierarchy_statistics_cpp/link_hierarchy_statistics_cpp.cpython-312-darwin.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/./link_hierarchy_statistics_cpp.cpython-312-darwin.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/./link_hierarchy_statistics_cpp.cpython-312-darwin.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" -x "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/./link_hierarchy_statistics_cpp.cpython-312-darwin.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/Users/jmrtnza/Documents/Documents - Jorge’s MacBook Air/Work/Research/LINKPROJECT/LinkNodeCommunity/build/cp312-cp312-macosx_15_0_arm64/cpp/link_hierarchy_statistics_cpp/CMakeFiles/link_hierarchy_statistics_cpp.dir/install-cxx-module-bmi-Release.cmake" OPTIONAL)
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "/Users/jmrtnza/Documents/Documents - Jorge’s MacBook Air/Work/Research/LINKPROJECT/LinkNodeCommunity/build/cp312-cp312-macosx_15_0_arm64/cpp/link_hierarchy_statistics_cpp/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
