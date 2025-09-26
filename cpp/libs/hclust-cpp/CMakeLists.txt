add_library(fastcluster fastcluster.cpp fastcluster.h)

set_target_properties(fastcluster PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(fastcluster PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)