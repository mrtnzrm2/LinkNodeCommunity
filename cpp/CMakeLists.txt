## Aggregate C++ build for submodules

# Define the shared hclust-cpp library once
add_subdirectory(libs/hclust-cpp)

# Add subprojects that build Python extensions
add_subdirectory(link_hierarchy_statistics_cpp)
add_subdirectory(node_community_hierarchy_cpp)
add_subdirectory(utlis_cpp)
add_subdirectory(linksim_cpp)
