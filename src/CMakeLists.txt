set(AUTO_GRAPH_SOURCE_DIR ${PROJECT_SOURCE_DIR}/src)
file(GLOB AUTO_GRAPH_SOURCE
        "${AUTO_GRAPH_SOURCE_DIR}/*.h"
        "${AUTO_GRAPH_SOURCE_DIR}/*.cpp")

# Create shared library AutoGraph
set(AUTO_GRAPH_NAME auto_graph)
add_library(${AUTO_GRAPH_NAME} SHARED ${AUTO_GRAPH_SOURCE})
target_include_directories(${AUTO_GRAPH_NAME} PUBLIC .)

# Find and link Taichi runtime library.
set(CMAKE_MODULE_PATH ${AUTO_GRAPH_SOURCE_DIR}/cmake)
find_package(Taichi REQUIRED)
target_link_libraries(${AUTO_GRAPH_NAME} Taichi::Runtime)
target_include_directories(${AUTO_GRAPH_NAME} PUBLIC ${TAICHI_C_API_INSTALL_DIR}/include)

# Find and link libzip.
find_library(LIBZIP_LIBRARY NAMES zip)
target_link_libraries(${AUTO_GRAPH_NAME} ${LIBZIP_LIBRARY})

# Include rapidjson.
target_include_directories(${AUTO_GRAPH_NAME} PUBLIC external/rapidjson/include)