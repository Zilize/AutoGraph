cmake_minimum_required(VERSION 3.17)

set(AUTO_GRAPH_PROJECT_NAME AutoGraph)
project(${AUTO_GRAPH_PROJECT_NAME} LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Declare executable target.
set(AUTO_GRAPH_SOURCE_DIR ${PROJECT_SOURCE_DIR}/src)
file(GLOB AUTO_GRAPH_SOURCE
     "${AUTO_GRAPH_SOURCE_DIR}/*.h"
     "${AUTO_GRAPH_SOURCE_DIR}/*.cpp")
add_executable(${AUTO_GRAPH_PROJECT_NAME} AUTO_GRAPH_SOURCE)
target_include_directories(${AUTO_GRAPH_PROJECT_NAME} PUBLIC ${TAICHI_C_API_INSTALL_DIR}/include)

# Find and link Taichi runtime library.
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Taichi REQUIRED)
target_link_libraries(${AUTO_GRAPH_PROJECT_NAME} Taichi::Runtime)

# Find and link libzip.
find_library(LIBZIP_LIBRARY NAMES zip)
target_link_libraries(${AUTO_GRAPH_PROJECT_NAME} ${LIBZIP_LIBRARY})