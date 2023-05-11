cmake_minimum_required(VERSION 3.17)

set(DEMO_PROJ_NAME Demo)
project(${DEMO_PROJ_NAME} LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(src)

# Declare executable target.
add_executable(${DEMO_PROJ_NAME} main.cpp)

# Find and link Taichi runtime library.
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Taichi REQUIRED)

target_link_libraries(${DEMO_PROJ_NAME} PUBLIC Taichi::Runtime auto_graph)
target_include_directories(${DEMO_PROJ_NAME} PUBLIC ${TAICHI_C_API_INSTALL_DIR}/include)