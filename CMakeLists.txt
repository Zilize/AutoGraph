cmake_minimum_required(VERSION 3.17)

set(DEMO_PROJ_NAME Demo)
project(${DEMO_PROJ_NAME} LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(src)

# Declare executable target.
add_executable(${DEMO_PROJ_NAME} main.cpp)
target_link_libraries(${DEMO_PROJ_NAME} PUBLIC auto_graph)
