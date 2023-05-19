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

# Link libzip.
option(ENABLE_COMMONCRYPTO "Enable use of CommonCrypto" OFF)
option(ENABLE_GNUTLS "Enable use of GnuTLS" OFF)
option(ENABLE_MBEDTLS "Enable use of mbed TLS" OFF)
option(ENABLE_OPENSSL "Enable use of OpenSSL" OFF)
option(ENABLE_WINDOWS_CRYPTO "Enable use of Windows cryptography libraries" OFF)
option(ENABLE_BZIP2 "Enable use of BZip2" OFF)
option(ENABLE_LZMA "Enable use of LZMA" OFF)
option(ENABLE_ZSTD "Enable use of Zstandard" OFF)
add_subdirectory(external/libzip)
target_link_libraries(${AUTO_GRAPH_NAME} zip)

# Include rapidjson.
target_include_directories(${AUTO_GRAPH_NAME} PUBLIC external/rapidjson/include)