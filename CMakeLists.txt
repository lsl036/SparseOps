cmake_minimum_required(VERSION 3.10)

# Set policy to use <PackageName>_ROOT variables for find_package
# This allows OpenMP_ROOT environment variable to work correctly
if(POLICY CMP0074)
    cmake_policy(SET CMP0074 NEW)
endif()

# Set a default C++ compiler if one wasn't provided
# Only set if user hasn't already specified one via command line
if(NOT DEFINED CMAKE_CXX_COMPILER)
  # Try to find Intel C++ compiler (icpx) first
  find_program(INTEL_CXX_COMPILER icpx)
  if(INTEL_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER "${INTEL_CXX_COMPILER}")
  else()
    # Fall back to g++ if Intel compiler is not available
    set(CMAKE_CXX_COMPILER "g++")
  endif()
endif()

set(CMAKE_CXX_FLAGS "-O3 -std=c++17")
# set(CMAKE_CXX_FLAGS "-O2 -g -std=c++17")

# Check if using Intel C++ compiler (icpx or icpc)
if(CMAKE_CXX_COMPILER MATCHES "icpx|icpc")      # == 如果是intel编译器
  # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qopt-zmm-usage=high -xHost")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -xHost -mavx512f -Wno-c++17-extensions -qopenmp")

  # 对于Intel编译器 使用 KMP_AFFINITY 设置线程亲和
  add_definitions(-DKMP_AFFINITY=compact,1,0,granularity=fine)
else()
  # For non-Intel compilers, use platform-specific settings
  if(APPLE)
    # macOS: use -march=native, OpenMP will be linked via FindOpenMP
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
  else()
    # Linux and other platforms: use -march=native and -fopenmp
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -fopenmp")
  endif()
endif()

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBSR_ANA ")

# Set the project name and version
project(SpOps VERSION 1.0 LANGUAGES CXX)

# Find OpenMP support (must be after project() command)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found: ${OpenMP_CXX_VERSION}")
else()
    message(WARNING "OpenMP not found. Some features may not work correctly.")
endif()

# Option to enable CSR5 support (requires AVX512)
option(ENABLE_CSR5 "Enable CSR5 SpMV support (requires AVX512)" OFF)

# Option to enable MKL support
option(ENABLE_MKL "Enable MKL baseline test support" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Specify the C++ standard
# set(CMAKE_CXX_STANDARD 11)
# set(CMAKE_CXX_STANDARD_REQUIRED True)

# make a directory named features for save matrix features
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/features)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/performance)

# Include directories for the library
include_directories(include)

# Add preprocessor definition for CSR5 if enabled
if(ENABLE_CSR5)
    add_definitions(-DENABLE_CSR5)
    message(STATUS "CSR5 support: ENABLED (requires AVX512)")
else()
    message(STATUS "CSR5 support: DISABLED")
endif()

# Add preprocessor definition for MKL if enabled
if(ENABLE_MKL)
    add_definitions(-DENABLE_MKL)
    message(STATUS "MKL support: ENABLED")
else()
    message(STATUS "MKL support: DISABLED")
endif()

# Gather the library sources
file(GLOB LIB_SOURCES "src/*.cpp")
file(GLOB LIB_UTILS   "utils/*.cpp")

# Conditionally exclude CSR5 source files
if(NOT ENABLE_CSR5)
    list(FILTER LIB_SOURCES EXCLUDE REGEX ".*spmv_csr5\\.cpp$")
    list(FILTER LIB_UTILS EXCLUDE REGEX ".*test_spmv_csr5\\.cpp$")
endif()

# Create the static library
add_library(SpOPs_static STATIC ${LIB_SOURCES} ${LIB_UTILS})

# Create the shared library
add_library(SpOPs_shared SHARED ${LIB_SOURCES} ${LIB_UTILS})

# Link OpenMP to libraries if available
if(OpenMP_CXX_FOUND)
    target_link_libraries(SpOPs_static PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(SpOPs_shared PUBLIC OpenMP::OpenMP_CXX)
endif()

# It's common practice to output libraries with the same name, with different extensions.
# CMake automatically appends the appropriate extension for static (.a) and shared (.so or .dll) libraries.
set_target_properties(SpOPs_shared PROPERTIES OUTPUT_NAME SpOPs)
set_target_properties(SpOPs_static PROPERTIES OUTPUT_NAME SpOPs)

# 定义源脚本目录和目标脚本目录变量
set(SOURCE_SCRIPT_DIR ${CMAKE_SOURCE_DIR}/script)
set(DESTINATION_SCRIPT_DIR ${CMAKE_BINARY_DIR})

add_custom_command(TARGET SpOPs_static POST_BUILD
                  COMMAND ${CMAKE_COMMAND} -E copy_directory
                          ${SOURCE_SCRIPT_DIR} ${DESTINATION_SCRIPT_DIR}
                  COMMENT "Copying scripts to build directory")

# Gather the test sources
file(GLOB TEST_SOURCES "test/*.cpp")

# Conditionally exclude CSR5 test files
if(NOT ENABLE_CSR5)
    list(FILTER TEST_SOURCES EXCLUDE REGEX ".*test_csr5_format\\.cpp$")
    list(FILTER TEST_SOURCES EXCLUDE REGEX ".*benchmark_spmv_csr5\\.cpp$")
endif()

# Conditionally exclude MKL test files
if(NOT ENABLE_MKL)
    list(FILTER TEST_SOURCES EXCLUDE REGEX ".*baseline_mkl_csr\\.cpp$")
endif()

# Allow the test executables to find the library headers
include_directories(${PROJECT_SOURCE_DIR}/include)

# Create an executable for each test source
foreach(test_src ${TEST_SOURCES})
  get_filename_component(test_name ${test_src} NAME_WE)
  add_executable(${test_name} ${test_src})
  # Add compiler options here if they are specific to the test executables
  target_compile_options(${test_name} PRIVATE -O3 -Wall)

  # target_link_libraries(${test_name} SpOPs_shared stdc++)
  
  # Check if the current source file is the baseline MKL test
  if(${test_name} STREQUAL "baseline_mkl_csr")
    # Link with MKL. The method to link may depend on your setup.
    # This example assumes you're using the Intel Compiler or you have MKL setup to integrate with CMake.
    if(ENABLE_MKL)
      find_package(MKL CONFIG REQUIRED)
      target_link_libraries(${test_name} MKL::MKL SpOPs_shared stdc++)
    else()
      message(WARNING "baseline_mkl_csr test requires ENABLE_MKL=ON, skipping...")
      # This should not happen as we already filtered it out, but just in case
      target_link_libraries(${test_name} SpOPs_shared stdc++)
    endif()
  else()
    target_link_libraries(${test_name} SpOPs_shared stdc++)
    # Link OpenMP to test executables if available
    if(OpenMP_CXX_FOUND)
        target_link_libraries(${test_name} OpenMP::OpenMP_CXX)
    endif()
  endif()

endforeach(test_src)

# Message to print out the compiler being used
message(STATUS "Using CXX compiler: ${CMAKE_CXX_COMPILER}")
# Also directly access the CMAKE_CXX_FLAGS variable
message(STATUS "CXX Flags: ${CMAKE_CXX_FLAGS}")