cmake_minimum_required(VERSION 3.18)
project(RunwayMC LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build options
option(USE_OPENMP "Enable OpenMP multi-threading" ON)
option(USE_CUDA "Enable CUDA acceleration" OFF)
option(BUILD_TESTS "Build Catch2 unit tests" ON)

if(USE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "all-major")
    endif()
endif()

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-O3 -march=native -Wall -Wextra -Wno-unused-parameter)
elseif(MSVC)
    add_compile_options(/O2 /W3)
endif()

# OpenMP
if(USE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        message(STATUS "OpenMP CXX found: ${OpenMP_CXX_VERSION}")
        add_compile_definitions(RUNWAY_HAS_OPENMP=1)
    else()
        message(WARNING "OpenMP requested but not found. Disabling OpenMP.")
        set(USE_OPENMP OFF)
    endif()
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Core sources
set(CORE_SOURCES
    src/core/instance_io.cpp
    src/core/validator.cpp
    src/core/objective.cpp
    src/gen/instance_gen.cpp
    src/gen/noise_cpu.cpp
    src/baselines/fcfs.cpp
    src/baselines/random_search.cpp
    src/baselines/brute_force.cpp
    src/serial/sa_serial.cpp
)

if(USE_OPENMP)
    list(APPEND CORE_SOURCES src/omp/sa_omp.cpp)
endif()

if(USE_CUDA)
    add_compile_definitions(RUNWAY_HAS_CUDA=1)
    set(CUDA_SOURCES
        src/cuda/sa_gpu.cu
    )
    add_library(runway_cuda STATIC ${CUDA_SOURCES})
    target_include_directories(runway_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda)
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(runway_cuda PUBLIC CUDA::curand CUDA::cudart)
endif()

add_library(runway_lib STATIC ${CORE_SOURCES})
target_include_directories(runway_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
if(USE_OPENMP)
    target_link_libraries(runway_lib PUBLIC OpenMP::OpenMP_CXX)
endif()
if(USE_CUDA)
    target_link_libraries(runway_lib PUBLIC runway_cuda)
endif()

# CLI application
add_executable(runway src/app/main.cpp)
target_link_libraries(runway PRIVATE runway_lib)

# Tests
if(BUILD_TESTS)
    enable_testing()
    include(FetchContent)
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG        v3.5.2
    )
    FetchContent_MakeAvailable(Catch2)

    set(TEST_SOURCES
        tests/test_main.cpp
        tests/test_worked_example.cpp
        tests/test_closure.cpp
        tests/test_validator.cpp
        tests/test_decoder.cpp
        tests/test_moves.cpp
        tests/test_baselines.cpp
    )

    if(USE_OPENMP)
        list(APPEND TEST_SOURCES tests/test_omp.cpp)
    endif()

    if(USE_CUDA)
        list(APPEND TEST_SOURCES tests/test_cuda.cpp)
    endif()

    add_executable(runway_tests ${TEST_SOURCES})
    target_link_libraries(runway_tests PRIVATE runway_lib Catch2::Catch2WithMain)
    add_test(NAME runway_tests COMMAND runway_tests)
endif()
