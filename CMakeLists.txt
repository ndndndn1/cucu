cmake_minimum_required(VERSION 3.18)
project(BareMetal-SGEMM LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA Toolkit (for Driver API)
find_package(CUDAToolkit REQUIRED)

# Detect GPU architecture
include(FindCUDA/select_compute_arch)
CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")

# Default to Ampere if detection fails
if(NOT CUDA_ARCH_LIST)
    set(CUDA_ARCH_LIST "80;86")
    message(STATUS "GPU detection failed, defaulting to SM 80/86 (Ampere)")
endif()

message(STATUS "Detected GPU architectures: ${CUDA_ARCH_LIST}")

# Compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")

# Generate PTX for JIT compilation
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptx")

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CUDAToolkit_INCLUDE_DIRS})

# ============================================================================
# PTX Kernel Generation (for JIT compilation)
# ============================================================================
set(KERNEL_SOURCES
    src/kernels/sgemm_naive.cu
    src/kernels/sgemm_coalesced.cu
    src/kernels/sgemm_tiled.cu
    src/kernels/sgemm_register_blocking.cu
    src/kernels/sgemm_double_buffer.cu
    src/kernels/sgemm_async_copy.cu
)

# Custom target to generate PTX files
foreach(KERNEL_SRC ${KERNEL_SOURCES})
    get_filename_component(KERNEL_NAME ${KERNEL_SRC} NAME_WE)
    set(PTX_OUTPUT ${CMAKE_SOURCE_DIR}/ptx/${KERNEL_NAME}.ptx)

    add_custom_command(
        OUTPUT ${PTX_OUTPUT}
        COMMAND ${CMAKE_CUDA_COMPILER}
                -ptx
                -o ${PTX_OUTPUT}
                -arch=sm_80
                -lineinfo
                --use_fast_math
                -I${CMAKE_SOURCE_DIR}/include
                ${CMAKE_SOURCE_DIR}/${KERNEL_SRC}
        DEPENDS ${CMAKE_SOURCE_DIR}/${KERNEL_SRC}
        COMMENT "Generating PTX for ${KERNEL_NAME}"
    )
    list(APPEND PTX_OUTPUTS ${PTX_OUTPUT})
endforeach()

add_custom_target(ptx_kernels ALL DEPENDS ${PTX_OUTPUTS})

# ============================================================================
# Main Library (Driver API based)
# ============================================================================
add_library(baremetal_sgemm STATIC
    src/driver/cuda_driver.cpp
    src/driver/kernel_launcher.cpp
    src/driver/memory_manager.cpp
)

target_link_libraries(baremetal_sgemm
    CUDA::cuda_driver
    CUDA::nvrtc
)

# ============================================================================
# Benchmark Executable
# ============================================================================
add_executable(sgemm_benchmark
    src/benchmark/main.cpp
    src/benchmark/benchmark_runner.cpp
    src/benchmark/cublas_reference.cu
)

target_link_libraries(sgemm_benchmark
    baremetal_sgemm
    CUDA::cuda_driver
    CUDA::cublas
    CUDA::cudart
)

add_dependencies(sgemm_benchmark ptx_kernels)

# ============================================================================
# Test Executable
# ============================================================================
add_executable(sgemm_test
    src/benchmark/test_correctness.cpp
)

target_link_libraries(sgemm_test
    baremetal_sgemm
    CUDA::cuda_driver
    CUDA::cublas
    CUDA::cudart
)

add_dependencies(sgemm_test ptx_kernels)

# ============================================================================
# Installation
# ============================================================================
install(TARGETS baremetal_sgemm DESTINATION lib)
install(TARGETS sgemm_benchmark sgemm_test DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY ptx/ DESTINATION share/baremetal-sgemm/ptx)
