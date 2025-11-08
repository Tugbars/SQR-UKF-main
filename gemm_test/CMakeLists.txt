cmake_minimum_required(VERSION 3.16)
project(gemm_test C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

message(STATUS "Building GEMM test suite")

# ---- GEMM library ----
set(GEMM_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/gemm/gemm.c
)

set(GEMM_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/gemm/gemm_helper.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/gemm/gemm.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/gemm/gemm_simd_ops.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/gemm/gemm_kernels_avx2.h
)

add_library(gemm_avx2 STATIC ${GEMM_SOURCES})

target_include_directories(gemm_avx2 PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/gemm
)

# AVX2/FMA optimization
if (MSVC)
    target_compile_options(gemm_avx2 PRIVATE /arch:AVX2 /fp:fast)
else()
    target_compile_options(gemm_avx2 PRIVATE 
        -mavx2 -mfma -O3 -ffast-math
        -Wall -Wextra -Wpedantic -Wno-unused-parameter
    )
endif()

# Feature flags
target_compile_definitions(gemm_avx2 PRIVATE
    LINALG_SIMD_ENABLE=1
)

# ---- Test executable ----
add_executable(test_gemm 
    test_main.c 
    test_kernels.c 
    test_reference.c
    test_benchmark.c
)

target_link_libraries(test_gemm PRIVATE gemm_avx2)

# ⚠️ CRITICAL: test_kernels.c uses AVX2 intrinsics, so it needs AVX2 flags too!
if (MSVC)
    target_compile_options(test_gemm PRIVATE /arch:AVX2)
else()
    target_compile_options(test_gemm PRIVATE -mavx2 -mfma)
endif()

# Link math library on Unix/MinGW
if(UNIX OR MINGW)
    target_link_libraries(test_gemm PRIVATE m)
endif()

message(STATUS "GEMM targets: gemm_avx2, test_gemm")