cmake_minimum_required(VERSION 3.16)
project(qr_test C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

message(STATUS "Building QR test suite")

# ============================================
# Build GEMM library (needed by mul())
# ============================================
add_library(gemm_avx2 STATIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/gemm/gemm.c
)

target_include_directories(gemm_avx2 PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/gemm
    ${CMAKE_CURRENT_SOURCE_DIR}/../includes
)

if (MSVC)
    target_compile_options(gemm_avx2 PRIVATE /arch:AVX2 /fp:fast)
else()
    target_compile_options(gemm_avx2 PRIVATE 
        -mavx2 -mfma -O3 
        -Wall -Wextra -Wpedantic -Wno-unused-parameter
    )
endif()

target_compile_definitions(gemm_avx2 PRIVATE LINALG_SIMD_ENABLE=1)

# ============================================
# QR library with dependencies
# ============================================
set(QR_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/qr/qr.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/inv.c           # ← ADD for inv()
)

set(QR_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/qr/qr.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/qr/qr_avx2_kernels.h
)

add_library(qr_avx2 STATIC ${QR_SOURCES})

target_include_directories(qr_avx2 PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/qr
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/gemm                # ← ADD for mul()
    ${CMAKE_CURRENT_SOURCE_DIR}/../includes
)

# AVX2/FMA optimization
if (MSVC)
    target_compile_options(qr_avx2 PRIVATE /arch:AVX2 /fp:fast)
else()
    target_compile_options(qr_avx2 PRIVATE 
        -mavx2 -mfma -O3 -ffast-math
        -Wall -Wextra -Wpedantic -Wno-unused-parameter
    )
endif()

# Feature flags
target_compile_definitions(qr_avx2 PRIVATE
    LINALG_SIMD_ENABLE=1
)

# Link GEMM to QR (provides mul())
target_link_libraries(qr_avx2 PUBLIC gemm_avx2)

# ============================================
# Test executable
# ============================================
add_executable(test_qr 
    test_main.c 
    test_workspace.c 
    test_correctness.c
    test_cpqr.c
    test_benchmark.c
)

target_link_libraries(test_qr PRIVATE qr_avx2)

# Test code uses AVX2 intrinsics
if (MSVC)
    target_compile_options(test_qr PRIVATE /arch:AVX2)
else()
    target_compile_options(test_qr PRIVATE -mavx2 -mfma)
endif()

# Link math library on Unix/MinGW
if(UNIX OR MINGW)
    target_link_libraries(test_qr PRIVATE m)
endif()

message(STATUS "QR targets: gemm_avx2, qr_avx2, test_qr")