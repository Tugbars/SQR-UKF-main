cmake_minimum_required(VERSION 3.16)
project(gemm_test_standalone C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

# ---- GEMM library (standalone, no parent dependencies) ----
set(GEMM_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/gemm/gemm.c
)

add_library(gemm_avx2 STATIC ${GEMM_SOURCES})

# Include only the gemm folder headers
target_include_directories(gemm_avx2 PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/gemm
)

# AVX2/FMA optimization
if (MSVC)
    target_compile_options(gemm_avx2 PRIVATE /arch:AVX2 /fp:fast)
else()
    target_compile_options(gemm_avx2 PRIVATE 
        -mavx2 -mfma -O3 
    )
endif()

# Feature flags (same as root project uses)
target_compile_definitions(gemm_avx2 PRIVATE
    LINALG_SIMD_ENABLE=1
)

# ---- Test executable ----
add_executable(test_gemm_simple test_main.c)
target_link_libraries(test_gemm_simple PRIVATE gemm_avx2)

# Link math library on Unix
if(UNIX)
    target_link_libraries(test_gemm_simple PRIVATE m)
endif()

message(STATUS "Building standalone GEMM test (isolated from root CMake)")