cmake_minimum_required(VERSION 3.16)
project(qr_test_suite C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

message(STATUS "========================================")
message(STATUS "Configuring QR Test Suite")
message(STATUS "========================================")

# ============================================================================
# COMPILER FLAGS
# ============================================================================

if(MSVC)
    set(SIMD_FLAGS /arch:AVX2 /fp:fast)
    set(WARNING_FLAGS /W4)
else()
    set(SIMD_FLAGS -mavx2 -mfma -msse4.2)
    set(OPT_FLAGS -O3 -march=native)
    set(WARNING_FLAGS -Wall -Wextra -Wpedantic -Wno-unused-parameter)
endif()

# ============================================================================
# GEMM LIBRARY (Reuse or Build)
# ============================================================================

set(GEMM_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../src/gemm_2)
set(GEMM_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/../includes)

# Check if gemm_lib already exists (from gemm_test subdirectory)
if(TARGET gemm_lib)
    message(STATUS "Reusing existing gemm_lib from gemm_test")
else()
    message(STATUS "Building GEMM library for QR tests...")
    
    file(GLOB GEMM_SOURCES 
        ${GEMM_DIR}/gemm.c
        ${GEMM_DIR}/gemm_small.c
        ${GEMM_DIR}/gemm_large.c
        ${GEMM_DIR}/gemm_planning.c
        ${GEMM_DIR}/gemm_kernels_avx2.c
        ${GEMM_DIR}/gemm_utils.c
    )
    
    add_library(gemm_lib STATIC ${GEMM_SOURCES})
    
    target_include_directories(gemm_lib PUBLIC 
        ${GEMM_DIR}
        ${GEMM_INCLUDES}
    )
    
    if(MSVC)
        target_compile_options(gemm_lib PRIVATE ${SIMD_FLAGS})
    else()
        target_compile_options(gemm_lib PRIVATE 
            ${SIMD_FLAGS} 
            ${OPT_FLAGS} 
            ${WARNING_FLAGS}
        )
    endif()
    
    target_compile_definitions(gemm_lib PUBLIC LINALG_SIMD_ENABLE=1)
    
    message(STATUS "  GEMM sources: ${GEMM_SOURCES}")
endif()

# ============================================================================
# QR LIBRARY
# ============================================================================

message(STATUS "Building QR library...")

set(QR_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../src/qr)

# Only qr.c exists
set(QR_SOURCES 
    ${QR_DIR}/qr.c
)

add_library(qr_lib STATIC ${QR_SOURCES})

target_include_directories(qr_lib PUBLIC 
    ${QR_DIR}
    ${GEMM_DIR}
    ${GEMM_INCLUDES}
)

if(MSVC)
    target_compile_options(qr_lib PRIVATE ${SIMD_FLAGS})
else()
    target_compile_options(qr_lib PRIVATE 
        ${SIMD_FLAGS} 
        ${OPT_FLAGS} 
        ${WARNING_FLAGS}
    )
endif()

target_compile_definitions(qr_lib PUBLIC LINALG_SIMD_ENABLE=1)

target_link_libraries(qr_lib PUBLIC gemm_lib)

message(STATUS "  QR sources: ${QR_SOURCES}")

# ============================================================================
# TEST COMMON INFRASTRUCTURE
# ============================================================================

set(TEST_COMMON
    ${CMAKE_CURRENT_SOURCE_DIR}/test_common.h
)

# ============================================================================
# INDIVIDUAL TEST EXECUTABLES (STANDALONE MODE)
# ============================================================================

message(STATUS "Building QR test executables...")

# Helper function to create standalone test
function(add_qr_standalone_test test_name source_file)
    add_executable(${test_name} ${source_file})
    
    target_include_directories(${test_name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${QR_DIR}
        ${GEMM_DIR}
        ${GEMM_INCLUDES}
    )
    
    target_link_libraries(${test_name} PRIVATE qr_lib gemm_lib)
    
    if(MSVC)
        target_compile_options(${test_name} PRIVATE ${SIMD_FLAGS})
    else()
        target_compile_options(${test_name} PRIVATE ${SIMD_FLAGS} ${WARNING_FLAGS})
    endif()
    
    target_compile_definitions(${test_name} PRIVATE STANDALONE=1)
    
    if(UNIX OR MINGW)
        target_link_libraries(${test_name} PRIVATE m)
    endif()
endfunction()

# ============================================================================
# ACTIVE TESTS (Currently Implemented)
# ============================================================================

# Correctness tests (ACTIVE)
add_qr_standalone_test(test_qr_correctness test_correctness.c)

# ============================================================================
# DISABLED TESTS (Not Yet Implemented)
# ============================================================================
# Uncomment when ready to enable:

# # Workspace tests
# add_qr_standalone_test(test_qr_workspace test_workspace.c)

# # Benchmark tests
# add_qr_standalone_test(test_qr_benchmark test_benchmark.c)

# ============================================================================
# CTEST INTEGRATION
# ============================================================================

enable_testing()

# Register active test suites
add_test(NAME QR_Correctness COMMAND test_qr_correctness)
set_tests_properties(QR_Correctness PROPERTIES 
    TIMEOUT 120
    LABELS "correctness"
)

# ============================================================================
# DISABLED CTEST ENTRIES
# ============================================================================
# Uncomment when ready to enable:

# add_test(NAME QR_Workspace COMMAND test_qr_workspace)
# set_tests_properties(QR_Workspace PROPERTIES 
#     TIMEOUT 60
#     LABELS "workspace"
# )

# add_test(NAME QR_Benchmark COMMAND test_qr_benchmark)
# set_tests_properties(QR_Benchmark PROPERTIES 
#     TIMEOUT 300
#     LABELS "benchmark"
# )

# ============================================================================
# CUSTOM TARGETS FOR CONVENIENCE
# ============================================================================

# Run all active tests via CTest
add_custom_target(run_qr_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS test_qr_correctness
    COMMENT "Running all QR tests via CTest..."
)

# Run correctness tests
add_custom_target(run_qr_correctness
    COMMAND test_qr_correctness
    DEPENDS test_qr_correctness
    COMMENT "Running QR correctness tests (reconstruction, orthogonality, triangular)..."
)

# Quick validation (same as correctness for now)
add_custom_target(run_qr_quick
    COMMAND test_qr_correctness
    DEPENDS test_qr_correctness
    COMMENT "Running quick QR validation..."
)

# ============================================================================
# DISABLED CUSTOM TARGETS
# ============================================================================
# Uncomment when ready to enable:

# # Run workspace tests
# add_custom_target(run_qr_workspace
#     COMMAND test_qr_workspace
#     DEPENDS test_qr_workspace
#     COMMENT "Running QR workspace tests (allocation, reuse, memory)..."
# )

# # Run benchmark
# add_custom_target(run_qr_benchmark
#     COMMAND test_qr_benchmark
#     DEPENDS test_qr_benchmark
#     COMMENT "Running QR performance benchmark..."
# )

# ============================================================================
# SUMMARY
# ============================================================================

message(STATUS "========================================")
message(STATUS "QR Test Configuration Summary:")
message(STATUS "  Compiler: ${CMAKE_C_COMPILER_ID}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  SIMD flags: ${SIMD_FLAGS}")
message(STATUS "  GEMM dir: ${GEMM_DIR}")
message(STATUS "  QR dir: ${QR_DIR}")
message(STATUS "  Libraries: gemm_lib (${TARGET_EXISTS}), qr_lib")
message(STATUS "")
message(STATUS "  ACTIVE Test Executables:")
message(STATUS "    ✓ test_qr_correctness")
message(STATUS "")
message(STATUS "  DISABLED Test Executables (not yet implemented):")
message(STATUS "    ✗ test_qr_workspace")
message(STATUS "    ✗ test_qr_benchmark")
message(STATUS "")
message(STATUS "  Available Custom Targets:")
message(STATUS "    - run_qr_tests        (all active tests via CTest)")
message(STATUS "    - run_qr_correctness  (correctness suite)")
message(STATUS "    - run_qr_quick        (quick validation)")
message(STATUS "")
message(STATUS "  To enable disabled tests:")
message(STATUS "    1. Implement test_workspace.c and test_benchmark.c")
message(STATUS "    2. Uncomment respective sections in CMakeLists.txt")
message(STATUS "========================================")