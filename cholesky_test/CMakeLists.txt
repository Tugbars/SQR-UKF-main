cmake_minimum_required(VERSION 3.16)
project(cholupdate_test_suite C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

message(STATUS "========================================")
message(STATUS "Configuring Cholesky Update Test Suite")
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

# Check if gemm_lib already exists (from gemm_test or qr_tests subdirectory)
if(TARGET gemm_lib)
    message(STATUS "Reusing existing gemm_lib from previous build")
else()
    message(STATUS "Building GEMM library for Cholesky tests...")
    
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
# QR LIBRARY (Reuse if exists - cholupdate depends on QR)
# ============================================================================

set(QR_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../src/qr)

if(TARGET qr_lib)
    message(STATUS "Reusing existing qr_lib from previous build")
else()
    message(STATUS "Building QR library for Cholesky tests...")
    
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
endif()

# ============================================================================
# CHOLUPDATE LIBRARY
# ============================================================================

message(STATUS "Building Cholesky Update library...")

set(CHOLUPDATE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../src)

set(CHOLUPDATE_SOURCES 
    ${CHOLUPDATE_DIR}/cholupdate.c
    ${CHOLUPDATE_DIR}/tran_pack.c
)

add_library(cholupdate_lib STATIC ${CHOLUPDATE_SOURCES})

target_include_directories(cholupdate_lib PUBLIC 
    ${CHOLUPDATE_DIR}
    ${QR_DIR}
    ${GEMM_DIR}
    ${GEMM_INCLUDES}
)

if(MSVC)
    target_compile_options(cholupdate_lib PRIVATE ${SIMD_FLAGS})
else()
    target_compile_options(cholupdate_lib PRIVATE 
        ${SIMD_FLAGS} 
        ${OPT_FLAGS} 
        ${WARNING_FLAGS}
    )
endif()

target_compile_definitions(cholupdate_lib PUBLIC LINALG_SIMD_ENABLE=1)

target_link_libraries(cholupdate_lib PUBLIC qr_lib gemm_lib)

message(STATUS "  Cholesky Update sources: ${CHOLUPDATE_SOURCES}")

# ============================================================================
# TEST COMMON INFRASTRUCTURE
# ============================================================================

set(TEST_COMMON
    ${CMAKE_CURRENT_SOURCE_DIR}/test_common.h
)

# ============================================================================
# INDIVIDUAL TEST EXECUTABLES (STANDALONE MODE)
# ============================================================================

message(STATUS "Building Cholesky Update test executables...")

# Helper function to create standalone test
function(add_cholupdate_standalone_test test_name source_file)
    add_executable(${test_name} ${source_file})
    
    target_include_directories(${test_name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CHOLUPDATE_DIR}
        ${QR_DIR}
        ${GEMM_DIR}
        ${GEMM_INCLUDES}
    )
    
    target_link_libraries(${test_name} PRIVATE cholupdate_lib qr_lib gemm_lib)
    
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
# ACTIVE TESTS
# ============================================================================

add_cholupdate_standalone_test(test_cholupdate test_cholupdate.c)

# ============================================================================
# CTEST INTEGRATION
# ============================================================================

enable_testing()

# Register test suite
add_test(NAME CholUpdate_Suite COMMAND test_cholupdate)
set_tests_properties(CholUpdate_Suite PROPERTIES 
    TIMEOUT 300
    LABELS "correctness;cholupdate"
)

# ============================================================================
# CUSTOM TARGETS FOR CONVENIENCE
# ============================================================================

# Run all tests via CTest
add_custom_target(run_cholupdate_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS test_cholupdate
    COMMENT "Running all Cholesky Update tests via CTest..."
)

# Run tests directly
add_custom_target(run_cholupdate
    COMMAND test_cholupdate
    DEPENDS test_cholupdate
    COMMENT "Running Cholesky Update tests (rank-1, rank-k, QR, workspace)..."
)

# Quick validation (same as full suite for now)
add_custom_target(run_cholupdate_quick
    COMMAND test_cholupdate
    DEPENDS test_cholupdate
    COMMENT "Running quick Cholesky Update validation..."
)

# ============================================================================
# SUMMARY
# ============================================================================

message(STATUS "========================================")
message(STATUS "Cholesky Update Test Configuration Summary:")
message(STATUS "  Compiler: ${CMAKE_C_COMPILER_ID}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  SIMD flags: ${SIMD_FLAGS}")
message(STATUS "  GEMM dir: ${GEMM_DIR}")
message(STATUS "  QR dir: ${QR_DIR}")
message(STATUS "  Cholupdate dir: ${CHOLUPDATE_DIR}")
message(STATUS "  Libraries: gemm_lib, qr_lib, cholupdate_lib")
message(STATUS "========================================")