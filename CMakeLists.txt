cmake_minimum_required(VERSION 3.16)
project(SQR_UKF_Root C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

message(STATUS "=== SQR-UKF Root Configuration ===")

# ============================================================================
# BUILD OPTIONS (Top-Level Toggles)
# ============================================================================

option(BUILD_GEMM_TESTS "Build GEMM test suite" ON)
option(BUILD_QR_TESTS "Build QR test suite" ON)

# ============================================================================
# CONDITIONAL SUBDIRECTORIES
# ============================================================================

if(BUILD_GEMM_TESTS)
    message(STATUS ">>> Adding GEMM tests subdirectory")
    add_subdirectory(gemm_test)
else()
    message(STATUS ">>> GEMM tests DISABLED")
endif()

if(BUILD_QR_TESTS)
    message(STATUS ">>> Adding QR tests subdirectory")
    add_subdirectory(qr_tests)
else()
    message(STATUS ">>> QR tests DISABLED")
endif()

# ============================================================================
# SUMMARY
# ============================================================================

message(STATUS "========================================")
message(STATUS "Build configuration:")
message(STATUS "  GEMM tests: ${BUILD_GEMM_TESTS}")
message(STATUS "  QR tests:   ${BUILD_QR_TESTS}")
message(STATUS "========================================")