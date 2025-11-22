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
option(BUILD_CHOLUPDATE_TESTS "Build Cholesky Update test suite" ON)
option(BUILD_LUP_TESTS "Build LUP Factorization test suite" ON)
option(BUILD_INV_TESTS "Build Matrix Inversion test suite" ON)

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

if(BUILD_CHOLUPDATE_TESTS)
    message(STATUS ">>> Adding Cholesky Update tests subdirectory")
    add_subdirectory(cholesky_test)
else()
    message(STATUS ">>> Cholesky Update tests DISABLED")
endif()

if(BUILD_LUP_TESTS)
    message(STATUS ">>> Adding LUP Factorization tests subdirectory")
    add_subdirectory(lup_test)          # ← Use lup_test (matches your directory)
else()
    message(STATUS ">>> LUP Factorization tests DISABLED")
endif()

if(BUILD_INV_TESTS)
    message(STATUS ">>> Adding Matrix Inversion tests subdirectory")
    add_subdirectory(test_inv)          # ← Keep test_inv (already exists)
else()
    message(STATUS ">>> Matrix Inversion tests DISABLED")
endif()

# ============================================================================
# SUMMARY
# ============================================================================

message(STATUS "========================================")
message(STATUS "Build configuration:")
message(STATUS "  GEMM tests:       ${BUILD_GEMM_TESTS}")
message(STATUS "  QR tests:         ${BUILD_QR_TESTS}")
message(STATUS "  CholUpdate tests: ${BUILD_CHOLUPDATE_TESTS}")
message(STATUS "  LUP tests:        ${BUILD_LUP_TESTS}")
message(STATUS "  INV tests:        ${BUILD_INV_TESTS}")
message(STATUS "========================================")
