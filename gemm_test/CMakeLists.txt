cmake_minimum_required(VERSION 3.16)
project(gemm_test_suite C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

message(STATUS "Building GEMM Library - Full Test Suite")

#===============================================================================
# VALIDATION LEVEL OPTION
#===============================================================================

option(GEMM_ENABLE_VALIDATION "Enable runtime validation (debug builds)" ON)

if(CMAKE_BUILD_TYPE MATCHES Debug)
    set(GEMM_VALIDATION_LEVEL 2 CACHE STRING "Validation level (0=none, 1=basic, 2=full)")
else()
    set(GEMM_VALIDATION_LEVEL 0 CACHE STRING "Validation level (0=none, 1=basic, 2=full)")
endif()

message(STATUS "Validation level: ${GEMM_VALIDATION_LEVEL}")

#===============================================================================
# Source Directory
#===============================================================================

set(GEMM_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../src/gemm_2)
set(TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})

#===============================================================================
# Compiler Flags
#===============================================================================

if(MSVC)
    set(GEMM_COMPILE_FLAGS /arch:AVX2 /W4 /O2)
    set(GEMM_LINK_FLAGS)
else()
    set(GEMM_COMPILE_FLAGS -march=native -mavx2 -mfma -O2 -Wall -Wextra)
    set(GEMM_LINK_FLAGS -lm)
endif()

# Add validation level definition
if(GEMM_VALIDATION_LEVEL GREATER 0)
    list(APPEND GEMM_COMPILE_FLAGS -DGEMM_VALIDATION_LEVEL=${GEMM_VALIDATION_LEVEL})
    if(NOT MSVC)
        # Add debug symbols for validation
        list(APPEND GEMM_COMPILE_FLAGS -g)
    endif()
endif()

#===============================================================================
# Core GEMM Library Sources
#===============================================================================

set(GEMM_CORE_SOURCES
    ${GEMM_SRC_DIR}/gemm_static.c
    ${GEMM_SRC_DIR}/gemm_planning.c
    ${GEMM_SRC_DIR}/gemm_small.c
    ${GEMM_SRC_DIR}/gemm_large.c
)

set(GEMM_CORE_HEADERS
    ${GEMM_SRC_DIR}/gemm.h
    ${GEMM_SRC_DIR}/gemm_static.h
    ${GEMM_SRC_DIR}/gemm_planning.h
    ${GEMM_SRC_DIR}/gemm_small.h
    ${GEMM_SRC_DIR}/gemm_utils.h
    ${GEMM_SRC_DIR}/gemm_simd_ops.h
    ${GEMM_SRC_DIR}/gemm_validation.h
    ${GEMM_SRC_DIR}/gemm_kernels_avx2.h
)

#===============================================================================
# GEMM Library Target
#===============================================================================

add_library(gemm_lib STATIC 
    ${GEMM_CORE_SOURCES}
    ${GEMM_CORE_HEADERS}
)

target_include_directories(gemm_lib PUBLIC 
    ${GEMM_SRC_DIR}
)

target_compile_options(gemm_lib PRIVATE ${GEMM_COMPILE_FLAGS})

#===============================================================================
# Test Common Header
#===============================================================================

set(TEST_COMMON_HEADER ${TEST_DIR}/test_common.h)

#===============================================================================
# Test Suite 1: Small Kernels (Tier 1)
#===============================================================================

add_executable(test_gemm_small 
    ${TEST_DIR}/test_gemm_small.c
    ${TEST_COMMON_HEADER}
)

target_link_libraries(test_gemm_small PRIVATE gemm_lib)
target_compile_definitions(test_gemm_small PRIVATE STANDALONE)
target_compile_options(test_gemm_small PRIVATE ${GEMM_COMPILE_FLAGS})

if(WIN32 AND MINGW)
    target_link_options(test_gemm_small PRIVATE -mconsole)
endif()

if(UNIX OR MINGW)
    target_link_libraries(test_gemm_small PRIVATE m)
endif()

#===============================================================================
# Test Suite 2: Planning Module
#===============================================================================

add_executable(test_gemm_planning 
    ${TEST_DIR}/test_planning.c
    ${TEST_COMMON_HEADER}
)

target_link_libraries(test_gemm_planning PRIVATE gemm_lib)
target_compile_definitions(test_gemm_planning PRIVATE STANDALONE)
target_compile_options(test_gemm_planning PRIVATE ${GEMM_COMPILE_FLAGS})

if(WIN32 AND MINGW)
    target_link_options(test_gemm_planning PRIVATE -mconsole)
endif()

if(UNIX OR MINGW)
    target_link_libraries(test_gemm_planning PRIVATE m)
endif()

#===============================================================================
# Test Suite 3: Individual Kernel Tests
#===============================================================================

add_executable(test_gemm_large 
    ${TEST_DIR}/test_gemm_large.c
    ${TEST_COMMON_HEADER}
)

target_link_libraries(test_gemm_large PRIVATE gemm_lib)
target_compile_definitions(test_gemm_large PRIVATE STANDALONE)
target_compile_options(test_gemm_large PRIVATE ${GEMM_COMPILE_FLAGS})

if(WIN32 AND MINGW)
    target_link_options(test_gemm_large PRIVATE -mconsole)
endif()

if(UNIX OR MINGW)
    target_link_libraries(test_gemm_large PRIVATE m)
endif()

#===============================================================================
# Test Suite 4: Validated Kernel Tests (Full Instrumentation)
#===============================================================================

add_executable(test_gemm_validated 
    ${TEST_DIR}/test_gemm_validated.c
    ${TEST_COMMON_HEADER}
)

target_link_libraries(test_gemm_validated PRIVATE gemm_lib)
target_compile_definitions(test_gemm_validated PRIVATE 
    STANDALONE
    GEMM_VALIDATION_LEVEL=${GEMM_VALIDATION_LEVEL}
)
target_compile_options(test_gemm_validated PRIVATE ${GEMM_COMPILE_FLAGS})

if(WIN32 AND MINGW)
    target_link_options(test_gemm_validated PRIVATE -mconsole)
endif()

if(UNIX OR MINGW)
    target_link_libraries(test_gemm_validated PRIVATE m)
endif()

#===============================================================================
# Test Suite 5: Execution Pipeline Tests (gemm_large.c orchestration)
#===============================================================================

add_executable(test_gemm_execute 
    ${TEST_DIR}/test_gemm_execute.c
    ${TEST_COMMON_HEADER}
)

target_link_libraries(test_gemm_execute PRIVATE gemm_lib)
target_compile_definitions(test_gemm_execute PRIVATE STANDALONE)
target_compile_options(test_gemm_execute PRIVATE ${GEMM_COMPILE_FLAGS})

if(WIN32 AND MINGW)
    target_link_options(test_gemm_execute PRIVATE -mconsole)
endif()

if(UNIX OR MINGW)
    target_link_libraries(test_gemm_execute PRIVATE m)
endif()

#===============================================================================
# Benchmark Suite: Performance Comparison (Optimized vs Naive)
#===============================================================================

add_executable(test_benchmark 
    ${TEST_DIR}/test_benchmark.c
)

target_link_libraries(test_benchmark PRIVATE gemm_lib)
target_compile_options(test_benchmark PRIVATE ${GEMM_COMPILE_FLAGS})

# Warn if not in Release mode
if(NOT CMAKE_BUILD_TYPE MATCHES Release)
    message(WARNING "Benchmark should be built in Release mode for accurate results!")
    message(WARNING "  Configure with: cmake -B build -DCMAKE_BUILD_TYPE=Release")
endif()

if(WIN32 AND MINGW)
    target_link_options(test_benchmark PRIVATE -mconsole)
endif()

if(UNIX OR MINGW)
    target_link_libraries(test_benchmark PRIVATE m)
endif()

#===============================================================================
# Unified Test Runner (All Test Suites)
#===============================================================================

add_executable(test_all 
    ${TEST_DIR}/test_main.c
    ${TEST_DIR}/test_gemm_small.c
    ${TEST_DIR}/test_planning.c
    ${TEST_DIR}/test_gemm_large.c
    ${TEST_DIR}/test_gemm_validated.c
    ${TEST_DIR}/test_gemm_execute.c
    ${TEST_COMMON_HEADER}
)

target_link_libraries(test_all PRIVATE gemm_lib)
target_compile_options(test_all PRIVATE ${GEMM_COMPILE_FLAGS})

if(WIN32 AND MINGW)
    target_link_options(test_all PRIVATE -mconsole)
endif()

if(UNIX OR MINGW)
    target_link_libraries(test_all PRIVATE m)
endif()

#===============================================================================
# CTest Integration
#===============================================================================

enable_testing()

# Register individual test suites
add_test(NAME SmallKernels COMMAND test_gemm_small)
add_test(NAME Planning COMMAND test_gemm_planning)
add_test(NAME KernelTests COMMAND test_gemm_large)
add_test(NAME ValidatedKernels COMMAND test_gemm_validated)
add_test(NAME ExecutePipeline COMMAND test_gemm_execute)
add_test(NAME AllTests COMMAND test_all)

# Set test timeouts
set_tests_properties(SmallKernels PROPERTIES TIMEOUT 60)
set_tests_properties(Planning PROPERTIES TIMEOUT 60)
set_tests_properties(KernelTests PROPERTIES TIMEOUT 120)
set_tests_properties(ValidatedKernels PROPERTIES TIMEOUT 180)
set_tests_properties(ExecutePipeline PROPERTIES TIMEOUT 120)
set_tests_properties(AllTests PROPERTIES TIMEOUT 240)

#===============================================================================
# Custom Targets for Convenience
#===============================================================================

# Run all tests via CTest
add_custom_target(run_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS test_all test_gemm_small test_gemm_planning test_gemm_large 
            test_gemm_validated test_gemm_execute
    COMMENT "Running all GEMM tests via CTest..."
)

# Run individual test suites
add_custom_target(run_small
    COMMAND test_gemm_small
    DEPENDS test_gemm_small
    COMMENT "Running small kernel tests (Tier 1)..."
)

add_custom_target(run_planning
    COMMAND test_gemm_planning
    DEPENDS test_gemm_planning
    COMMENT "Running planning module tests..."
)

add_custom_target(run_kernels
    COMMAND test_gemm_large
    DEPENDS test_gemm_large
    COMMENT "Running individual kernel unit tests..."
)

add_custom_target(run_validated
    COMMAND test_gemm_validated
    DEPENDS test_gemm_validated
    COMMENT "Running VALIDATED kernel tests (full instrumentation)..."
)

add_custom_target(run_execute
    COMMAND test_gemm_execute
    DEPENDS test_gemm_execute
    COMMENT "Running execution pipeline tests (gemm_large.c)..."
)

add_custom_target(run_unified
    COMMAND test_all
    DEPENDS test_all
    COMMENT "Running unified test suite (all tests)..."
)

# Run benchmark
add_custom_target(run_benchmark
    COMMAND test_benchmark
    DEPENDS test_benchmark
    COMMENT "Running GEMM performance benchmark (optimized vs naive)..."
)

#===============================================================================
# Address Sanitizer Build Option
#===============================================================================

option(ENABLE_ASAN "Enable AddressSanitizer" OFF)

if(ENABLE_ASAN)
    if(NOT MSVC)
        set(ASAN_FLAGS -fsanitize=address -fno-omit-frame-pointer -g)
        
        # Apply to validated and execution tests (most likely to catch bugs)
        target_compile_options(test_gemm_validated PRIVATE ${ASAN_FLAGS})
        target_link_options(test_gemm_validated PRIVATE ${ASAN_FLAGS})
        
        target_compile_options(test_gemm_execute PRIVATE ${ASAN_FLAGS})
        target_link_options(test_gemm_execute PRIVATE ${ASAN_FLAGS})
        
        message(STATUS "AddressSanitizer ENABLED for validated and execute tests")
    else()
        # MSVC 2019+ supports /fsanitize=address
        target_compile_options(test_gemm_validated PRIVATE /fsanitize=address)
        target_compile_options(test_gemm_execute PRIVATE /fsanitize=address)
        message(STATUS "AddressSanitizer ENABLED (MSVC)")
    endif()
endif()

#===============================================================================
# Valgrind Integration (Linux/macOS only)
#===============================================================================

option(ENABLE_VALGRIND "Add Valgrind test targets" OFF)

if(ENABLE_VALGRIND AND UNIX AND NOT APPLE)
    find_program(VALGRIND_EXECUTABLE valgrind)
    
    if(VALGRIND_EXECUTABLE)
        add_custom_target(valgrind_execute
            COMMAND ${VALGRIND_EXECUTABLE} 
                --leak-check=full 
                --show-leak-kinds=all 
                --track-origins=yes
                --verbose
                $<TARGET_FILE:test_gemm_execute>
            DEPENDS test_gemm_execute
            COMMENT "Running execution tests under Valgrind..."
        )
        
        add_custom_target(valgrind_validated
            COMMAND ${VALGRIND_EXECUTABLE} 
                --leak-check=full 
                --show-leak-kinds=all 
                --track-origins=yes
                --verbose
                $<TARGET_FILE:test_gemm_validated>
            DEPENDS test_gemm_validated
            COMMENT "Running validated tests under Valgrind..."
        )
        
        message(STATUS "Valgrind integration ENABLED")
        message(STATUS "  Run: make valgrind_execute")
        message(STATUS "  Run: make valgrind_validated")
    else()
        message(WARNING "Valgrind requested but not found")
    endif()
endif()

#===============================================================================
# Build Summary
#===============================================================================

message(STATUS "")
message(STATUS "╔════════════════════════════════════════════════════════════╗")
message(STATUS "║          GEMM Test Suite Configuration                     ║")
message(STATUS "╚════════════════════════════════════════════════════════════╝")
message(STATUS "")
message(STATUS "Directories:")
message(STATUS "  Source:       ${GEMM_SRC_DIR}")
message(STATUS "  Tests:        ${TEST_DIR}")
message(STATUS "")
message(STATUS "Build configuration:")
message(STATUS "  Build type:        ${CMAKE_BUILD_TYPE}")
message(STATUS "  Validation level:  ${GEMM_VALIDATION_LEVEL}")
message(STATUS "  AddressSanitizer:  ${ENABLE_ASAN}")
message(STATUS "  Valgrind:          ${ENABLE_VALGRIND}")
message(STATUS "")
message(STATUS "Build targets:")
message(STATUS "  gemm_lib                - Core GEMM library (static)")
message(STATUS "  test_gemm_small         - Small kernels test (Tier 1)")
message(STATUS "  test_gemm_planning      - Planning module test")
message(STATUS "  test_gemm_large         - Individual kernel tests")
message(STATUS "  test_gemm_validated     - Validated kernel tests (instrumented)")
message(STATUS "  test_gemm_execute       - Execution pipeline tests")
message(STATUS "  test_benchmark          - Performance benchmark (optimized vs naive)")
message(STATUS "  test_all                - Unified test runner")
message(STATUS "")
message(STATUS "Custom targets (make <target>):")
message(STATUS "  run_tests        - Run all tests via CTest")
message(STATUS "  run_small        - Run small kernel tests")
message(STATUS "  run_planning     - Run planning tests")
message(STATUS "  run_kernels      - Run individual kernel tests")
message(STATUS "  run_validated    - Run validated tests")
message(STATUS "  run_execute      - Run execution pipeline tests")
message(STATUS "  run_benchmark    - Run performance benchmark")
message(STATUS "  run_unified      - Run unified test suite")

if(ENABLE_VALGRIND AND VALGRIND_EXECUTABLE)
    message(STATUS "  valgrind_execute    - Run execute tests under Valgrind")
    message(STATUS "  valgrind_validated  - Run validated tests under Valgrind")
endif()

