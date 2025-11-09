cmake_minimum_required(VERSION 3.16)
project(gemm_planning_test C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

message(STATUS "Building GEMM Planning Test (minimal)")

#===============================================================================
# Source Directory
#===============================================================================

set(GEMM_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../src/gemm_2)

#===============================================================================
# Minimal Library - Planning Module Only
#===============================================================================

set(PLANNING_SOURCES
    ${GEMM_SRC_DIR}/gemm_planning.c
    ${GEMM_SRC_DIR}/gemm_static.c
  
)

set(PLANNING_HEADERS
    ${GEMM_SRC_DIR}/gemm_planning.h
    ${GEMM_SRC_DIR}/gemm_static.h
    ${GEMM_SRC_DIR}/gemm_utils.h      # Header-only utils
    ${GEMM_SRC_DIR}/gemm.h            # ADDED: For error codes
)

# Build minimal library (just planning + static pool)
add_library(gemm_planning_lib STATIC 
    ${PLANNING_SOURCES} 
    ${PLANNING_HEADERS}
)

target_include_directories(gemm_planning_lib PUBLIC 
    ${GEMM_SRC_DIR}
)

# AVX2 flags (needed for mask generation in planning)
if (MSVC)
    target_compile_options(gemm_planning_lib PRIVATE 
        /arch:AVX2 
        /W4
    )
else()
    target_compile_options(gemm_planning_lib PRIVATE 
        -mavx2 
        -mfma 
        -O2
        -Wall 
        -Wextra 
    )
endif()

#===============================================================================
# Test Executable - Planning Only
#===============================================================================

add_executable(test_planning 
    test_planning.c
)

target_link_libraries(test_planning PRIVATE gemm_planning_lib)

# Test needs AVX2 for mask verification
if (MSVC)
    target_compile_options(test_planning PRIVATE /arch:AVX2)
else()
    target_compile_options(test_planning PRIVATE -mavx2 -mfma)
endif()

# Link math library on Unix/MinGW
if(UNIX OR MINGW)
    target_link_libraries(test_planning PRIVATE m)
endif()

#===============================================================================
# Summary
#===============================================================================

message(STATUS "====================================")
message(STATUS "Minimal Planning Test Build")
message(STATUS "  Sources:     gemm_planning.c, gemm_static.c")
message(STATUS "  Test:        test_planning")
message(STATUS "  Source dir:  ${GEMM_SRC_DIR}")
message(STATUS "====================================")