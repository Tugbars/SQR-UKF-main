cmake_minimum_required(VERSION 3.16)
project(SQR_UKF_Root C)

message(STATUS "Root CMakeLists.txt - delegating to gemm_test/")

# Delegate everything to gemm_test subdirectory
add_subdirectory(gemm_test)