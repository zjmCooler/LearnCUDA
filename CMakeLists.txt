cmake_minimum_required(VERSION 3.15)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")

project(LearnCUDA CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_subdirectory(src)
