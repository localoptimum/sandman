cmake_minimum_required(VERSION 2.4.0)

project(sandmanCUDA)

FIND_PACKAGE(CUDA REQUIRED)

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/include /usr/local/cuda/include
~/CUDA/NVIDIA_CUDA-8.0_Samples/common/inc )


# Build list of source files
SET(SOURCES "")
SET(HEADERS "")

#C/C++ sources first
LIST(APPEND SOURCES "src/sandmanCUDA.cu")

#And also headers
LIST(APPEND HEADERS "include/sandmanCUDA.h")

#Build options for CUDA
LIST(APPEND CMAKE_CXX_FLAGS "-std=c++0x -O3 -ffast-math -Wall")
LIST(APPEND CUDA_NVCC_FLAGS "-std=c++11")
LIST(APPEND CUDA_NVCC_FLAGS --compiler-options -fno-strict-aliasing -lineinfo -use_fast_math -Xptxas -dlcm=cg)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_20,code=sm_20)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_35,code=sm_35)

#The actual building and linking
CUDA_ADD_LIBRARY(sandmanCUDA SHARED ${SOURCES})
TARGET_LINK_LIBRARIES(sandmanCUDA /usr/local/cuda/lib64/libcudart.so)

# Install library (needs root!)
install(TARGETS sandmanCUDA DESTINATION /usr/lib)

# Install library headers (needs root!)
install(FILES ${HEADERS} DESTINATION /usr/include)
