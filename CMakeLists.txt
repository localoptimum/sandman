cmake_minimum_required(VERSION 2.4.0)

project(sandmanCUDA)

FIND_PACKAGE(CUDA REQUIRED)

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/include /usr/local/cuda/include
/usr/local/cuda/samples/common/inc )

# Build list of source files
SET(SOURCES "")
SET(HEADERS "")
SET(SCRIPTS "")

#C/C++ sources first
LIST(APPEND SOURCES "src/sandmanCUDA.cu")

#And also headers
LIST(APPEND HEADERS "include/sandmanCUDA.h")

#And visualisation scripts
LIST(APPEND SCRIPTS "src/sandmanPlots.py")

#Build options for CUDA
#This is for deploy:
LIST(APPEND CMAKE_CXX_FLAGS "-std=c++0x -O3 -ffast-math -Wall")
#This is for debug:
#LIST(APPEND CMAKE_CXX_FLAGS "-std=c++0x --debug -Wall")

#...and the rest...
LIST(APPEND CUDA_NVCC_FLAGS "-std=c++11")
LIST(APPEND CUDA_NVCC_FLAGS --compiler-options -fno-strict-aliasing -lineinfo -use_fast_math -Xptxas -dlcm=cg)
#LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30)
#LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_35,code=sm_35)
#LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_50,code=sm_50)
#LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_52,code=sm_52)
#LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_53,code=sm_53)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_61,code=sm_61)

#The actual building and linking
CUDA_ADD_LIBRARY(sandmanCUDA SHARED ${SOURCES})
TARGET_LINK_LIBRARIES(sandmanCUDA /usr/local/cuda/lib64/libcudart.so)

# Install library (needs root!)
install(TARGETS sandmanCUDA DESTINATION /usr/lib)

# Install library headers (needs root!)
install(FILES ${HEADERS} DESTINATION /usr/include)
install(FILES ${SCRIPTS} DESTINATION /usr/bin PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_EXECUTE GROUP_READ WORLD_READ WORLD_EXECUTE)
