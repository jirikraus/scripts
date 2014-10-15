cmake_minimum_required(VERSION 2.8)

# project name
project (nvml_test)

find_package(CUDA)
find_package(NVML)

if(NVML_FOUND)
	include_directories(${NVML_INCLUDE_DIR})
	add_definitions(-DUSE_NVML)
endif(NVML_FOUND)

SET( CUDA_NVCC_FLAGS "-gencode arch=compute_35,code=sm_35" )
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native -O3")

CUDA_ADD_EXECUTABLE(nvml_test nvml_test.cu matrixMul.cu )

if(NVML_FOUND)
	target_link_libraries(nvml_test ${NVML_LIBRARY})
endif(NVML_FOUND)
