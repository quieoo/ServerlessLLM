cmake_minimum_required(VERSION 3.10)

project(mock_odcache)

set(CMAKE_CXX_STANDARD 17)

find_package(CUDA REQUIRED)
# set(CUDA_ARCHITECTURES 89)
set(CUDA_ARCHITECTURES "89" CACHE STRING "CUDA architectures to build for")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fopenmp -gencode arch=compute_89,code=sm_89")


set(CMAKE_PREFIX_PATH "/home/sdu/.conda/envs/sllm-worker-0.6/lib/python3.10/site-packages/torch/share/cmake/Torch")

find_package(Torch REQUIRED)

include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
)

set(CUDA_SRC segmented_atten.cu paged_atten.cu)
set(CXX_SRC main.cpp)
add_executable(ReshapeAndCacheSegmentTest ${CUDA_SRC} ${CXX_SRC})

target_link_libraries(ReshapeAndCacheSegmentTest
    ${TORCH_LIBRARIES}   # 正确链接 libtorch
    ${CUDA_LIBRARIES}
)



message(STATUS "CUDA include directories: ${CUDA_INCLUDE_DIRS}")
message(STATUS "Torch include directory: ${TORCH_INCLUDE_DIRS}")
message(STATUS "Linking against torch and CUDA libraries")

