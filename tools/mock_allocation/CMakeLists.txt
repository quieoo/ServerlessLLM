# 设置最低版本的 CMake
cmake_minimum_required(VERSION 3.10)

# 设置项目名称和语言
project(ServerlessLLM)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
find_package(CUDAToolkit QUIET)
find_package(CUDA REQUIRED)


# 查找所有源文件
file(GLOB SOURCES
    src/main.cpp
    src/model_pool.cpp
    src/binary_utils.cpp
)

# 查找所有头文件
file(GLOB HEADERS
    src/concurrent_array.h
    src/logger.h
    src/LRUList.h
    src/model_pool.h
    src/registered_model.h
    src/binary_utils.h
)

include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_LIBRARY_DIRS})



# 添加可执行文件
add_executable(Allocateion ${SOURCES} ${HEADERS})

# 链接库文件（如果有的话）
target_link_libraries(Allocateion ${CUDA_LIBRARIES})

# 设置输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# 如果使用了其他库或者有特殊配置，请在此处添加

