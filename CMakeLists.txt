cmake_minimum_required(VERSION 3.22)
project(cutlassTest LANGUAGES CXX CUDA)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# 引入CUTLASS头文件
include_directories(/home/miracle/cutlass/include) 
include_directories(/home/miracle/cutlass/tools/util/include)
# 指定GPU架构
set(GPU_ARCH "89")  # Ampere (A100/RTX 30系)

# 添加可执行文件
file(GLOB_RECURSE SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu)
add_executable(cutlass_test ${SRC})
