cmake_minimum_required(VERSION 3.22)
#指定项目名称并声明启用C++和CUDA
project(MiraOperator LANGUAGES CXX CUDA)

#开启详细构建日志
set(CMAKE_VERBOSE_MAKEFILE ON)

#为C++和C编译器追加相同的全局选项。-O3，启用最高级别优化， -fPIC，生成位置无关代码，-Wno-psabi，禁用与平台ABI相关警告
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -fPIC -Wno-psabi")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC -Wno-psabi")

find_package(CUDAToolkit REQUIRED)
find_package(pybind11 REQUIRED)
find_package(Torch REQUIRED)

# 设置C++标准
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

# 引入头文件和库
include_directories(MiraOperator/include 
third-party/cutlass/include 
third-party/cutlass/tools/util/include 
third-party/fmt/include)
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/include ${TORCH_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})
link_directories(${TORCH_INSTALL_PREFIX}/lib ${CUDA_TOOLKIT_ROOT_DIR}/lib64 ${CUDA_TOOLKIT_ROOT_DIR}/lib64/stubs)

# The main Python API entrance
pybind11_add_module(mira_operator_cpp csrc/python_api.cpp)
target_link_libraries(mira_operator_cpp PRIVATE ${TORCH_LIBRARIES} torch_python cuda)

# Enable kernel code indexing with CMake-based IDEs
add_library(mira_operator_indexing_cuda STATIC csrc/indexing/main.cu)