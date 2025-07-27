#pragma once
#include <string>
#include <fmt/format.h>
#include "../../jit/kernel_runtime.hpp"


inline std::string generate_add_kernel_code(const float* a, const float* b, float* c, int n) {
    // 生成包含实际kernel的代码
    return R"(
        #include <cuda.h>
        #include <cuda_runtime.h>
        
        __global__ void fp32_element_add(const float* a, const float* b, float* c, int n) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                c[idx] = a[idx] + b[idx];
            }
        }
        
        // 导出kernel函数指针
        extern "C" __global__ void* fp32_element_add_kernel = fp32_element_add;
    )";
}
namespace MiraOperator {
class ElementAddRuntime final : public LaunchRuntime<ElementAddRuntime>{
public:
    struct Args {
        int m, n, k, num_groups;
        const std::string& compiled_dims;
        LaunchArgs launch_args;

    };


    static std::string generate_impl(const Args& args){
            // 最简单的逐元素加法kernel
        return R"(
        #include <cuda.h>
        #include <string>
        #include <MiraOperator/elementadd/fp32_element_add.cuh>
        
        using namespace MiraOperator;
        
        static void __instantiate_kernel() {
            // Kernel instantiation placeholder
        }
        )";
    };
    
};

}; // namespace MiraOperator 