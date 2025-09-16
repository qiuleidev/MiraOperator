#pragma once
#include <string>
#include <fmt/format.h>
#include "../../jit/kernel_runtime.hpp"

namespace MiraOperator {
class ElementWiseRuntime final : public LaunchRuntime<ElementWiseRuntime>{
public:
    struct Args {
        int n;
        char op;
        float *a,*b,*c;
        LaunchArgs launch_args;
    };
    static std::string generate_impl(const Args& args){
            // 最简单的逐元素加法kernel
        return fmt::format(R"(
        #include <cuda.h>
        #include <string>
        #include <MiraOperator/elementwise/fp32_elementwise.cuh>
        
        using namespace MiraOperator;
        
        static void __fp32_elementwise_kernel(){{//双括号防止转义
            auto ptr = reinterpret_cast<void*>(&fp32_elementwise<{},'{}'>);
        }};
        )",args.n,args.op);
    };
    
    static void launch_impl(const cudaKernel_t& kernel, const cudaLaunchConfig_t& config, Args args){
        MO_CUDA_RUNTIME_CHECK(cudaLaunchKernelEx(&config, kernel,
            args.a,args.b,args.c));
    }
};

}; // namespace MiraOperator 