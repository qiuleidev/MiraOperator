#pragma once
#include <string>
#include <fmt/format.h>
#include <boost/type_index.hpp>
#include "../../jit/kernel_runtime.hpp"
namespace MiraOperator {

template <typename T>
class GlobalSoftmaxRuntime final : public LaunchRuntime<GlobalSoftmaxRuntime<T>>{
public:
    struct Args {
        int n;
        T *x,*y;
        //float *global_max,*global_exp_sum;
        LaunchArgs launch_args;
    };
    
    static std::string generate_impl(const Args& args){
        return fmt::format(R"(
        #include <cuda.h>
        #include <string>
        #include <MiraOperator/softmax/global_softmax.cuh>
        using namespace MiraOperator;
        
        
        static void __softmax_kernel() {{
            auto ptr = reinterpret_cast<void*>(&global_soft_max_f32_per_token_kernel<{}>);
        }}
        )",args.launch_args.num_threads);
    };
    
    static void launch_impl(const cudaKernel_t& kernel, const cudaLaunchConfig_t& config, Args args){

        MO_CUDA_RUNTIME_CHECK(cudaLaunchKernelEx(&config, kernel,
            args.x, args.y, args.n));
    }
};
}; // namespace MiraOperator 