#pragma once
#include <string>
#include <fmt/format.h>
#include "../../jit/kernel_runtime.hpp"

namespace MiraOperator{
template<typename T>
class TransposeRuntime final : public LaunchRuntime<TransposeRuntime<T>>{
public:
    struct Args{
        int m,n,Bm,Bn;
        T* input;
        T* output;
        LaunchArgs launch_args;
    };
    static std::string generate_impl(const Args& args){
        return fmt::format(R"(
        #include <cuda.h>
        #include <string>
        #include <MiraOperator/transpose/transpose.cuh>
        using namespace MiraOperator;
        static void __transpose_kernel(){{//双括号防止转义
            auto ptr = reinterpret_cast<void*>(&transpose_f32_kernel<{},{}>);
        }};
        )",args.Bm, args.Bn);
    };
    static void launch_impl(const cudaKernel_t& kernel, const cudaLaunchConfig_t& config, Args args){
        MO_CUDA_RUNTIME_CHECK(cudaLaunchKernelEx(&config, kernel,
            args.input, args.output, args.m, args.n));
    }
};

};