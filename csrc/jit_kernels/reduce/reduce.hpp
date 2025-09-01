#pragma once
#include <string>
#include <fmt/format.h>
#include "../../jit/kernel_runtime.hpp"

namespace MiraOperator{
template<typename T>
class ReduceRuntime final : public LaunchRuntime<ReduceRuntime<T>>{
public:
    struct Args{
        int n;
        T* input;
        T* output;
        LaunchArgs launch_args;
    };
    static std::string generate_impl(const Args& args){
        return fmt::format(R"(
        #include <cuda.h>
        #include <string>
        #include <MiraOperator/reduce/reduce.cuh>
        using namespace MiraOperator;
        static void __reduce_kernel(){{//双括号防止转义
            auto ptr = reinterpret_cast<void*>(&block_reduce<{},{}>);
        }};
        )",boost::typeindex::type_id_with_cvr<T>().pretty_name()
    ,args.launch_args.num_threads);
    };
    static void launch_impl(const cudaKernel_t& kernel, const cudaLaunchConfig_t& config, Args args){
        MO_CUDA_RUNTIME_CHECK(cudaLaunchKernelEx(&config, kernel,
            args.input,args.output,args.n));
    }
};

};