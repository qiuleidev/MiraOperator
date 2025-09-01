#pragma once
#include <string>
#include <fmt/format.h>
#include <boost/type_index.hpp>
#include "../../jit/kernel_runtime.hpp"
namespace MiraOperator {
template <typename T,typename TiledMMA>
class SimpleGEMMRuntime final : public LaunchRuntime<SimpleGEMMRuntime<T,TiledMMA>>{
public:
    struct Args {
        int m,n,k;
        T *a, *b, *c;
        int kTileM,kTileN,kTileK;
        LaunchArgs launch_args;
    };
    static std::string generate_impl(const Args& args){
            // 最简单的逐元素加法kernel
        return fmt::format(R"(
        #include <cuda.h>
        #include <string>
        #include <MiraOperator/gemm/simple_gemm.cuh>
        using namespace MiraOperator;
        static void __simple_gemm_kernel(){{//双括号防止转义
            auto ptr = reinterpret_cast<void*>(&simple_gemm<{},{},{},{},{}>);
        }};
        )",boost::typeindex::type_id_with_cvr<T>().pretty_name(),
        args.kTileM,args.kTileN,args.kTileK,
        boost::typeindex::type_id_with_cvr<TiledMMA>().pretty_name());
    };
    
    static void launch_impl(const cudaKernel_t& kernel, const cudaLaunchConfig_t& config, Args args){
        //参数不对会段错误
        printf("Launching kernel with args: m=%d, n=%d, k=%d, a=%p, b=%p, c=%p\n", 
               args.m, args.n, args.k, args.a, args.b, args.c);
        MO_CUDA_RUNTIME_CHECK(cudaLaunchKernelEx(&config, kernel,
            args.c,args.a,args.b,args.m,args.n,args.k));
    }
};

}; // namespace MiraOperator 