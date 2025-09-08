#pragma once
#include <string>
#include <fmt/format.h>
#include <boost/type_index.hpp>
#include "../../jit/kernel_runtime.hpp"
#include "MiraOperator/include/MiraOperator/gemm/cute_gemm_config_struct.hpp"
namespace MiraOperator {
using namespace cute;

template <typename T,typename GEMMConfig>
class CuteGEMMRuntime final : public LaunchRuntime<CuteGEMMRuntime<T,GEMMConfig>>{
public:
    struct Args {
        int m,n,k;
        T *a, *b, *c;
        LaunchArgs launch_args;
    };
    static std::string generate_impl(const Args& args){
        return fmt::format(R"(
        #include <cuda.h>
        #include <string>
        #include <MiraOperator/gemm/cute_gemm.cuh>
        #include <MiraOperator/gemm/cute_gemm_config_struct.hpp>
        using namespace MiraOperator;
        static void __cute_gemm_kernel(){{//双括号防止转义
            auto ptr = reinterpret_cast<void*>(&cute_gemm_fp16<{},{}>);
        }};
        )",boost::typeindex::type_id_with_cvr<T>().pretty_name(),
        boost::typeindex::type_id_with_cvr<GEMMConfig>().pretty_name());
    };
    
    static void launch_impl(const cudaKernel_t& kernel, const cudaLaunchConfig_t& config, Args args){
        //参数不对会段错误
        MO_CUDA_RUNTIME_CHECK(cudaLaunchKernelEx(&config, kernel,
            args.c,args.a,args.b,args.m,args.n,args.k));
    }
};
}; // namespace MiraOperator 