#pragma once
#include <string>
#include <fmt/format.h>
#include <boost/type_index.hpp>
#include "../../jit/kernel_runtime.hpp"
#include "MiraOperator/include/MiraOperator/gemm/cute_gemm_config_struct.hpp"
namespace MiraOperator {
using namespace cute;

template <typename T,typename GEMMConfig>
class CuteGEMMSafeRuntime final : public LaunchRuntime<CuteGEMMSafeRuntime<T,GEMMConfig>>{
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
        
        // Safe version with boundary checks
        template<typename T,typename Config>
        __global__ void cute_gemm_fp16_safe(T *Cptr, 
            const T *Aptr, 
            const T *Bptr,
            int m, int n, int k){{
                using namespace cute;

                using SmemLayoutA = typename Config::SmemLayoutA;
                using SmemLayoutB = typename Config::SmemLayoutB;
                using SmemLayoutC = typename Config::SmemLayoutC;
                using TiledMMA = typename Config::MMA;

                using S2RCopyA = typename Config::S2RCopyA;
                using S2RCopyB = typename Config::S2RCopyB;
                using G2SCopyA = typename Config::G2SCopyA;
                using G2SCopyB = typename Config::G2SCopyB;
                using R2SCopyC = typename Config::R2SCopyC;
                using S2GCopyC = typename Config::S2GCopyC;
                constexpr int kTileM = Config::kTileM;
                constexpr int kTileN = Config::kTileN;
                constexpr int kTileK = Config::kTileK;
                constexpr int kStage = Config::kStage;
                int bx = blockIdx.x;
                int by = blockIdx.y;
                int idx = threadIdx.x;

                // Add boundary checks
                if (bx * kTileN >= n || by * kTileM >= m) {{
                    return; // Early exit if block is out of bounds
                }}

                Tensor mA = make_tensor(make_gmem_ptr(Aptr),make_shape(m,k),make_stride(k,Int<1>{{}}));
                Tensor mB = make_tensor(make_gmem_ptr(Bptr),make_shape(n,k),make_stride(k,Int<1>{{}}));
                Tensor mC = make_tensor(make_gmem_ptr(Cptr),make_shape(m,n),make_stride(n,Int<1>{{}}));

                Tensor gA = local_tile(mA, make_tile(Int<kTileM>{{}}, Int<kTileK>{{}}), make_coord(by, _));
                Tensor gB = local_tile(mB, make_tile(Int<kTileN>{{}}, Int<kTileK>{{}}), make_coord(bx, _));
                Tensor gC = local_tile(mC, make_tile(Int<kTileM>{{}}, Int<kTileN>{{}}), make_coord(by, bx));

                extern __shared__ T smemA[];
                
                // Add shared memory bounds checking
                constexpr int smemA_size = cosize(SmemLayoutA{{}});
                constexpr int smemB_size = cosize(SmemLayoutB{{}});
                
                if (idx == 0) {{
                    // Only thread 0 checks shared memory bounds
                    if (smemA_size <= 0 || smemB_size <= 0) {{
                        return;
                    }}
                }}
                
                T *smemB = smemA + smemA_size;

                Tensor sA = make_tensor(make_smem_ptr(smemA), SmemLayoutA{{}});
                Tensor sB = make_tensor(make_smem_ptr(smemB), SmemLayoutB{{}});

                // Rest of the kernel implementation would go here...
                // For now, just a simple implementation to test bounds
                
                // Simple GEMM implementation with bounds checking
                for (int i = 0; i < kTileM && (by * kTileM + i) < m; ++i) {{
                    for (int j = 0; j < kTileN && (bx * kTileN + j) < n; ++j) {{
                        T sum = T(0);
                        for (int kk = 0; kk < k; ++kk) {{
                            if ((by * kTileM + i) < m && (bx * kTileN + j) < n && kk < k) {{
                                sum += gA(i, kk) * gB(j, kk);
                            }}
                        }}
                        if ((by * kTileM + i) < m && (bx * kTileN + j) < n) {{
                            gC(i, j) = sum;
                        }}
                    }}
                }}
            }}
        
        static void __cute_gemm_safe_kernel(){{
            auto ptr = reinterpret_cast<void*>(&cute_gemm_fp16_safe<{},{}>);
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
