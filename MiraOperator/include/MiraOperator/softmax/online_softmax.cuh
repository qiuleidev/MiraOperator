#pragma once
#include <cute/tensor.hpp>
#include <float.h>
#define WARP_SIZE 32
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])
namespace MiraOperator{
    //assume token dim less than 1024 
    struct __align__(8) MD{
        float m;
        float d;
    };

    template<const int kWarpSize = 32>
    __device__ __forceinline__ MD warp_reduce_MD(MD val){
        MD other;
        #pragma unroll
        for(int offset = kWarpSize >> 1;offset > 0;offset >>= 1){
            other.m = __shfl_xor_sync(0xffffffff,val.m,offset);
            other.d = __shfl_xor_sync(0xffffffff,val.d,offset);
            MD bigger = val.m > other.m ? val : other;
            MD smaller = val.m > other.m ? other : val;
            val.m = bigger.m;
            val.d = smaller.d * __expf(smaller.m - bigger.m) + bigger.d;
        }
        return val;
    }

    template<const int NUM_THREADS = 512>
    __global__ void online_soft_max_f32_per_token_kernel(float *x,float *y,int N){
        const int tid = threadIdx.x;
        const int idx = (blockIdx.x * blockDim.x + tid) * 4;
        float4 reg_x = reinterpret_cast<float4 *>(&x[idx])[0];
        reg_x.x = (idx + 0 < N) ? reg_x.x : -FLT_MAX;
        reg_x.y = (idx + 1 < N) ? reg_x.y : -FLT_MAX;
        reg_x.z = (idx + 2 < N) ? reg_x.z : -FLT_MAX;
        reg_x.w = (idx + 3 < N) ? reg_x.w : -FLT_MAX;
        MD local_md;
        local_md.m = fmaxf(fmaxf(reg_x.x,reg_x.y),fmaxf(reg_x.z,reg_x.w));
        local_md.d = 0.0f;
        local_md.d += (idx + 0 < N) ? __expf(reg_x.x - local_md.m) : 0.0f;
        local_md.d += (idx + 1 < N) ? __expf(reg_x.y - local_md.m) : 0.0f;
        local_md.d += (idx + 2 < N) ? __expf(reg_x.z - local_md.m) : 0.0f;
        local_md.d += (idx + 3 < N) ? __expf(reg_x.w - local_md.m) : 0.0f;

        constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
        const int warp = tid / WARP_SIZE;
        const int lane = tid % WARP_SIZE;
        local_md = warp_reduce_MD<WARP_SIZE>(local_md);
        __shared__ MD smem[NUM_WARPS];
        if(lane == 0)smem[warp] = local_md;
        __syncthreads();
        if(warp == 0){
            MD md = smem[lane];
            local_md.m = lane < NUM_WARPS ? md.m : 0.0f;
            local_md.d = lane < NUM_WARPS ? md.d : 0.0f;
            local_md = warp_reduce_MD<NUM_WARPS>(local_md);
            if(lane == 0)smem[0] = local_md;
        }
        __syncthreads();
        MD block_res = smem[0];
        float block_max = block_res.m;
        float block_d_inverse = __fdividef(1.0f,block_res.d);
        float4 reg_y;
        if(tid < N){
            reg_y.x = __expf(reg_x.x - block_max) * block_d_inverse;
            reg_y.y = __expf(reg_x.y - block_max) * block_d_inverse;
            reg_y.z = __expf(reg_x.z - block_max) * block_d_inverse;
            reg_y.w = __expf(reg_x.w - block_max) * block_d_inverse;
            reinterpret_cast<float4* >(&y[idx])[0] = reg_y;
        }
    }
}