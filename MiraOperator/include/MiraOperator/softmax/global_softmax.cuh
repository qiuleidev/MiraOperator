#pragma once
#include <float.h>
#define WARP_SIZE 32
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])
namespace MiraOperator{
    __device__ __forceinline__ void atomicMax(float* address,float val){
        int* address_as_int = (int *)address;
        int old = *address_as_int;
        int assumed;
        do{
            assumed = old;
            old = atomicCAS(address_as_int,assumed,fmaxf(val,__int_as_float(assumed)));
        }while(assumed!=old);
    }
    template<const int kWarpSize = 32>
    __device__ __forceinline__ float warp_reduce_sum(float val){
        #pragma unroll
        for(int offset = kWarpSize >> 1;offset > 0;offset >>=1){
            val += __shfl_xor_sync(0xffffffff,val,offset);
        }
        return val;
    }

    template<const int kWarpSize = 32>
    __device__ __forceinline__ float warp_reduce_max(float val){
        #pragma unroll
        for(int offset = kWarpSize >> 1;offset > 0;offset >>=1){
            val = fmaxf(val,__shfl_xor_sync(0xffffffff,val,offset));
        }
        return val;
    }

    template<const int NUM_THREADS = 512>
    __device__ float block_reduce_sum(float val){
        int tx = threadIdx.x;
        constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
        __shared__ float smem[NUM_WARPS];
        int warp = tx / WARP_SIZE;
        int lane = tx % WARP_SIZE;
        val = warp_reduce_sum<WARP_SIZE>(val);
        if(lane == 0)smem[warp] = val;
        __syncthreads();
        val = (lane < NUM_WARPS) ? smem[lane] : 0.0f;
        val = warp_reduce_sum<NUM_WARPS>(val);
        val = __shfl_sync(0xffffffff,val,0,32);
        return val;
    }

    template<const int NUM_THREADS = 512>
    __device__ float block_reduce_max(float val){
        int tx = threadIdx.x;
        constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
        __shared__ float smem[NUM_WARPS];
        int warp = tx / WARP_SIZE;
        int lane = tx % WARP_SIZE;
        val = warp_reduce_max<WARP_SIZE>(val);
        if(lane == 0)smem[warp] = val;
        __syncthreads();
        val = (lane < NUM_WARPS) ? smem[lane] : -FLT_MAX;
        val = warp_reduce_max<NUM_WARPS>(val);
        val = __shfl_sync(0xffffffff,val,0,32);
        return val;
    }

    template<const int NUM_THREADS = 512>
    __global__ void get_max_f32_kernel(float* x,float* global_max,int N){
        const int tx = threadIdx.x;
        const int idx = (blockIdx.x * blockDim.x + tx) * 4;
        float max_val = -1e38f;
        float4 vals = FLOAT4(x[idx]);
        float val = vals.x;
        val = fmaxf(val,vals.y);
        val = fmaxf(val,vals.z);
        val = fmaxf(val,vals.w);
        max_val = block_reduce_max<NUM_THREADS>(val);
        //只有warp == 0时返回block最大值，其他warp返回的是warp内规约值，但只有第一个线程更新最大值。
        if(tx == 0)atomicMax(global_max,max_val);
    }

    template<const int NUM_THREADS = 512>
    __global__ void get_exp_sum_f32_kernel(float* x,float *global_max,float* global_exp_sum,int N){
        const int tx = threadIdx.x;
        const int idx = (blockIdx.x * blockDim.x + tx) * 4;
        float exp_sum = 0.0f;
        float g_max = *global_max;
        if(idx < N){
            float4 vals = FLOAT4(x[idx]);
            vals.x = expf(vals.x - g_max);
            vals.y = expf(vals.y - g_max);
            vals.z = expf(vals.z - g_max);
            vals.w = expf(vals.w - g_max);
            exp_sum = vals.x + vals.y + vals.z + vals.w;
            exp_sum = block_reduce_sum<NUM_THREADS>(exp_sum);
        }
        if(tx == 0)atomicAdd(global_exp_sum,exp_sum);
    }

    template<const int NUM_THREADS = 512>
    __global__ void global_soft_max_f32_kernel(float *x,float *y,float *global_max,float *global_exp_sum,int N){
        const int tx = threadIdx.x;
        const int idx = (blockIdx.x * blockDim.x + tx) * 4;
        float g_max = *global_max;
        float g_exp_sum = *global_exp_sum;
        if(idx < N){
            float4 vals = FLOAT4(x[idx]);
            vals.x = expf(vals.x - g_max) / g_exp_sum;
            vals.y = expf(vals.y - g_max) / g_exp_sum;
            vals.z = expf(vals.z - g_max) / g_exp_sum;
            vals.w = expf(vals.w - g_max) / g_exp_sum;
            FLOAT4(y[idx]) = vals;
        }
    }
    
    //assume x/y dim <= NUM_THREADS * 4
    template<const int NUM_THREADS = 512>
    __global__ void global_soft_max_f32_per_token_kernel(float *x,float *y,int N){//N是token的维度
        const int tid = threadIdx.x;
        const int idx = blockIdx.x * N + tid << 2;
        int col_offset = tid << 2;
        float4 reg_x = reinterpret_cast<float4*>(&x[idx])[0];
        reg_x.x = (col_offset + 0 < N) ? reg_x.x : -FLT_MAX;
        reg_x.y = (col_offset + 1 < N) ? reg_x.y : -FLT_MAX;
        reg_x.z = (col_offset + 2 < N) ? reg_x.z : -FLT_MAX;
        reg_x.w = (col_offset + 3 < N) ? reg_x.w : -FLT_MAX;
        float local_max = fmaxf(fmaxf(reg_x.x,reg_x.y),fmaxf(reg_x.z,reg_x.w));
        float token_max = block_reduce_max<NUM_THREADS>(local_max);
        float local_exp_sum;
        reg_x.x =  (col_offset + 0 < N) ? __expf(reg_x.x - token_max) : 0.0f;
        reg_x.y =  (col_offset + 1 < N) ? __expf(reg_x.y - token_max) : 0.0f;
        reg_x.z =  (col_offset + 2 < N) ? __expf(reg_x.z - token_max) : 0.0f;
        reg_x.w =  (col_offset + 3 < N) ? __expf(reg_x.w - token_max) : 0.0f;
        local_exp_sum = reg_x.x + reg_x.y + reg_x.z + reg_x.w;
        float token_exp_sum = block_reduce_sum<NUM_THREADS>(local_exp_sum);
        float4 reg_y;
        reg_y.x = reg_x.x / token_exp_sum;
        reg_y.y = reg_x.y / token_exp_sum;
        reg_y.z = reg_x.z / token_exp_sum;
        reg_y.w = reg_x.w / token_exp_sum;
        reinterpret_cast<float4 *>(&y[idx])[0] = reg_y;
    }
}