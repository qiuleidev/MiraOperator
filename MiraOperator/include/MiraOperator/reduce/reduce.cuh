#include <cute/tensor.hpp>
#include <cuda_fp16.h> 
namespace MiraOperator{
    using namespace cute;
    #define WARP_SIZE 32
    template<typename T,int kWarpSize>
    __device__ __forceinline__ T warp_reduce(T val){
        #pragma unroll
        for(int offset = kWarpSize>>1; offset >=1; offset>>=1){
            if constexpr(std::is_same_v<T,half>)__hadd(val,__shfl_xor_sync(0xffffffff, val, offset));
            else val += __shfl_xor_sync(0xffffffff, val, offset);
        }
        return val;
    }

    template<typename T,int NUM_THREADS>
    __global__ void block_reduce(T *input,T *output,int n){
        int tid = threadIdx.x;
        int idx = (blockIdx.x * blockDim.x + tid) * 4;
        constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
        int warp = tid / WARP_SIZE;
        int lane = tid % WARP_SIZE;
        using T_DECAYED = std::decay_t<T>;
        if constexpr (std::is_same_v<T_DECAYED,float>){
            __shared__ float reduce_smem[NUM_WARPS];
            if(idx < n){
                //block reduce
                float4 reg_input = reinterpret_cast<float4 *> (&input[idx])[0];
                float sum = 0.0f;
                sum += reg_input.x + reg_input.y + reg_input.z + reg_input.w;
                sum = warp_reduce<float,WARP_SIZE>(sum);
                if(lane == 0)reduce_smem[warp] = sum;
                __syncthreads();
                sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
                if(warp == 0)sum = warp_reduce<float,WARP_SIZE>(sum);
                if(tid == 0)atomicAdd(output,sum);
                
                // //grid_reduce
                // int grid_reduce_size = (n + NUM_THREADS - 1)/ NUM_THREADS;
                // if(idx < grid_reduce_size){
                //     float4 reg_output = reinterpret_cast<float4 *>(&(output[idx + 32]))[0];
                //     sum = 0.0f;
                //     if (idx < grid_reduce_size) sum += reg_output.x;
                //     if (idx + 1 < grid_reduce_size) sum += reg_output.y;
                //     if (idx + 2 < grid_reduce_size) sum += reg_output.z;
                //     if (idx + 3 < grid_reduce_size) sum += reg_output.w;
                //     sum = warp_reduce<float,WARP_SIZE>(sum);
                //     if(lane == 0)reduce_smem[warp] = sum;
                //     __syncthreads();
                //     sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
                //     if(warp == 0)sum = warp_reduce<float,WARP_SIZE>(sum);
                //     if(tid == 0)atomicAdd(output,sum);
                // }
            }
        }
        else if constexpr (std::is_same_v<T_DECAYED,__half>){
            __shared__ __half reduce_smem[NUM_WARPS];
            if(idx < n){
                //block reduce
                __half reg_input[8] = reinterpret_cast<half4 *> (&input[idx])[0];
                __half sum = reg_input[0] + reg_input[1] + reg_input[2] + reg_input[3] + reg_input[4] + reg_input[5] + reg_input[6] + reg_input[7];
                sum = warp_reduce<__half,WARP_SIZE>(sum);
                if(lane == 0)reduce_smem[warp] = sum;
                __syncthreads();
                sum = (lane < NUM_WARPS) ? reduce_smem[lane] : __float2half(0.0f);
                if(warp == 0)sum = warp_reduce<__half,WARP_SIZE>(sum);
                if(tid == 0)atomicAdd(output, sum);
            }
        }
    }

    // template<typename T,int NUM_THREADS>
    // __global__ void block_reduce2(T *input,T *block_results,int n){
    //     int tid = threadIdx.x;
    //     int idx = (blockIdx.x * blockDim.x + tid) * 4;
    //     constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    //     int warp = tid / WARP_SIZE;
    //     int lane = tid % WARP_SIZE;
    //     using T_DECAYED = std::decay_t<T>;
    //     if constexpr (std::is_same_v<T_DECAYED,float>){
    //         __shared__ float reduce_smem[NUM_WARPS];
    //         if(idx < n){
    //             float4 reg_input = reinterpret_cast<float4 *> (&input[idx])[0];
    //             float sum = 0.0f;
    //             if (idx < n) sum += reg_input.x;
    //             if (idx + 1 < n) sum += reg_input.y;
    //             if (idx + 2 < n) sum += reg_input.z;
    //             if (idx + 3 < n) sum += reg_input.w;
    //             sum = warp_reduce<float,WARP_SIZE>(sum);
    //             if(lane == 0)reduce_smem[warp] = sum;
    //             __syncthreads();
    //             sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
    //             if(warp == 0)sum = warp_reduce<float,WARP_SIZE>(sum);
    //             if(tid == 0)atomicAdd(&block_results[blockIdx.x],sum);
    //         }
    //     }
    // }
}