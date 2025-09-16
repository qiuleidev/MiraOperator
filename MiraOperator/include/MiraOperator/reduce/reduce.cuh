#include <cuda_fp16.h> 
namespace MiraOperator{
    #define WARP_SIZE 32
    template<typename T,int kWarpSize>
    __device__ __forceinline__ T warp_reduce(T val){
        #pragma unroll
        for(int offset = kWarpSize>>1; offset >=1; offset>>=1){
            if constexpr(std::is_same_v<T,__half>)val = __hadd(val,__shfl_xor_sync(0xffffffff, val, offset));
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
                if(warp == 0)sum = warp_reduce<float,NUM_WARPS>(sum);
                if(tid == 0)atomicAdd(output,sum);
            }
        }
        else if constexpr (std::is_same_v<T_DECAYED,__half>){
            __shared__ __half reduce_smem[NUM_WARPS];
            idx = idx << 1;
            if(idx < n){
                //block reduce
                __half reg_input[8];
                reinterpret_cast<float4*>(&reg_input[0])[0] = reinterpret_cast<float4*> (&input[idx])[0];
                __half sum = __float2half(0.0f);
                #pragma unroll
                for(int i = 0;i < 8;i++){
                    sum += (idx + i) < n ? reg_input[i] : __float2half(0.0f);
                }
                sum = warp_reduce<__half,WARP_SIZE>(sum);
                if(lane == 0)reduce_smem[warp] = sum;
                __syncthreads();
                sum = (lane < NUM_WARPS) ? reduce_smem[lane] : __float2half(0.0f);
                if(warp == 0)sum = warp_reduce<__half,NUM_WARPS>(sum);
                if(tid == 0)atomicAdd(output, sum);
            }
        }
    }

}