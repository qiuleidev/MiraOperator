#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector>

#define WARP_SIZE 32
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

template<const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32(float val){
    #pragma unroll
    for(int offset = kWarpSize >> 1;offset >0;offset >>=1){
        val += __shfl_xor_sync(0xffffffff,val,offset);
    }
    return val;
}

template<const int NUM_THREADS = 256>
__global__ void block_all_reduce_sum_f32_f32_kernel(float* a,float* y,int N){
    int tid = threadIdx.x;
    int idx = blockIdx.x * NUM_THREADS + tid;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float reduce_smem[NUM_WARPS];
    // keep the data in register is enough for warp operaion.
    float sum = (idx < N) ? a[idx] : 0.0f;
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    // perform warp sync reduce.
    sum = warp_reduce_sum_f32<WARP_SIZE>(sum);
    // warp leaders store the data to shared memory.
    if (lane == 0)
        reduce_smem[warp] = sum;
    __syncthreads(); // make sure the data is in shared memory.
    // the first warp compute the final sum.
    sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
    if (warp == 0)
        sum = warp_reduce_sum_f32<NUM_WARPS>(sum);
    if (tid == 0){
        atomicAdd(y, sum);
    }
}

template<const int NUM_THREADS = 256/4>
__global__ void block_all_reduce_sum_f32x4_f32_kernel(float* a,float*y,int N){
    int tid = threadIdx.x;
    int idx = (blockIdx.x * NUM_THREADS + tid) * 4;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    float4 reg_a = LDST128BITS(a[idx]);
    __shared__ float reduce_smem[NUM_WARPS];
    float sum = (idx < N) ? (reg_a.x + reg_a.y + reg_a.z + reg_a.w) : 0.0f;
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    sum = warp_reduce_sum_f32<WARP_SIZE>(sum);
    if(lane == 0)reduce_smem[warp] = sum;
    __syncthreads();
    sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
    if(warp == 0)sum = warp_reduce_sum_f32<NUM_WARPS>(sum);
    if(tid == 0)atomicAdd(y,sum);
}

template<const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ half warp_reduce_sum_f16_f16(half val){
#pragma unroll
    for(int offset = kWarpSize >> 1;offset > 0;offset>>=1){
        val = __hadd(val,__shfl_xor_sync(0xffffffff,val,offset));
    }
    return val;
}
template<const int NUM_THREADS = 256>
__global__ void block_all_reduce_sum_f16_f16_kernel(half* a,float* y,int N){
    int tid = threadIdx.x;
    int idx = blockIdx.x * NUM_THREADS + tid;
    half sum_f16 = (idx < N)?a[idx] : __float2half(0.0f);
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1)/ WARP_SIZE;
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    __shared__ float reduce_smem[NUM_WARPS];
    sum_f16= warp_reduce_sum_f16_f16<WARP_SIZE>(sum_f16);
    if(lane == 0)reduce_smem[warp] = __half2float(sum_f16);
    __syncthreads();
    float sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
    if(warp == 0)sum = warp_reduce_sum_f32<NUM_WARPS>(sum);
    if(tid == 0)atomicAdd(y,sum);
}

template <const int NUM_THREADS = 256 / 2>
__global__ void block_all_reduce_sum_f16x2_f16_kernel(half *a, float *y,int N){
    int tid = threadIdx.x;
    int idx = (blockIdx.x * NUM_THREADS + tid) * 2;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    __shared__ float reduce_smem[NUM_WARPS];
    half2 reg_a = (reinterpret_cast<half2*>(&a[idx]))[0];
    half sum = (idx < N) ? (reg_a.x + reg_a.y) : __float2half(0.0f);
    sum = warp_reduce_sum_f16_f16<WARP_SIZE>(sum);
    if(lane == 0)reduce_smem[warp] = __half2float(sum);
    __syncthreads();
    float sum_32 = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
    if(warp == 0)sum_32 = warp_reduce_sum_f32<NUM_WARPS>(sum_32);
    if(tid == 0)atomicAdd(y,sum_32);
} 

template<const int NUM_THREADS = 256 / 8>
__global__ void block_all_reduce_sum_f16x8_pack_f16_kernel(half* a,float* y,int N){
    int tid = threadIdx.x;
    int idx = (blockIdx.x * NUM_THREADS + tid) * 8;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    int warp = tid /WARP_SIZE;
    int lane = tid %WARP_SIZE;
    __shared__ float reduce_smem[NUM_WARPS];
    half reg_a[8];
    LDST128BITS(reg_a[0]) = LDST128BITS(a[idx]);
    half sum = (idx < N) ? (reg_a[0] + reg_a[1] + reg_a[2] + reg_a[3] + reg_a[4] + reg_a[5] + reg_a[6] + reg_a[7]) : __float2half(0.0f);
    sum = warp_reduce_sum_f16_f16<WARP_SIZE>(sum);
    if(lane ==0)reduce_smem[warp] = __half2float(sum);
    __syncthreads();
    float sum_32 = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
    if(warp == 0)sum_32 = warp_reduce_sum_f32<NUM_WARPS>(sum_32);
    if(tid == 0)atomicAdd(y,sum_32);
}

template<const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f16_f32(half val){
    float val_32 = __half2float(val);
    #pragma unroll
    for(int offset = kWarpSize >> 1;offset >0;offset>>=1){
        val_32 += __shfl_xor_sync(0xffffffff,val_32,offset);
    }
    return val_32;
}

template<const int NUM_THREADS = 256>
__global__ void block_all_reduce_sum_f16_f32_kernel(half* a,float* y,int N){
    int tid = threadIdx.x;
    int idx = blockIdx.x * NUM_THREADS + tid;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float reduce_smem[NUM_WARPS];
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    half sum = (idx < N)?a[idx] : __float2half(0.0f);
    float sum_32 = warp_reduce_sum_f16_f32<WARP_SIZE>(sum);
    if(lane == 0)reduce_smem[warp] = sum_32;
    __syncthreads();
    sum_32 = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
    if(warp == 0)sum_32 = warp_reduce_sum_f32<NUM_WARPS>(sum_32);
    if(tid == 0)atomicAdd(y,sum_32);
}

template<const int NUM_THREADS = 256 / 2>
__global__ void block_all_reduce_sum_f16x2_f32_kernel(half* a,float *y,int N){
    int tid = threadIdx.x;
    int idx = (blockIdx.x * NUM_THREADS + tid) * 2;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    __shared__ float reduce_smem[NUM_WARPS];
    half2 reg_a = reinterpret_cast<half2*>(&a[idx])[0];
    half sum = (idx < N) ? (reg_a.x + reg_a.y): __float2half(0.0f);
    float sum_32 = warp_reduce_sum_f16_f32<WARP_SIZE>(sum);
    if(lane < NUM_WARPS)reduce_smem[warp] = sum_32;
    __syncthreads();
    sum_32 = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
    if(warp == 0)sum_32 = warp_reduce_sum_f32<NUM_WARPS>(sum_32);
    if(tid == 0)atomicAdd(y,sum_32);
}

template<const int NUM_THREADS = 256 / 8>
__global__ void block_all_reduce_sum_f16x8_pack_f32_kernel(half* a,float *y,int N){
    int tid = threadIdx.x;
    int idx = (blockIdx.x * NUM_THREADS + tid) * 8;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    __shared__ float reduce_smem[NUM_WARPS];
    half reg_a[8];
    LDST128BITS(reg_a[0]) = LDST128BITS(a[idx]);
    half sum = (idx < N) ? reg_a[0] + reg_a[1] + reg_a[2] + reg_a[3]+ reg_a[4]+ reg_a[5]+ reg_a[6]+ reg_a[7] : __float2half(0.0f);
    float sum_32 = warp_reduce_sum_f16_f32<WARP_SIZE>(sum);
    if(lane == 0)reduce_smem[warp] = sum_32;
    __syncthreads();
    sum_32 = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
    if(warp == 0)sum_32 = warp_reduce_sum_f32<NUM_WARPS>(sum_32);
    if(tid == 0)atomicAdd(y,sum_32);
}
template<const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ __nv_bfloat16 warp_reduce_sum_bf16_bf16(__nv_bfloat16 val){
    #pragma unroll
    for(int offset = kWarpSize >> 1;offset > 0; offset >>=1){
        val = __hadd(val,__shfl_xor_sync(0xffffffff,val,offset));
    }
    return val;
}

template<const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_bf16_f32(__nv_bfloat16 val){
    float val_32 = __bfloat162float(val);
    #pragma unroll
    for(int offset = kWarpSize >> 1;offset > 0;offset >>=1){
        val_32 += __shfl_xor_sync(0xffffffff,val_32,offset);
    }
    return val_32;
}

template<const int NUM_THREADS = 256>
__global__ void block_all_reduce_sum_bf16_bf16_kernel(__nv_bfloat16* a,float* y,int N){
    int tid = threadIdx.x;
    int idx = blockIdx.x * NUM_THREADS + tid;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ __nv_bfloat16 reduce_smem[NUM_WARPS];
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    __nv_bfloat16 sum = (idx < N) ? a[idx] : __float2bfloat16(0.0f);
    sum = warp_reduce_sum_bf16_bf16<WARP_SIZE>(sum);
    if(lane == 0)reduce_smem[warp] = sum;
    __syncthreads();
    sum = (lane < NUM_WARPS) ? reduce_smem[lane] : __float2bfloat16(0.0f);
    if(warp == 0)sum = warp_reduce_sum_bf16_bf16<NUM_WARPS>(sum);
    if(tid == 0)atomicAdd(y,__bfloat162float(sum));
}

template<const int NUM_THREADS = 256 / 2>
__global__ void block_all_reduce_sum_bf16x2_bf16_kernel(__nv_bfloat16* a,float* y,int N){
    int tid = threadIdx.x;
    int idx = (blockIdx.x * NUM_THREADS + tid) * 2;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    __shared__ __nv_bfloat16 reduce_smem[NUM_WARPS];
    __nv_bfloat162 reg_a = reinterpret_cast<__nv_bfloat162 *>(&a[idx])[0];
    __nv_bfloat16 sum = (idx < N) ? (reg_a.x + reg_a.y) : __float2bfloat16(0.0f);
    sum = warp_reduce_sum_bf16_bf16<WARP_SIZE>(sum);
    if(lane == 0)reduce_smem[warp] = sum;
    __syncthreads();
    sum = (lane < NUM_WARPS) ? reduce_smem[lane] : __float2bfloat16(0.0f);
    if(warp == 0)sum = warp_reduce_sum_bf16_bf16<NUM_WARPS>(sum);
    if(tid == 0)atomicAdd(y,__bfloat162float(sum));
}

template<const int NUM_THREADS = 256 / 8>
__global__ void block_all_reduce_sum_bf16x8_pack_bf16_kernel(__nv_bfloat16* a,float *y,int N){
    int tid = threadIdx.x;
    int idx = (blockIdx.x * NUM_THREADS + tid) * 8;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    __shared__ __nv_bfloat16 reduce_smem[NUM_WARPS]; 
    __nv_bfloat16 reg_a[8];
    reinterpret_cast<float4 *>(&(reg_a[0]))[0] = reinterpret_cast<float4 *>(&(a[idx]))[0];
    __nv_bfloat16 sum = __float2bfloat16(0.0f);
    #pragma unroll
    for(int i = 0;i < 8;i++){
        if(idx + i < N)sum = __hadd(sum,reg_a[i]);
    }
    sum = warp_reduce_sum_bf16_bf16<WARP_SIZE>(sum);
    if(lane == 0)reduce_smem[warp] = sum;
    __syncthreads();
    sum = (lane < NUM_WARPS) ? reduce_smem[lane] : __float2bfloat16(0.0f);
    if(warp == 0)sum = warp_reduce_sum_bf16_bf16<NUM_WARPS>(sum);
    if(tid == 0)atomicAdd(y,sum);
}
template<const int NUM_THREADS = 256>
__global__ void block_all_reduce_sum_bf16_f32_kernel(__nv_bfloat16 *a, float *y,int N){
    int tid = threadIdx.x;
    int idx = blockIdx.x * NUM_THREADS + tid;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float reduce_smem[NUM_WARPS];
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    __nv_bfloat16 sum = (idx < N) ? a[idx] : __float2bfloat16(0.0f);
    float sum_32 = warp_reduce_sum_bf16_f32<WARP_SIZE>(sum);
    if(lane == 0)reduce_smem[warp] = sum_32;
    __syncthreads();
    sum_32 = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
    if(warp == 0)sum_32 = warp_reduce_sum_f32<NUM_WARPS>(sum_32);
    if(tid == 0)atomicAdd(y,sum_32);
}

template<const int NUM_THREADS = 256 / 2>
__global__ void block_all_reduce_sum_bf16x2_f32_kernel(__nv_bfloat16 *a,float *y,int N){
    int tid = threadIdx.x;
    int idx = (blockIdx.x * NUM_THREADS + tid) * 2;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    __shared__ float reduce_smem[NUM_WARPS];
    __nv_bfloat162 reg_a = reinterpret_cast<__nv_bfloat162*>(&a[idx])[0];
    __nv_bfloat16 sum = (idx < N) ? (reg_a.x + reg_a.y) : __float2bfloat16(0.0f);
    float sum_32 = warp_reduce_sum_bf16_f32<WARP_SIZE>(sum);
    if(lane == 0)reduce_smem[warp] = sum_32;
    __syncthreads();
    sum_32 = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
    if(warp == 0)sum_32 = warp_reduce_sum_f32<NUM_WARPS>(sum_32);
    if(tid == 0)atomicAdd(y,sum_32);
}

template<const int NUM_THREADS = 256 / 8>
__global__ void block_all_reduce_sum_bf16x8_pack_f32_kernel(__nv_bfloat16 *a,float *y,int N){
    int tid = threadIdx.x;
    int idx = (blockIdx.x * NUM_THREADS + tid) * 8;
    const int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    __shared__ float reduce_smem[NUM_WARPS];
    __nv_bfloat16 reg_a[8],sum = __float2bfloat16(0.0f);
    reinterpret_cast<float4 *>(reg_a)[0] = reinterpret_cast<float4 *>(&a[idx])[0];
    #pragma unroll
    for(int i = 0;i < 8;i++){
        if(idx + i < N) sum =  __hadd(sum,reg_a[i]);
    }
    float sum_32 = warp_reduce_sum_bf16_f32<WARP_SIZE>(sum);
    if(lane == 0)reduce_smem[warp] = sum_32;
    __syncthreads();
    sum_32 = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
    if(warp == 0)sum_32 = warp_reduce_sum_bf16_f32<NUM_WARPS>(sum_32);
    if(tid == 0)atomicAdd(y,sum_32);
}
#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

#define LANUCH_REDUCE_KERNEL(NT, packed_type, acc_type, element_type,          \
                             out_type)                                         \
  block_all_reduce_sum_##packed_type##_##acc_type##_kernel<(NT)>               \
      <<<grid, block>>>(reinterpret_cast<element_type *>(x.data_ptr()),        \
                        reinterpret_cast<out_type *>(y.data_ptr()), N);

#define DISPATCH_REDUCE_KERNEL(K, packed_type, acc_type, element_type,         \
                               n_elements, out_type)                           \
  const int NT = (K) / (n_elements);                                           \
  dim3 block(NT);                                                              \
  dim3 grid((S));                                                              \
  switch (NT) {                                                                \
  case 32:                                                                     \
    LANUCH_REDUCE_KERNEL(32, packed_type, acc_type, element_type, out_type)    \
    break;                                                                     \
  case 64:                                                                     \
    LANUCH_REDUCE_KERNEL(64, packed_type, acc_type, element_type, out_type)    \
    break;                                                                     \
  case 128:                                                                    \
    LANUCH_REDUCE_KERNEL(128, packed_type, acc_type, element_type, out_type)   \
    break;                                                                     \
  case 256:                                                                    \
    LANUCH_REDUCE_KERNEL(256, packed_type, acc_type, element_type, out_type)   \
    break;                                                                     \
  case 512:                                                                    \
    LANUCH_REDUCE_KERNEL(512, packed_type, acc_type, element_type, out_type)   \
    break;                                                                     \
  case 1024:                                                                   \
    LANUCH_REDUCE_KERNEL(1024, packed_type, acc_type, element_type, out_type)  \
    break;                                                                     \
  default:                                                                     \
    throw std::runtime_error(                                                  \
        "only support (K)/(n_elements): 32/64/128/256/512/1024");              \
    break;                                                                     \
  }

#define TORCH_BINDING_REDUCE(packed_type, acc_type, th_type, element_type,     \
                             n_elements, out_type)                             \
  torch::Tensor block_all_reduce_sum_##packed_type##_##acc_type(               \
      torch::Tensor x) {                                                       \
    CHECK_TORCH_TENSOR_DTYPE(x, (th_type))                                     \
    auto y_th_type =                                                           \
        (th_type) == torch::kInt8 ? torch::kInt32 : torch::kFloat32;           \
    auto options =                                                             \
        torch::TensorOptions().dtype(y_th_type).device(torch::kCUDA, 0);       \
    auto y = torch::zeros({1}, options);                                       \
    const int ndim = x.dim();                                                  \
    if (ndim != 2) {                                                           \
      int N = 1;                                                               \
      for (int i = 0; i < ndim; ++i) {                                         \
        N *= x.size(i);                                                        \
      }                                                                        \
      dim3 block(1024 / (n_elements));                                         \
      dim3 grid((N + 1024 - 1) / 1024);                                        \
      block_all_reduce_sum_##packed_type##_##acc_type##_kernel<1024 /          \
                                                               (n_elements)>   \
          <<<grid, block>>>(reinterpret_cast<element_type *>(x.data_ptr()),    \
                            reinterpret_cast<out_type *>(y.data_ptr()), N);    \
    } else {                                                                   \
      const int S = x.size(0);                                                 \
      const int K = x.size(1);                                                 \
      const int N = S * K;                                                     \
      if ((K / (n_elements)) <= 1024) {                                        \
        DISPATCH_REDUCE_KERNEL(K, packed_type, acc_type, element_type,         \
                               n_elements, out_type)                           \
      } else {                                                                 \
        int N = 1;                                                             \
        for (int i = 0; i < ndim; ++i) {                                       \
          N *= x.size(i);                                                      \
        }                                                                      \
        dim3 block(1024 / (n_elements));                                       \
        dim3 grid((N + 1024 - 1) / 1024);                                      \
        block_all_reduce_sum_##packed_type##_##acc_type##_kernel<1024 /        \
                                                                 (n_elements)> \
            <<<grid, block>>>(reinterpret_cast<element_type *>(x.data_ptr()),  \
                              reinterpret_cast<out_type *>(y.data_ptr()), N);  \
      }                                                                        \
    }                                                                          \
    return y;                                                                  \
  }

// packed_type, acc_type, th_type, element_type, n_elements_per_pack, out_type
TORCH_BINDING_REDUCE(f32, f32, torch::kFloat32, float, 1, float)
TORCH_BINDING_REDUCE(f32x4, f32, torch::kFloat32, float, 4, float)
TORCH_BINDING_REDUCE(f16, f16, torch::kHalf, half, 1, float)
TORCH_BINDING_REDUCE(f16, f32, torch::kHalf, half, 1, float)
TORCH_BINDING_REDUCE(f16x2, f16, torch::kHalf, half, 2, float)
TORCH_BINDING_REDUCE(f16x2, f32, torch::kHalf, half, 2, float)
TORCH_BINDING_REDUCE(f16x8_pack, f16, torch::kHalf, half, 8, float)
TORCH_BINDING_REDUCE(f16x8_pack, f32, torch::kHalf, half, 8, float)
TORCH_BINDING_REDUCE(bf16, bf16, torch::kBFloat16, __nv_bfloat16, 1, float)
TORCH_BINDING_REDUCE(bf16x2, bf16, torch::kBFloat16, __nv_bfloat16, 2, float)
TORCH_BINDING_REDUCE(bf16x8_pack, bf16, torch::kBFloat16, __nv_bfloat16, 8,float)
TORCH_BINDING_REDUCE(bf16, f32, torch::kBFloat16, __nv_bfloat16, 1, float)
TORCH_BINDING_REDUCE(bf16x2, f32, torch::kBFloat16, __nv_bfloat16, 2, float)
TORCH_BINDING_REDUCE(bf16x8_pack, f32, torch::kBFloat16, __nv_bfloat16, 8,
                     float)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_f32_f32)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_f32x4_f32)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_f16_f16)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_f16_f32)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_f16x2_f16)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_f16x2_f32)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_f16x8_pack_f16)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_f16x8_pack_f32)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_bf16_bf16)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_bf16x2_bf16)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_bf16x8_pack_bf16)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_bf16_f32)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_bf16x2_f32)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_bf16x8_pack_f32)
}