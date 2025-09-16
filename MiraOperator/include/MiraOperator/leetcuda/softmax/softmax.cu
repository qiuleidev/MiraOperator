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
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])

struct __align__(8) MD{
    float m;
    float d;
};

template<const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ MD warp_reduce_md_op(MD value){
    #pragma unroll
    for(int offset = kWarpSize >> 1;offset >= 1;offset >>= 1){
        MD other;
        other.m = __shfl_xor_sync(0xffffffff,value.m,offset);
        other.d = __shfl_xor_sync(0xffffffff,value.d,offset);

        bool value_bigger = (value.m > other.m);
        MD bigger = value_bigger ? value : other;
        MD smaller = value_bigger ? other : value;

        value.d = bigger.d + smaller.d * __expf(smaller.m - bigger.m);
        value.m = bigger.m;
    }
    return value;
}

template<const int NUM_THREADS = 256 / 4>
__global__ void online_safe_softmax_f32x4_pack_per_token_kernel(float* x,float* y,int N){
    int tid = threadIdx.x;
    int idx = (blockIdx.x * NUM_THREADS + tid) * 4;
    const int NUM_WARPS = NUM_THREADS / WARP_SIZE;
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    float4 val = FLOAT4(x[idx]);
    float local_m = fmaxf(fmaxf(val.x,val.y),fmaxf(val.z,val.w));
    float local_d = __expf(val.x - local_m) + __expf(val.y - local_m) + __expf(val.z - local_m) + __expf(val.w - local_m);
    MD local_md = {local_m,local_d};
    MD res = warp_reduce_md_op<WARP_SIZE>(local_md);
    __shared__ MD smem[NUM_WARPS];
    if(lane == 0)smem[warp] = res;
    __syncthreads();
    if (warp == 0){//掩码为0xffffffff时必须是一整个warp线程参与，否则会导致死锁。
        MD block_res = smem[tid];
        block_res = warp_reduce_md_op<NUM_WARPS>(block_res);
        if(tid == 0)smem[0] = block_res;
    }
    __syncthreads();
    MD final_res = smem[0];
    float d_total_inverse = __fdividef(1.0f,final_res.d);
    if(idx < N){
        float4 reg_y;
        reg_y.x = __expf(val.x - final_res.m) * d_total_inverse;
        reg_y.y = __expf(val.y - final_res.m) * d_total_inverse;
        reg_y.z = __expf(val.z - final_res.m) * d_total_inverse;
        reg_y.w = __expf(val.w - final_res.m) * d_total_inverse;
        FLOAT4(y[idx]) = reg_y;
    }
}


#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

#define CHECK_TORCH_TENSOR_SHAPE(T1, T2)                                       \
  assert((T1).dim() == (T2).dim());                                            \
  for (int i = 0; i < (T1).dim(); ++i) {                                       \
    if ((T2).size(i) != (T1).size(i)) {                                        \
      throw std::runtime_error("Tensor size mismatch!");                       \
    }                                                                          \
  }

// grid memory fence
#define TORCH_BINDING_SOFTMAX(packed_type, th_type, element_type, n_elements)  \
  void softmax_##packed_type(torch::Tensor x, torch::Tensor y) {               \
    CHECK_TORCH_TENSOR_DTYPE(x, (th_type))                                     \
    CHECK_TORCH_TENSOR_DTYPE(y, (th_type))                                     \
    auto options =                                                             \
        torch::TensorOptions().dtype((th_type)).device(torch::kCUDA, 0);       \
    const int N = x.size(0);                                                   \
    CHECK_TORCH_TENSOR_SHAPE(x, y)                                             \
    auto total = torch::zeros({1}, options);                                   \
    dim3 block(256);                                                           \
    dim3 grid(((N + 256 - 1) / 256) / (n_elements));                           \
    softmax_##packed_type##_kernel<256><<<grid, block>>>(                      \
        reinterpret_cast<element_type *>(x.data_ptr()),                        \
        reinterpret_cast<element_type *>(y.data_ptr()),                        \
        reinterpret_cast<element_type *>(total.data_ptr()), N);                \
  }

// softmax per token
#define LANUCH_SOFTMAX_F32_PER_TOKEN_KERNEL(H)                                 \
  softmax_f32_per_token_kernel<(H)>                                            \
      <<<grid, block>>>(reinterpret_cast<float *>(x.data_ptr()),               \
                        reinterpret_cast<float *>(y.data_ptr()), N);

#define DISPATCH_SOFTMAX_F32_PER_TOKEN_KERNEL(S, H)                            \
  dim3 block((H));                                                             \
  dim3 grid((S));                                                              \
  switch ((H)) {                                                               \
  case 32:                                                                     \
    LANUCH_SOFTMAX_F32_PER_TOKEN_KERNEL(32)                                    \
    break;                                                                     \
  case 64:                                                                     \
    LANUCH_SOFTMAX_F32_PER_TOKEN_KERNEL(64)                                    \
    break;                                                                     \
  case 128:                                                                    \
    LANUCH_SOFTMAX_F32_PER_TOKEN_KERNEL(128)                                   \
    break;                                                                     \
  case 256:                                                                    \
    LANUCH_SOFTMAX_F32_PER_TOKEN_KERNEL(256)                                   \
    break;                                                                     \
  case 512:                                                                    \
    LANUCH_SOFTMAX_F32_PER_TOKEN_KERNEL(512)                                   \
    break;                                                                     \
  case 1024:                                                                   \
    LANUCH_SOFTMAX_F32_PER_TOKEN_KERNEL(1024)                                  \
    break;                                                                     \
  default:                                                                     \
    throw std::runtime_error("only support H: 64/128/256/512/1024");           \
    break;                                                                     \
  }

#define LANUCH_SOFTMAX_F32x4_PER_TOKEN_KERNEL(H)                               \
  softmax_f32x4_per_token_kernel<(H) / 4>                                      \
      <<<grid, block>>>(reinterpret_cast<float *>(x.data_ptr()),               \
                        reinterpret_cast<float *>(y.data_ptr()), N);

#define DISPATCH_SOFTMAX_F32x4_PER_TOKEN_KERNEL(S, H)                          \
  const int NT = (H) / 4;                                                      \
  dim3 block(NT);                                                              \
  dim3 grid((S));                                                              \
  switch (H) {                                                                 \
  case 32:                                                                     \
    LANUCH_SOFTMAX_F32x4_PER_TOKEN_KERNEL(32) break;                           \
  case 64:                                                                     \
    LANUCH_SOFTMAX_F32x4_PER_TOKEN_KERNEL(64) break;                           \
  case 128:                                                                    \
    LANUCH_SOFTMAX_F32x4_PER_TOKEN_KERNEL(128) break;                          \
  case 256:                                                                    \
    LANUCH_SOFTMAX_F32x4_PER_TOKEN_KERNEL(256) break;                          \
  case 512:                                                                    \
    LANUCH_SOFTMAX_F32x4_PER_TOKEN_KERNEL(512) break;                          \
  case 1024:                                                                   \
    LANUCH_SOFTMAX_F32x4_PER_TOKEN_KERNEL(1024) break;                         \
  case 2048:                                                                   \
    LANUCH_SOFTMAX_F32x4_PER_TOKEN_KERNEL(2048) break;                         \
  case 4096:                                                                   \
    LANUCH_SOFTMAX_F32x4_PER_TOKEN_KERNEL(4096) break;                         \
  default:                                                                     \
    throw std::runtime_error("only support H: 64/128/.../1024*4");             \
    break;                                                                     \
  }

// safe softmax per token
#define LANUCH_SAFE_SOFTMAX_F32_PER_TOKEN_KERNEL(H)                            \
  safe_softmax_f32_per_token_kernel<(H)>                                       \
      <<<grid, block>>>(reinterpret_cast<float *>(x.data_ptr()),               \
                        reinterpret_cast<float *>(y.data_ptr()), N);

#define DISPATCH_SATE_SOFTMAX_F32_PER_TOKEN_KERNEL(S, H)                       \
  dim3 block((H));                                                             \
  dim3 grid((S));                                                              \
  switch ((H)) {                                                               \
  case 32:                                                                     \
    LANUCH_SAFE_SOFTMAX_F32_PER_TOKEN_KERNEL(32)                               \
    break;                                                                     \
  case 64:                                                                     \
    LANUCH_SAFE_SOFTMAX_F32_PER_TOKEN_KERNEL(64)                               \
    break;                                                                     \
  case 128:                                                                    \
    LANUCH_SAFE_SOFTMAX_F32_PER_TOKEN_KERNEL(128)                              \
    break;                                                                     \
  case 256:                                                                    \
    LANUCH_SAFE_SOFTMAX_F32_PER_TOKEN_KERNEL(256)                              \
    break;                                                                     \
  case 512:                                                                    \
    LANUCH_SAFE_SOFTMAX_F32_PER_TOKEN_KERNEL(512)                              \
    break;                                                                     \
  case 1024:                                                                   \
    LANUCH_SAFE_SOFTMAX_F32_PER_TOKEN_KERNEL(1024)                             \
    break;                                                                     \
  default:                                                                     \
    throw std::runtime_error("only support H: 64/128/256/512/1024");           \
    break;                                                                     \
  }

// online softmax per token
#define LANUCH_ONLINE_SOFTMAX_F32_PER_TOKEN_KERNEL(H)                          \
  online_safe_softmax_f32_per_token_kernel<(H)>                                \
      <<<grid, block>>>(reinterpret_cast<float *>(x.data_ptr()),               \
                        reinterpret_cast<float *>(y.data_ptr()), N);

#define DISPATCH_ONLINE_SOFTMAX_F32_PER_TOKEN_KERNEL(S, H)                     \
  dim3 block((H));                                                             \
  dim3 grid((S));                                                              \
  switch ((H)) {                                                               \
  case 32:                                                                     \
    LANUCH_ONLINE_SOFTMAX_F32_PER_TOKEN_KERNEL(32)                             \
    break;                                                                     \
  case 64:                                                                     \
    LANUCH_ONLINE_SOFTMAX_F32_PER_TOKEN_KERNEL(64)                             \
    break;                                                                     \
  case 128:                                                                    \
    LANUCH_ONLINE_SOFTMAX_F32_PER_TOKEN_KERNEL(128)                            \
    break;                                                                     \
  case 256:                                                                    \
    LANUCH_ONLINE_SOFTMAX_F32_PER_TOKEN_KERNEL(256)                            \
    break;                                                                     \
  case 512:                                                                    \
    LANUCH_ONLINE_SOFTMAX_F32_PER_TOKEN_KERNEL(512)                            \
    break;                                                                     \
  case 1024:                                                                   \
    LANUCH_ONLINE_SOFTMAX_F32_PER_TOKEN_KERNEL(1024)                           \
    break;                                                                     \
  default:                                                                     \
    throw std::runtime_error("only support H: 64/128/256/512/1024");           \
    break;                                                                     \
  }

// online softmax per token
#define LANUCH_ONLINE_SOFTMAX_F32X4_PACK_PER_TOKEN_KERNEL(H)                   \
  online_safe_softmax_f32x4_pack_per_token_kernel<(H / 4)>                     \
      <<<grid, block>>>(reinterpret_cast<float *>(x.data_ptr()),               \
                        reinterpret_cast<float *>(y.data_ptr()), N);

#define DISPATCH_ONLINE_SOFTMAX_F32X4_PACK_PER_TOKEN_KERNEL(S, H)              \
  dim3 block((H / 4));                                                         \
  dim3 grid((S));                                                              \
  switch ((H)) {                                                               \
  case 128:                                                                    \
    LANUCH_ONLINE_SOFTMAX_F32X4_PACK_PER_TOKEN_KERNEL(128)                     \
    break;                                                                     \
  case 256:                                                                    \
    LANUCH_ONLINE_SOFTMAX_F32X4_PACK_PER_TOKEN_KERNEL(256)                     \
    break;                                                                     \
  case 512:                                                                    \
    LANUCH_ONLINE_SOFTMAX_F32X4_PACK_PER_TOKEN_KERNEL(512)                     \
    break;                                                                     \
  case 1024:                                                                   \
    LANUCH_ONLINE_SOFTMAX_F32X4_PACK_PER_TOKEN_KERNEL(1024)                    \
    break;                                                                     \
  case 2048:                                                                   \
    LANUCH_ONLINE_SOFTMAX_F32X4_PACK_PER_TOKEN_KERNEL(2048)                    \
    break;                                                                     \
  case 4096:                                                                   \
    LANUCH_ONLINE_SOFTMAX_F32X4_PACK_PER_TOKEN_KERNEL(4096)                    \
    break;                                                                     \
  default:                                                                     \
    throw std::runtime_error("only support H: 128/256/.../4096;");             \
    break;                                                                     \
  }

#define LANUCH_SAFE_SOFTMAX_F32x4_PER_TOKEN_KERNEL(H)                          \
  safe_softmax_f32x4_per_token_kernel<(H) / 4>                                 \
      <<<grid, block>>>(reinterpret_cast<float *>(x.data_ptr()),               \
                        reinterpret_cast<float *>(y.data_ptr()), N);

#define DISPATCH_SATE_SOFTMAX_F32x4_PER_TOKEN_KERNEL(S, H)                     \
  const int NT = (H) / 4;                                                      \
  dim3 block(NT);                                                              \
  dim3 grid((S));                                                              \
  switch (H) {                                                                 \
  case 32:                                                                     \
    LANUCH_SAFE_SOFTMAX_F32x4_PER_TOKEN_KERNEL(32) break;                      \
  case 64:                                                                     \
    LANUCH_SAFE_SOFTMAX_F32x4_PER_TOKEN_KERNEL(64) break;                      \
  case 128:                                                                    \
    LANUCH_SAFE_SOFTMAX_F32x4_PER_TOKEN_KERNEL(128) break;                     \
  case 256:                                                                    \
    LANUCH_SAFE_SOFTMAX_F32x4_PER_TOKEN_KERNEL(256) break;                     \
  case 512:                                                                    \
    LANUCH_SAFE_SOFTMAX_F32x4_PER_TOKEN_KERNEL(512) break;                     \
  case 1024:                                                                   \
    LANUCH_SAFE_SOFTMAX_F32x4_PER_TOKEN_KERNEL(1024) break;                    \
  case 2048:                                                                   \
    LANUCH_SAFE_SOFTMAX_F32x4_PER_TOKEN_KERNEL(2048) break;                    \
  case 4096:                                                                   \
    LANUCH_SAFE_SOFTMAX_F32x4_PER_TOKEN_KERNEL(4096) break;                    \
  default:                                                                     \
    throw std::runtime_error("only support H: 64/128/.../1024*4");             \
    break;                                                                     \
  }

#define LANUCH_SAFE_SOFTMAX_F16_F32_PER_TOKEN_KERNEL(H)                        \
  safe_softmax_f16_f32_per_token_kernel<(H)>                                   \
      <<<grid, block>>>(reinterpret_cast<half *>(x.data_ptr()),                \
                        reinterpret_cast<half *>(y.data_ptr()), N);

#define DISPATCH_SATE_SOFTMAX_F16_F32_PER_TOKEN_KERNEL(S, H)                   \
  dim3 block((H));                                                             \
  dim3 grid((S));                                                              \
  switch ((H)) {                                                               \
  case 32:                                                                     \
    LANUCH_SAFE_SOFTMAX_F16_F32_PER_TOKEN_KERNEL(32)                           \
    break;                                                                     \
  case 64:                                                                     \
    LANUCH_SAFE_SOFTMAX_F16_F32_PER_TOKEN_KERNEL(64)                           \
    break;                                                                     \
  case 128:                                                                    \
    LANUCH_SAFE_SOFTMAX_F16_F32_PER_TOKEN_KERNEL(128)                          \
    break;                                                                     \
  case 256:                                                                    \
    LANUCH_SAFE_SOFTMAX_F16_F32_PER_TOKEN_KERNEL(256)                          \
    break;                                                                     \
  case 512:                                                                    \
    LANUCH_SAFE_SOFTMAX_F16_F32_PER_TOKEN_KERNEL(512)                          \
    break;                                                                     \
  case 1024:                                                                   \
    LANUCH_SAFE_SOFTMAX_F16_F32_PER_TOKEN_KERNEL(1024)                         \
    break;                                                                     \
  default:                                                                     \
    throw std::runtime_error("only support H: 64/128/256/512/1024");           \
    break;                                                                     \
  }

#define LANUCH_SAFE_SOFTMAX_F16x2_F32_PER_TOKEN_KERNEL(H)                      \
  safe_softmax_f16x2_f32_per_token_kernel<(H) / 2>                             \
      <<<grid, block>>>(reinterpret_cast<half *>(x.data_ptr()),                \
                        reinterpret_cast<half *>(y.data_ptr()), N);

#define DISPATCH_SATE_SOFTMAX_F16x2_F32_PER_TOKEN_KERNEL(S, H)                 \
  const int NT = (H) / 2;                                                      \
  dim3 block(NT);                                                              \
  dim3 grid((S));                                                              \
  switch (H) {                                                                 \
  case 32:                                                                     \
    LANUCH_SAFE_SOFTMAX_F16x2_F32_PER_TOKEN_KERNEL(32) break;                  \
  case 64:                                                                     \
    LANUCH_SAFE_SOFTMAX_F16x2_F32_PER_TOKEN_KERNEL(64) break;                  \
  case 128:                                                                    \
    LANUCH_SAFE_SOFTMAX_F16x2_F32_PER_TOKEN_KERNEL(128) break;                 \
  case 256:                                                                    \
    LANUCH_SAFE_SOFTMAX_F16x2_F32_PER_TOKEN_KERNEL(256) break;                 \
  case 512:                                                                    \
    LANUCH_SAFE_SOFTMAX_F16x2_F32_PER_TOKEN_KERNEL(512) break;                 \
  case 1024:                                                                   \
    LANUCH_SAFE_SOFTMAX_F16x2_F32_PER_TOKEN_KERNEL(1024) break;                \
  case 2048:                                                                   \
    LANUCH_SAFE_SOFTMAX_F16x2_F32_PER_TOKEN_KERNEL(2048) break;                \
  default:                                                                     \
    throw std::runtime_error("only support H: 64/128/.../1024*2");             \
    break;                                                                     \
  }

#define LANUCH_SAFE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(H)                 \
  safe_softmax_f16x8_pack_f32_per_token_kernel<(H) / 8>                        \
      <<<grid, block>>>(reinterpret_cast<half *>(x.data_ptr()),                \
                        reinterpret_cast<half *>(y.data_ptr()), N);

#define DISPATCH_SATE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(S, H)            \
  const int NT = (H) / 8;                                                      \
  dim3 block(NT);                                                              \
  dim3 grid((S));                                                              \
  switch (H) {                                                                 \
  case 32:                                                                     \
    LANUCH_SAFE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(32) break;             \
  case 64:                                                                     \
    LANUCH_SAFE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(64) break;             \
  case 128:                                                                    \
    LANUCH_SAFE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(128) break;            \
  case 256:                                                                    \
    LANUCH_SAFE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(256) break;            \
  case 512:                                                                    \
    LANUCH_SAFE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(512) break;            \
  case 1024:                                                                   \
    LANUCH_SAFE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(1024) break;           \
  case 2048:                                                                   \
    LANUCH_SAFE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(2048) break;           \
  case 4096:                                                                   \
    LANUCH_SAFE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(4096) break;           \
  case 8192:                                                                   \
    LANUCH_SAFE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(8192) break;           \
  default:                                                                     \
    throw std::runtime_error("only support H: 64/128/.../1024*8");             \
    break;                                                                     \
  }

// per token fp32
// void softmax_f32_per_token(torch::Tensor x, torch::Tensor y) {
//   CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32)
//   CHECK_TORCH_TENSOR_DTYPE(y, torch::kFloat32)
//   CHECK_TORCH_TENSOR_SHAPE(x, y)
//   const int S = x.size(0); // seqlens
//   const int H = x.size(1); // head size/kv_len
//   const int N = S * H;
//   DISPATCH_SOFTMAX_F32_PER_TOKEN_KERNEL(S, H)
// }

// void softmax_f32x4_per_token(torch::Tensor x, torch::Tensor y) {
//   CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32)
//   CHECK_TORCH_TENSOR_DTYPE(y, torch::kFloat32)
//   CHECK_TORCH_TENSOR_SHAPE(x, y)
//   const int S = x.size(0); // seqlens
//   const int H = x.size(1); // head size/kv_len
//   const int N = S * H;
//   DISPATCH_SOFTMAX_F32x4_PER_TOKEN_KERNEL(S, H)
// }

// void safe_softmax_f32_per_token(torch::Tensor x, torch::Tensor y) {
//   CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32)
//   CHECK_TORCH_TENSOR_DTYPE(y, torch::kFloat32)
//   CHECK_TORCH_TENSOR_SHAPE(x, y)
//   const int S = x.size(0); // seqlens
//   const int H = x.size(1); // head size/kv_len
//   const int N = S * H;
//   DISPATCH_SATE_SOFTMAX_F32_PER_TOKEN_KERNEL(S, H)
// }

// void safe_softmax_f32x4_per_token(torch::Tensor x, torch::Tensor y) {
//   CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32)
//   CHECK_TORCH_TENSOR_DTYPE(y, torch::kFloat32)
//   CHECK_TORCH_TENSOR_SHAPE(x, y)
//   const int S = x.size(0); // seqlens
//   const int H = x.size(1); // head size/kv_len
//   const int N = S * H;
//   DISPATCH_SATE_SOFTMAX_F32x4_PER_TOKEN_KERNEL(S, H)
// }

// // per token fp16
// void safe_softmax_f16_f32_per_token(torch::Tensor x, torch::Tensor y) {
//   CHECK_TORCH_TENSOR_DTYPE(x, torch::kHalf)
//   CHECK_TORCH_TENSOR_DTYPE(y, torch::kHalf)
//   CHECK_TORCH_TENSOR_SHAPE(x, y)
//   const int S = x.size(0); // seqlens
//   const int H = x.size(1); // head size/kv_len
//   const int N = S * H;
//   DISPATCH_SATE_SOFTMAX_F16_F32_PER_TOKEN_KERNEL(S, H)
// }

// void safe_softmax_f16x2_f32_per_token(torch::Tensor x, torch::Tensor y) {
//   CHECK_TORCH_TENSOR_DTYPE(x, torch::kHalf)
//   CHECK_TORCH_TENSOR_DTYPE(y, torch::kHalf)
//   CHECK_TORCH_TENSOR_SHAPE(x, y)
//   const int S = x.size(0); // seqlens
//   const int H = x.size(1); // head size/kv_len
//   const int N = S * H;
//   DISPATCH_SATE_SOFTMAX_F16x2_F32_PER_TOKEN_KERNEL(S, H)
// }

// void safe_softmax_f16x8_pack_f32_per_token(torch::Tensor x, torch::Tensor y) {
//   CHECK_TORCH_TENSOR_DTYPE(x, torch::kHalf)
//   CHECK_TORCH_TENSOR_DTYPE(y, torch::kHalf)
//   CHECK_TORCH_TENSOR_SHAPE(x, y)
//   const int S = x.size(0); // seqlens
//   const int H = x.size(1); // head size/kv_len
//   const int N = S * H;
//   DISPATCH_SATE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(S, H)
// }

// void online_safe_softmax_f32_per_token(torch::Tensor x, torch::Tensor y) {
//   CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32)
//   CHECK_TORCH_TENSOR_DTYPE(y, torch::kFloat32)
//   CHECK_TORCH_TENSOR_SHAPE(x, y)
//   const int S = x.size(0); // seqlens
//   const int H = x.size(1); // head size/kv_len
//   const int N = S * H;
//   DISPATCH_ONLINE_SOFTMAX_F32_PER_TOKEN_KERNEL(S, H)
// }

void online_safe_softmax_f32x4_pack_per_token(torch::Tensor x,
                                              torch::Tensor y) {
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kFloat32)
  CHECK_TORCH_TENSOR_SHAPE(x, y)
  const int S = x.size(0);
  const int H = x.size(1);
  const int N = S * H;
  DISPATCH_ONLINE_SOFTMAX_F32X4_PACK_PER_TOKEN_KERNEL(S, H)
}

// grid memory fence fp32
// TORCH_BINDING_SOFTMAX(f32,   torch::kFloat32, float, 1)
// TORCH_BINDING_SOFTMAX(f32x4, torch::kFloat32, float, 4)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // TORCH_BINDING_COMMON_EXTENSION(softmax_f32)
  // TORCH_BINDING_COMMON_EXTENSION(softmax_f32x4)
  // TORCH_BINDING_COMMON_EXTENSION(softmax_f32_per_token)
  // TORCH_BINDING_COMMON_EXTENSION(softmax_f32x4_per_token)
  // TORCH_BINDING_COMMON_EXTENSION(safe_softmax_f32_per_token)
  // TORCH_BINDING_COMMON_EXTENSION(safe_softmax_f32x4_per_token)
  // TORCH_BINDING_COMMON_EXTENSION(safe_softmax_f16_f32_per_token)
  // TORCH_BINDING_COMMON_EXTENSION(safe_softmax_f16x2_f32_per_token)
  // TORCH_BINDING_COMMON_EXTENSION(safe_softmax_f16x8_pack_f32_per_token)
  // TORCH_BINDING_COMMON_EXTENSION(online_safe_softmax_f32_per_token)
  TORCH_BINDING_COMMON_EXTENSION(online_safe_softmax_f32x4_pack_per_token)
}