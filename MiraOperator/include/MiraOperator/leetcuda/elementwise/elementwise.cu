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
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

__global__ void elementwise_add_f32_kernel(float *a,float *b,float *c,int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N){
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void elementwise_add_f32x4_kernel(float *a,float *b,float *c,int N){//a粗化
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    idx = idx * 4;
    if(idx < N){
        float4 reg_a = FLOAT4(a[idx]);
        float4 reg_b = FLOAT4(b[idx]);
        float4 reg_c;
        reg_c.x = reg_a.x + reg_b.x;
        reg_c.y = reg_a.y + reg_b.y;
        reg_c.z = reg_a.z + reg_b.z;
        reg_c.w = reg_a.w + reg_b.w;
        FLOAT4(c[idx]) = reg_c;
    }
}

__global__ void elementwise_add_f16_kernel(half *a,half *b,half *c,int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N){
        c[idx] = __hadd(a[idx], b[idx]);
    }
}

__global__ void elementwise_add_f16x2_kernel(half *a,half *b,half *c,int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    idx = idx * 2;
    if(idx < N){
        half2 reg_a = HALF2(a[idx]);
        half2 reg_b = HALF2(b[idx]);
        half2 reg_c = __hadd2(reg_a, reg_b);
        HALF2(c[idx]) = reg_c;
    }
}

__global__ void elementwise_add_f16x8_kernel(half *a,half *b,half *c,int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    idx = idx * 8;
    if(idx < N){
        half2 reg_a0 = HALF2(a[idx]);
        half2 reg_a1 = HALF2(a[idx + 2]);
        half2 reg_a2 = HALF2(a[idx + 4]);
        half2 reg_a3 = HALF2(a[idx + 6]);
        half2 reg_b0 = HALF2(b[idx]);
        half2 reg_b1 = HALF2(b[idx + 2]);
        half2 reg_b2 = HALF2(b[idx + 4]);
        half2 reg_b3 = HALF2(b[idx + 6]);
        half2 reg_c0 = __hadd2(reg_a0, reg_b0);
        half2 reg_c1 = __hadd2(reg_a1, reg_b1);
        half2 reg_c2 = __hadd2(reg_a2, reg_b2);
        half2 reg_c3 = __hadd2(reg_a3, reg_b3);
        HALF2(c[idx]) = reg_c0;
        if(idx + 2 < N)HALF2(c[idx + 2]) = reg_c1;
        if(idx + 4 < N)HALF2(c[idx + 4]) = reg_c2;
        if(idx + 6 < N)HALF2(c[idx + 6]) = reg_c3;
    }
}

__global__ void elementwise_add_f16x8_pack_kernel(half *a,half *b,half *c,int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    idx = idx * 8;
    if(idx < N){
        half pack_a[8],pack_b[8],pack_c[8];
        LDST128BITS(pack_a[0]) = LDST128BITS(a[idx]);
        LDST128BITS(pack_b[0]) = LDST128BITS(b[idx]);
    #pragma unroll
        for(int i = 0;i<8;i+=2){
            //HALF2(c[idx + i]) = __hadd2(HALF2(pack_a[i]), HALF2(pack_b[i]));这样对c的全局内存还是每两个数据访问一次,而pack_c是在寄存器，因此可以优化全局内存的访问
            HALF2(pack_c[i]) = __hadd2(HALF2(pack_a[i]), HALF2(pack_b[i]));
        }
        if(idx + 7 < N)
        LDST128BITS(c[idx]) = LDST128BITS(pack_c[0]);
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

#define TORCH_BINDING_ELEM_ADD(packed_type, th_type, element_type, n_elements) \
  void elementwise_add_##packed_type(torch::Tensor a, torch::Tensor b,         \
                                     torch::Tensor c) {                        \
    CHECK_TORCH_TENSOR_DTYPE(a, (th_type))                                     \
    CHECK_TORCH_TENSOR_DTYPE(b, (th_type))                                     \
    CHECK_TORCH_TENSOR_DTYPE(c, (th_type))                                     \
    const int ndim = a.dim();                                                  \
    if (ndim != 2) {                                                           \
      int N = 1;                                                               \
      for (int i = 0; i < ndim; ++i) {                                         \
        N *= a.size(i);                                                        \
      }                                                                        \
      dim3 block(256 / (n_elements));                                          \
      dim3 grid((N + 256 - 1) / 256);                                          \
      elementwise_add_##packed_type##_kernel<<<grid, block>>>(                 \
          reinterpret_cast<element_type *>(a.data_ptr()),                      \
          reinterpret_cast<element_type *>(b.data_ptr()),                      \
          reinterpret_cast<element_type *>(c.data_ptr()), N);                  \
    } else {                                                                   \
      const int S = a.size(0);                                                 \
      const int K = a.size(1);                                                 \
      const int N = S * K;                                                     \
      if ((K / (n_elements)) <= 1024) {                                        \
        dim3 block(K / (n_elements));                                          \
        dim3 grid(S);                                                          \
        elementwise_add_##packed_type##_kernel<<<grid, block>>>(               \
            reinterpret_cast<element_type *>(a.data_ptr()),                    \
            reinterpret_cast<element_type *>(b.data_ptr()),                    \
            reinterpret_cast<element_type *>(c.data_ptr()), N);                \
      } else {                                                                 \
        int N = 1;                                                             \
        for (int i = 0; i < ndim; ++i) {                                       \
          N *= a.size(i);                                                      \
        }                                                                      \
        dim3 block(256 / (n_elements));                                        \
        dim3 grid((N + 256 - 1) / 256);                                        \
        elementwise_add_##packed_type##_kernel<<<grid, block>>>(               \
            reinterpret_cast<element_type *>(a.data_ptr()),                    \
            reinterpret_cast<element_type *>(b.data_ptr()),                    \
            reinterpret_cast<element_type *>(c.data_ptr()), N);                \
      }                                                                        \
    }                                                                          \
  }

TORCH_BINDING_ELEM_ADD(f32, torch::kFloat32, float, 1)
TORCH_BINDING_ELEM_ADD(f32x4, torch::kFloat32, float, 4)
TORCH_BINDING_ELEM_ADD(f16, torch::kHalf, half, 1)
TORCH_BINDING_ELEM_ADD(f16x2, torch::kHalf, half, 2)
TORCH_BINDING_ELEM_ADD(f16x8, torch::kHalf, half, 8)
TORCH_BINDING_ELEM_ADD(f16x8_pack, torch::kHalf, half, 8)
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f32)
  TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f32x4)
  TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f16)
  TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f16x2)
  TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f16x8)
  TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f16x8_pack)
}