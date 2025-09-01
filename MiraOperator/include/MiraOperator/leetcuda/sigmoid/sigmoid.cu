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
#define MAX_EXP_F32 88.3762626647949f
#define MIN_EXP_F32 -88.3762626647949f
#define MAX_EXP_F16 __float2half(11.089866488461016f)
#define MIN_EXP_F16 __float2half(-9.704060527839234f)
__global__ void sigmoid_f32_kernel(float* x,float* y,int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N){
        float v = fminf(fmaxf(x[idx],MIN_EXP_F32),MAX_EXP_F32);
        y[idx] = 1.0f / (1.0f + expf(-v));
    }
}

__global__ void sigmoid_f32x4_kernel(float* x,float* y,int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    idx *= 4;
    if(idx < N){
        float4 reg_x = FLOAT4(x[idx]);
        float4 reg_y;
        // float v0 = fminf(fmaxf(reg_x.x, MIN_EXP_F32), MAX_EXP_F32);这样会多用4个寄存器
        // float v1 = fminf(fmaxf(reg_x.y, MIN_EXP_F32), MAX_EXP_F32);
        // float v2 = fminf(fmaxf(reg_x.z, MIN_EXP_F32), MAX_EXP_F32);
        // float v3 = fminf(fmaxf(reg_x.w, MIN_EXP_F32), MAX_EXP_F32);
        // reg_y.x = 1.0f / (1.0f + expf(-v0));
        // reg_y.y = 1.0f / (1.0f + expf(-v1));
        // reg_y.z = 1.0f / (1.0f + expf(-v2));
        // reg_y.w = 1.0f / (1.0f + expf(-v3));
        reg_x.x = fminf(fmaxf(reg_x.x, MIN_EXP_F32), MAX_EXP_F32);
        reg_x.y = fminf(fmaxf(reg_x.y, MIN_EXP_F32), MAX_EXP_F32);
        reg_x.z = fminf(fmaxf(reg_x.z, MIN_EXP_F32), MAX_EXP_F32);
        reg_x.w = fminf(fmaxf(reg_x.w, MIN_EXP_F32), MAX_EXP_F32);
        reg_y.x = 1.0f / (1.0f + expf(-reg_x.x));
        reg_y.y = 1.0f / (1.0f + expf(-reg_x.y));
        reg_y.z = 1.0f / (1.0f + expf(-reg_x.z));
        reg_y.w = 1.0f / (1.0f + expf(-reg_x.w));
        FLOAT4(y[idx]) = reg_y;
    }
}

__global__ void sigmoid_f16_kernel(half* x,half *y,int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const half f = __float2half(1.0f);//直接用1.0f会把结果隐式转换成float
    if(idx < N){
        half v = __hmin(__hmax(x[idx],MIN_EXP_F16),MAX_EXP_F16);
        y[idx] = __hdiv(f,__hadd(f,hexp(-v)));
    }
}

__global__ void sigmoid_f16x2_kernel(half* x,half *y,int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    idx *= 2;
    const half f = __float2half(1.0f);
    if(idx < N){
        half2 reg_x = HALF2(x[idx]);
        half2 reg_y;
        reg_x.x = __hmin(__hmax(reg_x.x,MIN_EXP_F16),MAX_EXP_F16);
        reg_x.y = __hmin(__hmax(reg_x.y,MIN_EXP_F16),MAX_EXP_F16);
        reg_y.x = __hdiv(f,__hadd(f,hexp(-reg_x.x)));
        reg_y.y = __hdiv(f,__hadd(f,hexp(-reg_x.y)));
        HALF2(y[idx]) = reg_y;
    }
}

__global__ void sigmoid_f16x8_kernel(half* x,half *y,int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    idx *= 8;
    const half f = __float2half(1.0f);
    if(idx < N){
        half2 reg_x0 = HALF2(x[idx]);
        half2 reg_x1 = HALF2(x[idx + 2]);
        half2 reg_x2 = HALF2(x[idx + 4]);
        half2 reg_x3 = HALF2(x[idx + 6]);
        half2 reg_y0, reg_y1, reg_y2, reg_y3;
        reg_x0.x = __hmin(__hmax(reg_x0.x,MIN_EXP_F16),MAX_EXP_F16);
        reg_x0.y = __hmin(__hmax(reg_x0.y,MIN_EXP_F16),MAX_EXP_F16);
        reg_x1.x = __hmin(__hmax(reg_x1.x,MIN_EXP_F16),MAX_EXP_F16);
        reg_x1.y = __hmin(__hmax(reg_x1.y,MIN_EXP_F16),MAX_EXP_F16);
        reg_x2.x = __hmin(__hmax(reg_x2.x,MIN_EXP_F16),MAX_EXP_F16);
        reg_x2.y = __hmin(__hmax(reg_x2.y,MIN_EXP_F16),MAX_EXP_F16);
        reg_x3.x = __hmin(__hmax(reg_x3.x,MIN_EXP_F16),MAX_EXP_F16);
        reg_x3.y = __hmin(__hmax(reg_x3.y,MIN_EXP_F16),MAX_EXP_F16);
        reg_y0.x = __hdiv(f,__hadd(f,hexp(-reg_x0.x)));
        reg_y0.y = __hdiv(f,__hadd(f,hexp(-reg_x0.y)));
        reg_y1.x = __hdiv(f,__hadd(f,hexp(-reg_x1.x)));
        reg_y1.y = __hdiv(f,__hadd(f,hexp(-reg_x1.y)));
        reg_y2.x = __hdiv(f,__hadd(f,hexp(-reg_x2.x)));
        reg_y2.y = __hdiv(f,__hadd(f,hexp(-reg_x2.y)));
        reg_y3.x = __hdiv(f,__hadd(f,hexp(-reg_x3.x)));
        reg_y3.y = __hdiv(f,__hadd(f,hexp(-reg_x3.y)));
        HALF2(y[idx]) = reg_y0;
        if(idx + 2 < N){
            HALF2(y[idx + 2]) = reg_y1;
        }
        if(idx + 4 < N){
            HALF2(y[idx + 4]) = reg_y2;
        }
        if(idx + 6 < N){
            HALF2(y[idx + 6]) = reg_y3;
        }
    }
}

__global__ void sigmoid_f16x8_pack_kernel(half* x,half *y,int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    idx *= 8;
    const half f = __float2half(1.0f);
    if(idx < N){
        half pack_x[8],pack_y[8];
        LDST128BITS(pack_x[0]) = LDST128BITS(x[idx]); 
        #pragma unroll
        for(int i = 0;i < 8;i++){
            pack_x[i] = __hmin(__hmax(pack_x[i],MIN_EXP_F16),MAX_EXP_F16);
            pack_y[i] = __hdiv(f,__hadd(f,hexp(-pack_x[i])));
        }
        if(idx + 7 < N){
            LDST128BITS(y[idx]) = LDST128BITS(pack_y[0]);
        }
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

#define TORCH_BINDING_SIGMOID(packed_type, th_type, element_type, n_elements)  \
  void sigmoid_##packed_type(torch::Tensor x, torch::Tensor y) {               \
    CHECK_TORCH_TENSOR_DTYPE(x, (th_type))                                     \
    CHECK_TORCH_TENSOR_DTYPE(y, (th_type))                                     \
    const int ndim = x.dim();                                                  \
    if (ndim != 2) {                                                           \
      int N = 1;                                                               \
      for (int i = 0; i < ndim; ++i) {                                         \
        N *= x.size(i);                                                        \
      }                                                                        \
      dim3 block(256 / (n_elements));                                          \
      dim3 grid((N + 256 - 1) / 256);                                          \
      sigmoid_##packed_type##_kernel<<<grid, block>>>(                         \
          reinterpret_cast<element_type *>(x.data_ptr()),                      \
          reinterpret_cast<element_type *>(y.data_ptr()), N);                  \
    } else {                                                                   \
      const int S = x.size(0);                                                 \
      const int K = x.size(1);                                                 \
      const int N = S * K;                                                     \
      if ((K / (n_elements)) <= 1024) {                                        \
        dim3 block(K / (n_elements));                                          \
        dim3 grid(S);                                                          \
        sigmoid_##packed_type##_kernel<<<grid, block>>>(                       \
            reinterpret_cast<element_type *>(x.data_ptr()),                    \
            reinterpret_cast<element_type *>(y.data_ptr()), N);                \
      } else {                                                                 \
        int N = 1;                                                             \
        for (int i = 0; i < ndim; ++i) {                                       \
          N *= x.size(i);                                                      \
        }                                                                      \
        dim3 block(256 / (n_elements));                                        \
        dim3 grid((N + 256 - 1) / 256);                                        \
        sigmoid_##packed_type##_kernel<<<grid, block>>>(                       \
            reinterpret_cast<element_type *>(x.data_ptr()),                    \
            reinterpret_cast<element_type *>(y.data_ptr()), N);                \
      }                                                                        \
    }                                                                          \
  }

TORCH_BINDING_SIGMOID(f32, torch::kFloat32, float, 1)
TORCH_BINDING_SIGMOID(f32x4, torch::kFloat32, float, 4)
TORCH_BINDING_SIGMOID(f16, torch::kHalf, half, 1)
TORCH_BINDING_SIGMOID(f16x2, torch::kHalf, half, 2)
TORCH_BINDING_SIGMOID(f16x8, torch::kHalf, half, 8)
TORCH_BINDING_SIGMOID(f16x8_pack, torch::kHalf, half, 8)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(sigmoid_f32)
  TORCH_BINDING_COMMON_EXTENSION(sigmoid_f32x4)
  TORCH_BINDING_COMMON_EXTENSION(sigmoid_f16)
  TORCH_BINDING_COMMON_EXTENSION(sigmoid_f16x2)
  TORCH_BINDING_COMMON_EXTENSION(sigmoid_f16x8)
  TORCH_BINDING_COMMON_EXTENSION(sigmoid_f16x8_pack)
}