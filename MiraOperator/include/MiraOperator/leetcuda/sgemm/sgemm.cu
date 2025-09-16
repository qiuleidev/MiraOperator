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
__global__ void sgemm_naive_f32_kernel(float *a,float *b,float *c,int M,int N,int K){
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    if(m < M && n < N){
        float pSum = 0.0f;
        #pragma unroll
        for(int k = 0;k < K;k++){
            pSum += a[m * K + k] * b[k * N + n];
        }
        c[m * N + n] = pSum;
    }
}

template <const int BM = 32,const int BN = 32,const int BK = 32>
__global__ void sgemm_sliced_k_f32_kernel(float *a,float *b,float *c,int M,int N,int K){
    __shared__ float s_a[BM][BK],s_b[BK][BN];
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = threadIdx.y * blockDim.x + tx;
    int sam = tid / BM;
    int sak = tid % BK;
    int sbk = tid / BK;
    int sbn = tid % BN;
    int gam = by * BM + sam;
    int gbn = bx * BN + sbn;

    float sum = 0.0f;
    for(int bk = 0;bk < (K + BK - 1)/BK;++bk){
        int gak = bk * BK + sak;
        int ga = gam * K + gak;
        s_a[sam][sak] = a[ga];
        int gbk = bk * BK + sbk;
        int gb = gbk * N + gbn;
        s_b[sbk][sbn] = b[gb];
        __syncthreads();
        #pragma unroll
        for(int k = 0;k < BK;++k){
            sum += s_a[sam][k] * s_b[k][sbn]; 
        }
        __syncthreads();
    }
    int gc = gam * N + gbn;
    c[gc] = sum;
}

// template <const int BM = 128, const int BN = 128, const int BK = 8,const int TM = 8, const int TN = 8>
// __global__ void sgemm_t_8x8_sliced_k_f32x4_kernel(float *a, float *b, float *c,int M, int N, int K) {
//     int bx = blockIdx.x;
//     int by = blockIdx.y;
//     int tx = threadIdx.x;
//     int ty = threadIdx.y;
//     int tid = threadIdx.y * blockDim.x + tx;
//     __shared__ float s_a[BM][BK],s_b[BK][BN];
//     //float r_comp_a[TM];
//     float r_comp_b[TN];
//     int sam = tid >> 1;
//     int sak = (tid & 1) << 2;
//     int sbk = tid >> 5;
//     int sbn = (tid & 31) << 2;
//     int gam = by * BM + sam;
//     int gbn = bx * BN + sbn;
//     if(gam >= M || gbn >= N)return;
//     float r_c[TM][TN] = {0.0f};
//     for(int bk = 0;bk < (K + BK - 1) / BK;++bk){//一个线程计算8*8的数据，要计算整个K维度
//         //一个线程读取16B数据
//         int gak = bk * BK + sak;
//         int ga = gam * K + gak;
//         FLOAT4(s_a[sam][sak]) = FLOAT4(a[ga]);
//         int gbk = bk * BK + sbk;
//         int gb = gbk * N + gbn;
//         FLOAT4(s_b[sbk][sbn]) = FLOAT4(b[gb]);
//         __syncthreads();
//         //一个线程处理一个tile的计算,计算的结果存在r_c中
//         #pragma unroll
//         for(int k = 0;k < BK;++k){
//             // FLOAT4(r_comp_a[0]) = FLOAT4(s_a[ty * TM][0]);
//             // FLOAT4(r_comp_a[4]) = FLOAT4(s_a[ty * TM][4]);
//             FLOAT4(r_comp_b[0]) = FLOAT4(s_b[k][tx * TN]);
//             FLOAT4(r_comp_b[4]) = FLOAT4(s_b[k][tx * TN + 4]);
//             #pragma unroll
//             for(int m = 0;m < TM;++m){
//                 #pragma unroll
//                 for(int n = 0;n < TN;++n){
//                     r_c[m][n] = __fmaf_rn(s_a[ty * TM + m][k], r_comp_b[n], r_c[m][n]);
//                 }
//             }
//         }
//         __syncthreads();
//     }
//     #pragma unroll
//     for(int m = 0;m < TM;++m){
//         int gcm = by * BM + ty * TM + m;
//         #pragma unroll
//         for(int n = 0;n < TN;n+=4){
//             int gcn = bx * BN + tx * TN +n;
//             int gc = gcm * N + gcn;
//             FLOAT4(c[gc]) = FLOAT4(r_c[m][n]);
//         }
//     }
// }
template<const int BM = 128,const int BN = 128,const int BK = 8,const int TM = 8,const int TN = 8>
__global__ void sgemm_t_8x8_sliced_k_f32x4_kernel(float *a, float *b, float *c,int M, int N, int K){
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = ty * blockDim.x + tx;
    __shared__ float s_a[BM][BK],s_b[BK][BN];
    int sam = tid >> 1;
    int sak = (tid & 1) << 2;
    int sbk = tid >> 5;
    int sbn = (tid & 31) << 2;
    int gam = by * BM + sam;
    int gbn = bx * BN + sbn;
    if(gam >= M || gbn >= N)return;
    float r_c[TM][TN] = {0.0f};
    for(int k = 0;k < (K + BK - 1) / BK;++k){
        int gak = k * BK + sak;
        int ga = gam * K + gak;
        FLOAT4(s_a[sam][sak]) = FLOAT4(a[ga]);
        int gbk = k * BK + sbk;
        int gb = gbk * N + gbn;
        FLOAT4(s_b[sbk][sbn]) = FLOAT4(b[gb]);
        __syncthreads();
        #pragma unroll
        for(int bk = 0;bk < BK;++bk){
            #pragma unroll
            for(int m = 0;m < TM;++m){
                #pragma unroll
                for(int n = 0;n <TN;++n){
                    r_c[m][n] = s_a[ty * TM + m][bk] * s_b[bk][tx * TN + n];
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int m = 0;m <TM;++m){
        int gcm = by * BM + ty * TM + m;
        #pragma unroll
        for(int n = 0;n < TN;n+=4){
            int gcn = bx * BN + tx * TN + n;
            int gc = gcm * N + gcn;
            FLOAT4(c[gc]) = FLOAT4(r_c[m][n]);
        }
    }
}
template<const int BM = 128,const int BN = 128,const int BK = 8,const int TM = 8,const int TN = 8,const int OFFSET = 4>
__global__ void sgemm_t_8x8_sliced_k_f32x4_bcf_kernel(float* a,float* b,float* c,const int M,const int N,const int K){
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;
    __shared__ float s_a[BK][BM + OFFSET];
    __shared__ float s_b[BK][BM + OFFSET];
    float r_a[TM >> 1];
    float r_b[TN >> 1];
    float r_comp_a[TM];
    float r_comp_b[TN];
    float r_c[TM][TN] = {0.0};
    int sam = tid >> 1;
    int sak = (tid & 1) << 2;
    int sbk = tid >> 5;
    int sbn = (tid & 31) << 2;
    int gam = by * BM + sam;
    int gbn = bx * BN + sbn;
    if(gam >= M || gbn >= N)return;
    for(int bk = 0;bk < (K + BK - 1) / BK;++bk){
        int gak = bk * BK + sak;
        int ga = gam * K + gak;
        int gbk = bk * BK + sbk;
        int gb = gbk * N + sbn;
        FLOAT4(r_a[0]) = FLOAT4(a[ga]);
        FLOAT4(r_b[0]) = FLOAT4(b[gb]);
        s_a[sak][sam] = r_a[0];
        s_a[sak + 1][sam] = r_a[1];
        s_a[sak + 2][sam] = r_a[2];
        s_a[sak + 3][sam] = r_a[3];
        FLOAT4(s_b[sbk][sbn]) = FLOAT4(r_b[0]);
        __syncthreads();
    }
    #pragma unroll
    for(int tk = 0;tk < BK;++tk){
        
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

#define CHECK_TORCH_TENSOR_SHAPE(T, S0, S1)                                    \
  if (((T).size(0) != (S0)) || ((T).size(1) != (S1))) {                        \
    throw std::runtime_error("Tensor size mismatch!");                         \
  }

// SGEMM naive: compute one c[i,j] element per threads, all row major
void sgemm_naive_f32(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kFloat32)
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1);
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)
  constexpr int BM = 32;
  constexpr int BN = 32;

  dim3 block(BN, BM);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

  sgemm_naive_f32_kernel<<<grid, block>>>(
      reinterpret_cast<float *>(a.data_ptr()),
      reinterpret_cast<float *>(b.data_ptr()),
      reinterpret_cast<float *>(c.data_ptr()), M, N, K);

  
}

void sgemm_sliced_k_f32(torch::Tensor a,torch::Tensor b,torch::Tensor c){
  CHECK_TORCH_TENSOR_DTYPE(a,torch::kFloat32);
  CHECK_TORCH_TENSOR_DTYPE(b,torch::kFloat32);
  CHECK_TORCH_TENSOR_DTYPE(c,torch::kFloat32);
  const int M = a.size(0);
  const int N = b.size(1);
  const int K = a.size(1);
  CHECK_TORCH_TENSOR_SHAPE(a, M, K);
  CHECK_TORCH_TENSOR_SHAPE(b, K, N);
  CHECK_TORCH_TENSOR_SHAPE(c, M, N);
  constexpr int BM = 32;
  constexpr int BN = 32;
  constexpr int BK = 32;

  dim3 block(BN,BM);
  dim3 grid((N + BN - 1) / BN,(M + BM - 1) / BM);
  sgemm_sliced_k_f32_kernel<BM,BN,BK><<<grid,block>>>(reinterpret_cast<float *>(a.data_ptr()),reinterpret_cast<float *>(b.data_ptr()),reinterpret_cast<float *>(c.data_ptr()),M,N,K);
}

void sgemm_t_8x8_sliced_k_f32x4(torch::Tensor a,torch::Tensor b,torch::Tensor c){
  // CHECK_TORCH_TENSOR_DTYPE(a,torch::kFloat32);
  // CHECK_TORCH_TENSOR_DTYPE(b,torch::kFloat32);
  // CHECK_TORCH_TENSOR_DTYPE(c,torch::kFloat32);
  // const int M = a.size(0);
  // const int N = b.size(1);
  // const int K = a.size(1);
  // CHECK_TORCH_TENSOR_SHAPE(a, M, K);
  // CHECK_TORCH_TENSOR_SHAPE(b, K, N);
  // CHECK_TORCH_TENSOR_SHAPE(c, M, N);
  // constexpr int BM = 128;
  // constexpr int TM = 8;
  // constexpr int BN = 128;
  // constexpr int TN = 8;
  // constexpr int BK = 8;
  // dim3 block(BN/TN,BM/TM);
  // dim3 grid((N + BN - 1) / BN,(M + BM - 1) / BM);
  // sgemm_t_8x8_sliced_k_f32x4_kernel<BM,BN,BK,TM,TN><<<grid,block>>>(reinterpret_cast<float *>(a.data_ptr()),reinterpret_cast<float *>(b.data_ptr()),reinterpret_cast<float *>(c.data_ptr()),M,N,K);
  const int M = a.size(0);
  const int N = b.size(1);
  const int K = a.size(1);
  constexpr int BM = 128;
  constexpr int BN = 128;
  constexpr int BK = 8;
  constexpr int TM = 8;
  constexpr int TN = 8;
  dim3 block(BN / TN,BM / TM);
  dim3 grid((N + BN - 1) / BN,(M + BM - 1) / BM);
  sgemm_t_8x8_sliced_k_f32x4_kernel<BM,BN,BK,TM,TN><<<grid,block>>>(reinterpret_cast<float*>(a.data_ptr()),reinterpret_cast<float*>(b.data_ptr()),reinterpret_cast<float*>(c.data_ptr()),M,N,K);
}

void sgemm_t_8x8_sliced_k_f32x4_bcf(torch::Tensor a, torch::Tensor b,
                                    torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kFloat32)
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1);
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)
  constexpr int BM = 128;
  constexpr int BN = 128;
  constexpr int BK = 8;
  constexpr int TM = 8;
  constexpr int TN = 8;

  dim3 block(BN / TN, BM / TM);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

  sgemm_t_8x8_sliced_k_f32x4_bcf_kernel<BM, BN, BK, TM, TN>
      <<<grid, block>>>(reinterpret_cast<float *>(a.data_ptr()),
                        reinterpret_cast<float *>(b.data_ptr()),
                        reinterpret_cast<float *>(c.data_ptr()), M, N, K);
}

void sgemm_t_8x8_sliced_k_f32x4_bcf_offset(torch::Tensor a, torch::Tensor b,
                                           torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kFloat32)
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1);
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)
  constexpr int BM = 128;
  constexpr int BN = 128;
  constexpr int BK = 8;
  constexpr int TM = 8;
  constexpr int TN = 8;
  constexpr int OFFSET = 4;

  dim3 block(BN / TN, BM / TM);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

  sgemm_t_8x8_sliced_k_f32x4_bcf_kernel<BM, BN, BK, TM, TN, OFFSET>
      <<<grid, block>>>(reinterpret_cast<float *>(a.data_ptr()),
                        reinterpret_cast<float *>(b.data_ptr()),
                        reinterpret_cast<float *>(c.data_ptr()), M, N, K);
}
// from sgemm_async.cu
void sgemm_t_8x4_sliced_k16_f32x4_bcf_dbuf(torch::Tensor a, torch::Tensor b,
                                           torch::Tensor c);
void sgemm_t_8x4_sliced_k16_f32x4_bcf_dbuf_async(torch::Tensor a,
                                                 torch::Tensor b,
                                                 torch::Tensor c);
void sgemm_t_8x8_sliced_k16_f32x4_bcf_dbuf(torch::Tensor a, torch::Tensor b,
                                           torch::Tensor c);
void sgemm_t_8x8_sliced_k16_f32x4_bcf_dbuf_async(torch::Tensor a,
                                                 torch::Tensor b,
                                                 torch::Tensor c);
void sgemm_t_8x16_sliced_k16_f32x4_bcf_dbuf(torch::Tensor a, torch::Tensor b,
                                            torch::Tensor c);
void sgemm_t_8x16_sliced_k16_f32x4_bcf_dbuf_async(torch::Tensor a,
                                                  torch::Tensor b,
                                                  torch::Tensor c);
// from sgemm_cublas.cu
void sgemm_cublas(torch::Tensor a, torch::Tensor b, torch::Tensor c);
void sgemm_cublas_tf32(torch::Tensor a, torch::Tensor b, torch::Tensor c);
// from sgemm_wmma_tf32_stage.cu
void sgemm_wmma_m16n16k8_mma4x2_warp2x4_stage2(torch::Tensor a, torch::Tensor b,
                                               torch::Tensor c);
void sgemm_wmma_m16n16k8_mma4x2_warp2x4_stage2_offset(torch::Tensor a,
                                                      torch::Tensor b,
                                                      torch::Tensor c);
void sgemm_wmma_m16n16k8_mma4x2_warp2x4_stage3(torch::Tensor a, torch::Tensor b,
                                               torch::Tensor c);
void sgemm_wmma_m16n16k8_mma4x2_warp2x4_stage3_offset(torch::Tensor a,
                                                      torch::Tensor b,
                                                      torch::Tensor c);

void sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages(torch::Tensor a, torch::Tensor b,
                                               torch::Tensor c, int stages,
                                               bool swizzle,
                                               int swizzle_stride);
void sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages_dsmem(torch::Tensor a,
                                                     torch::Tensor b,
                                                     torch::Tensor c,
                                                     int stages, bool swizzle,
                                                     int swizzle_stride);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // CUDA Cores
  TORCH_BINDING_COMMON_EXTENSION(sgemm_naive_f32)
  TORCH_BINDING_COMMON_EXTENSION(sgemm_sliced_k_f32)
  TORCH_BINDING_COMMON_EXTENSION(sgemm_t_8x8_sliced_k_f32x4)
  TORCH_BINDING_COMMON_EXTENSION(sgemm_t_8x8_sliced_k_f32x4_bcf)
  TORCH_BINDING_COMMON_EXTENSION(sgemm_t_8x8_sliced_k_f32x4_bcf_offset)
//   TORCH_BINDING_COMMON_EXTENSION(sgemm_t_8x8_sliced_k_f32x4_bcf_dbuf)
//   TORCH_BINDING_COMMON_EXTENSION(sgemm_t_8x8_sliced_k_f32x4_bcf_dbuf_offset)
//   TORCH_BINDING_COMMON_EXTENSION(sgemm_t_8x4_sliced_k16_f32x4_bcf_dbuf)
//   TORCH_BINDING_COMMON_EXTENSION(sgemm_t_8x4_sliced_k16_f32x4_bcf_dbuf_async)
//   TORCH_BINDING_COMMON_EXTENSION(sgemm_t_8x8_sliced_k16_f32x4_bcf_dbuf)
//   TORCH_BINDING_COMMON_EXTENSION(sgemm_t_8x8_sliced_k16_f32x4_bcf_dbuf_async)
//   TORCH_BINDING_COMMON_EXTENSION(sgemm_t_8x16_sliced_k16_f32x4_bcf_dbuf)
//   TORCH_BINDING_COMMON_EXTENSION(sgemm_t_8x16_sliced_k16_f32x4_bcf_dbuf_async)
//   // cuBLAS Tensor Cores
//   TORCH_BINDING_COMMON_EXTENSION(sgemm_cublas)
//   TORCH_BINDING_COMMON_EXTENSION(sgemm_cublas_tf32)
//   // WMMA API Tensor Cores, stage, thread block swizzle, dsmem
//   TORCH_BINDING_COMMON_EXTENSION(sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages)
//   TORCH_BINDING_COMMON_EXTENSION(
//       sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages_dsmem)
}