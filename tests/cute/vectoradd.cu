#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cute/layout.hpp>
#include <cuda.h>
#include <cublas_v2.h>
#include <stdlib.h>
//z = ax +by +c
// template<int kNumElemThread = 8>
// __global__ void vector_add_local_tile_multi_elem_per_thread_half(half* z,int num,const half* x,const half* y,const half a,const half b,const half c){
//     using namespace cute;
//     int idx = threadIdx.x + blockDim.x * blockIdx.x;
//     if(idx >= num / kNumElemThread){
//         return;//忽略非对齐问题
//     }
//     Tensor tz = make_tensor(make_gmem_ptr(z),make_shape(num));
//     Tensor tx = make_tensor(make_gmem_ptr(x),make_shape(num));
//     Tensor ty = make_tensor(make_gmem_ptr(y),make_shape(num));

//     Tensor tzr = local_tile(tz, make_shape(Int<kNumElemThread>{}), make_coord(idx));//局部切片
//     Tensor txr = local_tile(tx, make_shape(Int<kNumElemThread>{}), make_coord(idx));
//     Tensor tyr = local_tile(ty, make_shape(Int<kNumElemThread>{}), make_coord(idx));

//     Tensor txR = make_tensor_like(txr);
//     Tensor tyR = make_tensor_like(tyr);
//     Tensor tzR = make_tensor_like(tzr);

//     //全局内存数据读入寄存器空间，生成LDG.128指令（128位），因此最好对齐16字节。
//     copy(txr, txR);
//     copy(tyr, tyR);

//     half2 a2 = {a, a};
//     half2 b2 = {b, b};
//     half2 c2 = {c, c};

//     auto tzR2 = recast<half2>(tzR);
//     auto txR2 = recast<half2>(txR);
//     auto tyR2 = recast<half2>(tyR);

// #pragma unroll
//     for(int i = 0; i < kNumElemThread; ++i){
//         //pragma 及后续for行实现了多个元素的z = ax + by + c的计算，并且通过括号将该计算通过两个HFMA2指令实现，如果没有括号，则其会生成 HMUL2 + HMUL2 + HADD2 + HADD2指令
//         tzR2(i) = txR2(i) * a2 + (tyR2(i) * b2 + c2);
//     }
//     auto tzRx = recast<half>(tzR2);
//     //寄存器空间数据写入全局内存，生成STG.128指令（128位），因此最好对齐16字节。
//     copy(tzRx, tzr);
// }

// int main(){
//     using namespace cute;

//     constexpr int N = 1024 * 1024 * 1024; // 元素总数
//     constexpr int kNumElemThread = 8;
//     constexpr int threads_per_block = 128;
//     constexpr int num_blocks = (N / kNumElemThread + threads_per_block - 1) / threads_per_block;

//     // 分配主机内存
//     std::vector<half> h_x(N, __float2half(1.0f));
//     std::vector<half> h_y(N, __float2half(2.0f));
//     std::vector<half> h_z(N, __float2half(0.0f));

//     half a = __float2half(2.0f);
//     half b = __float2half(3.0f);
//     half c = __float2half(4.0f);

//     // 分配设备内存
//     half *d_x, *d_y, *d_z;
//     cudaMalloc(&d_x, N * sizeof(half));
//     cudaMalloc(&d_y, N * sizeof(half));
//     cudaMalloc(&d_z, N * sizeof(half));

//     // 拷贝数据到设备
//     cudaMemcpy(d_x, h_x.data(), N * sizeof(half), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_y, h_y.data(), N * sizeof(half), cudaMemcpyHostToDevice);

//     // 启动 kernel
//     vector_add_local_tile_multi_elem_per_thread_half<kNumElemThread><<<num_blocks, threads_per_block>>>(
//         d_z, N, d_x, d_y, a, b, c
//     );
//     cudaDeviceSynchronize();

//     // 拷贝结果回主机
//     cudaMemcpy(h_z.data(), d_z, N * sizeof(half), cudaMemcpyDeviceToHost);

//     // 简单校验
//     for (int i = 0; i < 10; ++i) {
//         float expected = 2.0f * 1.0f + 3.0f * 2.0f + 4.0f; // a*x + b*y + c
//         float result = __half2float(h_z[i]);
//         std::cout << "z[" << i << "] = " << result << " (expected " << expected << ")\n";
//         assert(fabs(result - expected) < 1e-2);
//     }

//     // 释放内存
//     cudaFree(d_x);
//     cudaFree(d_y);
//     cudaFree(d_z);

//     std::cout << "Test passed!\n";
//     return 0;
// }

template <typename T>
void gen_rand_data(T *data, int n);

template <typename T, int kTileM, int kTileN, int kTileK, typename TiledMMA>
__global__ void gemm_simple(T *Cptr, const T *Aptr, const T *Bptr, int m, int n, int k) {

  using namespace cute;

  Tensor A = make_tensor(make_gmem_ptr(Aptr), make_shape(m, k), make_stride(k, Int<1>{}));
  Tensor B = make_tensor(make_gmem_ptr(Bptr), make_shape(n, k), make_stride(k, Int<1>{}));
  Tensor C = make_tensor(make_gmem_ptr(Cptr), make_shape(m, n), make_stride(n, Int<1>{}));

  int ix = blockIdx.x;
  int iy = blockIdx.y;

  Tensor gA = local_tile(A, make_tile(Int<kTileM>{}, Int<kTileK>{}), make_coord(iy, _));
  Tensor gB = local_tile(B, make_tile(Int<kTileN>{}, Int<kTileK>{}), make_coord(ix, _));
  Tensor gC = local_tile(C, make_tile(Int<kTileM>{}, Int<kTileN>{}), make_coord(iy, ix));
  //  gA(kTileM, kTileK, num_tile_k)
  //  gB(kTileN, kTileK, num_tile_k)
  //  gC(kTileM, kTileN) 

  TiledMMA tiled_mma;
  auto thr_mma = tiled_mma.get_slice(threadIdx.x);
  auto tAgA = thr_mma.partition_A(gA);  // (MMA, MMA_M, MMA_K, num_tile_k)
  auto tBgB = thr_mma.partition_B(gB);  // (MMA, MMA_N, MMA_K, num_tile_k)
  auto tCgC = thr_mma.partition_C(gC);  // (MMA, MMA_M, MMA_N)

  auto tArA = thr_mma.partition_fragment_A(gA(_, _, 0));  // (MMA, MMA_M, MMA_K)
  auto tBrB = thr_mma.partition_fragment_B(gB(_, _, 0));  // (MMA, MMA_N, MMA_K)
  auto tCrC = thr_mma.partition_fragment_C(gC(_, _));     // (MMA, MMA_M, MMA_N)
 
  clear(tCrC);
  
  int num_tile_k = size<2>(gA);
#pragma unroll 1
  for(int itile = 0; itile < num_tile_k; ++itile) {
    cute::copy(tAgA(_, _, _, itile), tArA);
    cute::copy(tBgB(_, _, _, itile), tBrB);

    cute::gemm(tiled_mma, tCrC, tArA, tBrB, tCrC);
  }

  cute::copy(tCrC, tCgC); 
}

int main() {
  srand(10086);

  using T = cute::half_t;
  using namespace cute;

  T *Cptr;
  T *Aptr;
  T *Bptr;

  int m = 81920;
  int n = 256;
  int k = 256;

  cudaMalloc(&Cptr, sizeof(T) * m * n);
  cudaMalloc(&Aptr, sizeof(T) * m * k);
  cudaMalloc(&Bptr, sizeof(T) * k * n);

  T *Aptr_host;
  T *Bptr_host;
  Aptr_host = (T*)malloc(sizeof(T) * m * k);
  Bptr_host = (T*)malloc(sizeof(T) * n * k);
  gen_rand_data(Aptr_host, m * k);
  gen_rand_data(Bptr_host, n * k);

  cudaMemcpy(Aptr, Aptr_host, sizeof(T) * m * k, cudaMemcpyHostToDevice);
  cudaMemcpy(Bptr, Bptr_host, sizeof(T) * n * k, cudaMemcpyHostToDevice);

  using mma_op = SM80_16x8x16_F16F16F16F16_TN;
  using mma_traits = MMA_Traits<mma_op>;
  using mma_atom = MMA_Atom<mma_traits>;

  using MMA = decltype(make_tiled_mma(mma_atom{}, 
                      make_layout(Shape<_2, _2, _1>{}), 
                      Tile<_32, _32, _16>{}));
  constexpr int kTileM = 128; 
  constexpr int kTileN = 128; 
  constexpr int kTileK = 32; 

  dim3 block(size(MMA{}));
  dim3 grid(n / kTileN, m / kTileM);
  for (int i = 0; i < 100; ++i) {
    gemm_simple<T, kTileM, kTileN, kTileK, MMA><<<grid, block>>>(Cptr, Aptr, Bptr, m, n, k);
  }
  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  printf("err = %d, str = %s\n", err, cudaGetErrorString(err));

  // cublas
  T *Cptr_cublas;

  cudaMalloc(&Cptr_cublas, sizeof(T) * m * n);

  cublasHandle_t handle;
  cublasCreate(&handle);

  half alpha = half(1.f);
  half beta = half(0.f);
  for (int i = 0; i < 100; ++i) {
    cublasStatus_t ret = cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
          	  n, m, k,
          	  &alpha,
          	  (half *)Bptr, k,
          	  (half *)Aptr, k,
          	  &beta,
          	  (half *)Cptr_cublas, n);
    if (ret != CUBLAS_STATUS_SUCCESS) {
      printf("blas err = %d, str = %s\n", ret, cublasGetStatusString(ret));
    }
  }

  cudaDeviceSynchronize();
  err = cudaGetLastError();
  printf("err = %d, str = %s\n", err, cudaGetErrorString(err));

  T *Cptr_host;
  T *Cptr_cublas_host;

  Cptr_host = (T*)malloc(sizeof(T) * m * n);
  Cptr_cublas_host = (T*)malloc(sizeof(T) * m * n);

  // compare
  cudaMemcpy(Cptr_host, Cptr, sizeof(T) * m * n, cudaMemcpyDeviceToHost);
  cudaMemcpy(Cptr_cublas_host, Cptr_cublas, sizeof(T) * m * n, cudaMemcpyDeviceToHost);

  float threshold = 0.1;
  for (int i = 0; i < m * n; ++i) {
    float v1 = Cptr_host[i];
    float v2 = Cptr_cublas_host[i];
    if (fabs(v2 - v1) > threshold) {
      printf("v1 = %f, v2 = %f\n", v1, v2);
    }
  }

  Tensor tensor_C = make_tensor(Cptr_host, make_shape(m, n), make_stride(n, 1));
  Tensor tensor_C_cublas = make_tensor(Cptr_cublas_host, make_shape(m, n), make_stride(n, 1));

  auto tile = make_tile(8, 8);
  auto coor = make_coord(0, 0);
  Tensor tc1 = local_tile(tensor_C, tile, coor);
  Tensor tc1_cublas = local_tile(tensor_C_cublas, tile, coor);

  print_tensor(tc1);
  print_tensor(tc1_cublas);
}

template <typename T>
void gen_rand_data(T *data, int n) {
  for (int i = 0; i < n; ++i) {
    float v = (rand() % 200 - 100) * 0.01;
    data[i] = v;
  }
}