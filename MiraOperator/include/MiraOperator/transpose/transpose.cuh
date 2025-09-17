#pragma once
namespace MiraOperator{
    template<const int Bm,const int Bn>
    __global__ void transpose_f32_kernel(float* A,float* B,const int M,const int N){
        __shared__ float smem[Bm][Bn];
        int r0 = blockIdx.y * Bm;
        int c0 = blockIdx.x * Bn;
        #pragma unroll
        for(int y = threadIdx.y;y < Bm;y += blockDim.y){
            int r = r0 + y;
            if(r >= M)break;
            #pragma unroll
            for(int x = threadIdx.x;x < Bn;x += blockDim.x){
                int c = c0 + x;
                if(c < N)smem[y][x ^ y] = A[r * N + c];
            }
        }
        __syncthreads();

        #pragma unroll
        for(int y = threadIdx.y;y < Bn;y += blockDim.y){
            int c = c0 + y;
            if(c >= N)break;
            #pragma unroll
            for(int x = threadIdx.x;x < Bm;x += blockDim.x){
                int r = r0 + x;
                if(r < M)B[c * M + r] = smem[x][x ^ y];
            }
        }
    }
    // //A,B均为行主序
    // //原来是 iy*N + ix，转置后还要行优先，但变成了N*M矩阵，也就是对应的ix，iy不变，但是ix代表纵轴，iy代表横轴了
    // __global__ void transposeRowV0(float* A,float* B,const int M,const int N){
    //     int ix = blockIdx.x * blockDim.x + threadIdx.x;
    //     int iy = blockIdx.y * blockDim.y + threadIdx.y;
    //     if(iy < M && ix < N)B[ix * M + iy] = A[iy * N + ix];
    // }
    // //A,B均为行主序
    // //按列读取，按行写入，相当于ix方向处理M，iy方向处理N，ix是纵轴，iy是横轴->ix是横轴，iy是纵轴
    // __global__ void transposeColV0(float* A,float* B,const int M,const int N){
    //     int ix = blockIdx.x * blockDim.x + threadIdx.x;
    //     int iy = blockIdx.y * blockDim.y + threadIdx.y;
    //     if(iy < N && ix < M)B[iy * M + ix] = A[ix * N + iy];
    // }
    
    // template<const int Bm,const int Bn>
    // __global__ void transposeColV1(float* A,float* B,const int M,const int N){
    //     int r0 = blockIdx.x * Bm;
    //     int c0 = blockIdx.y * Bn;
    //     #pragma unroll
    //     for(int x = threadIdx.x;x < Bm;x+=blockDim.x){
    //         int r = r0 + x;
    //         if(r >= M)break;
    //         #pragma unroll
    //         for(int y = threadIdx.y;y < Bn;y+=blockDim.y){
    //             int c = c0 + y;
    //             if(c < N)B[c * M + r] = A[r * N + c];
    //         }
    //     }
    // // }
    // //32-way bank conflict
    // template<const int Bm,const int Bn>
    // __global__ void transposeShared(float* A,float* B,const int M,const int N){
    //     __shared__ float smem[Bm][Bn];
    //     int r0 = blockIdx.y * Bm;
    //     int c0 = blockIdx.x * Bn;
    //     #pragma unroll
    //     for(int y = threadIdx.y;y < Bm;y+=blockDim.y){
    //         int r = r0 + y;
    //         if(r >= M)break;
    //         #pragma unroll
    //         for(int x = threadIdx.x;x < Bn;x+=blockDim.x){
    //             int c = c0 + x;
    //             if(c < n)smem[y][x] = A[r * N + c];
    //         }
    //     }
    //     __syncthreads();

    //     //（r0 , c0）M * N -> (c0 , r0) N * M
    //     #pragma unroll
    //     for(int y = threadIdx.y;y < Bn;y+=blockDim.y){
    //         int c = c0 + y;
    //         if(c >= N)break;
    //         #pragma unroll
    //         for(int x = threadIdx.x;x < Bm;x+=blockDim.x){
    //             int r = r0 + x;
    //             if(r < M)B[c * M + r] = smem[x][y];
    //         }
    //     }
    // }

}