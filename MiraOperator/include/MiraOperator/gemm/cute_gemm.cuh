#pragma once
#include <cute/arch/copy_sm75.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <cute/layout.hpp>
#include <cute/layout_composed.hpp>
#include <cute/numeric/math.hpp>
#include <cute/tensor.hpp>

namespace MiraOperator{
    template<typename T,typename Config>
    __global__ void cute_gemm_fp16(T *Cptr, 
        const T *Aptr, 
        const T *Bptr,
        int m, int n, int k){
            using namespace cute;

            using SmemLayoutA = typename Config::SmemLayoutA;
            using SmemLayoutB = typename Config::SmemLayoutB;
            using SmemLayoutC = typename Config::SmemLayoutC;
            using TiledMMA = typename Config::MMA;

            using S2RCopyA = typename Config::S2RCopyA;
            using S2RCopyB = typename Config::S2RCopyB;
            using G2SCopyA = typename Config::G2SCopyA;
            using G2SCopyB = typename Config::G2SCopyB;
            using R2SCopyC = typename Config::R2SCopyC;
            using S2GCopyC = typename Config::S2GCopyC;
            constexpr int kTileM = Config::kTileM;
            constexpr int kTileN = Config::kTileN;
            constexpr int kTileK = Config::kTileK;
            constexpr int kStage = Config::kStage;
            int bx = blockIdx.x;
            int by = blockIdx.y;
            int idx = threadIdx.x;

            Tensor mA = make_tensor(make_gmem_ptr(Aptr),make_shape(m,k),make_stride(k,Int<1>{}));
            Tensor mB = make_tensor(make_gmem_ptr(Bptr),make_shape(n,k),make_stride(k,Int<1>{}));//B Transpose
            Tensor mC = make_tensor(make_gmem_ptr(Cptr),make_shape(m,n),make_stride(n,Int<1>{}));

            Tensor gA = local_tile(mA, make_tile(Int<kTileM>{}, Int<kTileK>{}), make_coord(by, _));
            Tensor gB = local_tile(mB, make_tile(Int<kTileN>{}, Int<kTileK>{}), make_coord(bx, _));
            Tensor gC = local_tile(mC, make_tile(Int<kTileM>{}, Int<kTileN>{}), make_coord(by, bx));

            extern __shared__ T smemA[];//a and b //extern __shared__ T* smemA 只会取第一个地址！
            T *smemB = smemA + cosize(SmemLayoutA{});

            Tensor sA = make_tensor(make_smem_ptr(smemA), SmemLayoutA{});// (kTileM, kTileK, kStage)
            Tensor sB = make_tensor(make_smem_ptr(smemB), SmemLayoutB{});// (kTileN, kTileK, kStage)

            // gmem -cp.async-> shm -ldmatrix-> reg
            G2SCopyA g2s_tiled_copy_a;
            auto g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(idx);
            Tensor tAgA = g2s_thr_copy_a.partition_S(gA);  // (CPY, CPY_M, CPY_K, k)
            Tensor tAsA = g2s_thr_copy_a.partition_D(sA);  // (CPY, CPY_M, CPY_K, kStage)

            G2SCopyB g2s_tiled_copy_b;
            auto g2s_thr_copy_b = g2s_tiled_copy_b.get_slice(idx);
            Tensor tBgB = g2s_thr_copy_b.partition_S(gB);  // (CPY, CPY_N, CPY_K, k)
            Tensor tBsB = g2s_thr_copy_b.partition_D(sB);  // (CPY, CPY_N, CPY_K, kStage)
        //submit kStage - 1 tile
            // if (thread0())
            //     {
            //         // print("\n\nThrCopy A g2s: ");
            //         // print(thr_copy_a);
            //         print("\n\ntAgA_g2s: ");
            //         print(tAgA);
            //         print("\ntAsA_g2s: ");
            //         print(tAsA);
            //         print("\n");
            //         // print("\nThrCopy B g2s: ");
            //         // print(thr_copy_b);
            //         print("\ntBgB_g2s: ");
            //         print(tBgB);
            //         print("\ntBsB_g2s: ");
            //         print(tBsB);
            //         print("\n");
            //     }

            // Current tile index in gmem to read from
            int k_tile_next = 0;
            int k_tile_count = size<3>(tAgA);
            CUTE_UNROLL
            for (int k_pipe = 0; k_pipe < kStage - 1; ++k_pipe)
            {
                copy(g2s_tiled_copy_a, tAgA(_, _, _, k_tile_next), tAsA(_, _, _, k_pipe));
                copy(g2s_tiled_copy_b, tBgB(_, _, _, k_tile_next), tBsB(_, _, _, k_pipe));
                cp_async_fence();
                --k_tile_count;
                if (k_tile_count > 0)
                {
                    ++k_tile_next;
                }
            }

            TiledMMA tiled_mma;
            auto thr_mma = tiled_mma.get_slice(idx);
            Tensor tCrA = thr_mma.partition_fragment_A(gA(_, _, 0));  // (MMA, MMA_M, MMA_K)
            Tensor tCrB = thr_mma.partition_fragment_B(gB(_, _, 0));  // (MMA, MMA_N, MMA_K)
            Tensor tCrC = thr_mma.partition_fragment_C(gC);           // (MMA, MMA_M, MMA_N)

            // fill zero for accumulator
            clear(tCrC);

            S2RCopyA s2r_tiled_copy_a;
            auto s2r_thr_copy_a = s2r_tiled_copy_a.get_slice(idx);
            Tensor tCsA = s2r_thr_copy_a.partition_S(sA);  //  (CPY, CPY_M, CPY_K, kStage)
            Tensor tCrA_view = s2r_thr_copy_a.retile_D(tCrA);  //  (CPY, CPY_M, CPY_K)

            S2RCopyB s2r_tiled_copy_b;
            auto s2r_thr_copy_b = s2r_tiled_copy_b.get_slice(idx);
            Tensor tCsB = s2r_thr_copy_b.partition_S(sB);  //  (CPY, CPY_M, CPY_K, kStage)
            Tensor tCrB_view = s2r_thr_copy_b.retile_D(tCrB);  //  (CPY, CPY_M, CPY_K)
            // if (thread0())
            // {
            //     print("\n\ntCrA: ");
            //     print(tCrA);
            //     print("\ntCrB: ");
            //     print(tCrB);
            //     print("\ntCrC: ");
            //     print(tCrC);
            //     print("\ntCsA: ");
            //     print(tCsA);
            //     print("\ntCsB: ");
            //     print(tCsB);
            //     print("\n");

            // }
            // Current pipe index in smem to read from
            int smem_pipe_read = 0;
            // Current pipe index in smem to write to
            int smem_pipe_write = kStage - 1;

            // Pipe slice
            Tensor tCsA_p = tCsA(_, _, _, smem_pipe_read);
            Tensor tCsB_p = tCsB(_, _, _, smem_pipe_read);
            auto K_BLOCK_MAX = size<2>(tCrA);
            // if (thread0()){
            //     {
            //         print("\n\nK_BLOCK_MAX: %d\n", (int)K_BLOCK_MAX);
            //     }
            // }

            if(K_BLOCK_MAX > 1)
            {
                // Wait until our first prefetched tile is loaded in
                cp_async_wait<kStage - 2>();
                __syncthreads();

                // Prefetch the first rmem from the first k-tile
                copy(s2r_tiled_copy_a, tCsA_p(_, _, Int<0>{}), tCrA_view(_, _, Int<0>{}));
                copy(s2r_tiled_copy_b, tCsB_p(_, _, Int<0>{}), tCrB_view(_, _, Int<0>{}));
            }
            CUTE_NO_UNROLL
            while (k_tile_count > -(kStage - 1)){
                CUTE_UNROLL
                for(int k_block = 0;k_block < K_BLOCK_MAX;++k_block){
                    if(k_block == K_BLOCK_MAX - 1){//the last computation

                        //slice the smem
                        tCsA_p = tCsA(_, _, _, smem_pipe_read);
                        tCsB_p = tCsB(_, _, _, smem_pipe_read);

                        cp_async_wait<kStage - 2>();
                        __syncthreads();
                    }
                    // Load A, B shmem->regs for k_block+1
                    auto k_block_next = (k_block + 1) % K_BLOCK_MAX; // static
                    //if k_block == K_BLOCK_MAX - 1,the next two copys will copy next slice 
                    copy(s2r_tiled_copy_a, tCsA_p(_, _, k_block_next), tCrA_view(_, _, k_block_next));
                    copy(s2r_tiled_copy_b, tCsB_p(_, _, k_block_next), tCrB_view(_, _, k_block_next));

                    // Copy gmem to smem before computing gemm on each k-pipe
                    if(k_block == 0){
                        copy(g2s_tiled_copy_a, tAgA(_, _, _, k_tile_next), tAsA(_, _, _, smem_pipe_write));
                        copy(g2s_tiled_copy_b, tBgB(_, _, _, k_tile_next), tBsB(_, _, _, smem_pipe_write));
                        cp_async_fence();
                        --k_tile_count;
                        if (k_tile_count > 0)
                        {
                            ++k_tile_next;
                        }
                        smem_pipe_write = smem_pipe_read;
                        ++smem_pipe_read;
                        smem_pipe_read = (smem_pipe_read == kStage) ? 0 : smem_pipe_read;
                    }
                    
                    //thread_gemm
                    gemm(tiled_mma, tCrC, tCrA(_, _, k_block), tCrB(_, _, k_block), tCrC);
                }
            }
            Tensor sC = make_tensor(sA(_, _, smem_pipe_read).data(), SmemLayoutC{});
            R2SCopyC tiled_copy_c_r2s;
            auto thr_copy_c_r2s = tiled_copy_c_r2s.get_slice(idx);
            Tensor tCrC_r2s = thr_copy_c_r2s.retile_S(tCrC);// (CPY, CPY_M, CPY_N)
            Tensor tCsC_r2s = thr_copy_c_r2s.partition_D(sC);// (CPY, _1, _1, pipe)

            S2GCopyC tiled_copy_c_s2g;
            auto thr_copy_c_s2g = tiled_copy_c_s2g.get_slice(idx);
            Tensor tCsC_s2g = thr_copy_c_s2g.partition_S(sC);// (CPY, _1, _1, pipe)
            Tensor tCgC_s2g = thr_copy_c_s2g.partition_D(gC);// (CPY, CPY_M, CPY_N)

            Tensor tCgC_s2gx = group_modes<1, 3>(tCgC_s2g);// (CPY_, CPY_MN)
            Tensor tCrC_r2sx = group_modes<1, 3>(tCrC_r2s);// (CPY_, CPY_MN)
            int step = size<3>(tCsC_r2s);
            // if (thread0())
            // {
            //     // print("\n\nThrCopy C r2s: ");
            //     // print(thr_copy_c_r2s);
            //     print("\n\ntCrC_r2s: ");
            //     print(tCrC_r2s);
            //     print("\ntCsC_r2s: ");
            //     print(tCsC_r2s);

            //     // print("\n\nThrCopy C s2g: ");
            //     // print(thr_copy_c_s2g);
            //     print("\n\ntCsC_s2g: ");
            //     print(tCsC_s2g);
            //     print("\ntCgC_s2g: ");
            //     print(tCgC_s2g);
            //     print("\n");
            // }
            CUTE_UNROLL
            for (int i = 0; i < size<1>(tCrC_r2sx); i += step)
            {
                // reg -> shm
                CUTE_UNROLL
                for (int j = 0; j < step; ++j)
                {
                    auto t = make_tensor_like<T>(tCrC_r2sx(_, i + j));
                    copy(tCrC_r2sx(_, i + j), t);

                    copy(tiled_copy_c_r2s, t, tCsC_r2s(_, 0, 0, j));
                }
                __syncthreads();

                // shm -> global
                CUTE_UNROLL
                for (int j = 0; j < step; ++j)
                {
                    copy(tiled_copy_c_s2g, tCsC_s2g(_, 0, 0, j), tCgC_s2gx(_, i + j));
                }
                __syncthreads();
            }
        }
};