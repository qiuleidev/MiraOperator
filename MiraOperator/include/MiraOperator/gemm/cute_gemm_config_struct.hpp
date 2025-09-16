#pragma once
#include <cute/arch/copy_sm75.hpp>
#include <cute/arch/copy_sm80.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <cute/layout.hpp>
#include <cute/layout_composed.hpp>
#include <cute/numeric/math.hpp>
#include <cute/tensor.hpp>
using namespace cute;
template<typename T,int _kTileM = 128,int _kTileN = 128, int _kTileK = 32,int _kStage = 5,int _kSmemLayoutCBatch = 4>
    struct GEMMConfig{
        static constexpr int kTileM = _kTileM;
        static constexpr int kTileN = _kTileN;
        static constexpr int kTileK = _kTileK;
        static constexpr int kStage = _kStage;
        static constexpr int kSmemLayoutCBatch = _kSmemLayoutCBatch;
        
        
            //注意编译的时候设置arch=sm_80，否则用不了此tensorcore
        using mma_op = SM80_16x8x16_F16F16F16F16_TN;
        using mma_traits = MMA_Traits<mma_op>;
        using mma_atom = MMA_Atom<mma_traits>;
        //对应一个block
        using MMA = decltype(make_tiled_mma(mma_atom{}, 
                        Layout(Shape<_2, _2, _1>{}), //2*2*32 = 128个线程，线程在MN方向上分别重复两次，也就是一个block有4个warp协作完成
                        Tile<_32, _32, _16>{}));//最终一个block处理的分块大小，在MK方向上刚执行好1次，在N方向上要执行两次

        
        using g2s_copy_op = SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>;//一个线程读128位数据
        using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
        using g2s_copy_atom = Copy_Atom<g2s_copy_traits, T>;
        //这个TiledCopy一次处理(32,(4*8))的数据，即1024(32,32)个数据，G2S一共要处理A(128,32),B(32,128)个数据。
        using G2SCopyA =decltype(make_tiled_copy(g2s_copy_atom{},
                                Layout<Shape<_32,_4>,Stride<_4,_1>>{},//线程的layout布局
                                Layout<Shape<_1,_8>>{}));//每个线程读取的数据布局//默认列优先，这里是向量，无所谓行列优先
        using G2SCopyB = G2SCopyA;

        MMA tiled_mma = MMA{};
        using s2r_copy_op = SM75_U32x4_LDSM_N;//传入32个4B寄存器，对于half精度，为8*8的矩阵
        using s2r_copy_traits = Copy_Traits<s2r_copy_op>;
        using s2r_copy_atom = Copy_Atom<s2r_copy_traits, T>;
        
        using S2RCopyA = decltype(make_tiled_copy_A(s2r_copy_atom{},tiled_mma));//定义S2R的copy命令，传入MMA对象，底层会自动选择copy策略
        using S2RCopyB = decltype(make_tiled_copy_B(s2r_copy_atom{},tiled_mma));

        using SmemLayoutAtomAandB = decltype(composition(Swizzle<2,3,3>{},make_layout(make_shape(Int<8>{}, Int<kTileK>{}),make_stride(Int<kTileK>{}, Int<1>{}))));

        using SmemLayoutA = decltype(tile_to_shape(SmemLayoutAtomAandB{},make_shape(Int<kTileM>{}, Int<kTileK>{}, Int<kStage>{})));
        using SmemLayoutB = SmemLayoutA;

        using SmemLayoutAtomC = decltype(make_layout(make_shape(Int<32>{}, Int<32>{}),make_stride(Int<32>{}, Int<1>{})));

        //由于C的R2S用的是普通的ld指令，而非ldmatrix，因此不会有bank conflict。
        using SmemLayoutC = decltype(tile_to_shape(SmemLayoutAtomC{},make_shape(Int<32>{}, Int<32>{}, Int<kSmemLayoutCBatch>{})));

        using R2SCopyC = decltype(make_tiled_copy_C(Copy_Atom<UniversalCopy<int>,T>{},tiled_mma));//UniversalCopy是最基础的拷贝，底层不会映射到ldmatrix（S2R），cp.async(G2S)，而是最基础的读取指令ld（一次最多读16B，G2S,S2R,G2R都可用），st。//一个bank长度就是int，设置为32位可以最大限度利用bank。
        using S2GCopyC = decltype(make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, T>{},
                                            Layout<Shape<_32, _4>, Stride<_4, _1>>{}, // Thread layout
                                            Layout<Shape<_1, _8>>{}));//Value layout

        static_assert(size<0>(SmemLayoutA{}) * size<1>(SmemLayoutA{}) >=size(SmemLayoutC{}),"C shared memory request is large than A's one pipe");
        static constexpr int kThreadNum = size(MMA{});
        static constexpr int shm_size_AB = cute::cosize(SmemLayoutA{}) + cute::cosize(SmemLayoutB{});
        static constexpr int shm_size_C = cute::cosize(SmemLayoutC{});
        static constexpr int kShmSize = cute::max(shm_size_AB, shm_size_C) * sizeof(T);
    };