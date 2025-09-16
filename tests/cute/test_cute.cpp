#include <cute/layout.hpp>
#include <cute/tensor.hpp>
using namespace cute;
int main(){
    auto layoutA = make_layout(make_shape(Int<2>{},Int<2>{}),make_stride(Int<2>{},Int<1>{}));
    auto layoutB = make_layout(make_shape(Int<2>{},Int<3>{}),make_stride(Int<3>{},Int<1>{}));
    auto layoutC = logical_product(layoutA,layoutB);
    auto layoutA1 = make_layout(make_shape(Int<8>{},Int<8>{}),make_stride(Int<8>{},Int<1>{}));
    auto layoutC1 = logical_divide(layoutA1,layoutA);//除法会进行一个补齐操作，补齐到A1的元素个数能被A元素的个数整除为止。（若能通过断言）
    //auto layoutD = make_layout(make_shape(Int<4>{},make_stride(Int<2>{})));//编译时layout的complement都是(1,0)
    auto layoutD = make_layout(make_shape(4),make_stride(8));
    auto layoutE = make_layout(make_shape(Int<8>{},Int<4>{}),make_stride(Int<4>{},Int<1>{}));
    auto tile = make_tile(Shape<_2,_2>{});
    auto tile2 = Tile<_3,_3>{};
    auto layoutT = Layout<Shape<_8,_32>,Stride<_32,_1>>{};
    print_layout(layoutT);
    print('\n');
    auto layoutT1 = composition(Swizzle<3,3,3>{},layoutT);
    print_layout(layoutT1);
    print('\n');
    // print(logical_product(layoutA1, tile));
    // print('\n');
    print_layout(logical_divide(layoutA1, tile2));
    print('\n');
    print_layout(layoutC);
    print('\n');
    print(layoutC1);
    print('\n');
    print(cosize(layoutD));
    print('\n');
    print(complement(layoutD));//composition(A . B)就是把B的value作为A的coordinate，读取此处A的value。
    print('\n');
    print(left_inverse(layoutC));
    print('\n');
    print(right_inverse(layoutC));
    return 0;
}

