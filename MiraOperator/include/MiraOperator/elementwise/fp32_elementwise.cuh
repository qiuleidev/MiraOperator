#include <cute/layout.hpp>
#include <cute/tensor.hpp>
namespace MiraOperator{
    template<int N,char OP = '+'>
    __global__ void fp32_elementwise(const float* a, const float* b, float* c) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < N) {
            if constexpr (OP == '+') {
                c[idx] = a[idx] + b[idx];
            } else if constexpr (OP == '-') {
                c[idx] = a[idx] - b[idx];
            }
        }
        return;
    };
}