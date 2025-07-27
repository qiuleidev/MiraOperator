template<int N>
__global__ void fp32_element_add(const float* a, const float* b, float* c) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
};