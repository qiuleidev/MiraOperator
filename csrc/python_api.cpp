#include <pybind11/pybind11.h>
#include <torch/python.h>

#include "jit/compiler.hpp"
#include "jit/device_runtime.hpp"
#include "jit_kernels/elementwise/fp32_elementwise.hpp"
#include "jit_kernels/gemm/simple_gemm.hpp"
#include "jit_kernels/gemm/cute_gemm.hpp"
#include "jit_kernels/reduce/reduce.hpp"
#include "jit_kernels/softmax/global_softmax.hpp"
#include "jit_kernels/softmax/online_softmax.hpp"
#include "utils/exception.hpp"
#include <cutlass/core_io.h> 
#include <cutlass/numeric_types.h>
#include <cute/tensor.hpp>
#include <cute/numeric/math.hpp>
#include <cute/arch/copy_sm80.hpp>
#include <cute/arch/copy_sm75.hpp>
#ifndef TORCH_EXTENSION_NAME
#define TORCH_EXTENSION_NAME mira_operator_cpp
#endif
namespace MiraOperator{
torch::Tensor fp32_elementwise(const torch::Tensor& a, const torch::Tensor& b,const char op) {
    // 假设a, b均为float32且shape一致且在CUDA上
    MO_HOST_ASSERT(a.device().is_cuda() && b.device().is_cuda());
    MO_HOST_ASSERT(a.scalar_type() == torch::kFloat && b.scalar_type() == torch::kFloat);
    MO_HOST_ASSERT(a.sizes() == b.sizes());
    int n = a.numel();
    auto c = torch::empty_like(a);
    // 获取输入tensor的GPU指针
    float* pa = a.data_ptr<float>();
    float* pb = b.data_ptr<float>();
    float* pc = c.data_ptr<float>();
    
    // 生成kernel代码
    int threads = 512;
    int blocks = (n + threads - 1) / threads;
    const ElementWiseRuntime::Args& args = {
        .n = n,.op = op,
        .a = pa,.b = pb,.c = pc,
        .launch_args = LaunchArgs(blocks,threads)
    };
    const auto& code = ElementWiseRuntime::generate(args);

    // JIT编译并获取kernel runtime
    auto kernel_runtime = compiler->build("fp32_elementwise", code);

    // 启动kernel
    ElementWiseRuntime::launch(kernel_runtime,args);
    
    // 同步GPU，确保计算完成
    cudaDeviceSynchronize();
    
    return c;
}
torch::Tensor reduce(const torch::Tensor& input, const torch::Tensor& output){
    MO_HOST_ASSERT(input.device().is_cuda() && output.device().is_cuda());
    MO_HOST_ASSERT(input.sizes() == output.sizes());
    int n = input.numel();
    auto dtype = input.scalar_type();
    
    // 初始化 output 为 0
    output.zero_();
    int num_threads = 512;
    int num_blocks = (n + num_threads - 1) /num_threads;
    switch (dtype) {
        case torch::kFloat32: {
            auto input_ptr = input.data_ptr<float>();
            auto output_ptr = output.data_ptr<float>();
            const ReduceRuntime<float>::Args& args = {
                .n = n,
                .input = input_ptr,
                .output = output_ptr,
                .launch_args = LaunchArgs(num_blocks, num_threads)
            };
            const auto& code = ReduceRuntime<float>::generate(args);
            auto kernel_runtime = compiler->build("reduce",code);
            ReduceRuntime<float>::launch(kernel_runtime,args);
            cudaDeviceSynchronize();
            break;
        }
        case torch::kFloat16: {
            auto input_ptr = reinterpret_cast<half*>(input.data_ptr());
            auto output_ptr = reinterpret_cast<half*>(output.data_ptr());
            const ReduceRuntime<half>::Args& args = {
                .n = n,
                .input = input_ptr,
                .output = output_ptr,
                .launch_args = LaunchArgs(num_blocks, num_threads)
            };
            const auto& code = ReduceRuntime<half>::generate(args);
            auto kernel_runtime = compiler->build("reduce",code);
            ReduceRuntime<half>::launch(kernel_runtime,args);
            cudaDeviceSynchronize();
            break;
        }
        default: {
            throw std::runtime_error("Unsupported data type for reduction");
        }
    }
    return output;
}
torch::Tensor simple_gemm(const torch::Tensor& a, const torch::Tensor& b, torch::Tensor& c) {
    MO_HOST_ASSERT(a.device().is_cuda() && b.device().is_cuda() && c.device().is_cuda());
    MO_HOST_ASSERT(a.scalar_type() == torch::kHalf && b.scalar_type() == torch::kHalf && c.scalar_type() == torch::kHalf);
    MO_HOST_ASSERT(a.sizes()[0] == c.sizes()[0] && a.sizes()[1] == b.sizes()[1] && b.sizes()[0] == c.sizes()[1]);
    using namespace cute;
    using T = cute::half_t;
    int m = a.sizes()[0],n = c.sizes()[1],k = a.sizes()[1];
    constexpr int kTileM = 128; 
    constexpr int kTileN = 128; 
    constexpr int kTileK = 32; 

    static_assert(sizeof(cute::half_t) == sizeof(c10::Half), 
              "cute::half_t and c10::Half must have the same size");
    static_assert(alignof(cute::half_t) == alignof(c10::Half), 
              "cute::half_t and c10::Half must have the same alignment");
    //data_ptr()不要传模板参数，直接转化.要打印Aptr的值先复制到cpu上
    T* Aptr = reinterpret_cast<T*>(a.data_ptr());
    T* Bptr = reinterpret_cast<T*>(b.data_ptr());
    T* Cptr = reinterpret_cast<T*>(c.data_ptr());
    //注意编译的时候设置arch=sm_80，否则用不了此tensorcore
    using mma_op = SM80_16x8x16_F16F16F16F16_TN;
    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom = MMA_Atom<mma_traits>;

    //对应一个block
    using MMA = decltype(make_tiled_mma(mma_atom{}, 
                      Layout(Shape<_2, _2, _1>{}), //2*2*32 = 128个线程，线程在MN方向上分别重复两次，也就是一个block有4个warp协作完成
                      Tile<_32, _32, _16>{}));//最终一个block处理的分块大小，在MK方向上刚执行好1次，在N方向上要执行两次
    //make_tile();
    int block(size(MMA{}));
    std::pair<int,int> grid(n / kTileN, m / kTileM);
    // 生成kernel代码
    const SimpleGEMMRuntime<T,MMA>::Args& args = {
        .m = m,.n = n,.k = k,
        .a = Aptr,.b = Bptr,.c = Cptr,
        .kTileM = kTileM,.kTileN = kTileN,.kTileK = kTileK,
        .launch_args = LaunchArgs(grid, block)
    };
    const auto& code = SimpleGEMMRuntime<T,MMA>::generate(args);

    // JIT编译并获取kernel runtime
    auto kernel_runtime = compiler->build("simple_gemm", code);
    // 启动kernel
    SimpleGEMMRuntime<T,MMA>::launch(kernel_runtime,args);
    // 同步GPU，确保计算完成
    cudaDeviceSynchronize();
    return c;
}
//除了tensorCore以外增加了流水线和copy抽象进一步增加效率
torch::Tensor cute_gemm(const torch::Tensor& a, const torch::Tensor& b, torch::Tensor& c){
    MO_HOST_ASSERT(a.device().is_cuda() && b.device().is_cuda() && c.device().is_cuda());
    MO_HOST_ASSERT(a.scalar_type() == torch::kHalf && b.scalar_type() == torch::kHalf && c.scalar_type() == torch::kHalf);
    MO_HOST_ASSERT(a.sizes()[0] == c.sizes()[0] && a.sizes()[1] == b.sizes()[1] && b.sizes()[0] == c.sizes()[1]);
    using namespace cute;
    using T = cute::half_t;
    int m = a.sizes()[0],n = c.sizes()[1],k = a.sizes()[1];

    static_assert(sizeof(cute::half_t) == sizeof(c10::Half), 
              "cute::half_t and c10::Half must have the same size");
    static_assert(alignof(cute::half_t) == alignof(c10::Half), 
              "cute::half_t and c10::Half must have the same alignment");
    //data_ptr()不要传模板参数，直接转化.要打印Aptr的值先复制到cpu上
    T* Aptr = reinterpret_cast<T*>(a.data_ptr());
    T* Bptr = reinterpret_cast<T*>(b.data_ptr());
    T* Cptr = reinterpret_cast<T*>(c.data_ptr());

    GEMMConfig<T> gemm_config;
    int block = gemm_config.kThreadNum;
    std::pair<int,int> grid((n + gemm_config.kTileN - 1) / gemm_config.kTileN,(m + gemm_config.kTileM - 1) / gemm_config.kTileM);
    const CuteGEMMRuntime<T,GEMMConfig<T>>::Args& args = {
        .m = m,.n = n,.k = k,
        .a = Aptr,.b = Bptr,.c = Cptr,
        .launch_args = LaunchArgs(grid,block,gemm_config.kShmSize)
    };

    const auto& code = CuteGEMMRuntime<T,GEMMConfig<T>>::generate(args);

    auto kernel_runtime = compiler->build("cute_gemm", code);
    // 启动kernel
    CuteGEMMRuntime<T,GEMMConfig<T>>::launch(kernel_runtime,args);
    // 同步GPU，确保计算完成
    cudaDeviceSynchronize();
    return c;

}

torch::Tensor global_softmax(const torch::Tensor& input, torch::Tensor& output) {
    MO_HOST_ASSERT(input.device().is_cuda() && output.device().is_cuda());
    MO_HOST_ASSERT(input.scalar_type() == torch::kFloat && output.scalar_type() == torch::kFloat);
    MO_HOST_ASSERT(input.sizes() == output.sizes());
    int m = input.sizes()[0];
    int n = input.sizes()[1];
    
    // 获取输入和输出tensor的GPU指针
    float* input_ptr = input.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();

    // 计算grid和block大小
    const int NUM_THREADS = n >> 2;
    const int grid_size = m;
    
    const GlobalSoftmaxRuntime<float>::Args& args = {
        .n = n,
        .x = input_ptr,
        .y = output_ptr,
        .launch_args = LaunchArgs(grid_size, NUM_THREADS)
    };
    
    // 生成kernel代码
    const auto& code = GlobalSoftmaxRuntime<float>::generate(args);
    
    // JIT编译并获取kernel runtime
    auto kernel_runtime = compiler->build("global_softmax", code);
    
    // 启动kernel
    GlobalSoftmaxRuntime<float>::launch(kernel_runtime, args);
    
    // 同步GPU，确保计算完成
    cudaDeviceSynchronize();
    
    return output;
}

torch::Tensor online_softmax(const torch::Tensor& input, torch::Tensor& output) {
    MO_HOST_ASSERT(input.device().is_cuda() && output.device().is_cuda());
    MO_HOST_ASSERT(input.scalar_type() == torch::kFloat && output.scalar_type() == torch::kFloat);
    MO_HOST_ASSERT(input.sizes() == output.sizes());
    
    int m = input.sizes()[0];
    int n = input.sizes()[1];
    
    // 获取输入和输出tensor的GPU指针
    float* input_ptr = input.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();

    // 计算grid和block大小
    const int NUM_THREADS = n >> 2;
    const int grid_size = m;
    
    const OnlineSoftmaxRuntime<float>::Args& args = {
        .n = n,
        .x = input_ptr,
        .y = output_ptr,
        .launch_args = LaunchArgs(grid_size, NUM_THREADS)
    };
    
    // 生成kernel代码
    const auto& code = OnlineSoftmaxRuntime<float>::generate(args);
    
    // JIT编译并获取kernel runtime
    auto kernel_runtime = compiler->build("online_softmax", code);
    
    // 启动kernel
    OnlineSoftmaxRuntime<float>::launch(kernel_runtime, args);
    
    // 同步GPU，确保计算完成
    cudaDeviceSynchronize();
    
    return output;
}
} // namespace MiraOperator
// ReSharper disable once CppParameterMayBeConstPtrOrRef
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    using namespace MiraOperator;

    m.doc() = "MiraOperator C++ library";
    // Runtime
    m.def("get_num_sms", [&]() {
       return device_runtime->get_num_sms();
    });
    m.def("set_num_sms", [&](const int& new_num_sms) {
        device_runtime->set_num_sms(new_num_sms);
    });

    // JIT
    m.def("init", [&](const std::string& library_root_path, const std::string& cuda_home_path_by_torch) {
        MO_HOST_ASSERT(get_env("MO_JIT_USE_NVRTC", 0) == 0 and "Currently only support NVCC");
        compiler = std::make_shared<NVCCCompiler>(library_root_path, cuda_home_path_by_torch,"sm_80");
        KernelRuntime::set_cuda_home(cuda_home_path_by_torch);
    });

    m.def("fp32_elementwise", &fp32_elementwise, py::arg("a"), py::arg("b"),py::arg("op"), "Elementwise using JIT kernel");

    m.def("simple_gemm", &simple_gemm, py::arg("a"), py::arg("b"), py::arg("c"),"Simple GEMM using JIT kernel");

    m.def("cute_gemm", &cute_gemm, py::arg("a"), py::arg("b"), py::arg("c"),"Cute GEMM using JIT kernel");

    m.def("reduce",&reduce,py::arg("input"),py::arg("output"),"Reduce using JIT kernel");

    m.def("global_softmax",&global_softmax,py::arg("input"),py::arg("output"),"Global softmax using JIT kernel");
    
    m.def("online_softmax",&online_softmax,py::arg("input"),py::arg("output"),"Online softmax using JIT kernel");
}