#include <pybind11/pybind11.h>
#include <torch/python.h>

#include "jit/compiler.hpp"
#include "jit/device_runtime.hpp"
#include "jit_kernels/elementadd/fp32_element_add.hpp"
#include "utils/exception.hpp"

#ifndef TORCH_EXTENSION_NAME
#define TORCH_EXTENSION_NAME mira_operator_cpp
#endif
namespace MiraOperator{
torch::Tensor fp32_add(const torch::Tensor& a, const torch::Tensor& b) {
    // 假设a, b均为float32且shape一致且在CUDA上
    MO_HOST_ASSERT(a.device().is_cuda() && b.device().is_cuda());
    MO_HOST_ASSERT(a.scalar_type() == torch::kFloat && b.scalar_type() == torch::kFloat);
    MO_HOST_ASSERT(a.sizes() == b.sizes());
    int n = a.numel();
    auto c = torch::empty_like(a);

    // 获取输入tensor的GPU指针
    float* ta = a.data_ptr<float>();
    float* tb = b.data_ptr<float>();
    float* tc = c.data_ptr<float>();
    
    // 生成kernel代码
    std::string code = generate_add_kernel_code(ta, tb, tc, n);

    // JIT编译并获取kernel runtime
    auto kernel_runtime = compiler->build("fp32_element_add", code);

    std::cout<<code<<std::endl;

    // kernel launch参数
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    cudaLaunchConfig_t config;
    config.gridDim = {static_cast<unsigned>(blocks), 1, 1};
    config.blockDim = {static_cast<unsigned>(threads), 1, 1};
    config.dynamicSmemBytes = 0;
    config.stream = at::cuda::getCurrentCUDAStream();
    config.numAttrs = 0;

    // 启动kernel
    cudaLaunchKernelEx(&config, kernel_runtime->kernel, ta, tb, tc, n);
    
    // 同步GPU，确保计算完成
    cudaDeviceSynchronize();
    
    // 打印结果（从GPU内存复制到CPU）
    float result;
    cudaMemcpy(&result, tc, sizeof(float), cudaMemcpyDeviceToHost);
    return c;
}
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
        compiler = std::make_shared<NVCCCompiler>(library_root_path, cuda_home_path_by_torch);
        KernelRuntime::set_cuda_home(cuda_home_path_by_torch);
    });

    //element add kernel
    m.def("fp32_add", &fp32_add, py::arg("a"), py::arg("b"), "Elementwise add using JIT kernel");

    // Stable kernel APIs with automatic arch/layout dispatch
//     m.def("fp8_gemm_nt", &fp8_gemm_nt,
//           py::arg("a"), py::arg("b"), py::arg("d"),
//           py::arg("c") = std::nullopt, py::arg("recipe") = std::nullopt,
//           py::arg("compiled_dims") = "nk",
//           py::arg("disable_ue8m0_cast") = false);
//     m.def("fp8_gemm_nn", &fp8_gemm_nn,
//           py::arg("a"), py::arg("b"), py::arg("d"),
//           py::arg("c") = std::nullopt, py::arg("recipe") = std::nullopt,
//           py::arg("compiled_dims") = "nk",
//           py::arg("disable_ue8m0_cast") = false);
//     m.def("fp8_gemm_tn", &fp8_gemm_tn,
//           py::arg("a"), py::arg("b"), py::arg("d"),
//           py::arg("c") = std::nullopt, py::arg("recipe") = std::nullopt,
//           py::arg("compiled_dims") = "mn",
//           py::arg("disable_ue8m0_cast") = false);
//     m.def("fp8_gemm_tt", &fp8_gemm_tt,
//           py::arg("a"), py::arg("b"), py::arg("d"),
//           py::arg("c") = std::nullopt, py::arg("recipe") = std::nullopt,
//           py::arg("compiled_dims") = "mn",
//           py::arg("disable_ue8m0_cast") = false);
//     m.def("m_grouped_fp8_gemm_nt_contiguous", &m_grouped_fp8_gemm_nt_contiguous,
//           py::arg("a"), py::arg("b"), py::arg("d"), py::arg("m_indices"),
//           py::arg("recipe") = std::nullopt, py::arg("compiled_dims") = "nk",
//           py::arg("disable_ue8m0_cast") = false);
//     m.def("m_grouped_fp8_gemm_nn_contiguous", &m_grouped_fp8_gemm_nn_contiguous,
//           py::arg("a"), py::arg("b"), py::arg("d"), py::arg("m_indices"),
//           py::arg("recipe") = std::nullopt, py::arg("compiled_dims") = "nk",
//           py::arg("disable_ue8m0_cast") = false);
//     m.def("fp8_m_grouped_gemm_nt_masked", &fp8_m_grouped_gemm_nt_masked,
//           py::arg("a"), py::arg("b"), py::arg("d"), py::arg("masked_m"),
//           py::arg("expected_m"), py::arg("recipe") = std::nullopt,
//           py::arg("compiled_dims") = "nk", py::arg("disable_ue8m0_cast") = false);
//     m.def("k_grouped_fp8_gemm_tn_contiguous", &k_grouped_fp8_gemm_tn_contiguous,
//           py::arg("a"), py::arg("b"), py::arg("d"), py::arg("ks"),
//           py::arg("ks_tensor"), py::arg("c") = std::nullopt,
//           py::arg("recipe") = std::make_tuple(1, 1, 128),
//           py::arg("compiled_dims") = "mn");
//     m.def("transform_sf_into_required_layout", &transform_sf_into_required_layout);

//     // Raw kernels or functions
//     m.def("get_tma_aligned_size", &get_tma_aligned_size);
//     m.def("get_mk_alignment_for_contiguous_layout", &get_mk_alignment_for_contiguous_layout);
//     m.def("get_mn_major_tma_aligned_tensor", &get_mn_major_tma_aligned_tensor);
//     m.def("get_mn_major_tma_aligned_packed_ue8m0_tensor", &get_mn_major_tma_aligned_packed_ue8m0_tensor);
//     m.def("get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor", &get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor);
}

}