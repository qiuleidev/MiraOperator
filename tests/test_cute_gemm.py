import MiraOperator
from mira_operator_cpp import cute_gemm
from MiraOperator.testing.bench import bench_kineto
import torch
import time

def get_pytorch_kernel_name(M, N, K):
    """动态检测 PyTorch GEMM 使用的内核名称"""
    a = torch.randn(M, K, device='cuda', dtype=torch.float16)
    b = torch.randn(N, K, device='cuda', dtype=torch.float16)
    
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as profiler:
        result = a @ b.t()
        
    # 查找 GEMM 相关的内核
    for event in profiler.key_averages():
        key_lower = event.key.lower()
        if 'gemm' in key_lower and ('ampere' in key_lower or 'cutlass' in key_lower):
            # 返回一个可以唯一匹配的子字符串
            if 'ampere' in key_lower:
                # 对于 ampere 内核，使用更具体的模式
                parts = event.key.split('_')
                for i, part in enumerate(parts):
                    if 'gemm' in part and i > 0:
                        return '_'.join(parts[:i+2])  # 包含 gemm 和前面的部分
            elif 'cutlass' in key_lower:
                # 对于 cutlass 内核，使用 cutlass_80_tensorop_f16_s16816gemm
                return "cutlass_80_tensorop_f16_s16816gemm"
    
    # 如果没有找到，返回一个通用的模式
    return "gemm"

def test_correctness():
    """测试 cute_gemm 算子的正确性"""
    print("Testing correctness...")
    
    # 使用较小的矩阵进行正确性测试
    # cute_gemm(a, b, c) 其中 a是MxK, b是NxK, c是MxN
    a = torch.randn(1024, 512, device='cuda', dtype=torch.float16)  # M x K
    b = torch.randn(1024, 512, device='cuda', dtype=torch.float16)  # N x K
    c = torch.zeros(1024, 1024, device='cuda', dtype=torch.float16)  # M x N
    
    # 测试自定义算子
    result = cute_gemm(a, b, c)
    torch.cuda.synchronize()
    
    # 参考结果 - 注意b需要转置
    ref_result = torch.matmul(a, b.t())
    torch.cuda.synchronize()
    
    # 计算数值差异
    diff = torch.max(torch.abs(result - ref_result)).item()
    print(f"Max difference: {diff:.6f}")
    
    if diff < 1.0:  # 对于fp16，1.0的容差是合理的
        print("✓ Correctness test passed!")
        return True
    else:
        print(f"✗ Correctness test failed! Difference too large: {diff}")
        return False

def benchmark_performance():
    """性能基准测试"""
    print("\nPerformance Benchmark:")
    print("=" * 60)
    print(f"{'Size (MxNxK)':<15} {'CUTE (ms)':<12} {'PyTorch (ms)':<12} {'Speedup':<8}")
    print("-" * 60)
    
    # 测试不同大小的矩阵
    test_sizes = [
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (1024, 2048, 1024),
        (2048, 1024, 2048),
        (4096, 2048, 2048),
        (4096, 4096, 4096),
        (8192, 4096, 4096),
        (8192,8192,4096),
        (8192,8192,8192),
        (16384,8192,16384)
    ]
    
    results = []
    
    for M, N, K in test_sizes:
        try:
            
            # 创建测试数据 - 注意维度匹配
            a = torch.randn(M, K, device='cuda', dtype=torch.float16)  # M x K
            b = torch.randn(N, K, device='cuda', dtype=torch.float16)  # N x K
            c = torch.zeros(M, N, device='cuda', dtype=torch.float16)  # M x N
            
            # 测试 CUTE GEMM
            def cute_gemm_fn():
                return cute_gemm(a, b, c)
            
            num_tests = 30
            # 使用实际的内核名称片段进行匹配
            cute_time = bench_kineto(
                fn=cute_gemm_fn,
                kernel_names="cute_gemm_fp16",  # 使用内核名称的关键部分
                num_tests=num_tests,
                suppress_kineto_output=True
            )
            
            # 测试 PyTorch - 注意b需要转置
            def torch_gemm_fn():
                return a @ b.t()
            
            # 动态获取 PyTorch 内核名称
            pytorch_kernel_name = get_pytorch_kernel_name(M, N, K)
            
            torch_time = bench_kineto(
                torch_gemm_fn, 
                pytorch_kernel_name,
                num_tests=num_tests,
                suppress_kineto_output=True
            )
            
            # 计算加速比
            speedup = torch_time / cute_time if cute_time > 0 else 0
            
            size_str = f"{M}x{N}x{K}"
            print(f"{size_str:<15} {cute_time*1000:<12.3f} {torch_time*1000:<12.3f} {speedup:<8.2f}")
            
            results.append({
                'size': size_str,
                'cute_time': cute_time * 1000,  # 转换为毫秒
                'torch_time': torch_time * 1000,  # 转换为毫秒
                'speedup': speedup
            })
            
        except Exception as e:
            print(f"Error testing {M}x{N}x{K}: {str(e)}")
            continue
    
    # 计算平均加速比
    if results:
        avg_speedup = sum(r['speedup'] for r in results) / len(results)
        print("-" * 60)
        print(f"Average Speedup: {avg_speedup:.2f}x")
    
    return results

def main():
    """主测试函数"""
    print("CUTE GEMM Performance Test")
    print("=" * 40)
    
    # 1. 正确性测试
    if not test_correctness():
        print("Correctness test failed, skipping performance tests.")
        return
    
    # 2. 性能测试
    benchmark_performance()
    
    print("\n" + "=" * 40)
    print("All tests completed!")

if __name__ == '__main__':
    main()
