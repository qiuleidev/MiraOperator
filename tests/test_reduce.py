import MiraOperator
from mira_operator_cpp import reduce
import torch
from MiraOperator.testing.bench import bench_kineto
import time

def test_reduce_bench():
    size = 8192 * 4096
    num_tests = 30
    
    print("=== Testing Reduce Performance ===")
    
    print("\n--- FP32 Test ---")
    input_fp32 = torch.randn(size, device='cuda', dtype=torch.float32)
    output_fp32 = torch.zeros_like(input_fp32)
    
    def run_reduce_fp32():
        reduce(input_fp32, output_fp32)
        torch.cuda.synchronize()
    
    kernel_name = "reduce" 
    try:
        avg_time = bench_kineto(
            fn=run_reduce_fp32,
            kernel_names=kernel_name,
            num_tests=num_tests,
            suppress_kineto_output=False,
            trace_path="reduce_fp32_trace.json"
        )
        print(f"Custom FP32 Reduce Average Time: {avg_time:.6f} seconds")
    except Exception as e:
        print(f"Custom FP32 test failed: {e}")
    
    print("\n--- FP16 Test ---")
    input_fp16 = torch.randn(size, device='cuda', dtype=torch.float16)
    output_fp16 = torch.zeros_like(input_fp16)
    
    def run_reduce_fp16():
        reduce(input_fp16, output_fp16)
        torch.cuda.synchronize()
    
    try:
        avg_time_fp16 = bench_kineto(
            fn=run_reduce_fp16,
            kernel_names=kernel_name,
            num_tests=num_tests,
            suppress_kineto_output=False,
            trace_path="reduce_fp16_trace.json"
        )
        print(f"Custom FP16 Reduce Average Time: {avg_time_fp16:.6f} seconds")
        
        # 计算FP16 vs FP32性能对比
        if 'avg_time' in locals():
            fp16_vs_fp32 = avg_time / avg_time_fp16
            print(f"FP16 is {fp16_vs_fp32:.2f}x {'faster' if fp16_vs_fp32 > 1 else 'slower'} than FP32")
            
    except Exception as e:
        print(f"Custom FP16 test failed: {e}")
    
    print("\n--- PyTorch Native Comparison ---")
    
    def run_torch_reduce_fp32():
        torch.sum(input_fp32)
        torch.cuda.synchronize()
    
    def run_torch_reduce_fp16():
        torch.sum(input_fp16)
        torch.cuda.synchronize()
    
    try:
        torch_fp32_time = bench_kineto(
            fn=run_torch_reduce_fp32,
            kernel_names="sum",  
            num_tests=num_tests,
            suppress_kineto_output=False
        )
        print(f"PyTorch FP32 Sum Average Time: {torch_fp32_time:.6f} seconds")
        
        torch_fp16_time = bench_kineto(
            fn=run_torch_reduce_fp16,
            kernel_names="sum",
            num_tests=num_tests,
            suppress_kineto_output=False
        )
        print(f"PyTorch FP16 Sum Average Time: {torch_fp16_time:.6f} seconds")
        
        # 计算性能差异
        if 'avg_time' in locals():
            speedup_fp32 = torch_fp32_time / avg_time
            print(f"Custom FP32 kernel is {speedup_fp32:.2f}x {'faster' if speedup_fp32 > 1 else 'slower'} than PyTorch")
            
            if 'avg_time_fp16' in locals():
                speedup_fp16 = torch_fp16_time / avg_time_fp16
                print(f"Custom FP16 kernel is {speedup_fp16:.2f}x {'faster' if speedup_fp16 > 1 else 'slower'} than PyTorch")
            
            # 计算吞吐量
            custom_throughput_fp32 = size * 4 / avg_time / 1e9  # GB/s (float32 = 4 bytes)
            torch_throughput_fp32 = size * 4 / torch_fp32_time / 1e9  # GB/s
            print(f"Custom FP32 throughput: {custom_throughput_fp32:.2f} GB/s")
            print(f"PyTorch FP32 throughput: {torch_throughput_fp32:.2f} GB/s")
            
            if 'avg_time_fp16' in locals():
                custom_throughput_fp16 = size * 2 / avg_time_fp16 / 1e9  # GB/s (float16 = 2 bytes)
                torch_throughput_fp16 = size * 2 / torch_fp16_time / 1e9  # GB/s
                print(f"Custom FP16 throughput: {custom_throughput_fp16:.2f} GB/s")
                print(f"PyTorch FP16 throughput: {torch_throughput_fp16:.2f} GB/s")
            
    except Exception as e:
        print(f"PyTorch comparison test failed: {e}")

def test_reduce_correctness():
    """测试归约操作的正确性"""
    print("\n=== Testing Reduce Correctness ===")
    size = 1024
    input_tensor = torch.randn(size, device='cuda', dtype=torch.float32)
    output_tensor = torch.zeros_like(input_tensor)
    
    reduce(input_tensor, output_tensor)
    
    torch_result = torch.sum(input_tensor)
    
    custom_result = output_tensor[0]
    ds
    print(f"PyTorch result: {torch_result.item():.6f}")
    print(f"Custom result: {custom_result.item():.6f}")
    print(f"Difference: {abs(torch_result - custom_result).item():.6e}")
    
    tolerance = 1e-5
    if abs(torch_result - custom_result) < tolerance:
        print("✅ Correctness test passed!")
    else:
        print("❌ Correctness test failed!")
    
    print("\n--- FP16 Correctness Test ---")
    input_tensor_fp16 = torch.randn(size, device='cuda', dtype=torch.float16)
    output_tensor_fp16 = torch.zeros_like(input_tensor_fp16)
    
    reduce(input_tensor_fp16, output_tensor_fp16)
    
    torch_result_fp16 = torch.sum(input_tensor_fp16)
    
    custom_result_fp16 = output_tensor_fp16[0]
    
    print(f"PyTorch FP16 result: {torch_result_fp16.item():.6f}")
    print(f"Custom FP16 result: {custom_result_fp16.item():.6f}")
    print(f"Difference: {abs(torch_result_fp16 - custom_result_fp16).item():.6e}")
    
    # FP16的精度较低，使用更宽松的容差
    tolerance_fp16 = 1e-3
    if abs(torch_result_fp16 - custom_result_fp16) < tolerance_fp16:
        print("✅ FP16 Correctness test passed!")
    else:
        print("❌ FP16 Correctness test failed!")

def test_reduce_different_sizes():
    """测试不同大小的归约性能对比"""
    print("\n=== Testing Different Sizes ===")
    
    sizes = [1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216,67108864]
    
    print("FP32 Performance:")
    print(f"{'Size':<12} {'Custom(ms)':<12} {'PyTorch(ms)':<12} {'Speedup':<10} {'Custom(GB/s)':<12} {'PyTorch(GB/s)':<12}")
    print("-" * 80)
    
    for size in sizes:
        input_tensor = torch.randn(size, device='cuda', dtype=torch.float32)
        output_tensor = torch.zeros_like(input_tensor)
        
        def run_custom_reduce():
            reduce(input_tensor, output_tensor)
            torch.cuda.synchronize()
        
        def run_torch_reduce():
            torch.sum(input_tensor)
            torch.cuda.synchronize()
        
        try:
            # 测试custom kernel
            custom_time = bench_kineto(
                fn=run_custom_reduce,
                kernel_names=kernel_name,  
                num_tests=20,
                suppress_kineto_output=True
            )
            
            # 测试PyTorch kernel
            torch_time = bench_kineto(
                fn=run_torch_reduce,
                kernel_names="sum",
                num_tests=20,
                suppress_kineto_output=True
            )
            
            # 计算性能指标
            speedup = torch_time / custom_time
            custom_throughput = size * 4 / custom_time / 1e9  # GB/s (float32 = 4 bytes)
            torch_throughput = size * 4 / torch_time / 1e9  # GB/s
            
            print(f"{size:<12} {custom_time*1000:<12.3f} {torch_time*1000:<12.3f} {speedup:<10.2f} {custom_throughput:<12.2f} {torch_throughput:<12.2f}")
            
        except Exception as e:
            print(f"Size {size} test failed: {e}")
    
    print("\nFP16 Performance:")
    print(f"{'Size':<12} {'Custom(ms)':<12} {'PyTorch(ms)':<12} {'Speedup':<10} {'Custom(GB/s)':<12} {'PyTorch(GB/s)':<12}")
    print("-" * 80)
    
    for size in sizes:
        input_tensor = torch.randn(size, device='cuda', dtype=torch.float16)
        output_tensor = torch.zeros_like(input_tensor)
        
        def run_custom_reduce():
            reduce(input_tensor, output_tensor)
            torch.cuda.synchronize()
        
        def run_torch_reduce():
            torch.sum(input_tensor)
            torch.cuda.synchronize()
        
        try:
            # 测试custom kernel
            custom_time = bench_kineto(
                fn=run_custom_reduce,
                kernel_names=kernel_name,  
                num_tests=20,
                suppress_kineto_output=True
            )
            
            # 测试PyTorch kernel
            torch_time = bench_kineto(
                fn=run_torch_reduce,
                kernel_names="sum",
                num_tests=20,
                suppress_kineto_output=True
            )
            
            # 计算性能指标
            speedup = torch_time / custom_time
            custom_throughput = size * 2 / custom_time / 1e9  # GB/s (float16 = 2 bytes)
            torch_throughput = size * 2 / torch_time / 1e9  # GB/s
            
            print(f"{size:<12} {custom_time*1000:<12.3f} {torch_time*1000:<12.3f} {speedup:<10.2f} {custom_throughput:<12.2f} {torch_throughput:<12.2f}")
            
        except Exception as e:
            print(f"Size {size} test failed: {e}")

def test_reduce_performance_analysis():
    """详细的性能分析"""
    print("\n=== Performance Analysis ===")
    
    # 测试不同大小的性能
    sizes = [1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216, 67108864]
    
    print("FP32 Performance comparison:")
    print(f"{'Size':<10} {'Custom(ms)':<12} {'PyTorch(ms)':<12} {'Speedup':<10}")
    print("-" * 50)
    
    for size in sizes:
        input_tensor = torch.randn(size, device='cuda', dtype=torch.float32)
        output_tensor = torch.zeros_like(input_tensor)
        
        def run_custom():
            reduce(input_tensor, output_tensor)
            torch.cuda.synchronize()
        
        def run_torch():
            torch.sum(input_tensor)
            torch.cuda.synchronize()
        
        try:
            custom_time = bench_kineto(fn=run_custom, kernel_names="reduce", num_tests=20, suppress_kineto_output=True)
            torch_time = bench_kineto(fn=run_torch, kernel_names="sum", num_tests=20, suppress_kineto_output=True)
            
            speedup = torch_time / custom_time
            print(f"{size:<10} {custom_time*1000:<12.3f} {torch_time*1000:<12.3f} {speedup:<10.2f}")
            
        except Exception as e:
            print(f"Size {size}: {e}")
    
    print("\nFP16 Performance comparison:")
    print(f"{'Size':<10} {'Custom(ms)':<12} {'PyTorch(ms)':<12} {'Speedup':<10}")
    print("-" * 50)
    
    for size in sizes:
        input_tensor = torch.randn(size, device='cuda', dtype=torch.float16)
        output_tensor = torch.zeros_like(input_tensor)
        
        def run_custom():
            reduce(input_tensor, output_tensor)
            torch.cuda.synchronize()
        
        def run_torch():
            torch.sum(input_tensor)
            torch.cuda.synchronize()
        
        try:
            custom_time = bench_kineto(fn=run_custom, kernel_names="reduce", num_tests=20, suppress_kineto_output=True)
            torch_time = bench_kineto(fn=run_torch, kernel_names="sum", num_tests=20, suppress_kineto_output=True)
            
            speedup = torch_time / custom_time
            print(f"{size:<10} {custom_time*1000:<12.3f} {torch_time*1000:<12.3f} {speedup:<10.2f}")
            
        except Exception as e:
            print(f"Size {size}: {e}")

def test_reduce_ncu_analysis():
    """专门用于NCU性能分析的函数，每个size只调用一次"""
    print("\n=== NCU Performance Analysis ===")
    
    # 测试不同大小的性能，每个size只调用一次
    sizes = [1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216, 67108864]
    
    print("FP32 NCU analysis - single execution per size:")
    print(f"{'Size':<12} {'Custom(μs)':<15} {'PyTorch(μs)':<15} {'Speedup':<10}")
    print("-" * 70)
    
    for size in sizes:
        print(f"Testing size: {size}")
        
        # 创建输入tensor
        input_tensor = torch.randn(size, device='cuda', dtype=torch.float32)
        output_tensor = torch.zeros_like(input_tensor)
        
        # 预热GPU
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # 测试自定义reduce - 只调用一次
        try:
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()
            reduce(input_tensor, output_tensor)
            end_time.record()
            
            torch.cuda.synchronize()
            custom_time = start_time.elapsed_time(end_time) * 1000.0  # 转换为微秒
            
            print(f"  Custom reduce completed: {custom_time:.3f} μs")
            
        except Exception as e:
            print(f"  Custom reduce failed: {e}")
            custom_time = float('inf')
        
        # 测试PyTorch reduce - 只调用一次
        try:
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()
            torch_result = torch.sum(input_tensor)
            end_time.record()
            
            torch.cuda.synchronize()
            torch_time = start_time.elapsed_time(end_time) * 1000.0  # 转换为微秒
            
            print(f"  PyTorch reduce completed: {torch_time:.3f} μs")
            
        except Exception as e:
            print(f"  PyTorch reduce failed: {e}")
            torch_time = float('inf')
        
        # 计算性能对比
        if custom_time != float('inf') and torch_time != float('inf'):
            speedup = torch_time / custom_time
            print(f"  Speedup: {speedup:.2f}x")
        else:
            speedup = 0.0
            print(f"  Speedup: N/A")
        
        # 输出结果
        print(f"{size:<12} {custom_time:<15.3f} {torch_time:<15.3f} {speedup:<10.2f}")
        print("-" * 70)
        
        # 清理内存
        del input_tensor, output_tensor
        torch.cuda.empty_cache()
        
        # 短暂休息，确保GPU状态稳定
        time.sleep(0.1)
    
    print("\nFP16 NCU analysis - single execution per size:")
    print(f"{'Size':<12} {'Custom(μs)':<15} {'PyTorch(μs)':<15} {'Speedup':<10}")
    print("-" * 70)
    
    for size in sizes:
        print(f"Testing size: {size}")
        
        # 创建输入tensor
        input_tensor = torch.randn(size, device='cuda', dtype=torch.float16)
        output_tensor = torch.zeros_like(input_tensor)
        
        # 预热GPU
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # 测试自定义reduce - 只调用一次
        try:
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()
            reduce(input_tensor, output_tensor)
            end_time.record()
            
            torch.cuda.synchronize()
            custom_time = start_time.elapsed_time(end_time) * 1000.0  # 转换为微秒
            
            print(f"  Custom reduce completed: {custom_time:.3f} μs")
            
        except Exception as e:
            print(f"  Custom reduce failed: {e}")
            custom_time = float('inf')
        
        # 测试PyTorch reduce - 只调用一次
        try:
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()
            torch_result = torch.sum(input_tensor)
            end_time.record()
            
            torch.cuda.synchronize()
            torch_time = start_time.elapsed_time(end_time) * 1000.0  # 转换为微秒
            
            print(f"  PyTorch reduce completed: {torch_time:.3f} μs")
            
        except Exception as e:
            print(f"  PyTorch reduce failed: {e}")
            torch_time = float('inf')
        
        # 计算性能对比
        if custom_time != float('inf') and torch_time != float('inf'):
            speedup = torch_time / custom_time
            print(f"  Speedup: {speedup:.2f}x")
        else:
            speedup = 0.0
            print(f"  Speedup: N/A")
        
        # 输出结果
        print(f"{size:<12} {custom_time:<15.3f} {torch_time:<15.3f} {speedup:<10.2f}")
        print("-" * 70)
        
        # 清理内存
        del input_tensor, output_tensor
        torch.cuda.empty_cache()
        
        # 短暂休息，确保GPU状态稳定
        time.sleep(0.1)
    
    print("\nNCU analysis completed!")
    print("You can now run: ncu --set full -o reduce_ncu_report -f python tests/test_reduce.py")

if __name__ == "__main__":
    test_reduce_correctness()
    test_reduce_bench()
    test_reduce_different_sizes()
    test_reduce_performance_analysis()
    test_reduce_ncu_analysis()  # 启用NCU分析函数