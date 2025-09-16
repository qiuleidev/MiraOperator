import MiraOperator
from mira_operator_cpp import reduce
import torch
from MiraOperator.testing.bench import bench_kineto
import time

def test_reduce_bench():
    num_tests = 30
    sizes = [1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216]
    
    print("=== FP32 Reduce Performance ===")
    print("Size\t\tCustom(ms)\tPyTorch(ms)\tSpeedup")
    print("-" * 50)
    
    for size in sizes:
        input_tensor = torch.randn(size, device='cuda', dtype=torch.float32)
        output_tensor = torch.zeros_like(input_tensor)
        
        # Custom kernel - 使用reduce
        def run_custom():
            reduce(input_tensor, output_tensor)
            torch.cuda.synchronize()
        
        try:
            custom_time = bench_kineto(
                fn=run_custom,
                kernel_names="reduce",
                num_tests=num_tests,
                suppress_kineto_output=True
            )
        except Exception as e:
            print(f"Custom test failed for size {size}: {e}")
            custom_time = float('inf')
        
        # PyTorch - sum操作
        def run_torch():
            torch_result = torch.sum(input_tensor)
            torch.cuda.synchronize()
            return torch_result
        
        try:
            torch_time = bench_kineto(
                fn=run_torch,
                kernel_names="sum",
                num_tests=num_tests,
                suppress_kineto_output=True
            )
        except Exception as e:
            print(f"PyTorch test failed for size {size}: {e}")
            torch_time = float('inf')
        
        speedup = torch_time / custom_time if custom_time != float('inf') and torch_time != float('inf') else 0
        
        # 重新运行一次获取结果用于显示
        reduce(input_tensor, output_tensor)
        torch_result = torch.sum(input_tensor)
        
        print(f"{size:8d}\t\t{custom_time*1000:.3f}\t\t{torch_time*1000:.3f}\t\t{speedup:.2f}x")
        
        # 显示结果对比
        custom_result = output_tensor[0].item()
        torch_result_val = torch_result.item()
        
        print(f"  Custom:  {custom_result:.6f}")
        print(f"  PyTorch: {torch_result_val:.6f}")
        print()
    
    print("\n=== FP16 Reduce Performance ===")
    print("Size\t\tCustom(ms)\tPyTorch(ms)\tSpeedup")
    print("-" * 50)
    
    for size in sizes:
        input_tensor = torch.randn(size, device='cuda', dtype=torch.float16)
        output_tensor = torch.zeros_like(input_tensor)
        
        # Custom kernel - 使用reduce
        def run_custom():
            reduce(input_tensor, output_tensor)
            torch.cuda.synchronize()
        
        try:
            custom_time = bench_kineto(
                fn=run_custom,
                kernel_names="reduce",
                num_tests=num_tests,
                suppress_kineto_output=True
            )
        except Exception as e:
            print(f"Custom test failed for size {size}: {e}")
            custom_time = float('inf')
        
        # PyTorch - sum操作
        def run_torch():
            torch_result = torch.sum(input_tensor)
            torch.cuda.synchronize()
            return torch_result
        
        try:
            torch_time = bench_kineto(
                fn=run_torch,
                kernel_names="sum",
                num_tests=num_tests,
                suppress_kineto_output=True
            )
        except Exception as e:
            print(f"PyTorch test failed for size {size}: {e}")
            torch_time = float('inf')
        
        speedup = torch_time / custom_time if custom_time != float('inf') and torch_time != float('inf') else 0
        
        # 重新运行一次获取结果用于显示
        reduce(input_tensor, output_tensor)
        torch_result = torch.sum(input_tensor)
        
        print(f"{size:8d}\t\t{custom_time*1000:.3f}\t\t{torch_time*1000:.3f}\t\t{speedup:.2f}x")
        
        # 显示结果对比
        custom_result = output_tensor[0].item()
        torch_result_val = torch_result.item()
        
        print(f"  Custom:  {custom_result:.6f}")
        print(f"  PyTorch: {torch_result_val:.6f}")
        print()

if __name__ == "__main__":
    test_reduce_bench()