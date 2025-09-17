import MiraOperator
from mira_operator_cpp import transpose
import torch
from MiraOperator.testing.bench import bench_kineto
import time
import numpy as np

def test_transpose_bench():
    num_tests = 30
    # 测试不同的矩阵大小
    shapes = [
        (512, 512),     # 正方形矩阵
        (1024, 1024),   # 大正方形矩阵
        (2048, 1024),   # 矩形矩阵
        (1024, 2048),   # 矩形矩阵
        (4096, 2048),   # 大矩形矩阵
        (2048, 4096),   # 大矩形矩阵
        (8192, 1024),   # 长矩阵
        (1024, 8192),   # 宽矩阵
    ]
    
    print("Shape\t\tCustom(ms)\tPyTorch(ms)\tSpeedup")
    print("-" * 50)
    
    for M, N in shapes:
        input_tensor = torch.randn(M, N, device='cuda', dtype=torch.float32)
        output_tensor = torch.zeros(N, M, device='cuda', dtype=torch.float32)
        
        # Custom kernel - 使用我们的transpose实现
        def run_custom():
            transpose(input_tensor, output_tensor)
            torch.cuda.synchronize()
        
        try:
            custom_time = bench_kineto(
                fn=run_custom,
                kernel_names="transpose_f32_kernel",
                num_tests=num_tests,
                suppress_kineto_output=True
            )
        except Exception as e:
            print(f"Custom test failed for shape ({M},{N}): {e}")
            custom_time = float('inf')
        
        # PyTorch - 使用torch.transpose + contiguous() 来强制实际执行转置
        def run_torch():
            torch_result = torch.transpose(input_tensor, 0, 1).contiguous()
            torch.cuda.synchronize()
            return torch_result
        
        try:
            torch_time = bench_kineto(
                fn=run_torch,
                kernel_names="copy",
                num_tests=num_tests,
                suppress_kineto_output=True
            )
        except Exception as e:
            print(f"PyTorch test failed for shape ({M},{N}): {e}")
            torch_time = float('inf')
        
        speedup = torch_time / custom_time if custom_time != float('inf') and torch_time != float('inf') else 0
        
        # 重新运行一次获取结果用于显示
        transpose(input_tensor, output_tensor)
        torch_result = torch.transpose(input_tensor, 0, 1).contiguous()
        
        print(f"({M:4d},{N:4d})\t{custom_time*1000:.3f}\t\t{torch_time*1000:.3f}\t\t{speedup:.2f}x")
        
        # 显示前几个元素的对比
        custom_first = output_tensor[0, :8].cpu().numpy()
        torch_first = torch_result[0, :8].cpu().numpy()
        
        print(f"  Custom:  [{', '.join([f'{x:.4f}' for x in custom_first])}...]")
        print(f"  PyTorch: [{', '.join([f'{x:.4f}' for x in torch_first])}...]")
        print()

if __name__ == "__main__":
    test_transpose_bench()
