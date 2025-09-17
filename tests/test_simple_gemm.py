import MiraOperator
from mira_operator_cpp import simple_gemm
import torch
from MiraOperator.testing.bench import bench_kineto

def test_simple_gemm_bench():
    num_tests = 30
    # 测试不同的矩阵大小
    shapes = [
        (512, 512, 512),     # 小矩阵
        (1024, 1024, 1024),  # 中等矩阵
        (2048, 2048, 2048),  # 大矩阵
        (1024, 2048, 1024),  # 矩形矩阵
        (2048, 1024, 2048),  # 矩形矩阵
        (4096, 2048, 2048),  # 大矩形矩阵
        (4096, 4096, 4096),  # 超大矩阵
        (8192, 4096, 4096),  # 超大矩形矩阵
    ]
    
    print("Shape\t\tCustom(ms)\tPyTorch(ms)\tSpeedup")
    print("-" * 50)
    
    for M, N, K in shapes:
        a = torch.randn(M, K, device='cuda', dtype=torch.float16)
        b = torch.randn(N, K, device='cuda', dtype=torch.float16)
        c = torch.zeros(M, N, device='cuda', dtype=torch.float16)
        
        # Custom kernel - 使用我们的simple_gemm实现
        def run_custom():
            simple_gemm(a, b, c)
            torch.cuda.synchronize()
        
        try:
            custom_time = bench_kineto(
                fn=run_custom,
                kernel_names="simple_gemm",
                num_tests=num_tests,
                suppress_kineto_output=True
            )
        except Exception as e:
            print(f"Custom test failed for shape ({M},{N},{K}): {e}")
            custom_time = float('inf')
        
        # PyTorch - 使用torch.matmul
        def run_torch():
            torch_result = torch.matmul(a, b.t())
            torch.cuda.synchronize()
            return torch_result
        
        try:
            torch_time = bench_kineto(
                fn=run_torch,
                kernel_names="gemm",
                num_tests=num_tests,
                suppress_kineto_output=True
            )
        except Exception as e:
            print(f"PyTorch test failed for shape ({M},{N},{K}): {e}")
            torch_time = float('inf')
        
        speedup = torch_time / custom_time if custom_time != float('inf') and torch_time != float('inf') else 0
        
        # 重新运行一次获取结果用于显示
        simple_gemm(a, b, c)
        torch_result = torch.matmul(a, b.t())
        
        print(f"({M:4d},{N:4d},{K:4d})\t{custom_time*1000:.3f}\t\t{torch_time*1000:.3f}\t\t{speedup:.2f}x")
        
        # 显示前几个元素的对比
        custom_first = c[0, :8].cpu().numpy()
        torch_first = torch_result[0, :8].cpu().numpy()
        
        print(f"  Custom:  [{', '.join([f'{x:.4f}' for x in custom_first])}...]")
        print(f"  PyTorch: [{', '.join([f'{x:.4f}' for x in torch_first])}...]")
        print()

if __name__ == '__main__':
    test_simple_gemm_bench()