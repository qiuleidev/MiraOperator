import MiraOperator
from mira_operator_cpp import online_softmax
import torch
from MiraOperator.testing.bench import bench_kineto
import time
import torch.nn.functional as F

def test_online_softmax_bench():
    num_tests = 30
    #H must less than 2048
    shapes = [
        (1024, 512),   # S=1024, H=512
        (2048, 1024),  # S=2048, H=1024  
        (4096, 2048),  # S=4096, H=2048
        (8192, 1024),  # S=8192, H=1024
        (16384, 512),  # S=16384, H=512
        (16384,2048)
    ]
    
    print("Shape\t\tCustom(ms)\tPyTorch(ms)\tSpeedup")
    print("-" * 50)
    
    for S, H in shapes:
        input_tensor = torch.randn(S,H, device='cuda', dtype=torch.float32)
        output_tensor = torch.zeros_like(input_tensor)
        
        # Custom kernel - 使用修改后的online_softmax
        def run_custom():
            online_softmax(input_tensor, output_tensor)
            torch.cuda.synchronize()
        
        try:
            custom_time = bench_kineto(
                fn=run_custom,
                kernel_names="online_soft_max_f32_per_token",
                num_tests=num_tests,
                suppress_kineto_output=True
            )
        except Exception as e:
            print(f"Custom test failed for shape ({S},{H}): {e}")
            custom_time = float('inf')
        
        # PyTorch - 对每一行做softmax
        def run_torch():
            torch_result = torch.softmax(input_tensor, dim=-1)  # 在最后一个维度（列）上做softmax
            torch.cuda.synchronize()
            return torch_result
        
        try:
            torch_time = bench_kineto(
                fn=run_torch,
                kernel_names="softmax",
                num_tests=num_tests,
                suppress_kineto_output=True
            )
        except Exception as e:
            print(f"PyTorch test failed for shape ({S},{H}): {e}")
            torch_time = float('inf')
        
        speedup = torch_time / custom_time if custom_time != float('inf') and torch_time != float('inf') else 0
        
        # 重新运行一次获取结果用于显示
        online_softmax(input_tensor, output_tensor)
        torch_result = torch.softmax(input_tensor, dim=-1)
        
        print(f"({S:5d},{H:4d})\t{custom_time*1000:.3f}\t\t{torch_time*1000:.3f}\t\t{speedup:.2f}x")
        
        # 显示第一个token的对比结果
        custom_first = output_tensor[0, :8].cpu().numpy()
        torch_first = torch_result[0, :8].cpu().numpy()
        
        print(f"  Custom:  [{', '.join([f'{x:.4f}' for x in custom_first])}...]")
        print(f"  PyTorch: [{', '.join([f'{x:.4f}' for x in torch_first])}...]")
        print()

if __name__ == "__main__":
    test_online_softmax_bench()
