import MiraOperator
from mira_operator_cpp import global_softmax
import torch
from MiraOperator.testing.bench import bench_kineto
import time
import torch.nn.functional as F

def test_global_softmax_bench():
    size = 2048
    num_tests = 30
    
    print("=== Testing Global Softmax Performance ===")
    
    print("\n--- FP32 Test ---")
    input_fp32 = torch.randn(size, device='cuda', dtype=torch.float32)
    output_fp32 = torch.zeros_like(input_fp32)
    
    def run_global_softmax_fp32():
        global_softmax(input_fp32, output_fp32)
        torch.cuda.synchronize()
    
    kernel_name = "global_soft_max_f32_per_token" 
    try:
        avg_time = bench_kineto(
            fn=run_global_softmax_fp32,
            kernel_names=kernel_name,
            num_tests=num_tests,
            suppress_kineto_output=False,
            trace_path="global_softmax_fp32_trace.json"
        )
        print(f"Custom FP32 Global Softmax Average Time: {avg_time:.6f} seconds")
    except Exception as e:
        print(f"Custom FP32 test failed: {e}")
    
    print("\n--- PyTorch Native Comparison ---")
    
    def run_torch_softmax_fp32():
        torch.softmax(input_fp32, dim=0)
        torch.cuda.synchronize()
    
    try:
        torch_fp32_time = bench_kineto(
            fn=run_torch_softmax_fp32,
            kernel_names="softmax",  
            num_tests=num_tests,
            suppress_kineto_output=False
        )
        print(f"PyTorch FP32 Softmax Average Time: {torch_fp32_time:.6f} seconds")
        
        # 计算性能差异
        if 'avg_time' in locals():
            speedup_fp32 = torch_fp32_time / avg_time
            print(f"Custom FP32 kernel is {speedup_fp32:.2f}x {'faster' if speedup_fp32 > 1 else 'slower'} than PyTorch")
            
            # 计算吞吐量
            custom_throughput_fp32 = size * 4 / avg_time / 1e9  # GB/s (float32 = 4 bytes)
            torch_throughput_fp32 = size * 4 / torch_fp32_time / 1e9  # GB/s
            print(f"Custom FP32 throughput: {custom_throughput_fp32:.2f} GB/s")
            print(f"PyTorch FP32 throughput: {torch_throughput_fp32:.2f} GB/s")
            
    except Exception as e:
        print(f"PyTorch comparison test failed: {e}")

def test_global_softmax_correctness():
    """测试全局softmax操作的正确性"""
    print("\n=== Testing Global Softmax Correctness ===")
    size = 2048
    input_tensor = torch.randn(size, device='cuda', dtype=torch.float32)
    output_tensor = torch.zeros_like(input_tensor)
    
    global_softmax(input_tensor, output_tensor)
    
    torch_result = F.softmax(input_tensor, dim=0)
    
    print(f"PyTorch result res: {torch_result[:5].cpu().numpy()}")
    print(f"Custom result res: {output_tensor[:5].cpu().numpy()}")
    print(f"PyTorch result sum: {torch_result.sum().item():.6f}")
    print(f"Custom result sum: {output_tensor.sum().item():.6f}")
    print(f"Max difference: {(torch_result - output_tensor).abs().max().item():.6e}")
    
    tolerance = 1e-5
    if (torch_result - output_tensor).abs().max() < tolerance:
        print("✅ Correctness test passed!")
    else:
        print("❌ Correctness test failed!")

if __name__ == "__main__":
    test_global_softmax_correctness()
    test_global_softmax_bench()
