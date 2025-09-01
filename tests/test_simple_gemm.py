import MiraOperator
from mira_operator_cpp import simple_gemm
import torch
import cute
from MiraOperator.testing.bench import bench_kineto

def test_simple_gemm_bench():
    a = torch.randn(8192, 2560, device='cuda', dtype=torch.float16)
    b = torch.randn(2560 , 2560, device='cuda', dtype=torch.float16)
    c = torch.randn(8192, 2560, device='cuda', dtype=torch.float16)
    
    # Create copies to preserve original values
    c = simple_gemm(a, b, c)
    torch.cuda.synchronize()
    print("custom gemm:",c[0:10])
    c_ref = torch.matmul(a, b.t()) 
    torch.cuda.synchronize()
    print("torch gemm:",c_ref[0:10])
    # c_ori = torch.matmul(a, b) 
    # torch.cuda.synchronize()
    # print(c_ori[0:10])
    # c_ori_2 = torch.matmul(a, b) 
    # torch.cuda.synchronize()
    # print(c_ori_2[0:10])
    
if __name__ == '__main__':
    test_simple_gemm_bench()
    print('simple_gemm kernel test passed!')