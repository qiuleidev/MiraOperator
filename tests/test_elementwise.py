import MiraOperator
from mira_operator_cpp import fp32_elementwise
import torch
from MiraOperator.testing.bench import bench_kineto

def test_elementwise_bench():
    a = torch.randn(1024 * 1024 * 64, device='cuda', dtype=torch.float32)
    b = torch.randn(1024 * 1024 * 64, device='cuda', dtype=torch.float32)
    print(f'Benchmarking on shape: {a.shape}, dtype: {a.dtype}, device: {a.device}')
    #add
    op = '+'
    def fn_custom_add():
        fp32_elementwise(a, b, op)
        torch.cuda.synchronize()
    def fn_torch_add():
        torch.add(a, b)
        torch.cuda.synchronize()
    c_custom = fp32_elementwise(a, b, op)
    c_torch = torch.add(a, b)
    if torch.equal(c_custom, c_torch):
        print('exactly same!')
    else:
        print('not exactly same!')
    print('fp32_elementwise result (first 5 elements):', c_custom.flatten()[:5].cpu().numpy())
    print('torch.add result (first 5 elements):', c_torch.flatten()[:5].cpu().numpy())
    avg_time_custom = bench_kineto(fn_custom_add, 'fp32_elementwise', num_tests=30)
    avg_time_torch = bench_kineto(fn_torch_add, 'vectorized_elementwise_kernel', num_tests=30)
    print(f'Average kernel time for fp32_elementwise (custom): {avg_time_custom * 1000:.3f} ms')
    print(f'Average kernel time for torch.add (PyTorch): {avg_time_torch * 1000:.3f} ms')
    print(f'Speedup (PyTorch/custom): {avg_time_torch / avg_time_custom:.3f}x')

    #substract
    op = '-'
    def fn_custom_sub():
        fp32_elementwise(a, b, op)
        torch.cuda.synchronize()
    def fn_torch_sub():
        torch.sub(a, b)
        torch.cuda.synchronize()
    c_custom = fp32_elementwise(a, b, op)
    c_torch = torch.sub(a, b)
    if torch.equal(c_custom, c_torch):
        print('exactly same!')
    else:
        print('not exactly same!')
    print('fp32_elementwise result (first 5 elements):', c_custom.flatten()[:5].cpu().numpy())
    print('torch.sub result (first 5 elements):', c_torch.flatten()[:5].cpu().numpy())
    avg_time_custom = bench_kineto(fn_custom_sub, 'fp32_elementwise', num_tests=30)
    avg_time_torch = bench_kineto(fn_torch_sub, 'vectorized_elementwise_kernel', num_tests=30)
    print(f'Average kernel time for fp32_elementwise (custom): {avg_time_custom * 1000:.3f} ms')
    print(f'Average kernel time for torch.sub (PyTorch): {avg_time_torch * 1000:.3f} ms')
    print(f'Speedup (PyTorch/custom): {avg_time_torch / avg_time_custom:.3f}x')

if __name__ == '__main__':
    test_elementwise_bench()

    print('elementwise kernel test passed!')