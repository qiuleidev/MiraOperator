import MiraOperator


import torch
def test_add_basic():
    a = torch.randn(10, device='cuda', dtype=torch.float32)
    b = torch.randn(10, device='cuda', dtype=torch.float32)
    c = MiraOperator.mira_operator_cpp.fp32_add(a, b)
    print(a+b)
    print(c)
    torch.testing.assert_close(c, a + b)

if __name__ == '__main__':
    test_add_basic()
    print('add kernel test passed!')