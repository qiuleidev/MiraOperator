
import MiraOperator


from mira_operator_cpp import add
import torch
def test_add_basic():
    a = torch.randn(100, device='cuda', dtype=torch.float32)
    b = torch.randn(100, device='cuda', dtype=torch.float32)
    c = add(a, b)
    print(c)
    torch.testing.assert_close(c, a + b)

if __name__ == '__main__':
    test_add_basic()
    print('add kernel test passed!') 