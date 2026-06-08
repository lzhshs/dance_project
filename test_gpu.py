import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
try:
    x = torch.randn(3, 3, device='cuda')
    y = x @ x.T
    print(f'GPU tensor computation: OK')
    print(f'Result: {y[0,0].item():.4f}')
except Exception as e:
    print(f'GPU computation FAILED: {e}')
