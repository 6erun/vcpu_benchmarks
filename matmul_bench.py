import torch
import time

device = torch.device('cuda')
# matrix multiply — stresses both GPU compute and CPU↔GPU transfer
a = torch.randn(8192, 8192, device=device)
b = torch.randn(8192, 8192, device=device)

# warmup
for _ in range(10):
    c = torch.mm(a, b)
torch.cuda.synchronize()

# timed
start = time.perf_counter()
for _ in range(100):
    c = torch.mm(a, b)
torch.cuda.synchronize()
elapsed = time.perf_counter() - start
print(f"100 matmuls: {elapsed:.3f}s ({elapsed/100*1000:.1f}ms each)")
