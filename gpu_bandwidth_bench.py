"""GPU↔CPU memory bandwidth benchmark using PyTorch/HIP.

Measures pinned host-to-device (H2D) and device-to-host (D2H) bandwidth for
each GPU, then device-to-device (D2D) copy bandwidth on GPU 0.
Equivalent to what bandwidthTest / rocm-bandwidth-test used to measure.
"""
import torch
import time

SIZE_MB = 256
ITERS   = 20
WARMUP  = 5
BYTES   = SIZE_MB * 1024 * 1024

n_gpus = torch.cuda.device_count()
print(f"Detected {n_gpus} GPU(s)")
print()

for dev in range(n_gpus):
    torch.cuda.set_device(dev)
    name = torch.cuda.get_device_name(dev)

    host_buf = torch.empty(BYTES // 4, dtype=torch.float32).pin_memory()
    gpu_buf  = torch.empty(BYTES // 4, dtype=torch.float32, device=f'cuda:{dev}')

    # H2D
    for _ in range(WARMUP):
        gpu_buf.copy_(host_buf)
    torch.cuda.synchronize(dev)
    t0 = time.perf_counter()
    for _ in range(ITERS):
        gpu_buf.copy_(host_buf)
    torch.cuda.synchronize(dev)
    h2d_bw = BYTES * ITERS / (time.perf_counter() - t0) / 1e9

    # D2H
    for _ in range(WARMUP):
        host_buf.copy_(gpu_buf)
    torch.cuda.synchronize(dev)
    t0 = time.perf_counter()
    for _ in range(ITERS):
        host_buf.copy_(gpu_buf)
    torch.cuda.synchronize(dev)
    d2h_bw = BYTES * ITERS / (time.perf_counter() - t0) / 1e9

    print(f"GPU {dev} [{name}]")
    print(f"  Host-to-Device: {h2d_bw:.2f} GB/s")
    print(f"  Device-to-Host: {d2h_bw:.2f} GB/s")

    # D2D on the same GPU (memory copy bandwidth — stress HBM)
    src = torch.empty(BYTES // 4, dtype=torch.float32, device=f'cuda:{dev}')
    dst = torch.empty(BYTES // 4, dtype=torch.float32, device=f'cuda:{dev}')
    for _ in range(WARMUP):
        dst.copy_(src)
    torch.cuda.synchronize(dev)
    t0 = time.perf_counter()
    for _ in range(ITERS):
        dst.copy_(src)
    torch.cuda.synchronize(dev)
    d2d_bw = BYTES * ITERS / (time.perf_counter() - t0) / 1e9
    print(f"  Device-to-Device: {d2d_bw:.2f} GB/s")
    print()
