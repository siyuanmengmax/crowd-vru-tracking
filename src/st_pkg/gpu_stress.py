#!/usr/bin/env python3
"""GPU correctness stress test. Detects a degraded GPU: if sustained compute produces
WRONG or non-deterministic results, the hardware/driver is the problem. Env-agnostic
(only needs torch)."""
import torch, time, sys

assert torch.cuda.is_available(), "no CUDA"
dev = torch.cuda.get_device_name(0)
print(f"torch {torch.__version__} (cuda {torch.version.cuda}) on {dev}", flush=True)
torch.manual_seed(0)
N = 4096
A = torch.randn(N, N, device='cuda'); B = torch.randn(N, N, device='cuda')
ref = (A @ B)                                   # GPU reference
ref_cpu = (A.cpu() @ B.cpu())
print(f"matmul GPU-vs-CPU max err: {(ref.cpu()-ref_cpu).abs().max():.2e}", flush=True)

t0 = time.time(); iters = 0; bad = 0
while time.time() - t0 < 90:                     # sustained ~90s
    C = A @ B
    d = (C - ref).abs().max().item()             # same op, same inputs -> must match ref
    if d > 1e-3:
        bad += 1; print(f"  [MISMATCH] iter {iters} maxdiff={d:.3e}", flush=True)
    # scatter/gather stress with valid indices
    idx = torch.randint(0, N, (N, 64), device='cuda')
    g = torch.gather(A, 1, idx)                   # gather
    oh = torch.zeros(N, N, device='cuda')
    oh.scatter_(1, idx, 1.0)                      # scatter
    iters += 1
torch.cuda.synchronize()
print(f"\nDONE: {iters} iters in 90s, matmul mismatches={bad}", flush=True)
print("VERDICT:", "GPU COMPUTE UNSTABLE (hardware/driver suspect)" if bad else
      "GPU compute STABLE under sustained load", flush=True)
