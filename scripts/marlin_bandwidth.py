#!/usr/bin/env python3
"""What does Marlin itself reach on this card, at the shapes infero's GEMM runs?

infero's `mmq` sits at 701 GB/s across a decode step and 1154 on `ffn_gate`
alone. Whether that is 60% of what the kernel design allows or 95% of it is not
answerable from infero's own numbers, and every variant tried against it has been
measured against infero's other variants rather than against the thing vLLM
actually runs. This times that thing, on the same card, at the same shapes and
the same token count.

Only shapes and dtypes affect the time, so the weights are random: the kernel
unpacks integers either way and no branch depends on their value.
"""
import time

import torch
from vllm import _custom_ops as ops
from vllm.scalar_type import scalar_types

TOKENS = 32
GROUP = 128
# (label, k, n): the four widths a Llama-3.1-8B layer touches, plus the two
# vLLM gets by fusing q/k/v and gate/up at load time.
SHAPES = [
    ("attn_k    ", 4096, 1024),
    ("attn_q    ", 4096, 4096),
    ("qkv_fused ", 4096, 6144),
    ("ffn_gate  ", 4096, 14336),
    ("gate_up_f ", 4096, 28672),
    ("ffn_down  ", 14336, 4096),
]


def bench(k, n):
    dev = "cuda"
    groups = k // GROUP
    # AWQ's own packing, which is what `awq_marlin_repack` consumes.
    qweight = torch.randint(
        -(2**31), 2**31 - 1, (k, n // 8), dtype=torch.int32, device=dev
    )
    b_q = ops.awq_marlin_repack(qweight, size_k=k, size_n=n, num_bits=4)
    scales = torch.rand((groups, n), dtype=torch.half, device=dev) * 0.01 + 0.01
    zeros = torch.randint(
        -(2**31), 2**31 - 1, (groups, n // 8), dtype=torch.int32, device=dev
    )
    a = torch.randn((TOKENS, k), dtype=torch.half, device=dev)
    workspace = torch.zeros(n // 64 * 16, dtype=torch.int, device=dev)

    def run():
        return ops.marlin_gemm(
            a, None, b_q, None, scales, None, None, zeros, None, None,
            workspace, scalar_types.uint4, TOKENS, n, k,
        )

    for _ in range(5):
        run()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(50):
        run()
    torch.cuda.synchronize()
    secs = (time.perf_counter() - t0) / 50
    # Quants plus one fp16 scale per group per column, which is what infero's
    # 68-bytes-per-128-weights also holds.
    payload = n * k * 0.5 + groups * n * 2
    return secs * 1e6, payload / secs / 1e9


print(f"marlin, {TOKENS} tokens, {torch.cuda.get_device_name(0)}")
for label, k, n in SHAPES:
    try:
        us, gbs = bench(k, n)
        print(f"  {label} {us:>7.1f} us  {gbs:>5.0f} GB/s")
    except Exception as e:  # noqa: BLE001 - a shape Marlin declines is a result
        print(f"  {label}   declined: {type(e).__name__}: {str(e)[:70]}")
