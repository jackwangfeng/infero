#!/usr/bin/env python3
"""What does vLLM's decode attention cost, at the shape a decode step runs?

Companion to `marlin_bandwidth.py`, and written for the same reason: infero's
three attention kernels have only ever been measured against each other and
against infero's own fused attempt, which cannot say how much of the gap to vLLM
lives here. This times the kernel vLLM actually runs at decode under the
`FLASH_ATTN` backend, on the same card, at Llama-3.1-8B's shape.

The infero number this was first written against — 85.8 us a layer against
vLLM's 58.1 — was taken under `INFERO_PROFILE`, which serializes the stream and
puts a CUDA event pair around every launch. Measured the way this script
measures, by the host clock over back-to-back launches, infero's three kernels
take 57.7 us at the same shape. The two are within noise of each other; see
`crates/kernels/tests/bwidth_attn.rs`, which is the other half of this
comparison and prints the same denominator.
"""
import os
import time

import torch

BATCH = int(os.environ.get("BENCH_BATCH", 32))
# The engine's median, read off 31k traced launches, is 384 — not the 512 this
# script first used. Sweep it: the two engines' kernels are within noise of each
# other at 512 and that is not the shape a decode step spends its time at.
HISTORY = int(os.environ.get("BENCH_HIST", 512))
N_HEADS = 32
N_KV_HEADS = 8
D_HEAD = 128
PAGE = 16


def main():
    from vllm.vllm_flash_attn import flash_attn_varlen_func

    dev = "cuda"
    pages_per_seq = HISTORY // PAGE
    total_pages = BATCH * pages_per_seq

    # One query token per sequence: this is decode. The varlen form is what the
    # v1 FLASH_ATTN backend calls, with a paged KV cache and a block table.
    q = torch.randn((BATCH, N_HEADS, D_HEAD), dtype=torch.half, device=dev)
    k_cache = torch.randn(
        (total_pages, PAGE, N_KV_HEADS, D_HEAD), dtype=torch.half, device=dev
    )
    v_cache = torch.randn_like(k_cache)
    block_table = (
        torch.arange(total_pages, dtype=torch.int32, device=dev)
        .reshape(BATCH, pages_per_seq)
    )
    cu_q = torch.arange(BATCH + 1, dtype=torch.int32, device=dev)
    seqused = torch.full((BATCH,), HISTORY, dtype=torch.int32, device=dev)

    def run():
        return flash_attn_varlen_func(
            q=q,
            k=k_cache,
            v=v_cache,
            max_seqlen_q=1,
            cu_seqlens_q=cu_q,
            max_seqlen_k=HISTORY,
            seqused_k=seqused,
            softmax_scale=D_HEAD**-0.5,
            causal=True,
            block_table=block_table,
        )

    for _ in range(10):
        run()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(200):
        run()
    torch.cuda.synchronize()
    us = (time.perf_counter() - t0) / 200 * 1e6

    # K and V once each, which is the floor any correct kernel pays.
    kv = BATCH * HISTORY * N_KV_HEADS * D_HEAD * 2 * 2
    print(f"{torch.cuda.get_device_name(0)}")
    print(f"  batch {BATCH}, history {HISTORY}, {N_HEADS}q/{N_KV_HEADS}kv x {D_HEAD}")
    print(f"  vllm flash_attn_with_kvcache  {us:>7.1f} us  {kv / (us * 1e-6) / 1e9:>5.0f} GB/s")
    print(f"  per step over 32 layers: vllm {us * 32 / 1000:.2f} ms")
    print("  infero, same shape: cargo test -p infero-kernels --test bwidth_attn "
          f"-- --nocapture, with INFERO_ATTN_HIST={HISTORY}")


if __name__ == "__main__":
    main()
