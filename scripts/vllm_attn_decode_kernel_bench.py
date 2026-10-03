import time
import torch

from vllm.vllm_flash_attn import flash_attn_varlen_func

torch.manual_seed(0)
torch.cuda.set_device(0)

# Real Qwen3.8-27B-NVFP4 decode attention shape (matches infero's
# attn_decode kernel bench: N_HEADS=24, N_KV_HEADS=4, D_HEAD=256) and a
# real decode batch (B=16), realistic kv_len for a mid-generation decode
# step. block_size=16 matches this vLLM build's paged KV cache layout
# (FlashAttentionBackend.get_supported_kernel_block_sizes multiples of 16).
B = 16
N_HEADS = 24
N_KV_HEADS = 4
D_HEAD = 256
KV_LEN = 2048
BLOCK_SIZE = 16
BLOCKS_PER_SEQ = KV_LEN // BLOCK_SIZE
assert KV_LEN % BLOCK_SIZE == 0

dev = "cuda"
dtype = torch.float16  # matches this session's kv_cache_dtype="float16" workaround

# q: (total_q, nheads, headdim); one query token per sequence (decode step).
q = torch.randn(B, N_HEADS, D_HEAD, device=dev, dtype=dtype)
cu_seqlens_q = torch.arange(0, B + 1, device=dev, dtype=torch.int32)
seqused_k = torch.full((B,), KV_LEN, device=dev, dtype=torch.int32)

num_blocks = B * BLOCKS_PER_SEQ
k_cache = torch.randn(num_blocks, BLOCK_SIZE, N_KV_HEADS, D_HEAD, device=dev, dtype=dtype)
v_cache = torch.randn(num_blocks, BLOCK_SIZE, N_KV_HEADS, D_HEAD, device=dev, dtype=dtype)
block_table = torch.arange(0, num_blocks, device=dev, dtype=torch.int32).view(B, BLOCKS_PER_SEQ)

out = torch.empty(B, N_HEADS, D_HEAD, device=dev, dtype=dtype)
scale = D_HEAD ** -0.5

print(f"q={tuple(q.shape)} k_cache={tuple(k_cache.shape)} v_cache={tuple(v_cache.shape)} "
      f"block_table={tuple(block_table.shape)} kv_len={KV_LEN} group={N_HEADS // N_KV_HEADS}")


def run(n_iters, fa_version):
    for _ in range(n_iters):
        flash_attn_varlen_func(
            q=q,
            k=k_cache,
            v=v_cache,
            out=out,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=1,
            seqused_k=seqused_k,
            max_seqlen_k=KV_LEN,
            softmax_scale=scale,
            causal=True,
            block_table=block_table,
            fa_version=fa_version,
        )


for fa_version in (2, 3):
    try:
        run(5, fa_version)
        torch.cuda.synchronize()
    except Exception as e:
        print(f"fa_version={fa_version}: FAILED warmup ({e})")
        continue

    reps = 500
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    run(reps, fa_version)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / reps
    print(f"fa_version={fa_version}: {ms*1000:.2f} us/call "
          f"(vLLM real flash_attn_varlen_func decode, B={B}, kv_len={KV_LEN}, "
          f"{reps} reps, block_size={BLOCK_SIZE})")
