# Mixture of experts, on the AWQ path first

Target: `Qwen3-30B-A3B` AWQ 4-bit, already on the box at
`/home/jeff/models/qwen3-30b-moe-awq`. 48 layers, all of them sparse
(`decoder_sparse_step = 1`, `mlp_only_layers = []`), 128 experts, top-8,
`moe_intermediate_size = 768` against a `hidden_size` of 2048. The router
(`mlp.gate`) is excluded from quantization and ships as BF16; every expert
projection is AWQ. No shared expert, which is what Qwen1.5-MoE and DeepSeek have
and this does not — so v1 does not grow one.

## What the checkpoint looks like

56115 tensors. 55296 of them are experts:

```
model.layers.N.mlp.experts.E.{gate,up,down}_proj.{qweight,qzeros,scales}
model.layers.N.mlp.gate.weight        # the router, BF16, [128, 2048]
```

Per expert: `gate`/`up` are `[2048, 768]`, `down` is `[768, 2048]`. Small enough
that the *number* of matmuls, not their size, is what a decode step pays for.

## Expert weights are concatenated per layer, per projection

Copying the checkpoint's shape would mean 18432 `Matrix` values and as many
device allocations. Instead one buffer per (layer, projection) holds all 128
experts back to back, expert `e` at `e * stride`:

```rust
pub struct Experts {
    ty: WeightType,
    k: usize,             // per expert
    n: usize,             // per expert
    n_experts: usize,
    stride: usize,        // bytes per expert
    storage: Storage,
}
impl Experts { fn view_of(&self, e: usize) -> Result<CudaView<'_, u8>>; }
```

Three allocations a layer instead of 384. The reason it is worth naming as a
decision rather than an optimization: **this is the GGUF `_exps` layout**
(`[k, n, n_expert]`, one tensor per projection per layer). Adding the GGUF MoE
reader later fills the same struct a different way and touches neither the
kernels nor the forward pass. `Matrix` and every dense path stay as they are.

## Two routing paths, because decode and prefill are different problems

**Decode (one token, eight experts).** One kernel per projection, batched over
active experts rather than over tokens — `grid.y` indexes the active expert and
the kernel offsets its weights by `expert_ids[y] * stride`. The existing
`mmvq_batch` batches over *tokens* against one matrix, which is the wrong axis
here, so `mmvq_moe` is a sibling of it rather than a caller.

```
router      -> [k_active] ids + [k_active] weights
mmvq_moe    -> gate[k_active, d_ff], up[k_active, d_ff]
silu_mul    -> existing kernel, over k_active * d_ff
mmvq_moe    -> down[k_active, d_model]
moe_combine -> out[d_model] = sum_e w_e * down[e]
```

Three matmul launches a layer, the same order as a dense layer.

**Prefill (n tokens, each with its own eight).** Group the rows by expert on the
device — histogram, prefix sum, scatter — then one GEMM per expert over its
gathered rows, scatter-adding the results back. This is what vLLM's `fused_moe`
does and for the same reason.

Two alternatives are recorded here as rejected, so they are not re-tried:
looping the decode path per token is 2.3M launches on a 2000-token prompt, and
computing all 128 experts and masking is 16x the arithmetic top-8 asked for.

## Steps, each with the check that closes it

1. **Config and loader.** `qwen3_moe` into `SUPPORTED`; a
   `moe: Option<MoeConfig>` beside the existing `linear_attn` and `vision`
   options, carrying `n_experts`, `n_active`, `d_ff_expert`, `norm_topk_prob`
   and the `mlp_only_layers` exceptions. `Experts` and the AWQ concatenation.
   *Check:* the model loads and reports about 16 GiB.
2. **Decode.** `moe_router`, `mmvq_moe`, `moe_combine`. *Check:* one token's
   logits against `transformers`, by the criteria `tests/forward.rs` already
   uses — argmax exact, top-10 overlap, std within tolerance.
3. **Prefill.** The counting sort and the per-expert GEMM. *Check:* many-token
   logits against `transformers`, and against feeding the same tokens one at a
   time through step 2's path.
4. **Serve.** Batch and offload paths, then `/v1/chat/completions` on the box,
   with a tok/s number measured the way `docs/catching-vllm.md` says to measure
   one.

## Two risks worth writing down before they bite

**Load time.** 55296 AWQ tensors go through `AwqTensor::repack` on the CPU. The
27B FP8 checkpoint is ready ten seconds after launch; this one will not be.
Measure it at step 1 and parallelize the repack with rayon if it is over two
minutes — the work is per-tensor and embarrassingly parallel.

**No speculation.** The MTP head belongs to the 27B checkpoint. `load_mtp_head`
returns false here, so `TUILI_SPEC_K` turns itself off and the tok/s number from
step 4 is a no-speculation number. Not a conflict, just not comparable to the
27B's.
