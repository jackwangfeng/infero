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

**Prefill (n tokens, each with its own eight).** The design said: group the rows
by expert on the device — histogram, prefix sum, scatter — then one GEMM per
expert, scatter-adding the results back, which is what vLLM's `fused_moe` does.
**That is not what landed, and the simpler thing is better.**

`grid.y` already indexes a slot rather than a token. Let the slot be the
`(token, expert)` pair and add one parameter — `y_group`, how many consecutive
slots share an activation row — and the same launch serves both: `gate` and `up`
take `y_group = n_active`, because a token's `k` slots all read that token's
residual, and `down` takes 1, because each slot has its own SwiGLU product. At
one token every slot reads row zero, which is the decode case falling out of the
general one.

So there is no sort, no scatter-add, no histogram, and no separate prefill path
— `feed_forward_moe` has no `if n == 1` in it. What it costs is that the rows
are mat-vecs rather than a GEMM: with 2048 slots over 128 experts each expert's
weights are re-read ~16 times where a grouped GEMM would read them once. That is
the thing to fix when prefill throughput matters, and the numbers to beat should
come from `docs/catching-vllm.md`'s methodology rather than from a guess.

Two alternatives stay rejected: looping the decode path per token is 2.3M
launches on a 2000-token prompt, and computing all 128 experts and masking is
16x the arithmetic top-8 asked for.

## What the widths cost, which was not in the plan

Two bugs found on the way, both pre-existing and both in the `d_attn != d_model`
path that `Config`'s own comment admits has no regression test. This checkpoint
is the first quantized one to exercise it — 32 heads of 128 against a 2048-wide
residual makes the attention interior exactly twice the stream.

* `store_kv2_packed` was told the keys start at `d` inside the packed
  `[q | k | v]` row. They start at `da`. So the KV cache was filled from the
  middle of `q`, and the attention output came out 200x its right magnitude —
  but only above one token, because the single-token path does not pack. Decode
  was correct and prefill was word salad.
* The fused-residual output projection quantized `d` of a `da`-wide row and told
  the mat-vec `k = d`. That does not read half the weights, it reads the wrong
  *layout*: `nb` comes out 16 where the rows are 32 blocks long, so the scale
  block lands inside the quants. Every logit was NaN by layer 0, and the
  illegal-address it eventually raised was the *next* step's embedding gather on
  a garbage sampled id — three layers downstream of the cause.

The lesson worth keeping: `TUILI_AWQ_PACKED=1` made the NaN go away, which
looked like the transposed layout being wrong. It was not — the extended
`q4g128` test passes at all three of this model's real shapes. The env var
changed the *type*, which changed which path ran.

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
