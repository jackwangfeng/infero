# NVFP4 W4A4 FFN Quantization — Design

## Problem

infero has no FP4 support at all — `WeightType` (`crates/kernels/src/weight.rs`) has 12 variants (F32, F16, Q4_0/Q4_1/Q5_0/Q5_1/Q8_0, Q4K/Q6K, Q4G128/Q4G128T, Q8_0S, F8E4M3) and none of them is a 4-bit float. There is a real, downloaded, byte-verified checkpoint to target — `RadixArk/Qwen3.8-27B-NVFP4` (21.9 GiB, `/home/jeff/models/Qwen3.8-27B-NVFP4` on `bw`) — a NVIDIA ModelOpt-quantized version of the exact same base architecture infero's own production checkpoint (`qwen38-27b-fp8`) already serves (`model_type: qwen3_5`, `Qwen3_5ForConditionalGeneration`, identical `hidden_size=5120`, `num_hidden_layers=64`, `num_attention_heads=24`, `num_key_value_heads=4`, `head_dim=256`, GDN dims unchanged).

vLLM has real, mature native support for this exact format (`ModelOptNvFp4Config`, `crates/quantization/utils/nvfp4_*.py` on the real installed `vllm312` venv on `bw`) — both W4A4 (weights and activations both FP4, CUTLASS NVFP4 GEMM) and W4A16 (weights FP4, bf16/fp16 activations, FP4 Marlin GEMM). infero has neither.

### What NVFP4 actually is, verified against real sources

Fetched directly from CUTLASS's own vendored source (`test/unit/gemm/device/sm120_blockscaled_tensorop_gemm/sm120_bs_gemm_nvf4_nvf4_f32_f32.cu`, `gh api` against `NVIDIA/cutlass`, 2026-09-10) and cross-checked against NVIDIA's own real published block-quant spec:

- Element type: `cutlass::float_e2m1_t` — 2 exponent bits, 1 mantissa bit, 4 bits total. The representable nonzero magnitudes within one block are exactly `{0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}` (plus sign and zero) — a real, verified, much coarser value ladder than F8E4M3's 3-mantissa-bit, 15-exponent-value ladder.
- Block scale: `cutlass::float_ue4m3_t`, one scale per **16 elements** (not F8E4M3's 128×128 grid — a materially finer, and differently-shaped, physical block).
- A real, second scaling stage exists in the actual checkpoint format beyond the raw CUTLASS primitive: every quantized tensor in `RadixArk/Qwen3.8-27B-NVFP4` carries three real, distinct tensors (verified directly against the real safetensors header, shard 1, `model.language_model.layers.0.mlp.gate_proj`):

  | tensor | dtype | shape | role |
  |---|---|---|---|
  | `.weight` | `U8` | `[N, K/2]` | packed FP4, two `e2m1` values per byte |
  | `.weight_scale` | `F8_E4M3` | `[N, K/16]` | per-16-element block scale |
  | `.weight_scale_2` | `F32` | scalar | single per-tensor global correction factor |
  | `.input_scale` | `F32` | scalar | single per-tensor activation quantization scale (this checkpoint is W4A4 — activations are quantized to FP4 at runtime too, calibrated once at export time) |

  (`K=5120` for `gate_proj`; `K/2=2560` and `K/16=320` match the real header exactly — `weight`'s shape `[17408, 2560]`, `weight_scale`'s shape `[17408, 320]`.) `weight_scale_2` exists because a single FP8-E4M3 block scale's own dynamic range (~22 binades) is not enough headroom for the full range a real trained tensor's values span — this is a documented, real NVFP4 recipe detail (NVIDIA's own "per-tensor scaling" step), not an infero-specific wrinkle.

- Real, exhaustive mixed-precision scheme (`hf_quant_config.json`'s `config_groups`, all 193 real `group_1` targets enumerated and checked, not sampled): **only** `lm_head` and each layer's `mlp.gate_proj`/`mlp.up_proj`/`mlp.down_proj` are FP4. Every attention projection (Q/K/V/O, all 64 layers), every GDN projection (`in_proj_qkv`/`in_proj_z`/`out_proj`/`in_proj_a`/`in_proj_b`, all 48 GDN layers), and the input embedding stay at 8-bit (`group_0`). `mtp.*` is explicitly in the config's own `ignore` list — the MTP head ships as plain, unquantized `.weight` tensors (verified: `mtp.layers.0.mlp.down_proj.weight` has no accompanying `.weight_scale`/`.input_scale` in the index) — consistent with this design's own decision to defer MTP (see Scope).

This is a real, deliberate, standard recipe (attention/softmax is numerically sensitive to e2m1's 1-mantissa-bit coarseness; FFN's large, redundant matmuls tolerate it) — not a checkpoint-specific oddity infero needs to special-case away. infero's own `Matrix`/per-tensor `WeightType` tagging already supports exactly this "different layers, different quant scheme" shape (this is how AWQ's `modules_to_not_convert` already works in this codebase) — no dispatch-architecture change is needed to host a second type alongside `F8E4M3`, only the second type itself.

## Goal

Load `RadixArk/Qwen3.8-27B-NVFP4` and serve real, correct, reasonably fast inference from it: `lm_head` and every layer's FFN (`gate_proj`/`up_proj`/`down_proj`) run through a new NVFP4 W4A4 CUTLASS GEMM path; every other real weight (attention, GDN, embedding) continues through the existing, unmodified `F8E4M3` path exactly as it does for the current production checkpoint. No speculative decoding in this design (see Scope).

**Real, explicit performance target (user-set): infero's real end-to-end throughput on this checkpoint reaches at least 90% of vLLM's own real throughput on the same checkpoint, for both prefill and decode.** vLLM already has real, native, working NVFP4 support for this exact checkpoint (`ModelOptNvFp4Config`, confirmed this session) — the comparison baseline is a real, live vLLM run against `RadixArk/Qwen3.8-27B-NVFP4` on the same real hardware, not a recalled or assumed number (the same "934.30 tok/s" mistake this whole investigation already made once with the FP8 checkpoint does not get repeated here: measure vLLM's own real number on this real checkpoint, at matched real input shape/content, before computing a percentage against it). This target governs Testing §7/§8 below and is the real bar Step 4's tile-shape tuning and any further optimization work is measured against, not a nice-to-have.

## Scope

- **This checkpoint's real mixed scheme only.** Not a general "any FP4 ratio" framework. The new `WeightType::F4E2M1` variant and its loader/kernel path are built and validated against this real checkpoint's real tensor set (FFN + `lm_head`, W4A4, 16-element block + per-tensor `weight_scale_2` + `input_scale`). A future checkpoint with a different mix is out of scope until one exists to validate against (YAGNI).
- **No MTP in this design.** `mtp.*` stays exactly as it loads today (plain, unquantized weights, whatever `WeightType` the loader already assigns them) — speculative decoding is disabled (`INFERO_SPEC_K` unset or the FP4 model simply doesn't wire an MTP head) for this checkpoint until a follow-up design covers it. Standing production behavior for the existing FP8 checkpoint is untouched by this design either way — this is a new, separate model/checkpoint, not a change to `qwen38-27b-fp8`'s own load path.
- **W4A4, not W4A16.** Activations get a real, per-call FP4 quantization step (a new `quantize_act_e2m1_cutlass`-shaped kernel, analogous to `quantize_act_e4m3_cutlass`) using the checkpoint's own real, static `input_scale` — not the dynamic per-token scale F8E4M3's activation quantizer computes today. (Verify at implementation time whether NVFP4 W4A4 activation quantization is expected to be static-per-tensor like the weight side, or needs its own dynamic per-block scale in addition to the static `input_scale` — real vLLM source, `nvfp4_utils.py`, is the reference to check this against before assuming either way.)
- **SM120 only**, matching every existing FP8 CUTLASS entry point in this file (`fp8_bw_gemm.cu` has no non-SM120 NVFP4 precedent to extend, and this box only has SM120 hardware to validate against).
- **Correctness-and-then-speed, mirroring how `swap_ab` was actually built and validated this session**: a host-side reference implementation first (Step 1 below), a plain device dequant kernel checked against it (Step 2), a CUTLASS GEMM with the default (non-swapped) wide tile checked against both (Step 3), and only then real tile-shape tuning (Step 4) — no CUTLASS kernel is trusted before a slower, simpler kernel has already validated the same shape's numerics.

## Data Model

### New `WeightType` variant (`crates/kernels/src/weight.rs`)

```rust
/// NVFP4 (e2m1, 2-exponent/1-mantissa 4-bit float) with a two-level block
/// scale, the encoding NVIDIA ModelOpt ships (verified against
/// RadixArk/Qwen3.8-27B-NVFP4's real safetensors header, 2026-09-10).
///
/// Laid out as `n * k / 2` packed quant bytes (two e2m1 values per byte,
/// low nibble then high nibble -- verify against a real ModelOpt/ CUTLASS
/// packing-order reference at implementation time, do not assume), followed
/// by the block-scale grid as f8_e4m3, `n * ceil(k/16)` entries row-major
/// (one scale per 16-element run along k, UNLIKE F8E4M3's 128x128 grid),
/// followed by two f32 scalars: the per-tensor weight correction
/// (`weight_scale_2`) and the per-tensor activation quantization scale
/// (`input_scale`). `block_size` is 16 (the k-direction scale granularity);
/// `n_bytes` must come from the buffer, not from `k * n / block_size *
/// type_size`, for the same reason `F8E4M3`'s doc comment already gives --
/// the trailing scalars are not part of that formula.
F4E2M1,
```

### Loader (`crates/model/src/weights.rs`)

A new recognizer, parallel to the existing FP8 loader path, reads this checkpoint's real `hf_quant_config.json`/per-tensor `.weight_scale`/`.weight_scale_2`/`.input_scale` siblings (the same shape of lookup the FP8 loader already does for its own scale grid, generalized to also handle the two extra per-tensor scalars this format adds) and produces `WeightType::F4E2M1` matrices for exactly the real tensor set found in Problem (`lm_head`, every layer's three `mlp.*_proj`). Every other real tensor in this checkpoint (attention, GDN, embedding) loads through the **existing, unmodified** F8E4M3 path — this checkpoint's own `group_0` (8-bit) tensors use the same real physical layout (128×128 block F8E4M3) the current production checkpoint already uses; nothing new is needed there. (Verify at implementation time: does `group_0`'s block scale really match `F8E4M3`'s existing 128×128 grid exactly, or does ModelOpt use a different F8E4M3 block shape for its own 8-bit tier? Check the real header for one `group_0` tensor — e.g. an attention `q_proj` — before assuming reuse is free.)

### New kernel files (mirroring the existing FP8 pair)

- `crates/kernels/src/cutlass/fp4_bw_gemm.cu` — CUTLASS NVFP4 GEMM, structured the same way `fp8_bw_gemm.cu` is (a default wide-tile namespace first, `small_m`/`small_m_swap` variants added only after Step 4's real measurement finds them worth it — do not pre-build tiers this design hasn't measured a need for).
- `crates/kernels/src/cutlass_fp4.rs` — Rust FFI + dispatch, mirroring `cutlass_fp8.rs`'s structure (`mod ffi`, a public `mma_e2m1_cutlass_...` entry point, a `CutlassWeight`-shaped prepared-weight type if NVFP4 needs its own transpose/repack step at load time — check against `prepare_cutlass_weight`'s real F8E4M3 logic for what, if anything, carries over).
- A new activation quantizer, default choice: a new `crates/kernels/src/fp4.rs` module (mirroring `fp8.rs`'s own role for the F8E4M3 path) rather than adding e2m1-specific code into `fp8.rs` — the two formats' real block/nibble-packing math is different enough (16-element vs 128×128 blocks, 2-values-per-byte packing vs 1-byte-per-value) that sharing one file would mean threading a format switch through code that is otherwise straight-line per-format math, the same reasoning `cutlass_fp8.rs` vs. a hypothetical shared file already implies for the GEMM side. Implementing `quantize_act_e2m1_cutlass` there. Revisit only if implementation reveals real, non-trivial shared logic worth factoring out — do not pre-emptively share code two formats haven't been shown to actually share.

### Kernel development order (Scope's correctness-first requirement, made concrete)

1. **Host reference** (`crates/kernels/src/weight.rs` or a new pure-Rust test module, no GPU): dequantize one packed FP4 tensor by hand against the real math (e2m1's 7-value ladder × 16-element block scale × `weight_scale_2`), matmul in f64, compare against a from-scratch reference matmul. This is the numeric ground truth every later step is checked against — same role `mma_e4m3_block`'s own from-scratch reference played for the F8E4M3 path.
2. **Device dequant kernel**, checked bit-for-bit (or within a documented float-rounding floor, the same `worst_ratio`-style tolerance this codebase's other FP8 tests use) against Step 1.
3. **CUTLASS NVFP4 GEMM, default wide tile only** (no swap_ab, no small-M variant yet), checked against Step 1/2 at this checkpoint's real FFN shapes (`K=5120,N=17408` gate/up, `K=17408,N=5120` down, `K=5120,N=248320` lm_head — real vocab_size, confirmed via the downloaded config).
4. **Tile-shape tuning**, only after Step 3 is correctness- and memcheck-clean: sweep real `n_tokens` on real hardware (`examples/`-style bench, mirroring `swap_ab_vs_small_m_bench.rs`) to find NVFP4's own real small-M crossover — **do not assume `SWAP_AB_MAX_TOKENS=32` transfers**; NVFP4's 16-element block granularity and different CUTLASS template family make this a fresh measurement, not a copy.

## Error Handling

- A `hf_quant_config.json`/tensor-scale-sibling lookup that doesn't find what this design expects (missing `.weight_scale_2`, a `group_1` target this design didn't enumerate, an unexpected packed-tensor shape) must fail loud at load time, not silently fall back to treating the tensor as some other type — this is a real, new correctness-risk class (a misread FP4 tensor produces plausible-looking garbage, not a crash), the same standing concern this session's own risk-tolerance guidance has repeatedly flagged for new quantization work.
- The host-reference/device-kernel agreement in Steps 1-2 is the primary guard against a packing-order or scale-application bug; Step 3's CUTLASS kernel is checked against the *already-validated* Step 2 output, not directly against Step 1, so a CUTLASS-specific bug (e.g. a `Major::K`/`Major::MN` mistake of the same shape `swap_ab` hit) is isolated to one step rather than conflated with a possible dequant-math bug.

## Testing

1. **Host reference correctness** (Step 1 above) — pure-Rust unit test, no GPU required, real e2m1 value table checked against the documented `{0.5,1,1.5,2,3,4,6}` ladder.
2. **Device dequant kernel vs. host reference** (Step 2) — GPU test, real packed bytes from a synthetic (not the real checkpoint, for a fast/deterministic test) tensor.
3. **CUTLASS GEMM vs. Steps 1-2**, at this checkpoint's real FFN/lm_head shapes, both `accum=false` and `accum=true` (mirroring `the_f32out_gemm_matches_the_bf16_path`'s own real coverage pattern).
4. **`compute-sanitizer --tool memcheck` and `--tool racecheck`** on every new kernel — standing practice this whole session, not optional.
5. **Real end-to-end load** of `RadixArk/Qwen3.8-27B-NVFP4` on `bw`, confirming the server starts, the real mixed dispatch (F4E2M1 for FFN/lm_head, F8E4M3 for everything else) produces coherent, non-garbage generations for a real prompt.
6. **Real accuracy comparison against the existing FP8 production checkpoint** — same real prompts, same sampling config, a real side-by-side read of output quality (not just "it doesn't crash"). This checkpoint is a real, different quantization of the real same base model `qwen38-27b-fp8` already serves, so a direct quality comparison is possible and is the real bar, not a synthetic perplexity number alone.
7. **Real decode throughput vs. vLLM, matched shape** — live vLLM run against this exact checkpoint (`ModelOptNvFp4Config`, real, already confirmed working this session) at the same real batch size, same real chat-templated input (learn from this session's own real mistake: match input length/content exactly, per `LLM.chat()` not a bare `generate()` call, before computing any percentage), same real sampling config. Compare against infero's own real number on the same checkpoint/shape. Target: infero >= 90% of vLLM's real decode number (the user's explicit goal above).
8. **Real prefill throughput vs. vLLM, matched shape** — same live-vLLM-comparison methodology as §7, but for prefill (a real long-prompt, single-pass-through-the-model measurement, not decode's per-token steady state). This session has not yet built a prefill-specific vLLM-vs-infero benchmark for any checkpoint — build one here (real prompt lengths at this checkpoint's real shape, e.g. mirroring the real 30552-token shape this session's own earlier attention-prefill investigation used) rather than assuming decode's methodology transfers unchanged. Target: infero >= 90% of vLLM's real prefill number.
9. **If §7 or §8 falls short of 90%**, the real next step is the same kernel-level investigation methodology this whole session already validated for the FP8 path (module-split roofline, `ncu` occupancy checks, real head-to-head kernel benchmarks against vLLM's own compiled kernels) applied to the new FP4 GEMM path specifically — not a new investigation approach invented from scratch.

## Out of Scope

- MTP/speculative decoding for this checkpoint (a real, separate follow-up).
- A general N-bit-ratio-agnostic FP4 framework (Scope: this checkpoint's real mix only).
- W4A16 (activations stay bf16/f16) as an alternative mode — approved direction is W4A4 only.
- Porting `swap_ab`/stream-K tile variants before Step 4's real measurement justifies them.
- Any change to the existing `F8E4M3` path, its kernels, or its dispatch thresholds.
- Multi-GPU/TP interaction (deferred, per this codebase's existing TP design's own scope precedent).
