# NVFP4 W4A4 FFN Quantization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Load and serve `RadixArk/Qwen3.8-27B-NVFP4` from infero — its FFN (`gate_proj`/`up_proj`/`down_proj`, every layer) and `lm_head` run through a new NVFP4 (e2m1) W4A4 CUTLASS GEMM path; every other real weight (attention, GDN, embedding) continues through the existing, unmodified `F8E4M3` path.

**Architecture:** A new `WeightType::F4E2M1` variant with its own real physical layout (16-element `f8_e4m3` block scale + two per-tensor `f32` scalars — `weight_scale_2`, `input_scale`), a new pure-Rust host reference for correctness, a new device dequant kernel checked against it, a new CUTLASS NVFP4 GEMM (default wide tile first, tile-shape tuning only after correctness is nailed down), and loader logic that reads this checkpoint's real `hf_quant_config.json` to route exactly the real FFN+lm_head tensor set through the new type while everything else keeps using the existing F8E4M3 path.

**Tech Stack:** Rust, CUDA/CUTLASS (SM120), the existing `infero-kernels`/`infero-model` crate structure.

**Spec:** `docs/superpowers/specs/2026-09-10-nvfp4-w4a4-ffn-quantization-design.md`

## Global Constraints

- **This checkpoint's real mixed scheme only** — `lm_head` + every layer's `mlp.gate_proj`/`up_proj`/`down_proj` are `F4E2M1`; everything else (attention, GDN, embedding) stays `F8E4M3`. Not a general N-bit-ratio framework.
- **No MTP/speculative decoding** for this checkpoint in this plan. `mtp.*` loads unquantized, exactly as today; speculation is simply not wired up for this model.
- **W4A4 only** — activations get their own real, per-call FP4 quantization step using the checkpoint's own real static `input_scale`. Not W4A16.
- **SM120 only.**
- **Correctness before speed**: host reference → device dequant kernel checked against it → CUTLASS default wide tile checked against both → tile-shape tuning only after that's clean. No tile-shape tuning constant (`SWAP_AB_MAX_TOKENS`-equivalent) may be copied from the existing F8E4M3 path — measure fresh.
- **`compute-sanitizer --tool memcheck` and `--tool racecheck` clean** on every new kernel before it's considered done.
- **Real, explicit performance target**: infero's real end-to-end throughput on this checkpoint reaches >= 90% of vLLM's own real throughput on the same checkpoint, matched real input shape/content, for both prefill and decode. vLLM has real native `ModelOptNvFp4Config` support already confirmed working this session — the comparison baseline is a live vLLM run, never a recalled/assumed number.
- **NVFP4 real format facts** (verified this session, not to be re-derived or guessed): element type `float_e2m1_t` (2 exponent bits, 1 mantissa bit; representable nonzero magnitudes `{0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}`), block scale `float_ue4m3_t` one per 16 elements along `k`, plus a real per-tensor `f32` `weight_scale_2` correction and (for W4A4) a real per-tensor `f32` `input_scale` for activations. Real on-disk tensor names: `<matrix>.weight` (`U8`, `[N, K/2]`, two e2m1 values packed per byte), `<matrix>.weight_scale` (`F8_E4M3`, `[N, K/16]`), `<matrix>.weight_scale_2` (`F32`, scalar), `<matrix>.input_scale` (`F32`, scalar).
- **Real checkpoint location:** `/home/jeff/models/Qwen3.8-27B-NVFP4` on `bw` (already downloaded, byte-verified against the real HF API blob sizes).
- **Real comparison checkpoint:** infero's own production `qwen38-27b-fp8` (`F8E4M3`, same base architecture) for accuracy comparison; vLLM's own `ModelOptNvFp4Config` run against the real `Qwen3.8-27B-NVFP4` checkpoint for the performance target.

---

### Task 1: `WeightType::F4E2M1` + real layout constants

**Files:**
- Modify: `crates/kernels/src/weight.rs` (the `WeightType` enum, ~line 12-48, and its `ALL`/size-accounting logic)

**Interfaces:**
- Produces: `WeightType::F4E2M1` (a new enum variant), and whatever size/`n_bytes` helper functions the existing `F8E4M3` variant already has an equivalent for (check `F8E4M3`'s own real handling in this file before assuming the exact function names — mirror them, don't guess new ones).

- [ ] **Step 1: Read the existing `F8E4M3` variant and its every real consumer**

Read `crates/kernels/src/weight.rs`'s full `WeightType` enum and every `match`/`if` in this crate that handles `WeightType::F8E4M3` specifically (grep for `F8E4M3` across `crates/kernels/src/` and `crates/model/src/`). List every real call site that will need a new `F4E2M1` arm before writing code — this enum is almost certainly matched exhaustively in several places (Rust will not compile until every arm is covered), so the compiler itself will enumerate the real remaining sites once Step 2 lands, but knowing them up front avoids surprises.

- [ ] **Step 2: Add the `F4E2M1` variant**

Add to the enum, in the same style as `F8E4M3`'s own real doc comment:

```rust
/// NVFP4 (e2m1, 2-exponent/1-mantissa 4-bit float) with a two-level block
/// scale, the encoding NVIDIA ModelOpt ships (verified against
/// RadixArk/Qwen3.8-27B-NVFP4's real safetensors header, 2026-09-10).
///
/// Laid out as `n * k / 2` packed quant bytes (two e2m1 values per byte --
/// verify the real nibble packing order against a real ModelOpt/CUTLASS
/// reference before assuming low-then-high; Task 2's host reference is
/// where this gets pinned down for real, not here), followed by the
/// block-scale grid as f8_e4m3, `n * k.div_ceil(16)` entries row-major
/// (one scale per 16-element run along k -- UNLIKE F8E4M3's 128x128 grid),
/// followed by two f32 scalars: the per-tensor weight correction
/// (`weight_scale_2`) and the per-tensor activation quantization scale
/// (`input_scale`). `block_size` is 16 (the k-direction scale
/// granularity); `n_bytes` must come from the buffer, not computed from
/// `k * n / block_size * type_size`, for the same reason `F8E4M3`'s own
/// doc comment already gives -- the trailing scalars aren't part of that
/// formula.
F4E2M1,
```

- [ ] **Step 3: Fix every real compiler error from the new arm**

`cargo build -p infero-kernels -p infero-model` (no cutlass feature needed yet — this step is pure enum plumbing) and add an `F4E2M1` arm everywhere the compiler names. For any site whose correct real behavior isn't yet decided (e.g. a `WeightType::name()`-style debug string), pick the obvious real answer (`"f4e2m1"` or similar, matching this file's own existing naming convention for other variants) rather than leaving a `todo!()`.

- [ ] **Step 4: Commit**

```bash
git add crates/kernels/src/weight.rs
git commit -m "Add WeightType::F4E2M1 for NVFP4 (e2m1) FFN weights"
```

---

### Task 2: Host-side e2m1 dequantization reference

**Files:**
- Create: `crates/kernels/src/fp4.rs`
- Modify: `crates/kernels/src/lib.rs` (add `pub mod fp4;`, mirroring how `pub mod fp8;`/`pub mod turboquant;` are already declared)

**Interfaces:**
- Produces: `pub fn e2m1_value(nibble: u8) -> f32` (the real 4-bit → f32 lookup, sign bit + the `{0.5,1.0,1.5,2.0,3.0,4.0,6.0}` magnitude ladder), `pub fn dequant_f4e2m1_row(packed: &[u8], scale: &[f16_or_f32_as_decided], scale2: f32, k: usize) -> Vec<f32>` (dequantizes one real row of `k` e2m1 values, applying both the per-16-block scale and the per-tensor `weight_scale_2`), and `pub const F4E2M1_BLOCK: usize = 16;`.
- Consumes: nothing from other tasks — this is the numeric ground truth every later task is checked against, so it must not depend on any GPU/CUTLASS code.

- [ ] **Step 1: Write the failing test for the e2m1 value table**

```rust
#[test]
fn e2m1_matches_the_documented_ladder() {
    // Positive nibble codes 0..8 map to {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
    // per the real NVFP4 spec (verified this session against NVIDIA's own
    // published format description) -- pin the exact bit pattern -> value
    // mapping here so every later kernel is checked against this, not a
    // re-derivation. The exact nibble->index mapping (is 0b0000 zero, is
    // the sign in the high bit) must be confirmed against a real CUTLASS
    // or ModelOpt reference (e.g. `cutlass::float_e2m1_t`'s own real
    // conversion table) before filling in this test's expected values --
    // do not guess the encoding, look it up.
    let expected_magnitudes = [0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    for (code, &want) in expected_magnitudes.iter().enumerate() {
        let got = e2m1_value(code as u8);
        assert_eq!(got, want, "code {code}");
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p infero-kernels e2m1_matches_the_documented_ladder`
Expected: FAIL (function not defined)

- [ ] **Step 3: Implement `e2m1_value` against the real reference**

Before writing this function's body, fetch the real `float_e2m1_t` conversion implementation from CUTLASS's own vendored source (`INFERO_CUTLASS_DIR/include/cutlass/float_subbyte.h` or wherever the real vendored copy defines it — `grep -rn "float_e2m1_t" $INFERO_CUTLASS_DIR/include/cutlass/` on `bw`) and implement the exact same bit-pattern-to-value mapping, not a re-derivation from the abstract "2 exponent bits, 1 mantissa bit" description alone (subnormal/zero handling and sign-bit position are real details a from-scratch derivation can get wrong).

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p infero-kernels e2m1_matches_the_documented_ladder`
Expected: PASS

- [ ] **Step 5: Write the failing test for a full dequantized row**

```rust
#[test]
fn dequant_f4e2m1_row_applies_both_scale_levels() {
    // 16 packed e2m1 values (8 bytes), one block-scale, one tensor-scale.
    // Construct a real, hand-computable case: every nibble encodes the
    // value "1.0" (whatever bit pattern Step 3 pinned down for that),
    // block scale = 2.0 (f8_e4m3, but use a value f8_e4m3 represents
    // exactly, e.g. 2.0), weight_scale_2 = 0.5. Expected dequantized
    // value for every element: 1.0 * 2.0 * 0.5 = 1.0.
    let packed: Vec<u8> = vec![/* 8 bytes, both nibbles = the code for 1.0 */];
    let scale: Vec<f32> = vec![2.0]; // one block covering all 16 elements
    let scale2 = 0.5;
    let row = dequant_f4e2m1_row(&packed, &scale, scale2, 16);
    for (i, &v) in row.iter().enumerate() {
        assert!((v - 1.0).abs() < 1e-6, "element {i}: got {v}");
    }
}
```

- [ ] **Step 6: Run test to verify it fails, then implement, then verify it passes**

Run: `cargo test -p infero-kernels dequant_f4e2m1_row_applies_both_scale_levels -- --nocapture`, implement `dequant_f4e2m1_row` (unpack each byte's two nibbles via `e2m1_value`, multiply by the real per-16-block scale for that element's position, multiply by `scale2`), re-run until it passes.

- [ ] **Step 7: Commit**

```bash
git add crates/kernels/src/fp4.rs crates/kernels/src/lib.rs
git commit -m "Add e2m1 host-side dequantization reference"
```

---

### Task 3: Loader — parse the real `hf_quant_config.json` and tag matrices

**Files:**
- Modify: `crates/model/src/weights.rs` (wherever the loader currently recognizes `F8E4M3` tensors and their `.weight`/scale siblings — read this real code path fully before writing the parallel `F4E2M1` one)
- Test: `crates/model/tests/fp4_loader.rs` (new)

**Interfaces:**
- Consumes: `WeightType::F4E2M1` (Task 1).
- Produces: a loader change such that, given `hf_quant_config.json`'s real `config_groups` (`group_0`: 8-bit, unspecified/empty target list meaning "everything not in group_1"; `group_1`: 4-bit, real target list of `lm_head` + every layer's `mlp.{gate,up,down}_proj`), matrices for those exact real targets load as `WeightType::F4E2M1` (reading `.weight`/`.weight_scale`/`.weight_scale_2`/`.input_scale`), and every other real tensor in the checkpoint loads through the existing, unmodified `F8E4M3` path.

- [ ] **Step 1: Write the failing test against the real checkpoint's real config**

```rust
// crates/model/tests/fp4_loader.rs
#[test]
fn radixark_nvfp4_targets_classified_correctly() {
    // Real, exhaustive target list, copied from this session's own real
    // verification of hf_quant_config.json's group_1 (193 entries: lm_head
    // + 64 layers x 3 mlp projections). Do not re-derive from a partial
    // read -- this is the ground truth the loader's classification is
    // checked against.
    let quant_config_path = "/home/jeff/models/Qwen3.8-27B-NVFP4/hf_quant_config.json";
    // Skip gracefully if the checkpoint isn't present on this machine
    // (it's real, large, and only downloaded to `bw`) -- mirror how this
    // codebase's own tests already skip when a real GPU/checkpoint isn't
    // available (e.g. `kernels()?` returning early elsewhere in this
    // crate's test suite).
    if !std::path::Path::new(quant_config_path).exists() {
        eprintln!("skipping: real checkpoint not present on this machine");
        return;
    }
    let targets = infero_model::weights::classify_fp4_targets(quant_config_path).unwrap();
    assert!(targets.contains("lm_head"));
    assert!(targets.contains("model.language_model.layers.0.mlp.gate_proj"));
    assert!(targets.contains("model.language_model.layers.63.mlp.down_proj"));
    assert!(!targets.contains("model.language_model.layers.0.self_attn.q_proj"));
    assert_eq!(targets.len(), 193);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p infero-model radixark_nvfp4_targets_classified_correctly`
Expected: FAIL (`classify_fp4_targets` not defined)

- [ ] **Step 3: Implement `classify_fp4_targets`**

Parse the real `hf_quant_config.json` structure (`config_groups.group_1.targets`, a real JSON array of strings) into a `HashSet<String>`. This is the same real JSON this session already parsed via Python to verify the 193-entry count — mirror that real structure in Rust (`serde_json`, already a dependency this codebase uses elsewhere — check `Cargo.toml` for the exact version already in use rather than adding a new one).

- [ ] **Step 4: Run test to verify it passes (on `bw`, where the checkpoint exists)**

Run (on `bw`): `cargo test -p infero-model radixark_nvfp4_targets_classified_correctly -- --nocapture`
Expected: PASS

- [ ] **Step 5: Wire `classify_fp4_targets` into the real loader**

Read the real, current loader code path that recognizes `F8E4M3` tensors (find it before writing anything — `grep -n "F8E4M3" crates/model/src/weights.rs`). For a checkpoint whose directory contains a real `hf_quant_config.json` with a non-empty NVFP4 `group_1`, route each real target tensor through a new `F4E2M1` loading path (reads `.weight`/`.weight_scale`/`.weight_scale_2`/`.input_scale`) instead of the `F8E4M3` one; every tensor NOT in the target set continues through the existing, completely unmodified `F8E4M3` path. A checkpoint with no `hf_quant_config.json` at all (i.e. the current production `qwen38-27b-fp8`) must be byte-for-byte unaffected by this change — add a real regression test loading a small piece of the existing FP8 checkpoint's own weights and confirming it still reports `WeightType::F8E4M3`, not a new code path silently misfiring on it.

- [ ] **Step 6: Write and run the FP8-regression test**

```rust
#[test]
fn a_non_nvfp4_checkpoint_is_unaffected() {
    // Load one real matrix from the existing production qwen38-27b-fp8
    // checkpoint (no hf_quant_config.json present) and confirm it still
    // classifies as F8E4M3, proving this task's new code path is inert
    // for checkpoints that don't opt into it.
    // ... (real assertion against a real loaded Matrix's WeightType)
}
```

Run: `cargo test -p infero-model a_non_nvfp4_checkpoint_is_unaffected`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add crates/model/src/weights.rs crates/model/tests/fp4_loader.rs
git commit -m "Route RadixArk NVFP4 checkpoint's real FFN+lm_head targets to WeightType::F4E2M1"
```

---

### Task 4: Device dequant kernel, checked against Task 2's host reference

**Files:**
- Create: CUDA kernel in `crates/kernels/src/cu/fp4.cu` (mirroring `crates/kernels/src/cu/fp8.cu`'s own real structure for `dequant`-shaped kernels)
- Modify: `crates/kernels/src/fp4.rs` (add the Rust-side launcher)
- Test: `crates/kernels/tests/fp4_dequant.rs` (new)

**Interfaces:**
- Consumes: `e2m1_value`/`dequant_f4e2m1_row` (Task 2), `WeightType::F4E2M1` (Task 1).
- Produces: `pub fn dequant_f4e2m1(&self, out: &mut ViewMut<f32>, w: &View<u8>, scale: &View<u8>, scale2: f32, k: usize, n: usize) -> Result<()>` on `Kernels` (mirroring the real signature shape of this crate's other dequant-style kernels — check `tq_dequant_kv`'s own real signature in `crates/kernels/src/lib.rs` for the established argument-ordering convention before finalizing this one).

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn device_dequant_matches_host_reference() -> Result<()> {
    let k = kernels()?; // this crate's existing real GPU-availability-checked helper
    // Real synthetic tensor: 128 rows x 256 cols (multiple of the real
    // 16-element block and CUTLASS's own real alignment requirements),
    // pseudo-random nibbles via this crate's existing pseudo-random-byte
    // helpers (see swap_ab_vs_small_m_bench.rs's own `quant_bytes` for
    // the established pattern).
    let (n, kk) = (128usize, 256usize);
    let packed = /* real pseudo-random packed e2m1 bytes, n*kk/2 of them */;
    let scale_bytes = /* real pseudo-random f8_e4m3 scale bytes, n*kk/16 of them */;
    let scale2 = 0.7f32;

    let want: Vec<f32> = (0..n)
        .flat_map(|row| dequant_f4e2m1_row(&packed[row_slice], &scale_row, scale2, kk))
        .collect();

    let d_w = /* upload packed */;
    let d_scale = /* upload scale_bytes */;
    let mut d_out = /* alloc n*kk f32 */;
    k.dequant_f4e2m1(&mut d_out.as_view_mut(), &d_w.as_view(), &d_scale.as_view(), scale2, kk, n)?;
    let got = /* download d_out */;

    let (worst, at) = max_rel_diff(&got, &want, 1e-3); // this codebase's own established tolerance helper
    assert!(worst <= 1.0, "element {at}: got {}, want {}", got[at], want[at]);
    Ok(())
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p infero-kernels device_dequant_matches_host_reference`
Expected: FAIL (`dequant_f4e2m1` not defined)

- [ ] **Step 3: Write the CUDA kernel**

One thread block per output row (or per real 16-element block, whichever this codebase's own existing dequant kernels use as their real grid granularity — check `tq_dequant_kv`'s real grid shape in `crates/kernels/src/cu/turboquant.cu` for precedent). Unpack each byte's two nibbles via the same real e2m1 lookup Task 2's host reference implements (a device-side `__device__` version of the same table, not a re-derivation), multiply by that element's real 16-element block scale (read from the `f8_e4m3` scale buffer — this codebase already has an `f8_e4m3`-to-`f32` device conversion helper somewhere in `fp8.cu`, reuse it rather than writing a new one), multiply by `scale2`.

- [ ] **Step 4: Write the Rust launcher, run test to verify it passes**

Run: `cargo test -p infero-kernels device_dequant_matches_host_reference -- --nocapture`
Expected: PASS

- [ ] **Step 5: `compute-sanitizer` memcheck and racecheck**

Run (on `bw`, real GPU): `compute-sanitizer --tool memcheck <test binary> device_dequant_matches_host_reference --test-threads=1` and the same with `--tool racecheck`. Both must report 0 errors before this task is done.

- [ ] **Step 6: Commit**

```bash
git add crates/kernels/src/cu/fp4.cu crates/kernels/src/fp4.rs crates/kernels/tests/fp4_dequant.rs
git commit -m "Add device-side e2m1 dequant kernel, verified against the host reference"
```

---

### Task 5: Activation quantizer — `quantize_act_e2m1_cutlass`

**Files:**
- Modify: `crates/kernels/src/cu/fp4.cu` (add the quantize kernel)
- Modify: `crates/kernels/src/fp4.rs` (add the Rust launcher)
- Test: `crates/kernels/tests/fp4_quantize_act.rs` (new)

**Interfaces:**
- Consumes: `e2m1_value` (Task 2, for the inverse/round-to-nearest direction), `WeightType::F4E2M1` (Task 1).
- Produces: `pub fn quantize_act_e2m1_cutlass(&self, xq: &mut ViewMut<u8>, x: &View<f32>, input_scale: f32, k: usize, n_tokens: usize) -> Result<()>` on `Kernels`.

- [ ] **Step 1: Confirm the real quantization scheme against vLLM's own source before writing anything**

Read `/home/jeff/vllm312/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/utils/nvfp4_utils.py` on `bw` directly (this is real, installed, working code — the spec flagged this as unverified, resolve it here, not by assumption): does NVFP4 W4A4 activation quantization use only the checkpoint's static per-tensor `input_scale`, or does it also compute a real per-block (16-element) dynamic scale at runtime the way weight quantization does? Write down which it is, with the real function/line reference, before Step 2.

- [ ] **Step 2: Write the failing test**

```rust
#[test]
fn quantize_act_e2m1_round_trips_within_tolerance() -> Result<()> {
    // Real pseudo-random f32 activations, quantize then dequantize (via
    // Task 4's dequant_f4e2m1), compare to the original -- e2m1's real
    // coarse ladder means this tolerance must be real and wide (the value
    // ladder's own worst-case relative gap, not a tight numerical-noise
    // tolerance), matching the ladder Task 2 pinned down.
    Ok(())
}
```

- [ ] **Step 3: Run test to verify it fails, implement the quantize kernel per Step 1's real findings, run test to verify it passes**

- [ ] **Step 4: `compute-sanitizer` memcheck and racecheck**

Run on `bw`, both tools, 0 errors required.

- [ ] **Step 5: Commit**

```bash
git add crates/kernels/src/cu/fp4.cu crates/kernels/src/fp4.rs crates/kernels/tests/fp4_quantize_act.rs
git commit -m "Add quantize_act_e2m1_cutlass, verified against a real round-trip tolerance"
```

---

### Task 6: CUTLASS NVFP4 GEMM — default wide tile only

**Files:**
- Create: `crates/kernels/src/cutlass/fp4_bw_gemm.cu` (mirroring `fp8_bw_gemm.cu`'s real structure)
- Create: `crates/kernels/src/cutlass_fp4.rs` (mirroring `cutlass_fp8.rs`'s real structure — read `cutlass_fp8.rs`'s `prepare_cutlass_weight`/`CutlassWeight`/`mod ffi` sections in full before writing this file, per this task's own Step 1)
- Test: `crates/kernels/tests/cutlass_fp4_gemm.rs` (new)

**Interfaces:**
- Consumes: `WeightType::F4E2M1` (Task 1), `dequant_f4e2m1`/host reference (Tasks 2/4), `quantize_act_e2m1_cutlass` (Task 5).
- Produces: `pub fn mma_e2m1_cutlass_sfa_f32out(&self, out: &mut ViewMut<f32>, w: &View<u8>, cw: &CutlassFp4Weight, xq: &View<u8>, sfa: &View<f32>, k: usize, n: usize, n_tokens: usize, accum: bool) -> Result<bool>` on `Kernels`, mirroring `mma_e4m3_cutlass_sfa_f32out`'s real signature shape exactly (same `bool` "did this run" return convention).

- [ ] **Step 1: Read the real reference material in full before writing any CUDA**

Read, in full: `crates/kernels/src/cutlass/fp8_bw_gemm.cu`'s header comment and its default (non-swapped) `f32out` namespace (this is the real, working, already-shipped reference this task's kernel structurally mirrors). Fetch, via `gh api repos/NVIDIA/cutlass/contents/test/unit/gemm/device/sm120_blockscaled_tensorop_gemm/sm120_bs_gemm_nvf4_nvf4_f32_f32.cu`, the real CUTLASS NVFP4 test kernel this session already partially read (real `ElementA = cutlass::float_e2m1_t`, `ElementSF = cutlass::float_ue4m3_t`, `ElementPairA = cutlass::nv_float4_t<cutlass::float_e2m1_t>`) — this is the real template instantiation to adapt, not a from-scratch design.

- [ ] **Step 2: Write the failing correctness test**

```rust
#[test]
fn the_nvfp4_gemm_matches_the_host_reference() -> Result<()> {
    let k = kernels()?;
    if !k.device().caps().fp4 { // add this capability flag if it doesn't exist yet -- check caps() first
        eprintln!("skipping: no NVFP4 tensor cores on this device");
        return Ok(());
    }
    // Real checkpoint FFN shape: K=5120, N=17408 (gate/up), or a smaller
    // multiple-of-real-alignment shape for a fast test -- check what
    // alignment NVFP4's real CollectiveBuilder requires (likely a multiple
    // of 16 along k, given the block size) before picking test dims.
    // Build a real packed e2m1 weight + real scale grid + real scale2,
    // real pseudo-random activations, quantize via Task 5, run the CUTLASS
    // kernel, compare against Task 2's host reference matmul (in f64,
    // then cast down) within a real, documented tolerance for e2m1's own
    // coarse ladder.
    Ok(())
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cargo test -p infero-kernels --features cutlass the_nvfp4_gemm_matches_the_host_reference`
Expected: FAIL (kernel not implemented / compile error until Step 4)

- [ ] **Step 4: Write `fp4_bw_gemm.cu`'s default wide-tile namespace**

Adapt Step 1's two real references: infero's own file structure (namespace-per-tile-variant, `extern "C"` workspace+gemm function pairs, f32-direct epilogue writing straight into the caller's `out` buffer — mirror `fp8_bw_gemm.cu`'s own `f32out` namespace exactly) with CUTLASS's own real NVFP4 element/scale-config types substituted for the FP8 ones. Start with ONE tile shape (CUTLASS's own real default for this element type — check what the real NVFP4 test file uses, do not guess a tile shape) — no `small_m`/`small_m_swap` variants yet (Task 8 measures whether they're needed).

- [ ] **Step 5: Write `cutlass_fp4.rs`'s FFI + dispatch + `CutlassFp4Weight` prep**

Mirror `cutlass_fp8.rs`'s real `mod ffi`/`CutlassWeight`/`prepare_cutlass_weight` structure (read Task 1 of this task again — Step 1 already had you read the real reference). If NVFP4's own real block-scale layout needs a transpose/repack step at load time analogous to `F8E4M3`'s `transpose_scale_b_f32`, add the equivalent CUDA kernel; if it doesn't (verify against the real CUTLASS NVFP4 test's own real `layout_SFA`/`layout_SFB` construction before assuming either way), skip it and say so in this file's own doc comment.

- [ ] **Step 6: Run test to verify it passes**

Run: `cargo test -p infero-kernels --features cutlass the_nvfp4_gemm_matches_the_host_reference -- --nocapture`
Expected: PASS

- [ ] **Step 7: `compute-sanitizer` memcheck and racecheck**

Run on `bw`, both tools, 0 errors required — this is exactly the class of kernel (a fresh CUTLASS template instantiation) that hit two real bugs during the FP8 `swap_ab` port this session (a stride/pointer pairing mistake, a `Major::K`/`Major::MN` mistake) — do not skip this even though correctness Step 6 passed; those two FP8 bugs both passed a shallow correctness check before racecheck/deeper testing caught them.

- [ ] **Step 8: Commit**

```bash
git add crates/kernels/src/cutlass/fp4_bw_gemm.cu crates/kernels/src/cutlass_fp4.rs crates/kernels/tests/cutlass_fp4_gemm.rs
git commit -m "Add CUTLASS NVFP4 GEMM (default wide tile), verified against the host reference"
```

---

### Task 7: Wire `F4E2M1` dispatch into the real forward pass

**Files:**
- Modify: `crates/model/src/lib.rs` (the real `dense_ffn`/`feed_forward` call sites, and the real `lm_head`/logits call site — read both fully before editing; find them via `grep -n "F8E4M3\|mma_e4m3_cutlass_sfa_f32out" crates/model/src/lib.rs`)
- Test: (extends the real end-to-end test infrastructure — see Task 8, this task's own unit-level check is the compiler plus Task 8's real load test)

**Interfaces:**
- Consumes: `mma_e2m1_cutlass_sfa_f32out` (Task 6), `quantize_act_e2m1_cutlass` (Task 5), the loader's real `F4E2M1` tagging (Task 3).

- [ ] **Step 1: Read the real current FFN/lm_head dispatch code in full**

Find every real call site that currently assumes a matrix is `F8E4M3` on the dense-FFN/lm_head path (not GDN, not attention — those stay untouched per Global Constraints). This is likely a `match matrix.ty()` or an `if ty == WeightType::F8E4M3` guard feeding into `mma_e4m3_cutlass_sfa_f32out`.

- [ ] **Step 2: Add the `F4E2M1` branch, real dispatch mirrors the existing `F8E4M3` one**

For a matrix whose `WeightType` is `F4E2M1`, quantize the input activations via `quantize_act_e2m1_cutlass` (Task 5) instead of `quantize_act_e4m3_cutlass`, then call `mma_e2m1_cutlass_sfa_f32out` (Task 6) instead of `mma_e4m3_cutlass_sfa_f32out` — same call-site shape, different kernel pair. Every other matrix (attention, GDN, embedding — all real `F8E4M3` on this checkpoint) is completely unaffected; this branch only ever fires for the real 193 targets Task 3 tagged.

- [ ] **Step 3: `cargo build -p infero-model --features cutlass` and fix any real compile error**

- [ ] **Step 4: Commit**

```bash
git add crates/model/src/lib.rs
git commit -m "Dispatch WeightType::F4E2M1 matrices through the new NVFP4 GEMM path"
```

---

### Task 8: Real end-to-end load against the downloaded checkpoint

**Files:**
- No new source files — this is a real, live validation task on `bw`.

- [ ] **Step 1: Build production-mode on `bw`**

```bash
ssh bw '
export PATH="$HOME/.cargo/bin:$PATH"
cd /home/jeff/infero_dev
INFERO_CUDA_DIR=/tmp/infero_cuda_dir INFERO_CUTLASS_DIR=/tmp/cutlass_src INFERO_NVCC=/usr/local/cuda-12.8/bin/nvcc \
cargo build --release -p infero-server --features infero-model/cutlass
'
```

(Reuse this session's own already-established real build recipe for `bw` — `vendor/cuda`'s symlinks are broken there, this env-var form is the confirmed-working one.)

- [ ] **Step 2: Real load, isolated from production (production stays running on its own real config throughout)**

Launch a real, separate infero instance pointed at `/home/jeff/models/Qwen3.8-27B-NVFP4` on a free GPU (check `nvidia-smi` for real free capacity first — do not touch GPU3's own real production instance or any GPU another real tenant is using). Confirm the real startup log shows both `F4E2M1` and `F8E4M3` matrices loading (whatever this codebase's own real startup logging already reports per-`WeightType` — check `vram_mib`/`quant` fields in the existing log line for the pattern to extend, if it needs extending to show a mixed-type breakdown).

- [ ] **Step 3: Real generation smoke test**

Send a real prompt via `/v1/chat/completions`, confirm a coherent (not garbage, not repeated-token, not NaN-driven) response comes back.

- [ ] **Step 4: Tear down the test instance, confirm no impact on real production**

`curl http://127.0.0.1:8301/health/live` (infero's real production endpoint) still reports healthy throughout and after.

---

### Task 9: Real accuracy comparison against FP8 production

- [ ] **Step 1: Real, identical prompt set, both checkpoints**

Run the same real set of prompts (a real mix of factual/reasoning/code content, at least 10 real distinct prompts, matching this session's own established "real, not synthetic" testing bar) through both the new NVFP4 instance (Task 8) and the existing production FP8 instance, same real sampling config (`temperature`, `max_tokens`).

- [ ] **Step 2: Real side-by-side read**

Read every real output pair. Flag any real, qualitative degradation (incoherence, factual regression, repetition) — this is a real judgment call, not an automated pass/fail, matching how this codebase's own TurboQuant accuracy work was actually validated.

---

### Task 10: Real tile-shape tuning sweep

**Files:**
- Create: `crates/kernels/examples/fp4_tile_bench.rs` (mirroring `swap_ab_vs_small_m_bench.rs`'s real structure)
- Modify: `crates/kernels/src/cutlass/fp4_bw_gemm.cu` / `cutlass_fp4.rs` (add whatever tile variant(s) this task's own real measurement justifies)

- [ ] **Step 1: Real sweep at this checkpoint's real FFN shapes**

`K=5120,N=17408` (gate/up), `K=17408,N=5120` (down), `K=5120,N=248320` (lm_head — real vocab_size), across real `n_tokens` values `[1,2,4,8,16,24,32,48,64]`, comparing the default wide tile (Task 6) against a real small-M and/or swap_ab variant if this task builds one. **Do not assume `SWAP_AB_MAX_TOKENS=32` transfers** — measure the real crossover for NVFP4's own real tile shapes on this real hardware.

- [ ] **Step 2: If a real, measured win exists at production's real decode shape (`n_tokens` around 16), build the tile variant and wire it into real dispatch**

Mirror Task 6's Steps 4-8 (write the kernel, correctness-test against the host reference, memcheck/racecheck, commit) for whichever new tile variant Step 1's real numbers justify. If Step 1 finds no real win, record that finding and leave the default wide tile as the only real path — do not add complexity Step 1's own numbers don't support.

- [ ] **Step 3: Commit**

```bash
git add crates/kernels/examples/fp4_tile_bench.rs crates/kernels/src/cutlass/fp4_bw_gemm.cu crates/kernels/src/cutlass_fp4.rs
git commit -m "Measure NVFP4's real tile-shape crossover, wire in the tuned dispatch"
```

---

### Task 11: Real vLLM-vs-infero decode throughput (target: infero >= 90%)

- [ ] **Step 1: Real, live vLLM run against `RadixArk/Qwen3.8-27B-NVFP4`**

Reuse this session's own real `vllm_e2e_bench.py` (on `bw`, `/home/jeff/vllm_e2e_bench.py`) pointed at `/home/jeff/models/Qwen3.8-27B-NVFP4` instead of the FP8 checkpoint, real `LLM.chat()` (not bare `generate()` — this session's own real, hard-learned lesson about matching real input shape/content), real batch=16, 5 real repetitions for confidence (mirroring this session's own established practice after the earlier single-sample MTP surprise).

- [ ] **Step 2: Real infero run, same real shape/prompts**

Same real batch=16, same real prompts, against the Task 8 NVFP4 instance.

- [ ] **Step 3: Real percentage, compare against the 90% target**

If below 90%, this is a real, open finding to report — not a task to silently mark done. The real next step (per the spec's own Testing §9) is the same kernel-level investigation methodology already validated for FP8 (module-split roofline, `ncu`, real head-to-head kernel benchmarks) applied to this new FP4 path.

---

### Task 12: Real vLLM-vs-infero prefill throughput (target: infero >= 90%)

- [ ] **Step 1: Build a real prefill-specific benchmark**

This session has not built one before (all prior real vLLM-vs-infero comparisons were decode-shaped). Real long prompt(s) at a real, representative length (mirroring the real 30552-token shape this session's own earlier attention-prefill investigation used, or a real shorter length if 30552 doesn't fit this checkpoint's real `max_model_len`/available VRAM — check before assuming), measuring real wall-clock time for the prefill pass alone (time-to-first-token, not full generation).

- [ ] **Step 2: Real vLLM run, real infero run, same real shape**

- [ ] **Step 3: Real percentage, compare against the 90% target**

Same real-finding-not-silent-pass handling as Task 11 Step 3 if short of 90%.

---

## Self-Review Notes

- **Spec coverage**: Problem/Goal/Scope → Global Constraints + task framing. Data Model's `WeightType`/loader/kernel-file sections → Tasks 1/3/4/5/6. Kernel development order (host ref → device dequant → CUTLASS default → tile tuning) → Tasks 2/4/6/10 in that exact order. Error Handling's loud-failure requirement → Task 3 Step 5's real routing logic (a target not found is a real bug to surface, not silently fall back on — implementer must not add a silent fallback). Testing §1-9 → Tasks 2/4/6 (host+device+CUTLASS correctness), Task 6 Step 7 + every kernel task's own memcheck/racecheck step (§4), Task 8 (§5), Task 9 (§6), Task 11 (§7), Task 12 (§8), the "if short of 90%" handling in Tasks 11/12 (§9). Out of Scope items (MTP, general framework, W4A16, pre-tuned tile variants, F8E4M3 changes, multi-GPU) → no task touches any of them; confirmed by absence.
- **Placeholder scan**: real, open sub-decisions (nibble packing order, e2m1 bit encoding, activation quantization static-vs-dynamic, whether a scale transpose step is needed) are each pointed at a real, concrete, fetchable reference (CUTLASS's own vendored source, vLLM's own real installed source) to resolve during implementation, not left as unresolved guesses — this mirrors the same pattern this codebase's own `tq4-prefill-dequant-dispatch` plan used for its real open questions.
- **Type consistency**: `WeightType::F4E2M1` (Task 1) is the same name used in Tasks 3/6/7. `dequant_f4e2m1`/`dequant_f4e2m1_row`/`e2m1_value` (Task 2/4) are used consistently by name across Tasks 4/5/6. `mma_e2m1_cutlass_sfa_f32out` (Task 6) is the name Task 7 dispatches to.
