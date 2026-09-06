# TurboQuant Prefill Dequant-Dispatch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix TurboQuant (TQ) KV-cache quantization's real `O(n_tokens × kv_len)` prefill slowness by dequantizing a long-enough run's cached KV once into a dense f16 scratch buffer and reusing the existing dense tile kernels, instead of re-unpacking every prior key/value per query token.

**Architecture:** A new `tq_dequant_kv` CUDA kernel unpacks one sequence's `0..kv_len` quantized span into a per-sequence dense f16 scratch buffer, staying in TQ's existing rotated basis (no inverse rotation, so it pairs directly with the already-rotated `tq.q_rot`/`tq.acc_rot` the rest of the TQ pipeline already uses). A synthetic, single-sequence, identity-mapped `BatchLayout` then lets the **existing, unmodified** `attn_prefill_ws4`/`attn_prefill_decoupled6_f16acc`/flash_attn2 kernels read that scratch buffer exactly as they'd read the F16 dense pool. Runs at or below a measured threshold keep using today's `tq_attn_decode` unchanged.

**Tech Stack:** Rust, CUDA (NVRTC-JIT via `crates/kernels/src/cu/turboquant.cu`), the existing `cudarc`-based `Kernels`/`Buf`/`View` device abstractions in `crates/gpu`/`crates/kernels`.

**Spec:** `docs/superpowers/specs/2026-09-06-tq4-prefill-dequant-dispatch-design.md`

## Global Constraints

- TurboQuant is **not enabled in production** (`kv_quant=f16` there) — every task in this plan validates against an isolated TQ-quantized load (the small test checkpoint, `INFERO_TEST_GGUF` or the default `models/qwen2.5-0.5b-instruct-q8_0.gguf` path), never against live traffic.
- **Two-tier dispatch, not three.** No task in this plan builds a "skip quantization for a brand-new sequence's first chunk" path — accepted spec trade-off, see spec §Problem.
- **No inverse rotation inside `tq_dequant_kv`.** The kernel's output stays in the same rotated basis every other TQ kernel already operates in. Do not add a `Πᵀ` step to it — that would be redundant with the single existing final-output un-rotate at `lib.rs:4947-4949` and would corrupt the result if applied twice.
- **The write side is untouched.** `tq_matvec` (rotate) and `tq_store_k`/`tq_store_v` (quantize + store) keep running unconditionally over the whole batch's tokens, exactly as today, regardless of which read path a given item takes.
- **No change to `tq_attn_decode`, `tq_attn_scores`, `tq_attn_output`, `attn_prefill_ws4`, `attn_prefill_decoupled6_f16acc`, `attn_prefill_ws4_nw`, flash_attn2, or the `AttnDispatch`/`plan_attn_dispatch`/`run_attn` machinery from the mixed-batch-dispatch-split plan.** This plan only adds a new kernel, new scratch buffers, and a new, TQ-local partition that calls existing things.
- Every kernel-level test in this plan runs single-threaded against the GPU exactly like the existing TQ tests do — see `crates/model/tests/turboquant.rs`'s `gpu_lock()`/`GPU: Mutex<()>` pattern (a CUDA-graph capture in one test dies if another thread allocates on the same context) and reuse it, don't reinvent it.
- Rust naming/signature conventions to match exactly (established precedent, not a choice to relitigate): kernel wrapper methods take `&mut ViewMut<'_, T>` outputs and `&View<'_, T>` inputs, `bits: u8`, dimensions as plain `usize` cast to `i32` at the FFI boundary, `#[allow(clippy::too_many_arguments)]` on any wrapper past ~8 params (see `tq_store_k`/`tq_attn_decode` for the exact style).

---

### Task 1: Data model scaffolding — `seq_slots` accessor, dequant scratch buffers, threshold constant, env resolution

No new kernel and no dispatch behavior change in this task — it only adds the plumbing later tasks consume, so everything added here must compile, allocate correctly, and leave every existing test's behavior identical.

**Files:**
- Modify: `crates/model/src/cache.rs` (`KvPool`, add `seq_slots` accessor near `slot_table()` at line 287 and the existing `tq_key`/`tq_value` accessors at lines 688-758)
- Modify: `crates/model/src/lib.rs` (`TqBuffers` struct at line 1163, its constructor near `TqBuffers::new`/`tables: TqTables::new(...)` at line 1933, the `MIN_PREFILL_RUN` constant region at line 94, `Model`'s field list and a new accessor mirroring `split_mixed_batch()` at line 2056)

**Interfaces:**
- Consumes: `KvPool::extend`, `SeqState::slots` (both already exist, `cache.rs:555-582`, `cache.rs:58`), `TqTables::new(dev, d_head, quant)` (exists), `Buf<f16>`/`stream.alloc_zeros::<f16>()` (exist).
- Produces:
  - `KvPool::seq_slots(&self, id: SeqId) -> &[i32]` — this sequence's physical slot list in logical order.
  - `TqBuffers::dequant_k: Buf<f16>`, `TqBuffers::dequant_v: Buf<f16>` — each `[n_kv_heads, max_seq, d_head]` f16, allocated once at load.
  - `const TQ_DEQUANT_THRESHOLD: usize = 128;` — an explicit, conservative interim value (matching vLLM's own `_CONTINUATION_DECODE_THRESHOLD`, chosen here only as a defensible starting point since infero has not measured its own crossover yet — Task 5 measures the real value and updates this constant with a comment recording the measurement).
  - `Model::tq_prefill_dequant(&self) -> bool` — resolved once at load from `INFERO_TQ_PREFILL_DEQUANT`, mirroring `split_mixed_batch()`'s doc-comment style: `="0"` off (forces every run through `tq_attn_decode`, i.e. behaves as if `TQ_DEQUANT_THRESHOLD` were `usize::MAX`), unset or `="1"` on. Store the resolved bool as a new `Model` field, `tq_prefill_dequant: bool`.

**No standalone test for `seq_slots` in this task, deliberately.** `KvPool::extend`/`tq_key`/`tq_value` (and the new `seq_slots`) are all `pub(crate)` — internal to `infero-model`, unreachable from an external `tests/*.rs` integration test regardless of `seq_slots`' own visibility (confirmed: `mixed_batch_dispatch.rs` only ever calls `KvPool`'s `pub fn` surface — `alloc`, via `Model::new_pool`/`forward_batch` — never `extend` directly). Testing it in isolation would additionally require a hand-built `Config` (no `Default` impl, ~20+ fields, `crates/model/src/config.rs:41`), which is disproportionate scaffolding for a 3-line accessor that directly mirrors the already-correct `slot_table()`/`SeqState::slots` pattern one function above it. Its actual correctness — reading back the right physical slots in the right order — is verified for real in Task 3 (`long_prefill_dequant_dispatch_agrees_with_tq_attn_decode`): a wrong `seq_slots` would dequantize the wrong physical positions and that test's cosine/argmax check would fail. This task adds the accessor and confirms it compiles as part of the crate; behavior is proven downstream.

- [ ] **Step 1: Implement `KvPool::seq_slots`**

In `crates/model/src/cache.rs`, alongside `slot_table()` (line 287):

```rust
/// This sequence's physical slots, in the logical order `extend` assigned
/// them -- position `i` in the returned slice is where logical position
/// `i` actually lives. `tq_dequant_kv`'s caller uploads this to the device
/// as its `slots` argument, the same role `tq_store_k`/`tq_store_v`'s own
/// `slots` parameter already plays for the write side.
pub(crate) fn seq_slots(&self, id: SeqId) -> &[i32] {
    &self.seqs[id.0]
        .as_ref()
        .expect("seq_slots on a sequence that was never allocated")
        .slots
}
```

- [ ] **Step 2: Confirm it compiles**

Run: `cargo check -p infero-model`
Expected: builds clean (this method has no caller yet, so `#[allow(dead_code)]` may be needed temporarily until Task 3 calls it — check whether it warns-as-error in this workspace's lint config before deciding whether to add it).

- [ ] **Step 3: Add `dequant_k`/`dequant_v` to `TqBuffers`, the threshold constant, and the env-resolved `Model` field**

In `crates/model/src/lib.rs`:

```rust
// Near MIN_PREFILL_RUN, line 94:
/// Below this many query tokens, a TurboQuant prefill run stays on
/// `tq_attn_decode` -- not worth a dedicated dequantize-and-dispatch call.
/// This starting value mirrors vLLM's own `_CONTINUATION_DECODE_THRESHOLD`
/// as a conservative placeholder; Task 5 of the
/// `2026-09-06-tq4-prefill-dequant-dispatch` plan measures infero's own
/// crossover on this codebase's actual kernels and updates this constant
/// with the real number.
const TQ_DEQUANT_THRESHOLD: usize = 128;
```

```rust
// TqBuffers, line 1163:
struct TqBuffers {
    tables: TqTables,
    k_rot: Buf<f32>,
    v_rot: Buf<f32>,
    q_rot: Buf<f32>,
    q_qjl: Buf<f32>,
    acc_rot: Buf<f32>,
    /// Dense, rotated-basis KV for one sequence's dequantized prefill span.
    /// `[n_kv_heads, max_seq, d_head]`, sized once at load and reused across
    /// layers and calls -- the same "resolved once, reused everywhere"
    /// pattern `Scratch`'s other buffers already use. Only written when
    /// `Model::tq_prefill_dequant()` is true and at least one item in a
    /// batch is longer than `TQ_DEQUANT_THRESHOLD`.
    dequant_k: Buf<f16>,
    dequant_v: Buf<f16>,
}
```

Find `TqBuffers`' constructor (search for where `k_rot`/`v_rot`/`acc_rot` are allocated, near `tables: TqTables::new(&dev, cfg.d_head, kv_quant)?` at line 1933) and add:

```rust
dequant_k: stream.alloc_zeros::<f16>(cfg.n_kv_heads * max_seq * cfg.d_head)?,
dequant_v: stream.alloc_zeros::<f16>(cfg.n_kv_heads * max_seq * cfg.d_head)?,
```

(match whatever the surrounding constructor already names its `stream`/`max_seq` locals — read the ~20 lines around line 1933 first; don't assume the exact local variable names without checking, since `k_rot`'s own allocation line right above it already shows the real pattern to copy.)

Add the `Model` field and accessor, mirroring `split_mixed_batch` exactly:

```rust
// Model struct field list, alongside `split_mixed_batch: bool,` (line 1245):
/// Whether a long-enough TurboQuant prefill run dequantizes once and
/// reuses the dense tile kernels, resolved once at load from
/// `INFERO_TQ_PREFILL_DEQUANT`. `="0"` forces every run through
/// `tq_attn_decode` unchanged (`TQ_DEQUANT_THRESHOLD` effectively becomes
/// `usize::MAX`); unset or `="1"` leaves the threshold in `lib.rs:94` in
/// effect.
tq_prefill_dequant: bool,
```

```rust
// Accessor, alongside `split_mixed_batch()` (line 2056):
pub fn tq_prefill_dequant(&self) -> bool {
    self.tq_prefill_dequant
}
```

In `from_parts` (wherever `split_mixed_batch` is resolved from its env var — search for `INFERO_SPLIT_MIXED_BATCH` to find that exact spot and copy its `std::env::var(...).is_ok_and(...)` style), add the sibling resolution:

```rust
let tq_prefill_dequant = !std::env::var("INFERO_TQ_PREFILL_DEQUANT").is_ok_and(|v| v == "0");
```

and thread `tq_prefill_dequant` into the `Model { ... }` struct literal alongside `split_mixed_batch`.

- [ ] **Step 4: Run the full existing test suite to confirm zero behavior change**

Run: `cargo test -p infero-model -p infero-kernels`
Expected: PASS, identical to the pre-change baseline (this task allocates new buffers and adds one new field/accessor; nothing reads `dequant_k`/`dequant_v`/`tq_prefill_dequant` yet, so no existing test's numbers can move). Note the total VRAM-allocation delta this task introduces (`n_kv_heads * max_seq * d_head * 2 bytes * 2 buffers`) in the commit message — it's small (a single sequence's span, not the pool), but should be visible and named, not silent.

- [ ] **Step 5: Commit**

```bash
git add crates/model/src/cache.rs crates/model/src/lib.rs
git commit -m "tq4 dequant-dispatch: add seq_slots accessor, dequant scratch buffers, threshold constant

No behavior change -- new plumbing only, unused until later tasks in this
plan wire it up. See docs/superpowers/plans/2026-09-06-tq4-prefill-dequant-dispatch.md."
```

---

### Task 2: `tq_dequant_kv` CUDA kernel

**Files:**
- Modify: `crates/kernels/src/cu/turboquant.cu` (new kernel, alongside `tq_store_k`/`tq_store_v`'s CUDA definitions)
- Modify: `crates/kernels/src/lib.rs` (new `Kernels::tq_dequant_kv` wrapper, alongside `tq_store_k`/`tq_store_v` at lines 7454-7559)
- Test: `crates/kernels/tests/turboquant.rs` (extend)

**Interfaces:**
- Consumes: `Kernels::tq_store_k`/`tq_store_v` (existing, same file, for the test's setup — quantize known vectors into a cache the kernel then reads back), `DeviceTables`/`Tables`/`Codebook` (existing, `infero_kernels::turboquant`), `per_vector_block(d)` (existing helper already used by `tq_store_k`/`tq_store_v`/`tq_attn_output`).
- Produces: `Kernels::tq_dequant_kv(&self, dequant_k: &mut ViewMut<'_, f16>, dequant_v: &mut ViewMut<'_, f16>, k_codes: &View<'_, u8>, k_signs: &View<'_, u8>, k_scale: &View<'_, f16>, k_gamma: &View<'_, f16>, v_codes: &View<'_, u8>, v_scale: &View<'_, f16>, slots: &View<'_, i32>, k_levels: &View<'_, f32>, k_bits: u8, v_levels: &View<'_, f32>, v_bits: u8, n_kv_heads: usize, d_head: usize, n_slots: usize, kv_len: usize) -> Result<()>` — called by Task 3.

- [ ] **Step 1: Write the failing kernel-level test**

In `crates/kernels/tests/turboquant.rs`, add (near the existing `store_keys`/`host_decode` helpers at the top of the file):

```rust
/// Unpack a cached value vector on the host: uniform dequant, no rotation
/// undone (values are never rotated back either -- see `TqBuffers`' doc
/// comment in `crates/model/src/lib.rs`). Independent of the kernel on
/// purpose, same rationale as `host_decode` above.
fn host_dequant_no_rotation(codes: &[u8], scale: f32, cb: &Codebook) -> Vec<f32> {
    let bits = cb.bits as usize;
    let per_byte = 8 / bits;
    let mask = (1u8 << bits) - 1;
    (0..D)
        .map(|i| {
            let byte = codes[i / per_byte];
            let code = (byte >> ((i % per_byte) * bits)) & mask;
            cb.levels[code as usize] * scale
        })
        .collect()
}

#[test]
fn tq_dequant_kv_matches_store_then_manual_unpack() -> Result<()> {
    let k = kernels()?;
    let tables = DeviceTables::new(k.device(), D, KvQuant::Tq4)?;
    // `DeviceTables` already carries `k_codebook`/`v_codebook: Codebook` as
    // plain host-computed fields (`Codebook::solve` runs on the host,
    // `turboquant.rs:810-811`) -- no separate device->host download needed.
    let n = 5usize;
    let k_bits = tables.quant.k_mse_bits();
    let v_bits = tables.quant.v_bits();

    let keys = unit_vectors(n, 11);
    let values = unit_vectors(n, 12);
    let k_cache = store_keys(&k, &tables, &keys, n)?;
    let v_cache = store_values(&k, &tables, &values, n)?; // mirror store_keys but for tq_store_v -- write this helper the same way store_keys is written, reusing tq_matvec + tq_store_v.

    let stream = k.device().stream().clone();
    let slots: Vec<i32> = (0..n as i32).collect();
    let d_slots = stream.clone_htod(&slots)?;
    let mut d_dequant_k = stream.alloc_zeros::<half::f16>(n * D)?;
    let mut d_dequant_v = stream.alloc_zeros::<half::f16>(n * D)?;

    k.tq_dequant_kv(
        &mut d_dequant_k.as_view_mut(),
        &mut d_dequant_v.as_view_mut(),
        &k_cache.codes.as_view(),
        &k_cache.signs.as_view(),
        &k_cache.scale.as_view(),
        &k_cache.gamma.as_view(),
        &v_cache.codes.as_view(),
        &v_cache.scale.as_view(),
        &d_slots.as_view(),
        &tables.k_levels.as_view(),
        k_bits,
        &tables.v_levels.as_view(),
        v_bits,
        1,
        D,
        n,
        n,
    )?;
    k.device().synchronize()?;

    let got_k = stream.clone_dtoh(&d_dequant_k)?;
    let host_k_scale = stream.clone_dtoh(&k_cache.scale)?;
    let host_k_codes = stream.clone_dtoh(&k_cache.codes)?;
    let bytes_per_vec = D * k_bits as usize / 8;
    for v in 0..n {
        let expect = host_dequant_no_rotation(
            &host_k_codes[v * bytes_per_vec..(v + 1) * bytes_per_vec],
            host_k_scale[v].to_f32(),
            &tables.k_codebook,
        );
        let got: Vec<f32> = got_k[v * D..(v + 1) * D].iter().map(|x| x.to_f32()).collect();
        for (g, e) in got.iter().zip(&expect) {
            assert!((g - e).abs() < 1e-3, "key vector {v} mismatch: {g} vs {e}");
        }
    }
    Ok(())
}
```

`store_values` (called above) does not exist in this file yet — Step 3 adds it.

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p infero-kernels --test turboquant tq_dequant_kv_matches_store_then_manual_unpack -- --nocapture`
Expected: FAIL — `tq_dequant_kv` and `store_values` don't exist yet (compile error).

- [ ] **Step 3: Write `store_values` test helper**

Mirror `store_keys` (line 38) but call `tq_store_v` instead of `tq_store_k`, matching `tq_store_v`'s actual signature (no `qjl`, no `signs`, no `gamma` — it only has `codes`/`scale`, per `crates/kernels/src/lib.rs:7454-7465`):

```rust
fn store_values(k: &Kernels, tables: &DeviceTables, vectors: &[f32], n: usize) -> Result<Cache> {
    let stream = k.device().stream().clone();
    let bits = tables.quant.v_bits();
    let mut cache = alloc_cache(k, n, bits)?;

    let src = stream.clone_htod(vectors)?;
    let mut rotated = stream.alloc_zeros::<f32>(n * D)?;
    k.tq_matvec(&mut rotated.as_view_mut(), &src.as_view(), &tables.rotation.as_view(), D, n)?;

    let positions: Vec<i32> = (0..n as i32).collect();
    let dpos = stream.clone_htod(&positions)?;
    k.tq_store_v(
        &mut cache.codes.as_view_mut(),
        &mut cache.scale.as_view_mut(),
        &rotated.as_view(),
        &dpos.as_view(),
        &tables.v_levels.as_view(),
        bits,
        1,
        D,
        n,
        n,
    )?;
    k.device().synchronize()?;
    Ok(cache)
}
```

(`Cache` already has unused `signs`/`gamma` fields for a value-only cache — that's fine, matching `alloc_cache`'s existing shape reused for both key and value caches in this file already.)

- [ ] **Step 4: Write the CUDA kernel**

In `crates/kernels/src/cu/turboquant.cu`, alongside `tq_store_k`/`tq_store_v`'s definitions, add `tq_dequant_kv` — one thread block per `(kv_head, position)`, reading this position's physical slot from `slots[position]` and this position's packed key/value codes, writing the unpacked f16 vector straight to `dequant_k`/`dequant_v` at `[kv_head, position, :]` (no `slots` indirection on the *output* side — the destination is the compact per-item scratch buffer, addressed by logical position directly). The unpack math itself (MSE-level lookup + scale multiply for keys, uniform-level lookup + scale/zero for values) must exactly match what `tq_attn_decode_f32`'s own inline unpacking already does in this same file — read that kernel's unpack section before writing this one, and reuse its exact bit-shift/mask/level-table indexing rather than re-deriving it, since a subtly different unpack here is a silent accuracy bug (see spec §Error Handling).

```c
extern "C" __global__ void tq_dequant_kv(
    __half* dequant_k,        // [n_kv_heads, kv_len, d_head]
    __half* dequant_v,        // [n_kv_heads, kv_len, d_head]
    const unsigned char* k_codes,
    const unsigned char* k_signs,
    const __half* k_scale,
    const __half* k_gamma,
    const unsigned char* v_codes,
    const __half* v_scale,
    const int* slots,          // [kv_len], this sequence's slots in logical order
    const float* k_levels,     // [2^k_bits]
    int k_bits,
    const float* v_levels,     // [2^v_bits]
    int v_bits,
    int n_kv_heads,
    int d_head,
    int n_slots,               // pool-wide stride for k_codes/v_codes/... addressing
    int kv_len
) {
    int kv_head = blockIdx.x;
    int pos = blockIdx.y;
    if (kv_head >= n_kv_heads || pos >= kv_len) return;
    int slot = slots[pos];

    // Key: MSE-quantized, scale-only (no sign/gamma correction needed here --
    // `tq_attn_decode_f32`'s own key unpack already folds sign/gamma into the
    // *score* computation, not into a per-element dequantized value, so check
    // its exact treatment before assuming a plain level*scale is sufficient;
    // if it isn't, this kernel's key output must reproduce whatever
    // per-element correction that kernel applies before its dot product).
    int k_per_byte = 8 / k_bits;
    int k_bytes = (d_head * k_bits) / 8;
    const unsigned char* k_code_vec = k_codes + (size_t)(kv_head * n_slots + slot) * k_bytes;
    __half kscale = k_scale[kv_head * n_slots + slot];
    __half* k_out = dequant_k + ((size_t)kv_head * kv_len + pos) * d_head;
    for (int i = threadIdx.x; i < d_head; i += blockDim.x) {
        unsigned char byte = k_code_vec[i / k_per_byte];
        int mask = (1 << k_bits) - 1;
        int code = (byte >> ((i % k_per_byte) * k_bits)) & mask;
        k_out[i] = __float2half(k_levels[code] * __half2float(kscale));
    }

    // Value: uniform-quantized, scale-only (see `tq_attn_output`'s own value
    // unpack for the exact same "is scale alone sufficient" check).
    int v_per_byte = 8 / v_bits;
    int v_bytes = (d_head * v_bits) / 8;
    const unsigned char* v_code_vec = v_codes + (size_t)(kv_head * n_slots + slot) * v_bytes;
    __half vscale = v_scale[kv_head * n_slots + slot];
    __half* v_out = dequant_v + ((size_t)kv_head * kv_len + pos) * d_head;
    for (int i = threadIdx.x; i < d_head; i += blockDim.x) {
        unsigned char byte = v_code_vec[i / v_per_byte];
        int mask = (1 << v_bits) - 1;
        int code = (byte >> ((i % v_per_byte) * v_bits)) & mask;
        v_out[i] = __float2half(v_levels[code] * __half2float(vscale));
    }
}
```

Before finalizing this step, read `tq_attn_decode_f32`'s key-unpack section in this same file end to end and confirm whether `k_gamma`/`k_signs` participate in a per-element value correction (in which case this kernel's key loop needs the same correction term) or only in the score-level estimator's bias/variance correction *after* the dot product (in which case a plain `level * scale` per element, as drafted above, is correct and `k_signs`/`k_gamma` are unused by this kernel — remove them from the signature if so, since an unused parameter here would be dead weight, not caution). This is the one real open numerical question this task must resolve by reading the existing kernel, not by guessing; the test in Step 1 is the check that whichever choice is made is actually correct.

- [ ] **Step 5: Write the Rust wrapper**

In `crates/kernels/src/lib.rs`, alongside `tq_store_v`/`tq_store_k` (before `tq_attn_scores` at line 7561):

```rust
/// Unpack one sequence's `0..kv_len` cached span into a dense, rotated-basis
/// f16 buffer -- the read-side counterpart to `tq_store_k`/`tq_store_v`,
/// used when a prefill run is long enough that re-unpacking through
/// `tq_attn_decode` per query token stops paying for itself (see
/// `docs/superpowers/specs/2026-09-06-tq4-prefill-dequant-dispatch-design.md`).
/// Does **not** undo the rotation -- the output pairs with the
/// already-rotated `q_rot`/`acc_rot` the rest of the TQ pipeline already
/// uses, same convention as every other TQ kernel in this file.
#[allow(clippy::too_many_arguments)]
pub fn tq_dequant_kv(
    &self,
    dequant_k: &mut ViewMut<'_, f16>,
    dequant_v: &mut ViewMut<'_, f16>,
    k_codes: &View<'_, u8>,
    k_signs: &View<'_, u8>,
    k_scale: &View<'_, f16>,
    k_gamma: &View<'_, f16>,
    v_codes: &View<'_, u8>,
    v_scale: &View<'_, f16>,
    slots: &View<'_, i32>,
    k_levels: &View<'_, f32>,
    k_bits: u8,
    v_levels: &View<'_, f32>,
    v_bits: u8,
    n_kv_heads: usize,
    d_head: usize,
    n_slots: usize,
    kv_len: usize,
) -> Result<()> {
    let f = self
        .dev
        .kernels()
        .get("infero_turboquant", tq_src(), "tq_dequant_kv")?;
    let cfg = LaunchConfig {
        grid_dim: (n_kv_heads as u32, kv_len as u32, 1),
        block_dim: (per_vector_block(d_head), 1, 1),
        shared_mem_bytes: 0,
    };
    let (kb, vb, kh, dh, ns, kl) = (
        k_bits as i32,
        v_bits as i32,
        n_kv_heads as i32,
        d_head as i32,
        n_slots as i32,
        kv_len as i32,
    );
    let mut b = self.dev.stream().launch_builder(&f);
    b.arg(dequant_k)
        .arg(dequant_v)
        .arg(k_codes)
        .arg(k_signs)
        .arg(k_scale)
        .arg(k_gamma)
        .arg(v_codes)
        .arg(v_scale)
        .arg(slots)
        .arg(k_levels)
        .arg(&kb)
        .arg(v_levels)
        .arg(&vb)
        .arg(&kh)
        .arg(&dh)
        .arg(&ns)
        .arg(&kl);
    self.dev
        .profile()
        .time("tq_dequant_kv", self.dev.stream(), || {
            unsafe { b.launch(cfg) }.context("tq_dequant_kv")?;
            Ok(())
        })?;
    Ok(())
}
```

Adjust the argument list to match whatever Step 4 actually concluded about `k_signs`/`k_gamma` (drop them from both the CUDA kernel and this wrapper together if they turn out unused).

- [ ] **Step 6: Run test to verify it passes**

Run: `cargo test -p infero-kernels --test turboquant tq_dequant_kv_matches_store_then_manual_unpack -- --nocapture`
Expected: PASS

- [ ] **Step 7: `compute-sanitizer` pass on the new kernel**

Run: `compute-sanitizer --tool memcheck cargo test -p infero-kernels --test turboquant tq_dequant_kv_matches_store_then_manual_unpack --release -- --nocapture` and the same with `--tool racecheck`.
Expected: 0 hazards reported for both.

- [ ] **Step 8: Run the full kernels test suite**

Run: `cargo test -p infero-kernels`
Expected: PASS, no existing test's numbers changed (this task only adds a new kernel and test; nothing existing calls it yet).

- [ ] **Step 9: Commit**

```bash
git add crates/kernels/src/cu/turboquant.cu crates/kernels/src/lib.rs crates/kernels/tests/turboquant.rs
git commit -m "tq4 dequant-dispatch: add tq_dequant_kv kernel, correctness-tested against a host reference

Unpacks one sequence's cached span into dense rotated-basis f16, staying
in TQ's existing rotated basis rather than inverse-rotating -- see the
kernel's doc comment. Not called from the model's dispatch path yet."
```

---

### Task 3: Dispatch wiring — partition, dequantize, reuse the dense kernels

**Files:**
- Modify: `crates/model/src/lib.rs` (the `Some(tq)` branch, `lib.rs:4785` onward, specifically the point today's single `tq_attn_decode` call happens at `lib.rs:4872-4905`)
- Test: `crates/model/tests/turboquant.rs` (extend)

**Interfaces:**
- Consumes: `Kernels::tq_dequant_kv` (Task 2), `KvPool::seq_slots` (Task 1), `Model::tq_prefill_dequant()`/`TQ_DEQUANT_THRESHOLD`/`TqBuffers::dequant_k`/`dequant_v` (Task 1), `BatchItemKind` (existing, from the mixed-batch-dispatch-split plan), `Kernels::attn_prefill_ws4`/`attn_prefill_decoupled6_f16acc`/`prefill_attention` (existing, unmodified).
- Produces: the actual performance fix — no new public API, this task changes `attention()`'s internal behavior under TQ.

- [ ] **Step 1: Write the failing model-level test — a single long first-chunk prefill**

In `crates/model/tests/turboquant.rs`, add (reusing this file's existing `load`/`setup!`/`argmax`/`cosine` helpers, and its `qwen2.5-0.5b-instruct-q8_0.gguf` test model, which is `d_head=64` — confirmed by `mixed_batch_dispatch.rs`'s own comment to satisfy `attn_prefill_ws4`'s `prefill_attention` eligibility gate even though `attn_prefill_decoupled6_f16acc`/flash_attn2 require `d_head=256` and won't fire on this model; that's fine, this test only needs *some* dense tile kernel to be reachable):

```rust
/// A long TurboQuant prefill run must produce the same next-token
/// prediction whether or not the dequant-dispatch path serves it --
/// `INFERO_TQ_PREFILL_DEQUANT=0` and the (measured) default must agree,
/// because they're computing the identical mathematical result through
/// two different kernel paths.
#[test]
fn long_prefill_dequant_dispatch_agrees_with_tq_attn_decode() -> Result<()> {
    let _gpu = gpu_lock();
    let Some(path) = model_path() else { return Ok(()); };
    let gguf = infero_gguf::Gguf::open(&path)?;
    let tok = infero_tokenizer::Tokenizer::from_gguf(&gguf)?;

    // Long enough to clear even a generously large threshold -- see
    // TQ_DEQUANT_THRESHOLD's value in lib.rs at the time this runs.
    let prompt = ("The quick brown fox jumps over the lazy dog. ").repeat(40);
    let ids = tok.encode(&prompt, Some(false), false);
    assert!(ids.len() > 256, "prompt too short to exercise the long-run path: {} tokens", ids.len());

    unsafe { std::env::set_var("INFERO_TQ_PREFILL_DEQUANT", "0") };
    let mut model_off = infero_model::Model::load_quantized(
        infero_cuda::Device::new(0)?, &gguf, 1024, infero_model::KvCacheQuant::Tq4,
    )?;
    unsafe { std::env::remove_var("INFERO_TQ_PREFILL_DEQUANT") };
    let mut session_off = model_off.new_session()?;
    let logits_off: Vec<f32> = model_off
        .forward(&ids, infero_model::BatchItemKind::Prefill, &mut session_off)?
        .to_vec();

    let mut model_on = infero_model::Model::load_quantized(
        infero_cuda::Device::new(0)?, &gguf, 1024, infero_model::KvCacheQuant::Tq4,
    )?;
    assert!(model_on.tq_prefill_dequant(), "dequant-dispatch should default on");
    let mut session_on = model_on.new_session()?;
    let logits_on: Vec<f32> = model_on
        .forward(&ids, infero_model::BatchItemKind::Prefill, &mut session_on)?
        .to_vec();

    assert_eq!(argmax(&logits_off), argmax(&logits_on));
    let cos = cosine(&logits_off, &logits_on);
    eprintln!("  dequant-dispatch vs tq_attn_decode logit cosine: {cos:.6}");
    assert!(cos > 0.999, "cosine {cos:.6} -- two paths computing the same thing should match tightly");
    Ok(())
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p infero-model --test turboquant long_prefill_dequant_dispatch_agrees_with_tq_attn_decode -- --nocapture --test-threads=1`
Expected: FAIL — either a compile error (`tq_prefill_dequant` not found, if Task 1 wasn't merged first) or a numeric mismatch/panic once Task 1 is present but Task 3's dispatch isn't wired (the "on" path is currently identical to "off" since nothing reads `tq_prefill_dequant` yet — in which case this specific test would trivially pass for the wrong reason). **Before writing Step 3, confirm this test actually exercises the new path once wired** by checking (temporarily, not committed) that a deliberately-wrong dequant call changes the answer — if the test can't tell the two paths apart, it isn't testing what this task claims.

- [ ] **Step 3: Implement the partition and dispatch, inside the `Some(tq)` branch**

At `lib.rs:4872`, replace the unconditional `tq_attn_decode` call with a partition. The write side (`tq_matvec`/`tq_store_k`/`tq_store_v`, lines 4790-4857) stays exactly as-is, above this point, unconditional over the whole batch — do not move or gate it.

A batch can interleave short and long prefill items in any order (a decode-prefix invariant only guarantees *decode* items come first, not that every short *prefill* item is adjacent to another). Rather than collecting two flat lists and hoping the short ones happen to be contiguous, coalesce `items` into maximal contiguous runs of the same class up front — one `tq_attn_decode` call per short/decode run, one dequant+`ws4` call per long-prefill item (long-prefill items don't coalesce with each other: each needs its own `kv_len` and its own dequantized scratch, exactly like the long-run loop already does one call per item):

```rust
#[derive(PartialEq, Eq)]
enum RunClass { ShortOrDecode, Long }

let dequant_threshold = if self.tq_prefill_dequant { TQ_DEQUANT_THRESHOLD } else { usize::MAX };
// `items` here is whatever this function already has in scope to build
// `seq_of`/`positions`/`table` above (the same source `BatchItemKind`
// tagging the mixed-batch-dispatch-split plan added to `BatchItem`).
let mut short_runs: Vec<(usize, usize)> = Vec::new(); // (base, total_tokens)
let mut long_items: Vec<(usize, usize, usize, SeqId)> = Vec::new(); // (base, len, kv_len, seq)
let mut base = 0usize;
let mut cur_short_run: Option<(usize, usize)> = None; // (run_base, run_len_so_far)
for item in items {
    let len = item.tokens.len();
    let class = if item.kind == BatchItemKind::Prefill && len > dequant_threshold {
        RunClass::Long
    } else {
        RunClass::ShortOrDecode
    };
    match class {
        RunClass::ShortOrDecode => {
            cur_short_run = Some(match cur_short_run.take() {
                Some((run_base, run_len)) => (run_base, run_len + len),
                None => (base, len),
            });
        }
        RunClass::Long => {
            if let Some(run) = cur_short_run.take() {
                short_runs.push(run);
            }
            // `kv_len` for this item is `pool.len(item.seq)` -- the write
            // side (tq_matvec/tq_store_k/tq_store_v, unconditional above
            // this point) already extended the pool with this call's own
            // tokens, so the pool's tracked length already IS this
            // sequence's real, post-extension history, not just `len`
            // (its own new tokens). `KvPool::len(&self, id: SeqId) ->
            // usize` already exists (`cache.rs:537`) -- no new accessor.
            long_items.push((base, len, pool.len(item.seq), item.seq));
        }
    }
    base += len;
}
if let Some(run) = cur_short_run.take() {
    short_runs.push(run);
}

for (run_base, run_len) in &short_runs {
    // Unchanged tq_attn_decode call, but scoped to this one contiguous
    // run instead of the whole batch -- everything below this point
    // through the existing `if self.kern.tq_decode_attention(&dims) {
    // ... } else { ... }` block (lines 4872-4946) stays byte-for-byte
    // identical to today EXCEPT that `n`/`dims.n_tokens`/the buffer
    // slices it reads must now cover only `run_base..run_base+run_len`,
    // not the whole batch, and `kv_len` for this call is the max over
    // just this run's own items (mirroring how the F16 path's decode
    // prefix already computes its own `decode_kv_len` rather than
    // reusing the batch-wide max, `AttnDispatch::decode_kv_len`).
}

for (base, len, kv_len, seq_id) in &long_items {
    let n_kv_heads = dims.n_kv_heads;
    let d_head = dims.d_head;
    let host_slots = pool.seq_slots(*seq_id);
    let d_slots = self.dev.stream().clone_htod(host_slots)?;
    let (kcodes, ksigns, kscale, kgamma) = pool.tq_key(layer);
    let (vcodes, vscale) = pool.tq_value(layer);
    self.kern.tq_dequant_kv(
        &mut tq.dequant_k.slice_mut(..n_kv_heads * kv_len * d_head),
        &mut tq.dequant_v.slice_mut(..n_kv_heads * kv_len * d_head),
        &kcodes.as_view(), &ksigns.as_view(), &kscale.as_view(), &kgamma.as_view(),
        &vcodes.as_view(), &vscale.as_view(),
        &d_slots.as_view(),
        &tq.tables.k_levels.as_view(), k_bits,
        &tq.tables.v_levels.as_view(), v_bits,
        n_kv_heads, d_head, dims.n_slots, *kv_len,
    )?;

    // Synthetic single-sequence, identity-mapped layout: physical slot i
    // IS logical position i in the compact scratch buffer above, distinct
    // from the real pool's slot_table (which indexes the full pool width
    // and would be the wrong stride here).
    let identity_seq_of = vec![0i32; *kv_len];
    let identity_positions: Vec<i32> = (0..*kv_len as i32).collect();
    let identity_slots: Vec<i32> = (0..*kv_len as i32).collect();
    let d_seq_of = self.dev.stream().clone_htod(&identity_seq_of)?;
    let d_positions = self.dev.stream().clone_htod(&identity_positions)?;
    let d_slot_table = self.dev.stream().clone_htod(&identity_slots)?;
    let synth_batch = BatchLayout {
        seq_of: &d_seq_of.as_view(),
        positions: &d_positions.as_view(),
        slot_table: &d_slot_table.as_view(),
        table_stride: *kv_len,
    };
    let run_dims = AttnDims { n_tokens: *len, n_slots: *kv_len, ..dims };
    let q_lo = base * da;
    let q_hi = (base + len) * da;
    // Deliberately always `attn_prefill_ws4`, never `decoupled6`/flash_attn2
    // -- see the rationale below. `attn_prefill_ws4` has its own
    // `anyhow::ensure!(self.prefill_attention(&dims), ...)` internally, so
    // an unsupported shape fails loud rather than silently, exactly like
    // every other caller of this kernel already gets.
    self.kern.attn_prefill_ws4(
        &mut tq.acc_rot.slice_mut(q_lo..q_hi),
        &tq.q_rot.slice(q_lo..q_hi),
        &tq.dequant_k.slice(..n_kv_heads * kv_len * d_head),
        &tq.dequant_v.slice(..n_kv_heads * kv_len * d_head),
        synth_batch,
        run_dims,
        0,
        *len,
        *kv_len,
        attn_scale,
        &mut self.act.attn_partial.as_view_mut(),
    )?;
}
```

This is intentionally more conservative than the spec's illustrative sketch: it always calls `attn_prefill_ws4` for long runs, never `decoupled6`/flash_attn2, because `ws4` alone is eligible across every shape this plan's own tests can exercise (the `d_head=64` test model included, since `attn_prefill_ws4`'s own `prefill_attention` gate has no `d_head==256` requirement, unlike `decoupled6`/flash_attn2) and adding the other two kernels' own eligibility branches (matching the F16 path's real three-way dispatch at `lib.rs:4601-4650`, with its `INFERO_PREFILL_T6` handling and flash_attn2's `#[cfg(feature = "flash_attn2")]` gate) is separable, lower-priority work the spec's Out of Scope section already didn't promise. If a later measurement (Task 5/6) shows `ws4` alone leaves real performance on the table for the production `d_head=256` shape, that's a follow-up, not a gap in this task.

Resolve the marked-unclear points (this item's `SeqId` source, whether short items form one contiguous prefix) by reading `attention()`'s full body from its start down to line 4785 before writing this step for real — both are almost certainly already resolved by data this function already has in scope (the same `items`/`seq_of`/`table` construction the F16 path and the write-side `tq_store_k`/`tq_store_v` calls both already consume above this point), not new information this task must invent.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p infero-model --test turboquant long_prefill_dequant_dispatch_agrees_with_tq_attn_decode -- --nocapture --test-threads=1`
Expected: PASS

- [ ] **Step 5: Add a loud failure for untested bit-width combinations**

Per spec §Error Handling: this task's own tests (this file, Task 4) only exercise `KvCacheQuant::Tq4` (`k_bits=4, v_bits=4`). Rather than silently trusting `tq_dequant_kv` on a combination nothing here has verified, add an explicit check at the point the long-run path is chosen, mirroring `TqBuffers::new`'s existing loud-failure style (`lib.rs:1924`, `"KV cache quantization ({kv_quant:?}) has no kernels on this backend yet; ..."`):

```rust
// Immediately before the `for (base, len, kv_len) in &long_items` loop:
if !long_items.is_empty() {
    anyhow::ensure!(
        k_bits == 4 && v_bits == 4,
        "tq_dequant_kv: only k_bits=4/v_bits=4 has been validated against a \
         host reference (see crates/kernels/tests/turboquant.rs); this load's \
         KvCacheQuant is k_bits={k_bits}/v_bits={v_bits}. Set \
         INFERO_TQ_PREFILL_DEQUANT=0 to use the (slower but validated for \
         every bit width) tq_attn_decode path instead, or extend Task 2's \
         test coverage to this combination before removing this guard."
    );
}
```

This keeps the new path from ever running silently-unverified math in production, while never blocking the one combination (`Tq4`) this whole plan is scoped to fix — TurboQuant is not enabled in production today regardless (Global Constraints), so this guard cannot regress anything currently deployed.

- [ ] **Step 6: Run the full existing TurboQuant and mixed-batch test suites**

Run: `cargo test -p infero-model --test turboquant --test mixed_batch_dispatch -- --test-threads=1`
Expected: PASS — this task must not change any F16-path test's answer (it never touches `AttnDispatch`) nor any existing TQ test's answer for runs that stay under threshold.

- [ ] **Step 7: Commit**

```bash
git add crates/model/src/lib.rs crates/model/tests/turboquant.rs
git commit -m "tq4 dequant-dispatch: wire the long-run path into the TQ attention branch

Long prefill runs now dequantize once and reuse attn_prefill_ws4 instead
of tq_attn_decode's per-query-token unpack. Gated by
INFERO_TQ_PREFILL_DEQUANT (default on) and TQ_DEQUANT_THRESHOLD."
```

---

### Task 3.5: QJL-aware `tq_dequant_kv` (inserted during execution — see ledger)

**Why this task exists.** Task 3's implementer found, and measured, a real problem Task 2's own review did not catch because it wasn't the question asked: `tq_attn_decode` evaluates TurboQuant's real score estimator as **two separate terms**, not one —

```
est = scale·⟨q_rot, cb_k[code]⟩  +  qjl_scale·(√(π/2)/d)·γ·⟨q_qjl, s⟩
```

(`turboquant.cu:387`) — and `tq_dequant_kv` (Task 2) only materializes the first (MSE-codebook) term. Task 2's own finding, "`k_signs`/`k_gamma` are not a per-element correction to the codebook value," is correct — but the QJL term isn't a per-element key correction at all, it's a **second, independent score-level dot product** (`⟨q_qjl, s⟩`), and dropping it changes the answer for any QJL-enabled config, including `Tq4` — the config this whole plan exists to speed up. Measured end to end (Task 3's report, §2): `tq4` vs `tq4`-without-QJL cosine **0.289**, a different next token, not a rounding difference.

Task 3 shipped a safe stopgap — `uses_qjl()` keeps any QJL config on the existing, slower, correct `tq_attn_decode` path, pinned by a bit-exact-equality test — and correctly declined to force a fix under its own scope. **Ruling (user, verbatim: "1" / "哪能怂啊" / "补上QJL修正项，把Tq4真正跑通加速"): do the full fix. `Tq4` must actually get the speedup, not just the QJL-free configs.** A second ruling (controller): the bit-width guard `ensure!` Task 3 added per the original brief should become a graceful per-item fallback to `tq_attn_decode` (same pattern as the QJL gate), not a hard error — `k8v4` is a real, previously-recommended preset in this codebase's own tests and must not start 500ing on a long prompt. Verify Task 3's review addressed this before starting this task; if not, this task's Step 1 covers it too.

**The fix is known and cheap, not a research problem.** The QJL term is linear in `q_rot`. `q_qjl = tables.qjl · q_rot` (`Kernels::tq_matvec`, a `d×d` mat-vec), so for any real matrix `Q` and vectors `x, y`: `⟨Q·x, y⟩ = ⟨x, Qᵀ·y⟩`. Substituting `Q = tables.qjl`, `x = q_rot`, `y = s` (the sign sketch a cached key decodes to):

```
⟨q_qjl, s⟩ = ⟨Q·q_rot, s⟩ = ⟨q_rot, Qᵀ·s⟩
```

so

```
k_eff = scale·cb_k[code]  +  qjl_scale·(√(π/2)/d)·γ·(Qᵀ·s)
```

is a per-key vector — independent of the query — that reproduces the *exact* two-term estimator when dotted against `q_rot` by a standard dense kernel. Cost: one `d×d` mat-vec against the sign sketch per cached key, `O(H·L·d²)` against attention's own `O(H·L²·d)` — noise at any `L` this path serves.

**Files:**
- Modify: `crates/kernels/src/turboquant.rs` — `Tables`/`DeviceTables` gain a `qjl_t: Vec<f32>`/`Buf<f32>` field. `Tables::new` (~line 81-99) already computes `rotation_t = transpose(&rotation, d)` for exactly this "need both directions" reason (see its own doc comment, "`Πᵀ`, for mapping the attention output back"); add `qjl_t = transpose(&qjl, d)` immediately beside it, uploaded in `DeviceTables::new` the same way `rotation_t` is (`stream.clone_htod(&tables.qjl_t)?`). This is the same one-time, load-time cost `rotation_t` already pays — not a per-call cost.
- Modify: `crates/kernels/src/cu/turboquant.cu` — extend `tq_dequant_kv` (added in Task 2) to add the QJL correction term to its **key** output only (values are unaffected — `tq_attn_output`'s value unpack has no sign/gamma term at all, confirmed in Task 2's review, and this fix doesn't change that).
- Modify: `crates/kernels/src/lib.rs` — `Kernels::tq_dequant_kv`'s wrapper re-gains `k_signs: &View<'_, u8>` and `k_gamma: &View<'_, f16>` (removed in Task 2 because the *old*, incomplete kernel didn't need them — it does now), plus new `qjl_t: &View<'_, f32>` and `qjl_scale: f32` parameters.
- Modify: `crates/model/src/lib.rs` — Task 3's call site passes the new arguments; the `uses_qjl()` gate is **deleted**, not just loosened — once this lands, a QJL-enabled long-prefill run is exactly as eligible for the dequant path as any other. Also apply the bit-width-guard ruling above if Task 3's review didn't already.
- Test: `crates/kernels/tests/turboquant.rs` — extend with a QJL-aware correctness test.
- Test: `crates/model/tests/turboquant.rs` — Task 3 added `the_qjl_estimator_keeps_long_runs_off_the_dequant_path`, pinning the *old* (correct, for its own scope) behavior that QJL configs stay off the dequant path. That test's premise is now wrong on purpose — replace it with the mirror image: a `Tq4`-quantized long prefill run now agrees with `tq_attn_decode` (same cosine bar Task 3's non-QJL tests already clear, `> 0.999`), proving the fold actually closes the gap Task 3's report measured at 0.289.

**Interfaces:**
- Consumes: `Kernels::tq_matvec` (existing, for reference/comparison only — not called by the new kernel path itself, which does its own in-kernel mat-vec), `tq_sign_of` (existing `__device__` helper in `turboquant.cu`, decodes one sign bit to ±1 — reuse it, don't re-derive the bit-unpack), `TQ_SQRT_HALF_PI` (existing `#define`, `turboquant.cu:25`), `KvQuant::uses_qjl()`/`qjl_scale()` (existing, `crates/kernels/src/turboquant.rs:718-726`).
- Produces: `Tables::qjl_t: Vec<f32>` / `DeviceTables::qjl_t: Buf<f32>`; `Kernels::tq_dequant_kv`'s updated signature (adds `k_signs`, `k_gamma`, `qjl_t`, `qjl_scale`); the deleted `uses_qjl()` gate at the call site.

**The one thing this task must resolve empirically, not by hand-derivation (the way Task 2 resolved its own open question): the exact index/stride convention for reading `qjl_t` inside the new kernel code.** `Tables`' matrices are stored column-major (`m[j*d+i]` is row `i`, column `j` — see the struct's own doc comment) specifically so a mat-vec's inner loop reads consecutive addresses across threads; get the transpose direction backwards and the kernel will silently compute the wrong per-key vector rather than crash. **Do not guess the indexing from this description — derive it by writing a host-side reference first** (mirroring `crates/kernels/tests/turboquant.rs`'s existing `host_decode`/`host_dequant_no_rotation` pattern: independently reconstruct `k_eff` on the host from downloaded `qjl_t`, `k_signs`, `k_gamma`, `k_codes`, `k_scale`, using the formula above verbatim), then match the kernel to that reference until the test passes — the same TDD discipline this whole plan has used throughout, not an exception for this task.

- [ ] **Step 1: Confirm/fix the bit-width guard from Task 3, if not already done**

Check `crates/model/src/lib.rs` for the `ensure!(k_bits == 4 && v_bits == 4, ...)` guard Task 3 added. If Task 3's own review already changed this to a graceful per-item fallback (check the ledger / Task 3's final commits before starting), skip this step. Otherwise: change it from a hard `anyhow::ensure!` into an eligibility condition — an item whose bit widths don't match falls through to the short/decode `tq_attn_decode` path for that one item, the same way a QJL-enabled item currently does, rather than erroring the whole forward pass.

- [ ] **Step 2: Write the failing kernel-level test**

In `crates/kernels/tests/turboquant.rs`, add a host-side reference that reconstructs the *full* two-term estimator's effective key vector — not just the MSE term `host_dequant_no_rotation` (Task 2) already covers:

```rust
/// The full per-key effective vector the QJL-aware `tq_dequant_kv` must
/// produce: `k_eff = scale·cb[code] + qjl_scale·(sqrt(pi/2)/d)·gamma·(Qt·s)`,
/// where `Qt` is `tables.qjl`'s transpose and `s` is this key's sign sketch
/// decoded to +-1. Reproduces `tq_attn_decode_f32`'s two-term estimator
/// exactly when dotted against a real `q_rot` -- see turboquant.cu:387.
fn host_qjl_aware_key_eff(
    codes: &[u8],
    scale: f32,
    signs: &[u8],
    gamma: f32,
    cb: &Codebook,
    qjl_t: &[f32],       // column-major, D*D
    qjl_scale: f32,
    d: usize,
) -> Vec<f32> {
    let mse = host_dequant_no_rotation(codes, scale, cb); // Task 2's existing helper
    let sign = |i: usize| -> f32 {
        let byte = signs[i / 8];
        if (byte >> (i % 8)) & 1 == 0 { 1.0 } else { -1.0 } // match tq_sign_of's real bit convention -- verify against the CUDA source, don't assume 0=+1
    };
    let c = qjl_scale * (std::f32::consts::PI / 2.0).sqrt() / d as f32;
    (0..d)
        .map(|i| {
            let qt_s: f32 = (0..d).map(|j| qjl_t[j * d + i] * sign(j)).sum(); // column-major: qjl_t[j*d+i] is row i, column j -- this is the (Qt * s)[i] convention to verify against the kernel, per this task's own "resolve empirically" instruction
            mse[i] + c * gamma * qt_s
        })
        .collect()
}

#[test]
fn tq_dequant_kv_reproduces_the_full_qjl_estimator() -> Result<()> {
    let k = kernels()?;
    let tables = DeviceTables::new(k.device(), D, KvQuant::Tq4)?; // Tq4 uses QJL -- KvQuant::Tq4.uses_qjl() must be true
    assert!(tables.quant.uses_qjl(), "test setup: Tq4 must use QJL");
    let n = 5usize;
    let k_bits = tables.quant.k_mse_bits();
    let v_bits = tables.quant.v_bits();

    let keys = unit_vectors(n, 21);
    let values = unit_vectors(n, 22);
    let k_cache = store_keys(&k, &tables, &keys, n)?; // Task 2's existing helper -- unchanged
    let v_cache = store_values(&k, &tables, &values, n)?;

    let stream = k.device().stream().clone();
    let slots: Vec<i32> = (0..n as i32).collect();
    let d_slots = stream.clone_htod(&slots)?;
    let mut d_dequant_k = stream.alloc_zeros::<half::f16>(n * D)?;
    let mut d_dequant_v = stream.alloc_zeros::<half::f16>(n * D)?;

    k.tq_dequant_kv(
        &mut d_dequant_k.as_view_mut(),
        &mut d_dequant_v.as_view_mut(),
        &k_cache.codes.as_view(),
        &k_cache.signs.as_view(),   // re-added this task
        &k_cache.scale.as_view(),
        &k_cache.gamma.as_view(),   // re-added this task
        &v_cache.codes.as_view(),
        &v_cache.scale.as_view(),
        &d_slots.as_view(),
        &tables.k_levels.as_view(),
        k_bits,
        &tables.v_levels.as_view(),
        v_bits,
        &tables.qjl_t.as_view(),   // new
        tables.quant.qjl_scale(),  // new
        1,
        D,
        n,
        n,
    )?;
    k.device().synchronize()?;

    let got_k = stream.clone_dtoh(&d_dequant_k)?;
    let host_k_scale = stream.clone_dtoh(&k_cache.scale)?;
    let host_k_codes = stream.clone_dtoh(&k_cache.codes)?;
    let host_k_signs = stream.clone_dtoh(&k_cache.signs)?;
    let host_k_gamma = stream.clone_dtoh(&k_cache.gamma)?;
    let host_qjl_t = stream.clone_dtoh(&tables.qjl_t)?; // downloading a device buffer back to host for the reference -- check the real accessor pattern this file already uses elsewhere (e.g. how `rotation_is_an_isometry_on_the_device` downloads `Tables` fields) rather than assuming `clone_dtoh` is the exact right call here
    let bytes_per_vec = D * k_bits as usize / 8;
    let sign_bytes_per_vec = D / 8;
    for v in 0..n {
        let expect = host_qjl_aware_key_eff(
            &host_k_codes[v * bytes_per_vec..(v + 1) * bytes_per_vec],
            host_k_scale[v].to_f32(),
            &host_k_signs[v * sign_bytes_per_vec..(v + 1) * sign_bytes_per_vec],
            host_k_gamma[v].to_f32(),
            &tables.k_codebook,
            &host_qjl_t,
            tables.quant.qjl_scale(),
            D,
        );
        let got: Vec<f32> = got_k[v * D..(v + 1) * D].iter().map(|x| x.to_f32()).collect();
        for (g, e) in got.iter().zip(&expect) {
            assert!((g - e).abs() < 1e-2, "key {v} mismatch: {g} vs {e}"); // looser tolerance than Task 2's MSE-only test -- two summed terms compound more f16/f32 rounding; tighten if the real numbers allow once this passes
        }
    }
    Ok(())
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cargo test -p infero-kernels --test turboquant tq_dequant_kv_reproduces_the_full_qjl_estimator -- --nocapture`
Expected: FAIL — compile error (`qjl_t`/`k_signs`/`k_gamma`/`qjl_scale` not yet accepted by `tq_dequant_kv`'s current signature).

- [ ] **Step 4: Add `qjl_t` to `Tables`/`DeviceTables`**

`crates/kernels/src/turboquant.rs`, in `Tables::new` (~line 81-99):

```rust
let rotation = random_rotation(d, seed);
let rotation_t = transpose(&rotation, d);
let s = gaussian_matrix(d, seed ^ 0x5151_5151_5151_5151);
let qjl = matmul_t(&s, &rotation, d);
let qjl_t = transpose(&qjl, d);

Ok(Self {
    d,
    rotation: to_column_major(&rotation, d),
    rotation_t: to_column_major(&rotation_t, d),
    qjl: to_column_major(&qjl, d),
    qjl_t: to_column_major(&qjl_t, d),
})
```

Add the `qjl_t: Vec<f32>` field to the `Tables` struct (beside `qjl`) and `qjl_t: Buf<f32>` to `DeviceTables`, uploaded in `DeviceTables::new` beside the existing `qjl: stream.clone_htod(&tables.qjl)?,` line: `qjl_t: stream.clone_htod(&tables.qjl_t)?,`.

- [ ] **Step 5: Extend the CUDA kernel**

In `crates/kernels/src/cu/turboquant.cu`, extend `tq_dequant_kv` (added in Task 2): after computing each key element's MSE-based value (unchanged), stage this key's decoded sign vector into shared memory once per block (every output thread needs the *whole* d-length sign vector, not just its own index — read `tq_attn_decode_f32`'s own handling of `sign`/`qs`/shared-memory staging around `turboquant.cu:370-390` for the established pattern in this exact file, and follow it rather than inventing a new one), then have each thread accumulate its own output element's `Qᵀ·s` term against `qjl_t` and add the scaled result to the MSE value already computed. Value output (`dequant_v`) is unchanged by this task — no sign/gamma term applies to values.

Do not guess the `qjl_t` indexing — get Step 2's test passing, which is the actual verification that the indexing (and the sign-bit convention: confirm `tq_sign_of`'s real mapping of bit value to +1/-1 against `turboquant.cu`'s own source, don't assume) is correct.

- [ ] **Step 6: Update the Rust wrapper**

`crates/kernels/src/lib.rs`, `Kernels::tq_dequant_kv`: re-add `k_signs: &View<'_, u8>`, `k_gamma: &View<'_, f16>` to the signature (removed in Task 2), add `qjl_t: &View<'_, f32>`, `qjl_scale: f32`. Update the `LaunchConfig`/`b.arg(...)` chain to match, keeping this file's existing FFI-wrapper conventions (`usize` cast to `i32` at the boundary, etc.).

- [ ] **Step 7: Run test to verify it passes**

Run: `cargo test -p infero-kernels --test turboquant tq_dequant_kv_reproduces_the_full_qjl_estimator -- --nocapture`
Expected: PASS

- [ ] **Step 8: `compute-sanitizer` pass**

Run memcheck and racecheck against the new test, same as Task 2's own precedent. Expected: 0 hazards for both.

- [ ] **Step 9: Update the model-level call site and delete the `uses_qjl()` gate**

In `crates/model/src/lib.rs`, find Task 3's `tq_dequant_kv` call site (the long-runs loop inside the TQ dispatch) and: pass the sequence's real `k_signs`/`k_gamma` (already fetched via `pool.tq_key(layer)`, which Task 3's code already calls — no new pool accessor needed), `tq.tables.qjl_t.as_view()`, and `quant.qjl_scale()`. Delete the `uses_qjl()` eligibility check entirely — a QJL-enabled item is now exactly as eligible for the long path as any other, gated only by run length and (per Step 1) bit width.

- [ ] **Step 10: Replace the now-obsolete model-level pinning test**

In `crates/model/tests/turboquant.rs`, Task 3 added `the_qjl_estimator_keeps_long_runs_off_the_dequant_path`, asserting bit-exact equality specifically *because* the dequant path was closed to QJL configs. That premise is gone. Replace it with a test proving the fold actually works end to end:

```rust
/// The whole point of this task: a real Tq4 (QJL-enabled) long prefill run
/// must now agree with tq_attn_decode just as well as the non-QJL configs
/// already do, closing the 0.289-cosine gap Task 3's report measured.
#[test]
fn a_qjl_enabled_long_prefill_now_agrees_via_the_dequant_path() -> Result<()> {
    // Mirror long_prefill_dequant_dispatch_agrees_with_tq_attn_decode
    // (Task 3), but load with KvCacheQuant::Tq4 specifically (not a
    // QJL-free variant) and confirm cosine > 0.999 between
    // INFERO_TQ_PREFILL_DEQUANT=0 and unset, same bar Task 3's own
    // non-QJL tests already clear.
}
```

Write out the real test body following Task 3's own `long_prefill_dequant_dispatch_agrees_with_tq_attn_decode` structure exactly (same file, same helpers) — don't leave it as the sketch above in the actual commit.

- [ ] **Step 11: Run the full existing TurboQuant and mixed-batch test suites**

Run: `cargo test -p infero-model --test turboquant --test mixed_batch_dispatch -- --test-threads=1`
Expected: PASS, same failure set as Task 3 left it (pre-existing, unrelated failures only — verify by comparing against Task 3's own documented failure list rather than assuming zero failures means success or new failures mean this task broke something already broken).

- [ ] **Step 12: Commit**

```bash
git add crates/kernels/src/turboquant.rs crates/kernels/src/cu/turboquant.cu crates/kernels/src/lib.rs crates/kernels/tests/turboquant.rs crates/model/src/lib.rs crates/model/tests/turboquant.rs
git commit -m "tq4 dequant-dispatch: add the QJL correction term, close the gap for Tq4 itself

tq_dequant_kv now reproduces TurboQuant's full two-term score estimator
(MSE codebook term + QJL sign-sketch term, linear-algebra fold verified
against a host reference), not just the MSE term Task 2 shipped. Deletes
the uses_qjl() eligibility gate Task 3 added as a stopgap -- Tq4 itself
now gets the long-run speedup this whole plan exists for."
```

---

### Task 4: Continuation-prefill, mixed-batch, and threshold-boundary scenarios

**Files:**
- Modify: `crates/model/tests/turboquant.rs` (extend)

**Interfaces:**
- Consumes: `BatchItem::new`/`BatchItem::without_logits` (existing, `crates/model/tests/mixed_batch_dispatch.rs:132,161` show the exact call shape), `Model::forward_batch(&[BatchItem], &mut KvPool)` (existing).
- Produces: no new production interfaces — this task is validation-only, closing the spec's Testing §2 scenarios (b), (c), (d).

- [ ] **Step 1: Write the failing test — continuation prefill above threshold with real prior TQ history**

```rust
/// A continuation chunk (real prior TQ-cached history, not just this
/// call's own tokens) above the threshold must also agree between the two
/// dispatch paths -- this is the scenario `tq_dequant_kv`'s `kv_len`
/// (`pool.len(item.seq)`, the sequence's real post-extension length, not
/// just `item.tokens.len()`) exists to get right.
#[test]
fn continuation_prefill_above_threshold_agrees() -> Result<()> {
    let _gpu = gpu_lock();
    let Some(path) = model_path() else { return Ok(()); };
    let gguf = infero_gguf::Gguf::open(&path)?;
    let tok = infero_tokenizer::Tokenizer::from_gguf(&gguf)?;
    let first = tok.encode("The history of the Roman Empire begins with", Some(false), false);
    let second_text = ("and then the legions marched further east. ").repeat(30);
    let second = tok.encode(&second_text, Some(false), false);
    assert!(second.len() > 200, "continuation too short: {} tokens", second.len());

    let run = |dequant: &str| -> Result<Vec<f32>> {
        unsafe { std::env::set_var("INFERO_TQ_PREFILL_DEQUANT", dequant) };
        let mut model = infero_model::Model::load_quantized(
            infero_cuda::Device::new(0)?, &gguf, 4096, infero_model::KvCacheQuant::Tq4,
        )?;
        unsafe { std::env::remove_var("INFERO_TQ_PREFILL_DEQUANT") };
        let mut session = model.new_session()?;
        let _ = model.forward(&first, infero_model::BatchItemKind::Prefill, &mut session)?;
        Ok(model.forward(&second, infero_model::BatchItemKind::Prefill, &mut session)?.to_vec())
    };

    let off = run("0")?;
    let on = run("1")?;
    assert_eq!(argmax(&off), argmax(&on));
    let cos = cosine(&off, &on);
    eprintln!("  continuation-prefill dequant-dispatch vs tq_attn_decode cosine: {cos:.6}");
    assert!(cos > 0.999, "cosine {cos:.6}");
    Ok(())
}
```

- [ ] **Step 2: Run test to verify it fails or passes for the right reason**

Run: `cargo test -p infero-model --test turboquant continuation_prefill_above_threshold_agrees -- --nocapture --test-threads=1`
Expected: PASS if Task 3 got `kv_len` right (`pool.len(item.seq)`, covering the real prior history, not just the current chunk). If this fails, Task 3's Step 3 has the exact bug the spec's Data Model section warned about — fix there, not here.

- [ ] **Step 3: Write the failing test — mixed decode + long-prefill batch in one call**

Mirror `mixed_batch_dispatch.rs`'s own real API (`BatchItem::new`, `model.forward_batch(&items, &mut pool)`) directly, not `Model::forward`'s single-session convenience wrapper, since this scenario needs one call spanning two sequences:

```rust
#[test]
fn mixed_decode_and_long_prefill_batch_agrees() -> Result<()> {
    let _gpu = gpu_lock();
    let Some(path) = model_path() else { return Ok(()); };
    let gguf = infero_gguf::Gguf::open(&path)?;
    let tok = infero_tokenizer::Tokenizer::from_gguf(&gguf)?;
    let decoding_prompt = tok.encode("Hello,", Some(false), false);
    let long_prompt_text = ("A long prefill chunk needs many tokens. ").repeat(30);
    let long_prompt = tok.encode(&long_prompt_text, Some(false), false);
    assert!(long_prompt.len() > 200);

    let run = |dequant: &str| -> Result<Vec<f32>> {
        unsafe { std::env::set_var("INFERO_TQ_PREFILL_DEQUANT", dequant) };
        let mut model = infero_model::Model::load_quantized(
            infero_cuda::Device::new(0)?, &gguf, 4096, infero_model::KvCacheQuant::Tq4,
        )?;
        unsafe { std::env::remove_var("INFERO_TQ_PREFILL_DEQUANT") };
        let mut pool = model.new_pool(4096, 4)?;
        let decoder = pool.alloc().expect("free sequence slot");
        // Prime the decoder with one real token first, exactly like
        // mixed_batch_dispatch.rs's own decode+prefill test does, so the
        // decode item in the mixed call below is a genuine decode step
        // (kv history already present) and not a length-1 prefill.
        let item = infero_model::BatchItem::new(decoder, &decoding_prompt, infero_model::BatchItemKind::Prefill);
        let _ = model.forward_batch(std::slice::from_ref(&item), &mut pool)?;
        let next_tok = decoding_prompt[decoding_prompt.len() - 1];

        let fresh = pool.alloc().expect("free sequence slot");
        let items = vec![
            infero_model::BatchItem::new(decoder, std::slice::from_ref(&next_tok), infero_model::BatchItemKind::Decode),
            infero_model::BatchItem::new(fresh, &long_prompt, infero_model::BatchItemKind::Prefill),
        ];
        // `mixed_batch_dispatch.rs:153,181` confirms both the accessor
        // (`model.config().vocab_size`) and that a combined-rows result is
        // ordered to match `items`' order, sliced in `vocab`-sized chunks.
        let vocab = model.config().vocab_size;
        let out = model.forward_batch(&items, &mut pool)?;
        Ok(out[vocab..2 * vocab].to_vec()) // the long-prefill item's row (items[1])
    };

    let off = run("0")?;
    let on = run("1")?;
    assert_eq!(argmax(&off), argmax(&on));
    assert!(cosine(&off, &on) > 0.999);
    Ok(())
}
```

This mirrors `mixed_batch_dispatch.rs`'s own mixed decode+prefill test (lines 170-181, building `items` with a `Decode` and a `Prefill` `BatchItem` together) directly.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p infero-model --test turboquant mixed_decode_and_long_prefill_batch_agrees -- --nocapture --test-threads=1`
Expected: PASS. If it fails, the most likely bug is exactly the one `mixed_batch_dispatch.rs`'s own module doc comment warns about for the F16 split — a per-item buffer offset or `kv_len` mixed up between items — check Task 3's Step 3 partition/offset math first.

- [ ] **Step 5: Write the exact-threshold-boundary test**

```rust
/// One token on either side of `TQ_DEQUANT_THRESHOLD` must land on the
/// path the threshold says it should, and both must still agree with the
/// `INFERO_TQ_PREFILL_DEQUANT=0` reference -- this is the test that would
/// catch an off-by-one in the `>` vs `>=` comparison at the partition site.
#[test]
fn threshold_boundary_both_sides_agree_with_reference() -> Result<()> {
    let _gpu = gpu_lock();
    let Some(path) = model_path() else { return Ok(()); };
    let gguf = infero_gguf::Gguf::open(&path)?;
    let tok = infero_tokenizer::Tokenizer::from_gguf(&gguf)?;
    // Build a prompt at least TQ_DEQUANT_THRESHOLD + 8 tokens long by
    // repetition, then test at exactly `threshold` and `threshold + 1`
    // tokens by truncating the encoded id list -- check
    // TQ_DEQUANT_THRESHOLD's current real value in lib.rs before writing
    // this test's repeat count, since it must exceed threshold + 8.
    let long_text = ("word ").repeat(64);
    let ids = tok.encode(&long_text, Some(false), false);
    assert!(ids.len() > 136, "need >136 tokens to bracket a 128 threshold with room to spare");

    for &n in &[128usize, 129] { // replace 128 with TQ_DEQUANT_THRESHOLD's real value, read from lib.rs, not hardcoded blind
        let slice = &ids[..n.min(ids.len())];
        let off_logits = {
            unsafe { std::env::set_var("INFERO_TQ_PREFILL_DEQUANT", "0") };
            let mut model = infero_model::Model::load_quantized(
                infero_cuda::Device::new(0)?, &gguf, 1024, infero_model::KvCacheQuant::Tq4,
            )?;
            unsafe { std::env::remove_var("INFERO_TQ_PREFILL_DEQUANT") };
            let mut session = model.new_session()?;
            model.forward(slice, infero_model::BatchItemKind::Prefill, &mut session)?.to_vec()
        };
        let on_logits = {
            let mut model = infero_model::Model::load_quantized(
                infero_cuda::Device::new(0)?, &gguf, 1024, infero_model::KvCacheQuant::Tq4,
            )?;
            let mut session = model.new_session()?;
            model.forward(slice, infero_model::BatchItemKind::Prefill, &mut session)?.to_vec()
        };
        assert_eq!(argmax(&off_logits), argmax(&on_logits), "mismatch at n={n}");
        assert!(cosine(&off_logits, &on_logits) > 0.999, "cosine too low at n={n}");
    }
    Ok(())
}
```

- [ ] **Step 6: Run test to verify it passes**

Run: `cargo test -p infero-model --test turboquant threshold_boundary_both_sides_agree_with_reference -- --nocapture --test-threads=1`
Expected: PASS

- [ ] **Step 7: `compute-sanitizer` pass on the mixed-batch scenario**

Run: `compute-sanitizer --tool memcheck cargo test -p infero-model --test turboquant mixed_decode_and_long_prefill_batch_agrees --release -- --nocapture --test-threads=1` and the same with `--tool racecheck`.
Expected: 0 hazards. The synthetic `BatchLayout`'s identity mapping and the per-item scratch-buffer slicing (Task 3, Step 3) are exactly the kind of new addressing this check exists for — see spec §Testing item 3.

- [ ] **Step 8: Commit**

```bash
git add crates/model/tests/turboquant.rs
git commit -m "tq4 dequant-dispatch: add continuation-prefill, mixed-batch, and threshold-boundary tests

Closes the remaining scenarios from the design spec's Testing section
(items b, c, d). All pass, memcheck/racecheck clean on the mixed-batch case."
```

---

### Task 5: Real threshold measurement

**Files:**
- Create: `crates/kernels/examples/tq_prefill_dequant_vs_decode_bench.rs` (mirrors `tq_attn_decode_bench.rs`'s structure exactly)
- Modify: `crates/model/src/lib.rs` (`TQ_DEQUANT_THRESHOLD`'s value and doc comment, updated with the real measured number)

**Interfaces:**
- Consumes: `Kernels::tq_attn_decode` (existing), `Kernels::tq_dequant_kv` (Task 2), `Kernels::attn_prefill_ws4` (existing) — all called directly with synthetic data, same style as `tq_attn_decode_bench.rs`, no `Model`/session involved.
- Produces: a real, recorded measurement and the final value of `TQ_DEQUANT_THRESHOLD`.

- [ ] **Step 1: Write the benchmark**

Copy `crates/kernels/examples/tq_attn_decode_bench.rs` verbatim as a starting point (same constants: `N_HEADS=24, N_KV_HEADS=4, D_HEAD=256, N_SLOTS=8192, K_BITS=4, V_BITS=4` — this checkpoint's real shape), then for a sweep of `RUN_TOKENS` values (e.g. `[8, 16, 32, 64, 96, 128, 192, 256, 384, 512]`) and a fixed `KV_LEN=2048`, time two things per point:

1. `tq_attn_decode` called with `dims.n_tokens = RUN_TOKENS` (today's path for any run length) — same call shape as the existing bench, just varying `n_tokens` instead of fixing it at 1.
2. `tq_dequant_kv` (once, at this `KV_LEN`) immediately followed by `attn_prefill_ws4` reading the dequantized scratch with a synthetic identity `BatchLayout` — same construction Task 3 uses, built directly here with pseudo-random data exactly like the rest of this bench file already does (no real `Model`/quantized cache needed, matching this file's own existing "measures timing and occupancy, not numerics" scope note at its top).

```rust
//! `tq_attn_decode` vs. `tq_dequant_kv`+`attn_prefill_ws4`, across prefill
//! run lengths, to find the real crossover on this codebase's own kernels
//! -- see docs/superpowers/specs/2026-09-06-tq4-prefill-dequant-dispatch-design.md
//! and TQ_DEQUANT_THRESHOLD's doc comment in crates/model/src/lib.rs.
//!
//!     cargo run --release -p infero-kernels --example tq_prefill_dequant_vs_decode_bench

use anyhow::Result;
use half::f16;
use infero_cuda::Device;
use infero_kernels::{AttnDims, BatchLayout, Kernels};

const N_HEADS: usize = 24;
const N_KV_HEADS: usize = 4;
const D_HEAD: usize = 256;
const N_SLOTS: usize = 8192;
const KV_LEN: usize = 2048;
const K_BITS: u8 = 4;
const V_BITS: u8 = 4;
const RUN_LENGTHS: &[usize] = &[8, 16, 32, 64, 96, 128, 192, 256, 384, 512];

// (reuse pseudo_random_bytes/pseudo_random_f32 from tq_attn_decode_bench.rs verbatim)

fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter("warn").init();
    let dev = Device::new(0)?;
    let k = Kernels::new(dev.clone());
    let stream = dev.stream().clone();

    // (same synthetic k_codes/k_signs/k_scale/k_gamma/v_codes/v_scale/
    // k_levels/v_levels/slot_table setup as tq_attn_decode_bench.rs, at
    // N_SLOTS/KV_LEN scale -- copy verbatim, this bench doesn't need real
    // quantized data, only the same shapes.)

    println!("run_tokens | tq_attn_decode (us) | dequant+ws4 (us) | speedup");
    for &n in RUN_LENGTHS {
        // tq_attn_decode timing: same call as tq_attn_decode_bench.rs but
        // with dims.n_tokens = n, seq_of/positions sized to n (all in the
        // same sequence, positions KV_LEN-n..KV_LEN), 3 warm-up + 200 timed
        // iterations, host-clock bracketed exactly like the existing bench.

        // dequant+ws4 timing: tq_dequant_kv once (kv_len=KV_LEN) then
        // attn_prefill_ws4 with n_tokens=n, kv_len=KV_LEN, a synthetic
        // identity BatchLayout over a compact [N_KV_HEADS, KV_LEN, D_HEAD]
        // f16 buffer -- same construction as Task 3's dispatch code, built
        // directly here. 3 warm-up + 200 timed iterations covering BOTH
        // calls together (the dequant is part of what this run length pays,
        // same as it would in the real dispatch).

        println!("{n:>10} | {:>20.2} | {:>17.2} | {:.2}x", /* ... */);
    }
    Ok(())
}
```

This step's code block is intentionally structural rather than fully spelled out for the setup/timing bodies, because they are byte-for-byte copies of `tq_attn_decode_bench.rs`'s already-committed, already-correct patterns (lines 24-171 of that file) with only the swept parameter and the second code path changed — **read that file in full and copy its exact setup code**, rather than retyping it with any drift, since this bench's entire value is measuring the real kernels, not a reimplementation of them.

- [ ] **Step 2: Run the benchmark on `bw`**

Run: `ssh bw "cd /home/jeff/infero && cargo run --release -p infero-kernels --example tq_prefill_dequant_vs_decode_bench"` (or locally if this machine has the same GPU class available — check which is this session's real convention for kernel benchmarks before assuming `bw`).
Record the real output table in the commit message for Step 4 below — this is the artifact that justifies whatever threshold value gets chosen, matching this codebase's own established practice (`FA2_ROW_THRESHOLD`'s doc comment in `lib.rs:4527-4553` records exactly this kind of real measured table).

- [ ] **Step 3: Pick the crossover and update `TQ_DEQUANT_THRESHOLD`**

From the real table, find the smallest `run_tokens` where `dequant+ws4` is at or below `tq_attn_decode`'s time, with a small safety margin (round up to the next tested point if the crossover falls between two swept values, matching how `FA2_ROW_THRESHOLD` was chosen from its own sweep). Update the constant in `crates/model/src/lib.rs`:

```rust
/// Below this many query tokens, a TurboQuant prefill run stays on
/// `tq_attn_decode` -- not worth a dedicated dequantize-and-dispatch call.
/// Measured <DATE> on <GPU>, `tq_prefill_dequant_vs_decode_bench`,
/// kv_len=2048, this checkpoint's real N_HEADS=24/N_KV_HEADS=4/D_HEAD=256:
/// <PASTE THE REAL TABLE HERE>. Crossover at <N> tokens; this constant is
/// set to <CHOSEN VALUE> for a safety margin above it.
const TQ_DEQUANT_THRESHOLD: usize = /* the real measured value */;
```

- [ ] **Step 4: Re-run Tasks 3 and 4's test suites against the updated threshold**

Run: `cargo test -p infero-model --test turboquant --test mixed_batch_dispatch -- --test-threads=1`
Expected: PASS — the threshold-boundary test (Task 4, Step 5) reads `TQ_DEQUANT_THRESHOLD`'s value at compile time via the constant, not a hardcoded number, so it stays correct across this change **only if that test was written to reference the constant directly rather than the literal `128`/`129` used while drafting it — fix Task 4's Step 5 to import and reference `TQ_DEQUANT_THRESHOLD` directly if it doesn't already**, otherwise this step will silently test the wrong boundary once the constant changes.

- [ ] **Step 5: Commit**

```bash
git add crates/kernels/examples/tq_prefill_dequant_vs_decode_bench.rs crates/model/src/lib.rs
git commit -m "tq4 dequant-dispatch: measure the real threshold, update TQ_DEQUANT_THRESHOLD

Real crossover from crates/kernels/examples/tq_prefill_dequant_vs_decode_bench.rs
on <GPU>: <one-line summary>. See the constant's doc comment for the full table."
```

---

### Task 6: End-to-end validation

**Files:**
- No source changes expected — this task is validation-only. If it finds a real bug, fix it in the file it's actually in and note that in the commit message rather than treating this task as pure documentation.

**Interfaces:**
- Consumes: everything from Tasks 1-5.
- Produces: the final go/no-go evidence for this plan.

- [ ] **Step 1: Real wall-clock end-to-end A/B on a TQ-quantized load**

Find the real server CLI's KV-quant flag (`crates/server/src/main.rs:46`, `--kv-quant`, parsed via `infero_model::KvCacheQuant::parse` at line 127) and start two isolated server instances on a free port (not the production port 8301), one with `--kv-quant tq4` and `INFERO_TQ_PREFILL_DEQUANT=0`, one with the same flag and the variable unset. Send the same long-prompt request to both (reuse a prompt shape similar to this session's own earlier `concurrent_long_test.py`-style script, or write a minimal single-request curl call with a multi-thousand-token prompt) and compare wall-clock time. Confirm the "on" run is measurably faster and does not time out, closing the loop on the original client-timeout report this whole plan started from.

- [ ] **Step 2: Full stress test against the TQ-quantized load**

Run: `scripts/server_stress_test.py --passes 5` (check its actual flag for pointing at a non-default port/model if it doesn't already default to one, matching how this script was invoked for the mixed-batch-dispatch-split plan's own final validation) against the `--kv-quant tq4` instance from Step 1.
Expected: all categories STABLE, no new failures relative to a same-script run against the F16 instance.

- [ ] **Step 3: Record results and close out**

Write a short summary of Steps 1-2's real numbers (before/after wall-clock, stress-test pass/fail) into this plan's own progress ledger if using `subagent-driven-development`, or directly into the final commit message if executed inline.

- [ ] **Step 4: Commit** (only if Step 1 or 2 required a real fix; otherwise this task ends without a commit, which is fine — validation-only tasks don't need to manufacture a diff)
