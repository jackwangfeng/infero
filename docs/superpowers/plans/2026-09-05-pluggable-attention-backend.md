# Pluggable Attention Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a second, vendor-sourced attention backend (`FlashAttn2Ffi`, a torch-free FFI shim around Dao-AILab/flash-attention's CUDA kernel source) selectable at runtime alongside infero's existing hand-rolled kernels, without touching GDN or regressing today's default path.

**Architecture:** A small `AttentionBackend` trait with two implementations — `InferoHandRolled` (today's existing dispatch, unmodified) and `FlashAttn2Ffi` (new). `Model` resolves which one to use once at load time by probing hardware capability + the model's `AttnDims`/`KvQuant`, and inserts exactly one new branch ahead of the existing prefill cascade in `attention()` — the existing cascade itself is untouched, so a build without the new `flash_attn2` cargo feature (or on a shape/kv_quant the new backend doesn't support) behaves identically to today.

**Tech Stack:** Rust, CUDA (hand-written `.cu` NVRTC-JIT kernels for the existing path, AOT `nvcc`-compiled static archive for the new FFI shim — mirrors the existing `cutlass` feature's build), `extern "C"` FFI, no libtorch/Python.

**Spec:** `docs/superpowers/specs/2026-09-05-pluggable-attention-backend-design.md`

## Global Constraints

- Scope is standard (non-GDN) attention only. Do not touch `gdn_*` kernels, `crates/kernels/src/cu/gdn.cu`, or anything under the `Some(tq)` TurboQuant arm of `Model::attention()`.
- No libtorch/Python runtime dependency, ever — the FFI shim links a `nvcc`-AOT-compiled static archive of extracted CUDA source, exactly like `crates/kernels/build.rs`'s existing `cutlass` feature.
- Runtime selection has no silent double-gate: once `flash_attn2` is compiled in, `FlashAttn2Ffi::supports()` alone decides eligibility — no second required env var to "actually" activate it (the exact footgun this session hit with `INFERO_CUTLASS_DIR`/`INFERO_FP8_UNIFIED`/`INFERO_ATTN_MMA` needing to all be set together).
- `KvQuant::F16` (dense, unquantized) is the only KV mode `FlashAttn2Ffi` may claim; anything with `is_quantized() == true` (TurboQuant Tq2/Tq4) is out of scope for this backend.
- This pass is scoped to the **prefill** path only (the exact shape `prefill_profile` exercises: one sequence, contiguous, causal). Decode-step attention keeps using infero's existing kernels unconditionally in this pass — a vendor decode backend is a separate, later addition, not required to prove the architecture works.
- Do not chase `FlashAttn2Ffi` beating infero's hand-rolled kernels on v6000/sm_120a. It is known to lose (FA2 here runs only via generic sm_80 PTX-JIT forward compatibility). Success is correctness + real runtime selection, not speed.
- Never commit to git unless explicitly told to in a task step below — none of these steps say to; leave changes staged in the working tree.
- `bw` build/run facts, already verified this session, reuse them rather than rediscovering:
  - `ssh bw` (ControlMaster active, no re-auth).
  - Synced repo copy at `/home/jeff/infero` on `bw` — verify sync via `md5sum` on changed files before building there, matching how this session confirmed sync earlier.
  - `cutlass` feature build: `cd /home/jeff/infero && source ~/.cargo/env && CUDA_VISIBLE_DEVICES=3 INFERO_CUTLASS_DIR=/tmp/cutlass_src cargo build --release -p infero-model --example prefill_profile --features cutlass`
  - Runtime fast path needs `INFERO_FP8_UNIFIED=1 INFERO_ATTN_MMA=1` set (unrelated existing gates, not part of this plan, but needed to get a real `prefill_profile` baseline number for comparison).
  - Checkpoint: `/home/jeff/models/qwen38-27b-fp8`. GPU 3 (`CUDA_VISIBLE_DEVICES=3`) has the most headroom.
  - `ncu`/hardware profiler access is blocked (`ERR_NVGPUCTRPERM`) — don't plan around it.
  - A pre-existing, already-cloned flash-attention source checkout is at `/tmp/flash_attn_src` on `bw` (used by an earlier research pass this session) — its torch-free CUDA kernel files are under `csrc/flash_attn/src/` (`flash_fwd_kernel.h`, `kernel_traits.h`, `flash_fwd_launch_template.h`). Reuse this checkout; don't reclone.

---

### Task 1: `AttentionBackend` trait and `HardwareCaps`

**Files:**
- Create: `crates/kernels/src/attn_backend.rs`
- Modify: `crates/kernels/src/lib.rs` — add `pub mod attn_backend;` near the other `pub mod` declarations at the top of the file, and `pub use attn_backend::{AttentionBackend, AttnCallCtx, HardwareCaps};`

**Interfaces:**
- Produces: `trait AttentionBackend`, `struct AttnCallCtx<'a>`, `struct HardwareCaps`, all `pub` from `infero_kernels`. Later tasks implement `AttentionBackend` for two concrete types and call `HardwareCaps::probe(&Device)`.

- [ ] **Step 1: Write `HardwareCaps` and its probe, plus a `#[test]` for it**

```rust
// crates/kernels/src/attn_backend.rs

use crate::{AttnDims, BatchLayout, View, ViewMut};
use infero_cuda::Device;
use anyhow::Result;

/// One-time device capability probe, cheap enough to call at `Model` load
/// time and cache — not per forward call.
#[derive(Debug, Clone, Copy)]
pub struct HardwareCaps {
    pub cc_major: u32,
    pub cc_minor: u32,
}

impl HardwareCaps {
    pub fn probe(dev: &Device) -> Result<Self> {
        let (major, minor) = dev.compute_capability()?;
        Ok(Self { cc_major: major, cc_minor: minor })
    }

    /// `cc_major.cc_minor >= major.minor`, the same comparison vLLM's own
    /// `DeviceCapability` uses for backend floors (e.g. FA2's real
    /// `>= (8, 0)` floor).
    pub fn at_least(&self, major: u32, minor: u32) -> bool {
        (self.cc_major, self.cc_minor) >= (major, minor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probes_a_real_device_capability() {
        let dev = Device::new(0).expect("no CUDA device for this test");
        let caps = HardwareCaps::probe(&dev).expect("probe failed");
        assert!(caps.cc_major >= 5, "implausible compute capability major: {}", caps.cc_major);
        assert!(caps.at_least(caps.cc_major, caps.cc_minor));
        assert!(!caps.at_least(caps.cc_major + 1, 0));
    }
}
```

Before writing this, check `infero_cuda::Device` for whatever method already exposes compute capability (grep `crates/cuda/src/*.rs` for `compute_capability` or `cuDeviceGetAttribute`/`CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY`) — reuse it verbatim if it exists under a different name, don't add a second device-capability query path. If none exists, add a minimal one in `crates/cuda/src/device.rs` (or wherever `Device` lives) using the driver API's `cuDeviceGetAttribute` for `CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR`/`_MINOR`, following whatever pattern nearby methods on `Device` already use for driver calls.

- [ ] **Step 2: Run it**

Run: `cargo test -p infero-kernels --lib attn_backend:: -- --nocapture`
Expected: PASS, printing nothing but exiting 0 (requires a CUDA device — run this on a machine with one, e.g. `bw`, or the local dev box's A4000 which also satisfies `cc_major >= 5`).

- [ ] **Step 3: Write the trait and `AttnCallCtx`**

```rust
/// The canonical, infero-native call shape every attention backend receives.
/// This is infero's own KV-pool layout, not a lowest-common-denominator one
/// — a backend that wants a different physical layout (e.g. contiguous
/// fp16 blocks) repacks internally in its own `prefill`/`decode`, on its own
/// time budget. See the design doc's "trait boundary" section.
pub struct AttnCallCtx<'a> {
    pub out: &'a mut ViewMut<'a, f32>,
    pub q: &'a View<'a, f32>,
    pub k_cache: &'a View<'a, f16>,
    pub v_cache: &'a View<'a, f16>,
    pub batch: BatchLayout<'a>,
    pub dims: AttnDims,
    pub run_base: usize,
    pub run_tokens: usize,
    pub kv_len: usize,
    pub scale: f32,
    /// `InferoHandRolled` ignores this (its kernels reach the stream via
    /// its own `&Kernels` handle) — it exists for `FlashAttn2Ffi`, which has
    /// no `Kernels` of its own to pull one from. See Task 3.
    pub stream: &'a std::sync::Arc<infero_gpu::Stream>,
}

/// One implementation of standard (non-GDN) attention's prefill path.
/// Decode-step attention is out of scope for this trait in this pass —
/// see the plan's Global Constraints.
pub trait AttentionBackend: Send + Sync {
    fn name(&self) -> &'static str;

    /// Lower wins when multiple backends are eligible for the same call.
    fn priority(&self) -> u32;

    /// Checked once at `Model` load time. A backend that returns `true`
    /// here and then fails at call time has a bug in `supports`, not a
    /// normal fallback condition — see the design doc's "Error handling".
    fn supports(&self, caps: &HardwareCaps, dims: &AttnDims, kv_quant: crate::KvQuant) -> bool;

    fn prefill(&self, ctx: &mut AttnCallCtx<'_>) -> Result<()>;
}

/// Picks the highest-priority backend whose `supports()` returns true.
/// Callers hold the result's index/name and re-fetch from their own
/// backend list — this fn borrows nothing so `Model` can cache the outcome
/// without a lifetime fight.
pub fn select_backend<'a>(
    backends: &'a [Box<dyn AttentionBackend>],
    caps: &HardwareCaps,
    dims: &AttnDims,
    kv_quant: crate::KvQuant,
    forced: Option<&str>,
) -> Result<&'a dyn AttentionBackend> {
    if let Some(name) = forced {
        return backends
            .iter()
            .find(|b| b.name() == name)
            .filter(|b| b.supports(caps, dims, kv_quant))
            .map(|b| b.as_ref())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "INFERO_ATTN_BACKEND={name} does not support this shape/kv_quant \
                     (dims={dims:?}, kv_quant={kv_quant:?})"
                )
            });
    }
    backends
        .iter()
        .filter(|b| b.supports(caps, dims, kv_quant))
        .min_by_key(|b| b.priority())
        .map(|b| b.as_ref())
        .ok_or_else(|| anyhow::anyhow!("no attention backend supports dims={dims:?} kv_quant={kv_quant:?}"))
}
```

Adjust the exact field types of `AttnCallCtx` (`View`/`ViewMut`/`f16` import paths, `BatchLayout`'s lifetime parameter) to match what's actually in scope in `crates/kernels/src/lib.rs` at the call site — read `attn_prefill_decoupled6_f16acc`'s signature there (already `pub fn attn_prefill_decoupled6_f16acc(&self, out: &mut ViewMut<'_, f32>, q: &View<'_, f32>, k_cache: &View<'_, f16>, v_cache: &View<'_, f16>, batch: BatchLayout<'_>, dims: AttnDims, run_base: usize, run_tokens: usize, kv_len: usize, scale: f32) -> Result<()>`) and mirror its exact types field-for-field — this task is defining the shape that call already has, not inventing a new one.

- [ ] **Step 4: `cargo check -p infero-kernels` compiles clean**

Run: `cargo check -p infero-kernels`
Expected: no errors (warnings about unused trait/fn are fine — nothing implements or calls it yet).

- [ ] **Step 5: Commit**

```bash
git add crates/kernels/src/attn_backend.rs crates/kernels/src/lib.rs
git commit -m "Add AttentionBackend trait and HardwareCaps probe"
```

---

### Task 2: `InferoHandRolled` backend (wraps existing dispatch, no behavior change)

**Files:**
- Modify: `crates/kernels/src/attn_backend.rs`

**Interfaces:**
- Consumes: `AttentionBackend`, `AttnCallCtx`, `HardwareCaps` from Task 1; `Kernels::prefill_attention`, `Kernels::attn_prefill_decoupled6_f16acc`, `Kernels::attn_prefill_ws4` from `crates/kernels/src/lib.rs` (unmodified).
- Produces: `struct InferoHandRolled<'k> { kern: &'k Kernels }` implementing `AttentionBackend`, used by Task 4's wiring.

This task **only** covers the `prefill_run`-eligible branch of the existing cascade (the `attn_prefill_decoupled6_f16acc`/`attn_prefill_ws4` choice at `crates/model/src/lib.rs:3256-3302`) — not `attn_decode`, `attn_flash`, or the 3-kernel score fallback, which stay exactly as they are in `Model::attention()` and are never reached through this trait in this pass (see Task 4: the new backend selection only intercepts the call when `prefill_run.is_some()`, identical to today's own gating condition, so the rest of the cascade is provably unreachable-different from today).

- [ ] **Step 1: Write the test — same oracle, called through the trait**

In `crates/kernels/tests/ops.rs`, find `attn_prefill_matches_the_three_kernels` (extended earlier this session for a `bulk48` variant — read its current body first, including that addition, to match its exact reference-computation and tolerance). Add one more block inside it that builds an `InferoHandRolled` wrapping the same `Kernels` instance, calls `.prefill()` with an `AttnCallCtx` built from the test's existing buffers, and asserts the same tolerance against the same reference:

```rust
{
    use infero_kernels::attn_backend::{AttentionBackend, AttnCallCtx, InferoHandRolled};
    let backend = InferoHandRolled::new(&kern);
    let mut ctx = AttnCallCtx {
        out: &mut out.slice_mut(..),
        q: &q.slice(..),
        k_cache: &k_cache.as_view(),
        v_cache: &v_cache.as_view(),
        batch,
        dims,
        run_base: 0,
        run_tokens,
        kv_len,
        scale,
        stream: kern.dev.stream(), // ignored by InferoHandRolled, still required by AttnCallCtx
    };
    backend.prefill(&mut ctx).expect("InferoHandRolled::prefill");
    assert_matches_reference(&out, &reference, TOLERANCE); // reuse this test's existing assertion helper/inline check
}
```

Match variable names (`kern`, `q`, `k_cache`, `v_cache`, `batch`, `dims`, `run_tokens`, `kv_len`, `scale`, `out`, `reference`, the tolerance constant, and whatever the existing assertion looks like — it may be an inline `assert!(diff < TOLERANCE)` rather than a named helper) to whatever the surrounding test in the file actually uses; this snippet is illustrative of intent, not a literal diff.

- [ ] **Step 2: Run it, confirm it fails to compile** (nothing named `InferoHandRolled` exists yet)

Run: `cargo test -p infero-kernels --test ops attn_prefill_matches_the_three_kernels`
Expected: FAIL to compile — `unresolved import` or `cannot find type InferoHandRolled`.

- [ ] **Step 3: Implement `InferoHandRolled`**

```rust
pub struct InferoHandRolled<'k> {
    kern: &'k crate::Kernels,
}

impl<'k> InferoHandRolled<'k> {
    pub fn new(kern: &'k crate::Kernels) -> Self {
        Self { kern }
    }
}

impl AttentionBackend for InferoHandRolled<'_> {
    fn name(&self) -> &'static str {
        "handrolled"
    }

    fn priority(&self) -> u32 {
        0
    }

    fn supports(&self, _caps: &HardwareCaps, dims: &AttnDims, _kv_quant: crate::KvQuant) -> bool {
        self.kern.prefill_attention(dims)
    }

    fn prefill(&self, ctx: &mut AttnCallCtx<'_>) -> Result<()> {
        // Identical to `Model::attention()`'s own `prefill_run` branch at
        // crates/model/src/lib.rs:3275-3302 — the `INFERO_PREFILL_T6=0`
        // escape hatch is preserved here too, so this backend's behavior
        // exactly matches today's default dispatch, not a simplified copy.
        if ctx.dims.d_head == 256 && !std::env::var("INFERO_PREFILL_T6").is_ok_and(|v| v == "0") {
            self.kern.attn_prefill_decoupled6_f16acc(
                ctx.out, ctx.q, ctx.k_cache, ctx.v_cache, ctx.batch, ctx.dims,
                ctx.run_base, ctx.run_tokens, ctx.kv_len, ctx.scale,
            )
        } else {
            // `attn_prefill_ws4` needs a `partial` scratch buffer this
            // trait's `AttnCallCtx` doesn't carry (Task 1 intentionally
            // scoped it to what the T=6/vendor paths need) — allocate a
            // same-shaped scratch here. Check `Model`'s own `partial`
            // buffer sizing at its allocation site (search
            // `crates/model/src/lib.rs` for `partial:` / `let partial`)
            // and match its element count exactly rather than guessing.
            let mut partial = self.kern.dev_scratch_like_model_partial(ctx.dims, ctx.run_tokens)?;
            self.kern.attn_prefill_ws4(
                ctx.out, ctx.q, ctx.k_cache, ctx.v_cache, ctx.batch, ctx.dims,
                ctx.run_base, ctx.run_tokens, ctx.kv_len, ctx.scale, &mut partial,
            )
        }
    }
}
```

The `dev_scratch_like_model_partial` call above is a placeholder name — find `Model`'s actual `partial` buffer (its allocation, element count formula, and dtype) by reading around `crates/model/src/lib.rs`'s `Activations`/`Scratch` construction (grep `partial` in that file), and either (a) add a small pub helper on `Kernels` that allocates a same-shaped scratch buffer given `dims`/`run_tokens`, matching that exact formula, or (b) if `partial`'s sizing depends on state only `Model` has (e.g. `max_seq`, batch concurrency) that isn't reachable from `Kernels` alone, change `InferoHandRolled::new` to take an extra `partial: &'k mut ViewMut<'k, f32>` (borrowed from `Model`'s existing buffer, not freshly allocated) and thread it through — prefer (b) if `Model` already owns a persistent `partial` buffer sized for the whole run, since that's what `Model::attention()` itself does today (reuses `partial` across calls rather than allocating fresh each time) and this backend should not allocate where the code it's replacing doesn't.

- [ ] **Step 4: Run the test, confirm it passes**

Run: `cargo test -p infero-kernels --test ops attn_prefill_matches_the_three_kernels`
Expected: PASS.

- [ ] **Step 5: memcheck** (no new kernel code was written in this task — `InferoHandRolled` only calls existing, already-verified kernels through a new Rust indirection — but run it anyway to catch a wiring mistake, e.g. a wrong buffer size on the `partial` scratch)

Run: `compute-sanitizer --tool memcheck cargo test -p infero-kernels --test ops attn_prefill_matches_the_three_kernels --release -- --test-threads=1` (adjust to however this repo's other memcheck invocations are actually run — grep for an existing `compute-sanitizer` invocation in `crates/kernels/` scripts/CI/notes and match its exact flags/target instead of guessing).
Expected: 0 errors.

- [ ] **Step 6: Commit**

```bash
git add crates/kernels/src/attn_backend.rs crates/kernels/tests/ops.rs
git commit -m "Add InferoHandRolled attention backend wrapping existing dispatch"
```

---

### Task 3: `FlashAttn2Ffi` backend — build plumbing and extern "C" shim

**Files:**
- Create: `crates/kernels/src/cu_vendor/flash_attn2_shim.cu` (the hand-written `extern "C"` wrapper — thin, calls into vendored FA2 template headers)
- Create: `crates/kernels/src/flash_attn2.rs` (Rust side: `extern "C"` declarations + safe wrapper + `AttentionBackend` impl)
- Modify: `crates/kernels/Cargo.toml` — add `flash_attn2 = ["cuda"]` feature next to the existing `cutlass = ["cuda"]` (crates/kernels/Cargo.toml:38)
- Modify: `crates/kernels/build.rs` — add a second AOT-compile block gated by `CARGO_FEATURE_FLASH_ATTN2`, mirroring the existing `cutlass` block exactly (same `resolve_nvcc`, a new `resolve_flash_attn_dir()` reading `INFERO_FLASH_ATTN_DIR`, same static-archive-and-link pattern)
- Modify: `crates/kernels/src/lib.rs` — `#[cfg(feature = "flash_attn2")] pub mod flash_attn2;`

**Interfaces:**
- Consumes: `AttentionBackend`, `AttnCallCtx`, `HardwareCaps` from Task 1.
- Produces: `struct FlashAttn2Ffi` implementing `AttentionBackend`, used by Task 4's wiring. `fn resolve_flash_attn_dir() -> PathBuf` in `build.rs`, mirroring `resolve_cutlass_dir`.

This is the substantial new engineering in this plan. Scope it down deliberately: **prefill only, one fixed real shape** — causal, `d_head <= 256`, single contiguous sequence (matching exactly what `prefill_profile` exercises and what `Model::attention()`'s `prefill_run` branch guarantees before it ever calls into T=6/`ws4` today — see `crates/model/src/lib.rs:3241-3245`'s own comment: "one sequence, `n` tokens, contiguous, causal"). Do not attempt to support the general varlen-multi-sequence-batch case FA2's own real `flash_attn_varlen_func` handles — that generality isn't needed to prove this architecture, and attempting it is where most of the real risk in a from-scratch FFI shim would live.

- [ ] **Step 1: Read the real reference before writing anything**

On `bw`: read `/tmp/flash_attn_src/csrc/flash_attn/src/kernel_traits.h`, `flash_fwd_kernel.h`, and `flash_fwd_launch_template.h` in full (already identified this session as the torch-free CUDA files — `flash_fwd_launch_template.h` is specifically the file that picks tile-size traits per head-dim and instantiates+launches the templated kernel, taking a plain `Flash_fwd_params` struct, not a `torch::Tensor` — confirm this by reading `flash_fwd_launch_template.h`'s function signatures directly rather than assuming). Identify: (a) the exact `Flash_fwd_params` fields needed for a causal, single-sequence, `d_head<=256`, fp16 forward pass (query/key/value pointers+strides, `seqlen_q`/`seqlen_k`, `d`, `is_causal`, `scale_softmax`, output pointer+strides — read the real struct definition, don't guess field names), (b) which `Kernel_traits` specialization the launch template picks for `d_head=256` causal fp16, and (c) the exact template function to call to run just that one instantiation directly from a plain C++ (non-torch) call site — this is very likely already demonstrated somewhere in the same source tree in a non-Python entry point (check for a `run_mha_fwd`-style top-level dispatcher function in `flash_fwd_launch_template.h` itself, callable with a filled `Flash_fwd_params` and a raw `cudaStream_t`, no torch involved) rather than needing to write a from-scratch template instantiation.

- [ ] **Step 2: Write the extern "C" shim**

`crates/kernels/src/cu_vendor/flash_attn2_shim.cu` — `#include`s the vendored `flash_fwd_launch_template.h` (path supplied via an `-I` build flag pointing at `INFERO_FLASH_ATTN_DIR/csrc/flash_attn/src`, set in `build.rs`), defines one function:

```cpp
extern "C" int infero_flash_attn2_fwd_causal_f16(
    const void* q, const void* k, const void* v, void* out,
    int seqlen_q, int seqlen_k, int n_heads, int n_kv_heads, int d_head,
    float softmax_scale, cudaStream_t stream
);
```

whose body fills a real `Flash_fwd_params` (zero-initialized, then only the fields Step 1 identified as needed for this exact case set explicitly — leaving irrelevant fields at their zero-init default, matching how FA2's own non-causal/non-varlen callers already do this in its reference `api.cpp`, which is worth reading for the *shape* of a correct params-fill even though its torch-tensor-unwrapping parts don't apply here) and calls the dispatcher identified in Step 1, returning `0` on success and a nonzero code if the launch's own return path signals an error (match whatever error-reporting convention the launch template already uses — a `cudaGetLastError()` check after the launch if nothing else does).

- [ ] **Step 3: `build.rs` — new AOT block**

Add, mirroring the existing `cutlass` block at `crates/kernels/build.rs:13-105` exactly in structure:

```rust
if std::env::var_os("CARGO_FEATURE_FLASH_ATTN2").is_some() {
    println!("cargo:rerun-if-changed=src/cu_vendor/flash_attn2_shim.cu");
    println!("cargo:rerun-if-env-changed=INFERO_FLASH_ATTN_DIR");
    let nvcc = resolve_nvcc();
    let fa_dir = resolve_flash_attn_dir();
    let fa_src = fa_dir.join("csrc/flash_attn/src");
    if !fa_src.is_dir() {
        panic!(
            "flash-attention checkout at {} is missing {} -- set INFERO_FLASH_ATTN_DIR to a \
             checkout of Dao-AILab/flash-attention",
            fa_dir.display(), fa_src.display()
        );
    }
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let obj = out_dir.join("flash_attn2_shim.o");
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let src = manifest.join("src/cu_vendor/flash_attn2_shim.cu");
    let status = Command::new(&nvcc)
        .args(["-std=c++17", "-O3", "-c", "--expt-relaxed-constexpr", "-DNDEBUG",
               "-Xcompiler", "-fPIC", "-gencode", "arch=compute_80,code=sm_80"])
        .arg("-I").arg(&fa_src)
        .arg(&src).arg("-o").arg(&obj)
        .status().unwrap_or_else(|e| panic!("failed to run {}: {e}", nvcc.display()));
    if !status.success() { panic!("nvcc failed compiling {}", src.display()); }
    let ar = /* same ar-resolution as the cutlass block */;
    let archive = out_dir.join("libinfero_flash_attn2.a");
    /* same ar invocation, link-search, rustc-link-lib pattern as the cutlass block */
    println!("cargo:rustc-link-lib=static=infero_flash_attn2");
}

fn resolve_flash_attn_dir() -> PathBuf {
    if let Ok(p) = std::env::var("INFERO_FLASH_ATTN_DIR") {
        return PathBuf::from(p);
    }
    panic!(
        "the `flash_attn2` feature needs a Dao-AILab/flash-attention checkout -- set \
         INFERO_FLASH_ATTN_DIR (e.g. /tmp/flash_attn_src on `bw`)"
    );
}
```

Use `arch=compute_80,code=sm_80` deliberately, not `sm_120a` — this matches this session's own finding that vLLM's own bundled FA2 is *also* only sm_80-compiled and runs on sm_120 via PTX forward compatibility; compiling for `sm_80` here is not a shortcut, it's the real, correct target for this specific vendor kernel family on this hardware generation (FA2 has no sm_120-specific tuning to target). Extract the shared `ar`-resolution/linking lines into a small helper fn both blocks call, rather than duplicating them — a real, small refactor of the existing `cutlass` block, in scope because this task is adding a second near-identical block right next to it.

- [ ] **Step 4: `cargo build -p infero-kernels --features flash_attn2` on `bw`** (needs `nvcc` + the flash-attn checkout; not buildable on a machine without both)

Run (on `bw`): `cd /home/jeff/infero && source ~/.cargo/env && INFERO_FLASH_ATTN_DIR=/tmp/flash_attn_src cargo build -p infero-kernels --features flash_attn2 2>&1 | tail -50`
Expected: builds clean, or fails with a concrete `nvcc` compile error against the shim — iterate on Step 2 until it compiles. This is the step most likely to need several rounds; budget for it.

- [ ] **Step 5: Rust-side wrapper + a standalone correctness test against the same oracle**

```rust
// crates/kernels/src/flash_attn2.rs
use crate::attn_backend::{AttentionBackend, AttnCallCtx, HardwareCaps};
use crate::{AttnDims, KvQuant};
use anyhow::{ensure, Result};

// Same FFI boundary shape `cutlass_fp8.rs` already uses for its own AOT
// kernel (`crates/kernels/src/cutlass_fp8.rs:188-216`): a `CUstream` from
// `cudarc::driver::sys`, obtained at the call site via `stream.cu_stream()`
// on `infero_gpu::Stream` — not a raw `*mut c_void`.
unsafe extern "C" {
    fn infero_flash_attn2_fwd_causal_f16(
        q: *const std::ffi::c_void, k: *const std::ffi::c_void, v: *const std::ffi::c_void,
        out: *mut std::ffi::c_void,
        seqlen_q: i32, seqlen_k: i32, n_heads: i32, n_kv_heads: i32, d_head: i32,
        softmax_scale: f32, stream: cudarc::driver::sys::CUstream,
    ) -> i32;
}

pub struct FlashAttn2Ffi;

impl AttentionBackend for FlashAttn2Ffi {
    fn name(&self) -> &'static str { "flash_attn2" }
    fn priority(&self) -> u32 { 100 }

    fn supports(&self, caps: &HardwareCaps, dims: &AttnDims, kv_quant: KvQuant) -> bool {
        caps.at_least(8, 0)
            && !kv_quant.is_quantized()
            && dims.d_head <= 256
            && dims.d_head % 8 == 0
    }

    fn prefill(&self, ctx: &mut AttnCallCtx<'_>) -> Result<()> {
        ensure!(ctx.run_base == 0, "flash_attn2 backend: only run_base=0 supported this pass");
        // Fill real pointers/strides from ctx.{q,k_cache,v_cache,out} -- read
        // each `View`/`ViewMut`'s own accessor methods (grep `impl<T> View`
        // in crates/kernels/src/lib.rs for `.as_ptr()`/`.stride()`-style
        // methods already used elsewhere in this file, e.g. inside
        // `attn_prefill_ws4`'s own body) rather than inventing new ones.
        let rc = unsafe {
            infero_flash_attn2_fwd_causal_f16(
                ctx.q.as_ptr() as *const _,
                ctx.k_cache.as_ptr() as *const _,
                ctx.v_cache.as_ptr() as *const _,
                ctx.out.as_mut_ptr() as *mut _,
                ctx.run_tokens as i32, ctx.kv_len as i32,
                ctx.dims.n_heads as i32, ctx.dims.n_kv_heads as i32, ctx.dims.d_head as i32,
                ctx.scale,
                ctx.stream.cu_stream(), // `ctx.stream: &Arc<infero_gpu::Stream>` -- add this field to `AttnCallCtx` in Task 1 if it isn't already there; `InferoHandRolled`'s kernels currently get their stream implicitly via `self.kern.dev.stream()`, but an FFI backend called through the trait needs it passed explicitly since it has no `Kernels` handle of its own.
            )
        };
        ensure!(rc == 0, "flash_attn2 fwd failed, code {rc}");
        Ok(())
    }
}
```

`FlashAttn2Ffi` has no `&Kernels` to pull a stream from the way `InferoHandRolled` does — go back to Task 1 and add `pub stream: &'a std::sync::Arc<infero_gpu::Stream>` to `AttnCallCtx`, and update Task 4's `AttnCallCtx { ... }` construction to pass `self.dev.stream()` (or however `Model` already names its stream accessor — grep it rather than guessing) alongside the other fields, and update `InferoHandRolled::prefill` to ignore it (it already has its own route to the stream via `self.kern`).

Add a new test in `crates/kernels/tests/ops.rs`, gated `#[cfg(feature = "flash_attn2")]`, that runs the exact same small-scale scenario `attn_prefill_matches_the_three_kernels` already builds, through `FlashAttn2Ffi::prefill()` this time, against the same reference and tolerance.

- [ ] **Step 6: Run it on `bw`**

Run: `cd /home/jeff/infero && INFERO_FLASH_ATTN_DIR=/tmp/flash_attn_src cargo test -p infero-kernels --features flash_attn2 --test ops flash_attn2 -- --nocapture`
Expected: PASS. If it fails numerically (not a compile error), suspect the `Flash_fwd_params` field-fill from Step 1/2 first — re-check against FA2's own `api.cpp` params-fill for the same causal/fp16 case rather than re-deriving the math; this is exactly the kind of bug this codebase's own history says to check the "obvious" reference-following details on before assuming something subtler is wrong.

- [ ] **Step 7: memcheck + racecheck**

Run: `compute-sanitizer --tool memcheck <the same test binary/invocation>` then `--tool racecheck` (match this repo's real existing invocation pattern — grep for one, as in Task 2 Step 5).
Expected: 0 errors, 0 hazards, both.

- [ ] **Step 8: Commit**

```bash
git add crates/kernels/src/cu_vendor/flash_attn2_shim.cu crates/kernels/src/flash_attn2.rs \
        crates/kernels/Cargo.toml crates/kernels/build.rs crates/kernels/src/lib.rs \
        crates/kernels/tests/ops.rs
git commit -m "Add FlashAttn2Ffi backend: torch-free FA2 FFI shim behind flash_attn2 feature"
```

---

### Task 4: Wire selection into `Model` and add the one new dispatch branch

**Files:**
- Modify: `crates/model/src/lib.rs` — `Model` struct (add one field), its load path (resolve+cache the backend once), and `attention()`'s existing cascade at `crates/model/src/lib.rs:3256` (add one new branch ahead of the existing `if let Some(run_tokens) = prefill_run.filter(...)`)

**Interfaces:**
- Consumes: `AttentionBackend`, `AttnCallCtx`, `HardwareCaps`, `select_backend`, `InferoHandRolled` from `infero_kernels::attn_backend`; `FlashAttn2Ffi` from `infero_kernels::flash_attn2` (feature-gated).
- Produces: `Model::attn_backend_name(&self) -> &'static str` (a small accessor Task 5's integration test/benchmark uses to confirm which backend actually got picked, without needing to parse log output).

- [ ] **Step 1: Add the field and resolve it once at load time**

Find where `Model`'s other load-time-cached decisions live (`needs_score_buffer`, `batch_tokens` — both computed once around `crates/model/src/lib.rs:1003-1034` per this session's earlier reading) and add, in the same place:

```rust
let hw_caps = infero_kernels::attn_backend::HardwareCaps::probe(&dev)?;
let mut backends: Vec<Box<dyn infero_kernels::attn_backend::AttentionBackend>> =
    vec![Box::new(infero_kernels::attn_backend::InferoHandRolled::new(&kern))];
#[cfg(feature = "flash_attn2")]
backends.push(Box::new(infero_kernels::flash_attn2::FlashAttn2Ffi));
let forced = std::env::var("INFERO_ATTN_BACKEND").ok();
let attn_backend_idx = {
    let chosen = infero_kernels::attn_backend::select_backend(
        &backends, &hw_caps, &AttnDims { n_heads: cfg.n_heads, n_kv_heads: cfg.n_kv_heads,
            d_head: cfg.d_head, n_slots: 0, n_tokens: 0 },
        KvQuant::F16, // this pass's dispatch only ever intercepts the dense (`None` quant) arm -- see Step 2
        forced.as_deref(),
    )?;
    tracing::info!(backend = chosen.name(), "attention backend selected");
    backends.iter().position(|b| std::ptr::eq(b.as_ref(), chosen)).unwrap()
};
```

`InferoHandRolled::new(&kern)` borrows `kern` — check whether `Model`'s own field is named `kern`/`self.kern` and whether it's constructed before or after this point in the load function; if `Kernels` isn't yet constructed this early, move this block to right after it is (it must run after `kern`/`dev` exist, same as `needs_score_buffer` already does). Store `backends` and `attn_backend_idx` on `Model` (two new fields) rather than re-resolving per call — mirrors how `batch_tokens`/`needs_scores` are already fields, not recomputed.

- [ ] **Step 2: Add the one new branch in `attention()`**

At `crates/model/src/lib.rs:3256`, immediately before the existing `if let Some(run_tokens) = prefill_run.filter(|_| self.kern.prefill_attention(&dims)) {`, add:

```rust
if let (Some(run_tokens), true) = (prefill_run, self.backends[self.attn_backend_idx].name() != "handrolled") {
    let mut ctx = infero_kernels::attn_backend::AttnCallCtx {
        out: &mut attn_out.slice_mut(..n * da),
        q: &self.act.q.slice(..n * da),
        k_cache: &pool.dense(layer).0.as_view(),
        v_cache: &pool.dense(layer).1.as_view(),
        batch,
        dims,
        run_base: 0,
        run_tokens,
        kv_len,
        scale: attn_scale,
        stream: self.dev.stream(), // check `Model`'s real accessor name for its stream Arc and use that instead of guessing "self.dev.stream()" verbatim
    };
    self.backends[self.attn_backend_idx].prefill(&mut ctx)?;
} else if let Some(run_tokens) = prefill_run.filter(|_| self.kern.prefill_attention(&dims)) {
    /* existing T=6/ws4 branch, entirely unchanged */
```

This only fires when a non-`handrolled` backend was selected at load time (which itself only happens if `flash_attn2` was compiled in AND `supports()` matched this model's real dims/kv_quant AND, per Task 1's `select_backend`, it beat `InferoHandRolled`'s priority — which never happens by default since `InferoHandRolled`'s priority is 0 and always supports every shape `prefill_attention` already covers, so this branch is provably dead code on every build without `flash_attn2`, and on a `flash_attn2` build it's still dead unless `INFERO_ATTN_BACKEND=flash_attn2` forces it — matching this plan's non-goal of not trying to beat the hand-rolled kernel by default). Confirm this reasoning holds against the *real* `select_backend` priority comparison (`min_by_key`) rather than assuming — `InferoHandRolled` always wins ties or lower values, so `flash_attn2`'s priority 100 never wins unless forced.

- [ ] **Step 3: Add the accessor**

```rust
impl Model {
    pub fn attn_backend_name(&self) -> &'static str {
        self.backends[self.attn_backend_idx].name()
    }
}
```

- [ ] **Step 4: `cargo build -p infero-model` (default features) still builds and the existing test suite is unaffected**

Run: `cargo build -p infero-model && cargo test -p infero-model`
Expected: builds and passes exactly as before this task — this is the regression check that the new branch is truly inert by default.

- [ ] **Step 5: `cargo build -p infero-model --features flash_attn2` on `bw` also builds**

Run (on `bw`): `INFERO_FLASH_ATTN_DIR=/tmp/flash_attn_src cargo build -p infero-model --features flash_attn2 2>&1 | tail -30`
Expected: builds clean.

- [ ] **Step 6: Commit**

```bash
git add crates/model/src/lib.rs
git commit -m "Wire AttentionBackend selection into Model's load path and prefill dispatch"
```

---

### Task 5: Real integration check on `bw` + honest report

**Files:** none (no new code — this task runs what Tasks 1-4 built and records the real result)

- [ ] **Step 1: Build the fast path with `flash_attn2` forced on, on `bw`**

```bash
cd /home/jeff/infero && source ~/.cargo/env
CUDA_VISIBLE_DEVICES=3 INFERO_CUTLASS_DIR=/tmp/cutlass_src INFERO_FLASH_ATTN_DIR=/tmp/flash_attn_src \
  cargo build --release -p infero-model --example prefill_profile --features "cutlass flash_attn2"
```

- [ ] **Step 2: Run `prefill_profile` twice, once per backend, same shape as this session's confirmed 5.91s baseline**

```bash
cd /home/jeff/infero
for backend in handrolled flash_attn2; do
  echo "=== $backend ==="
  CUDA_VISIBLE_DEVICES=3 INFERO_FP8_UNIFIED=1 INFERO_ATTN_MMA=1 INFERO_ATTN_BACKEND=$backend \
    ./target/release/examples/prefill_profile /home/jeff/models/qwen38-27b-fp8 30552
done
```

Expected: the `handrolled` run reproduces ~5.9s (this session's confirmed baseline — if it doesn't, something in Tasks 1-4 changed default behavior, which is a real regression to fix before proceeding, not to explain away). The `flash_attn2` run should complete without error (it is not expected to be faster — see Global Constraints — but it must run and produce a plausible, non-garbage token/s number, confirming the FFI shim is really being called inside a real forward pass, not just in isolation).

- [ ] **Step 3: Confirm which backend actually ran, per invocation**

Add a one-line `eprintln!("attn backend: {}", model.attn_backend_name());` temporarily to `prefill_profile.rs` (same pattern this session used earlier for the `fp8_unified`/`needs_scores` debug print — backed up the file first, reverted after), rebuild, rerun both, confirm the printed name matches the `INFERO_ATTN_BACKEND` requested for each run, then revert `prefill_profile.rs` back to its Task-4-committed state (verify via `md5sum`/`git diff` before moving on, exactly as this session did when it reverted its own earlier debug patch to `lib.rs`).

- [ ] **Step 4: Write the honest result**

Report, in the final message to the user (not a new file): whether `flash_attn2` ran correctly end-to-end, its real measured time next to the 5.9s `handrolled` baseline (expected to be slower — say by how much, and don't editorialize it as a problem), and confirm explicitly that the default (`INFERO_ATTN_BACKEND` unset) run still picks `handrolled` and still reproduces ~5.9s. If anything in Steps 1-3 failed and couldn't be resolved, report the real failure precisely (compile error text, numerical mismatch, whatever) rather than a vague "ran into issues."

- [ ] **Step 5: Do not commit the temporary debug print** (Step 3 already reverted it) — confirm `git status` shows no diff versus Task 4's last commit before finishing.
