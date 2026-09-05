# Pluggable attention backend

Goal: stop hand-tuning a new CUDA kernel for every GPU generation infero ever
runs on. Scope is **standard (softmax, non-linear) attention only** — GDN
(gated delta net) has no vendor-library equivalent anywhere in the industry
(vLLM and SGLang both hand-roll their own GDN kernels too, confirmed this
session by reading SGLang's `gdn_blackwell` source), so it stays exactly as it
is today, outside this design entirely.

Trigger for this design: v6000 (sm_120a, RTX PRO 6000 Blackwell) needed its
own hand-tuned `ws4`/`decoupled6` kernel family, tile sizes and all, because
neither FlashAttention-3 nor FA4 support this compute-capability family at
all (both are hardcoded to specific SM families in `vllm_flash_attn`'s own
source — FA3 to 9.x only, FA4 to 9.x/10.x/11.x, both excluding 12.x). FA2 is
the only vendor kernel that runs here at all, and only via generic sm_80
PTX-JIT forward compatibility (verified via `cuobjdump` on vLLM's own bundled
`.so` this session) — not because anyone tuned it for this card. The next
unfamiliar GPU infero targets will have its own version of this problem, and
the fix each time so far has been "hand-tune another kernel family." This
design breaks that cycle for the *standard-attention* half of it.

## What's already established, taken as given

- infero's own kernels (`attn_prefill_ws4`, `attn_prefill_decoupled6_f16acc`,
  `attn_decode`, the score-materializing fallback) stay in the tree,
  unmodified, and remain the default on hardware they've been tuned for.
- No libtorch/Python runtime dependency. FA2's most-maintained distribution
  form is a PyTorch extension, but infero stays pure Rust+CUDA — a new
  backend vendors just the CUDA kernel sources (a trimmed, torch-free
  extraction of `Dao-AILab/flash-attention`'s `csrc/flash_attn/src/`) behind
  a hand-written `extern "C"` shim, the same relationship infero already has
  with CUTLASS today (`INFERO_CUTLASS_DIR`, sparse checkout, no Python).
- Backend *selection* is runtime auto-detection, not a build-time either/or —
  a single infero binary should pick the right backend for whatever GPU it
  finds itself on at startup, the way vLLM's own `_get_backend_priorities` +
  `validate_configuration` + fallback chain does (read in full this session,
  `vllm/platforms/cuda.py`).
- Backend eligibility is gated by KV layout/quant mode, not just hardware.
  infero's FP8/packed KV layouts are custom and no vendor kernel understands
  them; a vendor backend is only ever a candidate for plain F16/BF16 KV.
- No second real GPU to validate against right now — only v6000 is in hand.
  Success for this pass is: the trait boundary is clean, runtime detection
  and selection actually work, the FA2 shim is numerically correct on v6000,
  and infero's own kernels remain reachable as the default/fallback. This
  pass is explicitly **not** trying to make FA2 beat `decoupled6` on v6000 —
  it's known to lose (0.85x-ish territory, per this session's own bulk-sync
  finding, a different kernel but the same "generic vs. tuned" story) and
  closing that gap isn't the point of this design.

## One footgun this design must not repeat

This session lost a benchmark run to three independent, silently-stacking
gates (`--features cutlass`, `INFERO_CUTLASS_DIR`, `INFERO_FP8_UNIFIED=1`,
`INFERO_ATTN_MMA=1`) where missing any one produced no error — just a quiet
fallback to a 6-9x slower path. Runtime backend selection in this design
must **never** have a silent "did you also remember to set the env var"
half-state: once a backend is compiled in, its eligibility is determined
entirely by `supports()` (hardware caps + shape + kv layout), with no
additional required toggle to make it "actually" active. An env var still
exists, but only as an override/escape hatch (force a specific backend, or
force-disable one for debugging) — never as a second mandatory gate.

## The trait boundary

```rust
/// One implementation of standard (non-GDN) attention.
trait AttentionBackend: Send + Sync {
    /// Static name, for logging ("Using %s attention backend" style).
    fn name(&self) -> &'static str;

    /// Relative preference when multiple backends are eligible for the same
    /// call — lower wins. infero's own tuned kernels sit at priority 0 on
    /// hardware they cover; a vendor backend is the fallback for hardware
    /// nothing here has been tuned for yet, not a default competitor on
    /// hardware that already has a tuned kernel.
    fn priority(&self) -> u32;

    /// Whether this backend can serve this exact call shape at all.
    /// Checked once at `Model` load time, not per forward call — a backend
    /// that claims `true` here and then fails at call time is a bug in
    /// `supports`, not a normal runtime fallback (see "Error handling").
    fn supports(&self, caps: &HardwareCaps, dims: &AttnDims, kv_quant: KvQuant) -> bool;

    fn prefill(&self, ctx: &AttnCallCtx) -> Result<()>;
    fn decode(&self, ctx: &AttnCallCtx) -> Result<()>;
}
```

`AttnCallCtx` is the one canonical shape every backend receives: device
pointers/strides for Q/K/V/O in infero's existing KV-pool layout, `seqlens`,
block table, causal flag, softmax scale, `AttnDims`. It is *infero's* layout,
not a lowest-common-denominator one — a backend that needs a different
physical layout (FA2 wants contiguous fp16/bf16 per block) repacks
internally, on its own time budget, rather than every caller and every other
backend paying for a layout neither of them needs. This is why KV layout
gates `supports()`: repacking infero's FP8/packed layout into what FA2 wants
would cost more than any of the last several GDN "more parallelism, more
copying" experiments this session already found net-negative — the FA2
backend simply declines FP8/packed KV rather than eating that cost silently.

`HardwareCaps` is a thin, one-time probe (compute capability major/minor,
shared-memory-per-block ceiling, SM count) wrapping the same device query
infero already does at load time elsewhere — not a new subsystem.

## Backend implementations, this pass

1. **`InferoHandRolled`** — wraps today's dispatch (`ws4`, `decoupled6`,
   `attn_decode`, the score-materializing fallback) behind the trait,
   behavior-preserving. `supports()` returns `true` for every `kv_quant` and
   every shape it already handles today (including the GQA/head-dim gate
   `prefill_attention` already encodes) — this is what guarantees a call
   always finds at least one eligible backend. Priority 0.
2. **`FlashAttn2Ffi`** — new. `supports()` requires: F16/BF16 KV (`kv_quant`
   not FP8/packed), causal, `d_head <= 256`, compute capability >= 8.0 (FA2's
   own real floor, confirmed in `_is_fa2_supported`). Priority 100 (loses to
   any hand-rolled kernel that also claims the shape). Its `build.rs` needs
   `INFERO_FLASH_ATTN_DIR` pointing at a trimmed source checkout, mirroring
   `INFERO_CUTLASS_DIR` exactly.

Whether `FlashAttn2Ffi` is *compiled into the binary at all* is a separate
question from runtime selection, and deliberately so: requiring every dev
build on every machine to have a flash-attention source checkout present
would be a real regression for anyone not touching this path (the same
reason `cutlass` is a cargo feature and not unconditional today). A cargo
feature (e.g. `flash_attn2`) controls *availability* at build time; once
available, there is no second runtime toggle required to make it
*considered* — that's the footgun this design is explicitly avoiding. A
build without the feature has exactly one backend, today's, unchanged.

## Selection

At `Model` load time, alongside where `needs_score_buffer`/`batch_tokens`
are already resolved once and cached: build the list of compiled-in
backends, probe `HardwareCaps` once, filter to `supports()==true` for this
model's actual `dims`/`kv_quant`, sort by priority, take the first, and cache
it on `Model` (same shape as today's `batch_tokens` field). Log which
backend won and why, matching vLLM's own
`logger.info_once("Using %s attention backend out of potential backends: %s")`
pattern read this session. `INFERO_ATTN_BACKEND=handrolled|flash_attn2`
forces a specific choice for debugging/bring-up, erroring loudly if the
forced backend doesn't actually support the shape (not silently falling back
to a different one — a forced choice that can't run is a config error to
surface, not paper over).

## Error handling

Because `supports()` is the only gate and it's checked once at load time,
there is no per-call fallback path to design — a backend either was selected
at load time (and is expected to work for every call with these fixed
`dims`/`kv_quant` for the rest of the process) or wasn't considered at all.
If `FlashAttn2Ffi::prefill`/`decode` returns a CUDA/FFI error at call time,
it propagates like any other kernel error in this codebase already does —
that's a real bug in the FFI shim or in `supports()`'s shape check, not a
condition to swallow and reroute around.

## Testing

- Correctness: extend the existing shared oracle test (`attn_prefill_matches_the_three_kernels`-shaped —
  same reference every prefill kernel in this file is already checked
  against) to run through the trait dispatch for every compiled-in backend,
  not just call kernels directly. Small scale, exact-match tolerance.
- memcheck/racecheck on the new FFI shim same as any new kernel in this
  codebase — non-negotiable, this file has twice lost time to skipping it.
- Real integration: `prefill_profile` with `INFERO_ATTN_BACKEND=flash_attn2`
  forced, on `bw`, confirms the shim actually runs end-to-end inside a real
  forward pass, not just a standalone kernel test. Explicitly not a
  performance gate — correctness and "did it actually get called" are what
  this pass checks.

## What this design does not do

- Does not touch GDN, decode-side batching, KV-pool allocation, or the
  score-materializing fallback's own internals.
- Does not attempt to make any vendor backend faster than infero's tuned
  kernels on hardware that already has one.
- Does not add FA3/FA4/FlashInfer backends now — FA2 is the only one that
  actually runs on hardware infero doesn't already have a tuned kernel for
  today (v6000). A `TritonAttn`-style or `FlashInfer`-style backend is a
  natural later addition once a concrete second GPU target exists to justify
  the extra vendored dependency and validate it against.
