# TQ4 long-prefill dispatch: the flash_attn2 / decoupled6 follow-up

Follow-up to `2026-09-06-tq4-prefill-dequant-dispatch.md` (spec:
`../specs/2026-09-06-tq4-prefill-dequant-dispatch-design.md`), which shipped
the dequant-dispatch mechanism but deliberately scoped its long-run call site
to `attn_prefill_ws4` only, with an explicit "separable follow-up work"
comment. This closes that.

## What shipped

The TurboQuant long-prefill call site (`crates/model/src/lib.rs`, inside
`Model::attention()`'s `Some(tq)` arm, the `plan.long_runs` loop) now makes
the same three-way choice the F16 dense path's `run_attn` closure makes:

1. **flash_attn2**, when the `flash_attn2` feature is compiled in *and*
   `FlashAttn2Ffi::supports(&self.hw_caps, &run_dims, KvCacheQuant::F16)`
   *and* `FlashAttn2Ffi::kv_run_is_contiguous(...)` on the synthetic layout.
2. **`attn_prefill_decoupled6_f16acc`**, when `run_dims.d_head == 256` and
   `INFERO_PREFILL_T6 != "0"`.
3. **`attn_prefill_ws4`** otherwise — unchanged, including its
   `ensure_partial_fits` guard (the other two don't use `attn_partial`).

The choice itself is a small pure function, `tq_long_run_kernel(fa2_ok,
d_head, t6_enabled) -> TqLongRunKernel`, placed next to `plan_tq_dispatch` so
it is unit-testable with no GPU. `t6_enabled` is passed in rather than read
from the environment inside it, so the test doesn't have to mutate process
env.

### The architectural point: why this does *not* gate on `attn_backend_name`

`attn_backend_name` is resolved once at load time by `select_backend()`
against the **pool's** real `KvQuant`. `FlashAttn2Ffi::supports()` contains
`!kv_quant.is_quantized()`, so on any TurboQuant model that selection always
resolves to `"handrolled"` for the whole model — and the F16 dense path's
`backend_name != "handrolled"` gate would therefore block FA2 here forever.

That gate answers the wrong question at this call site. By the time this loop
calls a dense kernel, `tq_dequant_kv` has already unpacked the span into
`tq.dequant_k` / `tq.dequant_v`, which are plain dense `f16` buffers —
physically and semantically indistinguishable from the dense pool's own
cache. So the call site asks `supports()` itself, per run, passing
`KvCacheQuant::F16` explicitly. That is not a fork of `supports()`'s logic:
it is the same function, called with the parameter that is actually true of
this call's data.

I checked the premise the task flagged as the thing to verify — whether
`supports()` or `prefill()` has any *other* dependency on the real
`kv_quant`. It does not. `supports()` reads `kv_quant` in exactly one place
(the `is_quantized()` term); `prefill()` never sees a `KvQuant` at all — it
takes `AttnCallCtx`, whose `k_cache`/`v_cache` are `View<'_, f16>`, i.e. the
dense buffers we hand it. The premise holds.

I also verified the layout arithmetic lines up, since this is the one place
FA2's shim could silently mis-address a *synthetic* pool:

- The shim indexes KV as flat `[n_kv_heads, dims.n_slots, d_head]` and offsets
  by `base_slot * d_head`. Here `run_dims.n_slots = seq_kv` (the compact
  scratch buffer's own row stride, already set by the existing code) and the
  identity slot table makes `base_slot = 0`, so the addressing is exact.
- The shim is causal with `seqlen_q = run_tokens`, `seqlen_k = kv_len`, and
  ignores `batch.positions` entirely. A continuation chunk here has exactly
  the same `(tokens, seq_kv)` relationship a continuation chunk has on the F16
  dense path, which already routes through FA2 in production — so this is
  parity, not a new case.
- `prefill_attention()` (the gate `plan_tq_dispatch` consults before
  classifying an item as long, and which `decoupled6` re-`ensure!`s
  internally) depends only on `n_heads`/`n_kv_heads`/`d_head` — all invariant
  between the batch-wide `dims` and this run's `run_dims`. So the plan-time
  gate and the call-time `ensure!` cannot disagree.

### The persistent `hw_caps` field

`HardwareCaps` was previously a load-time local inside the
`let attn_backend_name = { ... }` block, dropped immediately after
`select_backend()` used it, and `Kernels`' own private `caps` field has no
accessor. The probe is hoisted one scope out (same single
`HardwareCaps::probe(&dev)` call — not a second probe) and stored on `Model`
as `hw_caps: HardwareCaps`, following the same "resolved once at load, reused
everywhere" pattern as `batch_tokens` / `split_mixed_batch` /
`tq_prefill_dequant` / `attn_backend_name`.

The field is declared unconditionally (so `Model`'s shape doesn't depend on
the feature) but carries
`#[cfg_attr(not(feature = "flash_attn2"), allow(dead_code))]`, since the TQ
dispatch is its only reader and that reader is feature-gated. Without this the
default build gains a new `dead_code` warning.

## Deviations from the task description

- **The choice is factored into a named function + enum** rather than written
  as an inline `if / else if / else` at the call site. This is what makes the
  host-side test possible at all, and it is the only way to test the branch
  selection on a machine that can't reach `d_head == 256`.
- **`INFERO_PREFILL_T6` is read at the call site and passed in** as
  `t6_enabled`, rather than read inside the decision function — keeps the
  function pure and the test free of `set_var`.
- **No new long-prompt GPU test was added.** The task allowed for one, but it
  would prove nothing here: the local checkpoint is `d_head = 64`, so both new
  branches are unreachable regardless of prompt length, and `load_tq()` builds
  the pool at `max_seq = 1024`, so a 4096+ token prompt isn't constructible
  without changing that helper's contract for every other test in the file. A
  test that runs 4096 tokens through `ws4` on `d_head = 64` would be a new,
  slower test asserting the branch that was already covered. Said plainly
  rather than added for the appearance of coverage.

## Test results

### What I could actually verify on this machine (RTX A4000, sm_86)

| suite | result |
|---|---|
| `cargo test -p infero-model --lib` (22 tests, incl. all `tq_dispatch_tests`) | **22 passed, 0 failed** |
| `cargo test --release -p infero-model --test turboquant -- --test-threads=1` | 8 passed, **1 failed** — `incremental_and_batched_writes_agree` |
| `cargo test --release -p infero-model --test mixed_batch_dispatch -- --test-threads=1` | 5 passed, **1 failed** — `split_off_reproduces_the_old_dispatch` |
| `cargo clippy -p infero-model --lib`, diffed against HEAD | **no new warnings** |

**Both failures are pre-existing at HEAD, not regressions.** I re-ran each one
with my change stashed and got byte-identical failure values:

- `incremental_and_batched_writes_agree`: `left: 7407, right: 12095` — same
  numbers with and without the change.
- `split_off_reproduces_the_old_dispatch`: `scenario 1: cosine
  0.999664069861` — same value to all twelve printed digits with and without
  the change.

Both are outside this change's blast radius anyway (the second one is the F16
dense path, which this change does not touch at all), but they were checked
rather than assumed.

The behavior-neutrality claim is structural, not just empirical: on
`d_head = 64` (this checkpoint), `supports()` returns false on `d_head`, so
`fa2_ok` is false, and `tq_long_run_kernel(false, 64, _)` returns `Ws4` for
both values of `t6_enabled` — the identical `attn_prefill_ws4` call with
identical arguments. That is asserted directly in
`the_long_run_cascade_picks_the_right_kernel`.

### The `flash_attn2` feature cannot be built on this machine

`--features flash_attn2` requires `INFERO_NVCC` (a full CUDA Toolkit),
`INFERO_CUTLASS_DIR`, and `INFERO_FLASH_ATTN_DIR`. This box has
`/usr/local/cuda-13.1` with **only** `compute-sanitizer` — there is no `nvcc`
anywhere on the filesystem — so `crates/kernels/build.rs` cannot AOT-compile
`flash_attn2_shim.cu` and the feature build fails before rustc runs.

That would normally leave every `#[cfg(feature = "flash_attn2")]` line I wrote
completely unchecked, which is not acceptable for dispatch code. So I
type-checked it by temporarily un-gating the module (`cargo check` doesn't
link, so the unresolved `extern "C"` symbol is irrelevant):

1. dropped `#[cfg(feature = "flash_attn2")]` from `pub mod flash_attn2;` in
   `crates/kernels/src/lib.rs`,
2. rewrote every `feature = "flash_attn2"` in `crates/model/src/lib.rs` to
   `feature = "cuda"` (on by default, so the FA2 arms compile and the
   `not(...)` arms drop out),
3. `cargo check -p infero-model --lib --tests` → **0 errors**,
4. proved the check was real, not cached-past, by deliberately breaking the
   FA2 arm (`prefill(&mut ctx, 1)`) and confirming `error[E0061]: this method
   takes 1 argument but 2 arguments were supplied`,
5. ran `cargo test -p infero-model --lib tq_dispatch_tests` under the same
   un-gating → **9 passed**, including the otherwise-skipped
   `fa2_supports_answers_the_dequantized_buffers_shape_not_the_pools`, whose
   assertions are pure host arithmetic (`HardwareCaps` is two integers,
   `supports()` touches no device),
6. restored both files from byte-exact `/tmp` copies taken beforehand and
   confirmed `git status` shows `crates/kernels/src/lib.rs` unmodified.

So the FA2 branch is **type-checked and its eligibility logic is executed and
asserted**, on a build where the feature is nominally unavailable.

### New tests

Both in `mod tq_dispatch_tests` (`crates/model/src/lib.rs`), matching that
module's existing host-side-test pattern:

- `the_long_run_cascade_picks_the_right_kernel` — the branch table:
  `(fa2_ok=false, d_head=64)` → `Ws4` for both `t6_enabled` values (the local
  checkpoint's reality, and the behavior-neutrality proof);
  `(false, 256, true)` → `Decoupled6`; `(false, 256, false)` → `Ws4` (the
  `INFERO_PREFILL_T6=0` rollback lands on the *old* behavior, not on FA2);
  `(true, 256, *)` → `FlashAttn2`.
- `fa2_supports_answers_the_dequantized_buffers_shape_not_the_pools`
  (`#[cfg(feature = "flash_attn2")]`) — the other half, in `FlashAttn2Ffi`:
  with caps/dims held fixed at the production shape (`sm_86`, 24 heads / 4 kv
  / `d_head=256`, 8192 rows), `supports(..., KvCacheQuant::F16)` is **true**
  and `supports(..., KvCacheQuant::Tq4)` is **false**. That single pair is the
  whole architectural claim of this change, pinned. Plus the three declines:
  300 rows (below `FA2_ROW_THRESHOLD`), `d_head=64`, and `sm_70`.

## Files changed

- `crates/model/src/lib.rs` — the `hw_caps` field + hoisted probe, the
  `TqLongRunKernel` enum and `tq_long_run_kernel()` function, the rewritten
  long-run call site, and the two new tests.
- `docs/superpowers/specs/2026-09-06-tq4-prefill-dequant-dispatch-design.md`
  — a second amendment to the "Amended after Task 3" paragraph recording that
  the follow-up it deferred has now landed, and why the
  `attn_backend_name` gate is deliberately not copied.

## Self-review findings

- **This changes production behavior even without the `flash_attn2`
  feature.** On a `d_head == 256` TurboQuant model, long runs now go to
  `decoupled6` instead of `ws4` on *every* build. That is intended (it mirrors
  the dense path, where `decoupled6` was measured 1.082x over `ws4`), and
  `INFERO_PREFILL_T6=0` rolls it back, but it is not a no-op change gated
  behind an unbuilt feature — worth knowing before deploying.
- **`decoupled6` has no multi-chunk `partial` path**, unlike `ws4`'s `grid.z`
  chunking for small-`n_tiles`/huge-`kv_len` shapes. The dense path already
  accepts this exposure with an explicit comment ("fine for this checkpoint's
  real batch sizes ... but not yet generalized"), and the TQ long path's
  `kv_len` is a full sequence span, which can be larger than a dense run's.
  This is parity with a known, accepted dense-path limitation rather than a
  new hazard, but it is the shape most likely to surprise someone at
  `d_head=256` with a very long context and a narrow head count.
- **`kv_run_is_contiguous` costs a real D2H per long run per layer** when FA2
  is eligible — `seq_kv` `i32`s copied and a sync, so ~88KB × layer-count for
  a 22K-token prompt. It is provably redundant here (`tq.identity_slots` is an
  identity ramp by construction). I kept it because the task asked for it and
  because it is a real guard if that synthetic layout is ever changed, but if
  the FA2 path turns out to be D2H-bound on the production shape, this is the
  first thing to skip — the identity property is a local, checkable invariant.
- **`std::env::var("INFERO_PREFILL_T6")` is read per long run per layer.**
  The F16 dense path does exactly the same thing inline, so this is parity,
  and long runs are few (one per long prefill item per layer) — but neither
  site caches it in a `OnceLock` the way `prefill_attention` caches
  `INFERO_ATTN_MMA`.
- **`fa2_ok`'s per-run evaluation is not a re-litigation** of the dense path's
  measured "don't gate FA2 per run" regression. On a TurboQuant model there is
  no once-per-load answer to reuse, because the load-time one is structurally
  wrong here (it saw a quantized pool). Documented in the function's doc
  comment so nobody "fixes" it back into a load-time gate.
- The `#[cfg(not(feature = "flash_attn2"))] unreachable!(...)` arm is genuinely
  unreachable: on that build `fa2_ok` is a literal `false` and
  `tq_long_run_kernel(false, ..)` cannot return `FlashAttn2`.

## What remains unverified — and what `bw` needs to do to close it

**I have not run the FA2 branch. Not once.** No `d_head == 256` checkpoint and
no `nvcc` exist on this machine, so:

- The FA2 branch has never executed. Its correctness on the synthetic
  identity layout is argued from the shim's source (the `dims.n_slots` /
  `base_slot` arithmetic above), not measured.
- The `decoupled6` branch has never executed on the TQ path either — it is
  also hard-gated at `d_head == 256`.
- No performance claim of any kind is made here. The 15.66s-vs-4.86s gap that
  motivated this task is untouched by anything I can measure locally.

A validation pass on `bw`, against the real 27B checkpoint (24 heads / 4 kv /
`d_head=256`) with a TurboQuant (`tq4`) pool, would need to:

1. Build with `--features flash_attn2` (`INFERO_NVCC`, `INFERO_CUTLASS_DIR`,
   `INFERO_FLASH_ATTN_DIR`) and confirm `INFERO_ATTN_MMA=1` is set — without
   it `prefill_attention()` is false, `tq_dequant_threshold` collapses to
   `usize::MAX`, and there are no long runs to dispatch at all.
2. Confirm the FA2 branch actually *fires*: a prefill run must be ≥ 4096
   tokens (`FA2_ROW_THRESHOLD`) for `supports()` to claim it, so the run must
   be at least that wide — the 22060-token prompt from the motivating
   measurement, chunked at `CUTLASS_BATCH_TOKENS=8192`, qualifies. Add a
   temporary trace/counter at the match arm rather than inferring it from wall
   clock; "it got faster" is not proof the intended branch ran.
3. Check **correctness first, speed second**: compare generated logits (or at
   minimum argmax + cosine, the way `turboquant.rs` does) for the same prompt
   with `INFERO_PREFILL_T6=0` + FA2 unavailable (i.e. the old `ws4` path)
   versus the new path. A wrong-but-fast FA2 result would look like a win in
   wall clock alone — and this backend has already had one silent
   wrong-output bug (the f32-read-as-f16 reinterpretation, see
   `flash_attn2.rs`'s own comment).
4. Then A/B the three branches on prefill wall clock at the real chunk size,
   in both orders, with unique prompt nonces so the prefix cache can't serve a
   repeat — the same methodology the `FA2_ROW_THRESHOLD` investigation used.
   Specifically worth measuring whether `FA2_ROW_THRESHOLD = 4096` is even the
   right crossover *for this path*: it was chosen for the dense path, where
   the run reads the pool directly, whereas here every run has already paid
   `tq_dequant_kv`'s fixed cost, which shifts the cost ratio.
5. Re-check the `decoupled6` multi-chunk exposure noted above at a genuinely
   long context (a full 22K-token sequence span as `kv_len`), since the TQ
   long path can hand it a wider `kv_len` than the dense path's runs do.
