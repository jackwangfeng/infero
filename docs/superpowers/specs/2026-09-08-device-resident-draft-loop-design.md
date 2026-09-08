# Device-Resident MTP Draft Loop — Design

> **Real outcome (2026-09-08, after implementation): shipped, validated (full test suite, real correctness/determinism), and A/B'd for real at both shapes this design names — a real, measured NET LOSS at both.** Dual-stream (`n=2`): ~103.8 tok/s (5 clean samples) against the pre-existing ~115.2 tok/s baseline. Batch=16 with speculation forced on (`n=16`): 278.10 tok/s, worse than even the *old* host-sync forced-on path's 307.54 tok/s, and far worse than the real no-spec floor of 514.92 tok/s. Real accept rate dropped at both shapes (dual-stream ~1.77-1.84 vs baseline ~1.95-2.00; batch=16 ~1.66-1.75 vs the old path's ~1.74-1.97) — the real cost of Gumbel-max dropping repetition-penalty/top-p, this design's own explicitly-named tradeoff, turned out larger than the host-sync savings at both shapes actually tested. **`INFERO_DRAFT_DEVICE_RESIDENT` stays off by default; the code and its real, isolated kernel-level validation stay in the tree as tested infrastructure, matching this session's own established practice for a real kernel-level idea that does not translate to a real end-to-end win.** The rest of this document is the original design as approved and built, kept for the record.

## Problem

`Model::draft_with_head_sampled_batch` (`crates/model/src/mtp.rs`) drafts `k` tokens for every concurrently-drafting sequence in one round. Tonight's own real measurement (`INFERO_STEP_TIMING`, real dual-stream production traffic) found `draft_ms` barely moves between a graphed and an ungraphed run (4.77ms → 4.75ms), unlike `verify_ms` (already batched the same way, 4.2x cheaper graphed, 3.89ms → 0.92ms) — direct evidence that something inside the draft loop sits outside whatever CUDA graph capture covers, and cannot be hidden by it.

The cause: each of the `k` draft steps ends in a **host-blocking sample**. `MtpHead::sample_batch_from_logits` (added tonight, batches the *sampling itself* across sequences into one kernel call) still ends in one `dev.synchronize()` per step, because the sampled token has to come back to the host as a plain `Vec<u32>` before the *next* step's `step_tree` call can be built — `step_tree`'s `tokens: &[u32]` parameter is a host slice. A round therefore costs `k` host round-trips, not one — batching *within* a step (tonight's fix) never touches this; it only reduces `n` round-trips *inside* each step down to one.

Real, load-bearing consequence measured today: batch=16 concurrent decode, speculation forced on via `INFERO_SPEC_SKIP_MIN_CONCURRENCY=17`, real HTTP benchmark: **307.54 tok/s, against 514.92 tok/s with speculation cleanly skipped at that concurrency** — a real ~40% loss, not a rounding error. Accept rate itself was healthy (1.74–1.97 of a possible 3) — the *sampling* is fine; the *host round-trip count* is the cost, and it scales with `k`, which does not shrink as concurrency grows the way batch width can otherwise amortize a fixed cost.

### What vLLM does differently (real source read, `/home/jeff/vllm312`, v1 engine)

vLLM's `EagleProposer.propose()` runs its whole `k`-step draft loop — forward pass, sampling, and feeding the sampled token into the next step — **without a single host round-trip**. Concretely:

- Sampling is `compute_probs_and_sample_next_token(...)`, deterministic **Gumbel-max**: `argmax(logits/temperature + Gumbel_noise(seed, position))`, where the noise is a pure function of `(seed, position, vocab_index)` rather than a host-advanced RNG draw. No repetition penalty, no top-k/top-p.
- The sampled token never leaves the device between steps — the next step's `input_ids` is the previous step's own output tensor, sliced, not a value read back and re-uploaded.
- Per-step host-side "shadow" fields (things like `_seq_lens_cpu`) are explicitly set to `None` after each step specifically to *prevent* an accidental host sync from being triggered by later code that might otherwise read a stale cached copy — the comment in vLLM's own source states this outright: to avoid a host↔device sync.
- Rejection sampling (the accept/reject step analogous to infero's acceptance rule) is one batch-wide Triton kernel over the whole batch's candidates, not a per-sequence loop.

infero already has the matching low-level pieces for the sampling half of this, built and kernel-tested in an earlier round of this same investigation (`crates/kernels/src/lib.rs`'s `gumbel_sample_rows`, `crates/model/src/mtp.rs`'s `step_tree_device`/`TokenSource::Device`, `Sampler::draft_seed()`) but never wired into the real dispatch — this design is that wiring, for the draft loop specifically.

## Goal

`draft_with_head_sampled_batch`'s real round cost drops from `k` host round-trips to `1`, for every concurrently-drafting sequence in a round, with **no change to `Drafted`'s shape, no change to `finish_verify_batch`/the acceptance rule, and no change to `MtpHead`'s or `GdnRollback`'s per-branch state layout**. Verification keeps reading a `Drafted { token: u32, q: Vec<(u32, f32)> }` exactly as it does today; only how the draft loop *produces* that value changes.

## Scope

- **Only the draft loop's own internal `k`-step sampling round-trip.** `Model::verify_draft_sampled(_batch)`, `finish_verify_batch`, `Work::Verify` batching, `GdnRollback`, and `MtpHead`'s per-branch `Vec`-based state are all untouched. Explicitly out of scope for this design (a real, larger, separate project if ever pursued): making per-branch draft/rollback state itself slot-indexed the way vLLM's paged block-table is — infero has no block-table equivalent today and this design does not add one.
- **A new function, not a replacement — at first.** `MtpHead`/`Model` gain a new orchestrator (name TBD at implementation time, e.g. `draft_with_head_device_batch`) alongside the existing `draft_with_head_sampled_batch`, selected by an env var (`INFERO_DRAFT_DEVICE_RESIDENT`, default off until real-measured) — matching this session's own established rollout pattern (`INFERO_DRAFT_VOCAB_Q4`, `INFERO_FUSE_FFN`, etc.). The existing host-sync path stays in the tree as the fallback and, until proven otherwise, the shipped default.
- **Sampling semantics change, deliberately, only for the draft's own proposal**: temperature-scaled Gumbel-max, no repetition penalty, no top-k/top-p, matching `gumbel_sample_rows`'s own existing (already-shipped, already-tested) contract. This is the same tradeoff already accepted for tonight's Q4G128 draft-vocab-head change — the acceptance rule corrects the *served* distribution regardless of what the draft's own proposal was, so final output correctness is unaffected; only the accept rate can move, in either direction, and must be measured, not assumed.
- **Real accept-rate and real end-to-end A/B required before any default flip**, at both shapes this investigation has real baselines for: dual-stream (`n=2`, today's production shape) and batch=16 (`n=16`, today's real 514.92 tok/s no-spec floor and 307.54 tok/s forced-on-old-path floor). A change that helps one shape and hurts the other is a real, reportable finding, not grounds to silently pick a default.

## Current Data Flow (for contrast)

Per round, per draft step (`draft_with_head_sampled_batch`, `crates/model/src/mtp.rs`):

1. `step_tree(kern, embed, tokens: &[u32], positions: &[usize], src_rows: &[usize], branch_of: &[usize], tail)` — a real forward pass through the MTP head's own tiny transformer block, host-side `tokens` from the *previous* step's sampled result.
2. `logits_rows_batch_device` — vocab projection for every sequence's current row, batched (`mmvq_batch`, one call for the whole `n`).
3. `sample_batch_from_logits` (tonight's fix) — one `sample_rows_split` kernel call, one `dev.synchronize()`, returns `Vec<(u32, Vec<(u32,f32)>)>` — this is where the host round-trip happens, once a step.
4. The sampled `u32` tokens become next step's `tokens: &[u32]` (step 1 again) — a step cannot start until step 3's synchronize returns.

`k` iterations of 1–4 → `k` synchronizes a round.

## New Data Flow

Same four responsibilities, restructured so only step 4 (now the *whole loop's* step 4, not each step's) touches the host:

1. **Prime, unchanged.** `prime_batch` still runs once, exactly as today — it already takes real hidden states from the last forward pass and produces the first step's rows; nothing about priming is on the hot round-trip path today (measured: it's the per-*step* sampling that costs, not the one-time prime).
2. **`k` steps, no host sync between any of them:**
   - `step_tree_device(kern, embed, tokens: View<'_, u32>, positions: &[usize], src_rows: &[usize], branch_of: &[usize], tail)` — the position/`src_rows`/`branch_of` arguments stay host-side (they're pure host bookkeeping that never depends on what token was actually sampled: positions increment by a fixed 1 each step, branches don't change mid-draft, `src_rows` is this call's own row layout) — only `tokens` becomes a device `View` instead of a host slice.
   - `logits_rows_batch_device`, unchanged.
   - **New**: sample via `gumbel_sample_rows(out: &mut ViewMut<u32>, ..., scaled_logits: &mut ViewMut<f32>, temperature, seed, position, n_rows, vocab)`, writing the sampled token ids into a persistent device buffer (this step's slice of a `[k, n]`-shaped token-history buffer) and this step's temperature-scaled dense logits into a persistent `[k, n, vocab]` scratch (sized once, reused across rounds — at this checkpoint's real shape, `k≤8` (`GEMV_KSPLIT_TOKENS`-style cap, matching the codebase's existing convention of a compile-time-bounded unroll) `× n≤32 × vocab=248320 × 4 bytes` is a few tens of MB, not a real memory concern).
   - `seed`/`temperature` come from each sequence's own `Sampler::draft_seed()`/`SamplingParams::temperature`, uploaded once at round start (small, `[n]`-shaped, no different in kind from what today's per-step upload already does).
   - Feed this step's own token slice (a `View`, no copy needed beyond what `step_tree_device` itself already does internally) directly into the next step's `step_tree_device` call.
3. **One readback, at the end of the `k`-step loop:**
   - The final step's sampled tokens (needed on the host regardless — the scheduler's own bookkeeping, `DraftedRound`, `history` tracking, etc. all live on the host today and this design does not change that) come back in one `memcpy_dtoh` over the whole `[k, n]` token buffer.
   - The `[k, n, vocab]` accumulated `scaled_logits` buffer is reduced, in **one further batched kernel call**, into the exact `Drafted::q: Vec<(u32, f32)>` shape `finish_verify_batch` already expects — a top-k-over-already-scaled-logits extraction (no penalty window, no re-sampling: the token was already chosen), reusing the existing top-k-finding machinery's *shape of kernel* (`sample_topk_partial_f32`/`argmax_combine_f32`'s own pattern) rather than the existing `sample_rows_split`'s full penalty+sample pipeline, since neither a penalty window nor a fresh random draw applies here. Whether this reuses an existing kernel unchanged, needs a thin new host-side wrapper around existing device code, or needs a small new kernel is an implementation-time question, not a design one — the shape of the answer (`Vec<(u32,f32)>`, same truncation-error tradeoff already accepted and measured for the current top-k sampler) is fixed by this design; the exact kernel is not.
   - This is the **only** point in the whole round where the host waits on the GPU for the draft loop's own sake.

`k` iterations of step 2 (zero syncs) → one batched extraction + one readback. One synchronize a round, not `k`.

## Testing

Correctness for a *changed sampling algorithm* cannot be a token-for-token comparison against the old path (the old path used repetition-penalty/top-k/top-p categorical sampling; this one is temperature-scaled Gumbel-max — different, both valid, not required to agree token-for-token). The real correctness bar, matching this codebase's own existing test for exactly this class of question:

1. **Statistical composition test**, same shape as `crates/model/tests/spec_sampled.rs`'s existing `the_composition_of_draft_and_acceptance_is_a_draw_from_the_target`/`the_multi_candidate_composition_is_a_draw_from_the_target`: many repeated draws through the new device-resident draft path plus the *unchanged* acceptance rule must still reproduce draws from the target model's real distribution. This is the test that actually matters — it validates the whole point of the design (a different draft proposal is fine iff the composed process still serves the right distribution).
2. **Kernel-level correctness** for anything genuinely new (the top-k-from-already-scaled-logits extraction): an f64-host-accumulated reference, matching this session's own established rigor for every new kernel tonight (`gemv_f16_ksplit`, `quantize_f16_to_q4g128`).
3. **Determinism**: repeated calls with the same `(seed, position)` inputs must be bit-identical — `gumbel_sample_rows` is already designed to be deterministic from those inputs alone; a test pinning this (not just trusting the doc comment) belongs here, the same way tonight's `gemv_f16_ksplit` probe added an explicit 10-repeat bit-identical check after a *different* kernel's determinism bug was found the hard way earlier this session.
4. **`compute-sanitizer --tool memcheck`/`--tool racecheck`**, 0 errors/hazards, on whatever new device-resident probe exercises the full `k`-step loop.
5. **Full `cargo test -p infero-model` suite**, real GPU, before any deploy — same bar as every change shipped tonight.
6. **Real accept-rate and real end-to-end A/B**, both at dual-stream (`n=2`) and batch=16 (`n=16`), against the real numbers already on record tonight (dual-stream: draft_ms 4.35–4.81ms, ~115.2 tok/s post-Q4G128; batch=16: 514.92 tok/s no-spec, 307.54 tok/s forced-on-old-path) — the whole motivation for this design is a number this step either confirms or refutes, not something to assume from the design alone.

## Rollout

`INFERO_DRAFT_DEVICE_RESIDENT` (default off) selects the new path; `0`/unset keeps today's host-sync path unchanged, no rebuild needed to fall back. Default flips only after real A/B data at both shapes above shows a real, non-regressing (or honestly-characterized-and-accepted-tradeoff) win — matching this session's own repeated practice of shipping infrastructure behind a flag before trusting it as a default, and reverting dispatch (while keeping validated kernels in tree) when a real measurement doesn't support the win a change was built to chase.
