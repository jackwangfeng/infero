# Mixed-Batch Attention Dispatch Split — Design

## Problem

`attn_partial` (`crates/model/src/lib.rs:5231-5232`, sized via `Kernels::attn_partial_floats(n_heads, d_head, n_tokens) = 32*n_heads*n_tokens*(d_head+2)` floats, `crates/kernels/src/lib.rs:1999-2001`) is permanently allocated at `n_tokens = batch_tokens = 8192`, costing 6192 MiB — confirmed by exact match against the server's own startup log (`partial_mib=6192`). It is real, currently-necessary VRAM given today's dispatch logic, not a leftover artifact — but the dispatch logic itself is the actual root cause.

Today, `single_seq_run = (items.len() == 1).then(...)` (`crates/model/src/lib.rs:1827`) and `prefill_run = single_seq_run.filter(|&t| t >= MIN_PREFILL_RUN)` (line ~2005) gate whether the fast, single-sequence tile-based prefill kernels (`decoupled6`/`ws4`/`flash_attn2`) run at all. These kernels assume ONE sequence, contiguous KV, monotonically-increasing causal positions (`crates/model/src/lib.rs:1818-1826`) — a tile spanning two sequences would read one sequence's KV for another's rows. So ANY batch with `items.len() != 1` — including an all-prefill batch with 2+ prefill items and zero decode items — falls through to the generic `attn_decode` kernel, the only kernel that reads/writes `attn_partial`.

Under real continuous batching at production's `--max-seqs 2`, a batch mixing one decode item + one prefill item (or two simultaneous prefill items) is the **ordinary case** (`crates/server/src/scheduler.rs::plan()`, ~line 1662, fills decode items for every prompt-complete running sequence first, then fills remaining `batch_tokens` budget with prefill chunks from other sequences) — not a rare edge case. This forces `attn_decode` to regularly receive up to the full `batch_tokens`=8192-wide prefill chunks, which is why `attn_partial` must be sized for that worst case today.

## Goal

Restructure attention dispatch so:
- All `Work::Decode` items in a batch batch into ONE `attn_decode` call (already safe today — no single-sequence-tile constraint, which is why it can already handle multiple concurrently-decoding sequences).
- Each `Work::Prefill` item gets its own separate call to the appropriate single-sequence tile kernel (`decoupled6`/`ws4`/`flash_attn2`, via the existing per-item eligibility logic), instead of being swept into `attn_decode` whenever the batch isn't exactly one item.

This lets `attn_decode`'s real `n` (query-token count) ceiling revert to small (`max_seqs*(k+1) + MIN_PREFILL_RUN`, see below), so `attn_partial` can shrink to match — from 6192 MiB to roughly 9 MiB at production's current config (`max_seqs=2`, MTP `k=1`, `MIN_PREFILL_RUN=8`).

**Real payoff:** infero's total production VRAM would drop from the current real steady-state (~59630 MiB) to roughly ~53550 MiB (~52.3 GB) — matching vLLM's real measured total (52.23 GiB) at the same config almost exactly.

## Scope

- **Single-GPU only.** Tensor-parallel (`--tensor-parallel-size > 1`) is explicitly out of scope for this change. The GDN-hybrid architecture's TP support is itself only structurally validated, not in real production use, per this session's own findings. The TP broadcast path (`crate::tp::WorkMsg`) gets a `debug_assert!(rank.tp_size <= 1, ...)` rather than being silently wrong under TP — a future TP-aware extension is separate, explicitly deferred work.
- **Zero performance regression required**, not just "acceptable." The looped per-prefill-item dispatch must not measurably slow down mixed-batch or multi-prefill-batch throughput/latency versus today's single-call `attn_decode` fallback — this must be measured directly, not assumed.
- **Short prefill remainders stay folded into `attn_decode`.** `MIN_PREFILL_RUN=8` exists because very short prefill chunks aren't worth a dedicated tile-kernel launch. This behavior is preserved: `attn_partial`'s new bound includes headroom for `MIN_PREFILL_RUN`-scale remainders riding along with the decode batch, rather than shrinking to the absolute theoretical minimum.
- **Feature-flag gated**, matching this codebase's existing precedent for high-risk kernel-path changes (`INFERO_PREFILL_T6`, `INFERO_FUSE_FFN`): a new `INFERO_SPLIT_MIXED_BATCH` env var, default-on (new behavior), settable to `0` to instantly roll back to today's single-call-per-batch behavior without a rebuild.

## Data Model

`BatchItem` (`crates/model/src/lib.rs:277`) gains a `kind: BatchItemKind` field (`enum BatchItemKind { Decode, Prefill }`), set at both construction sites where `scheduler.rs`'s `Work::Decode`/`Work::Prefill` (lines 962, 989) become `BatchItem`s. The scheduler already knows this distinction when it builds the batch; today it's simply discarded on the way into `BatchItem` — carrying it through is not new information, just preserved information.

`attn_partial`'s sizing (`Activations::new`, ~line 1267):

```rust
let decode_n_ceiling = max_seqs * (mtp_k + 1) + MIN_PREFILL_RUN;
let partial_n = if split_mixed_batch_enabled { decode_n_ceiling } else { batch_tokens };
```

resolved once at load time, mirroring the existing "resolved once at load" pattern already used for `batch_tokens`/`attn_backend_name`.

## Dispatch Flow

Before the 64-layer loop, per `forward_batch_device` call:

```rust
let (decode_items, prefill_items) = partition_by_kind(&items);
debug_assert!(
    decode_items.iter().all(|i| i.kind == BatchItemKind::Decode)
        && prefill_items.iter().all(|i| i.kind == BatchItemKind::Prefill),
    "scheduler ordering invariant violated"
);
let decode_layout = (!decode_items.is_empty()).then(|| BatchLayout::build(&decode_items));
let prefill_layouts: Vec<BatchLayout> = prefill_items.iter().map(BatchLayout::build_single).collect();
```

This partition happens once per batch (not per layer). It relies on the scheduler-ordering invariant that decode items always precede prefill items in `items` (confirmed: `scheduler.rs::plan()` fills decode items first, then fills remaining budget with prefill chunks) — so decode items occupy a contiguous prefix of the flat per-token buffers (`x`, `q`, `k`, `v`, etc.) and each prefill item occupies its own contiguous region after that. This means offset-based sub-slicing is sufficient; no gather/scatter is needed. The `debug_assert!` above exists specifically so that if this invariant is ever violated by a future scheduler change, the failure is loud (a panic in a debug/test build) rather than silent misrouted data in release.

Inside each layer's attention step (replacing the current single dispatch at ~lines 3661-3763):

```rust
if let Some(layout) = &decode_layout {
    kern.attn_decode(..., &mut act.attn_partial, layout, /* offset */ 0)?;
}
let mut offset = decode_items.iter().map(|i| i.tokens.len()).sum();
for (item, layout) in prefill_items.iter().zip(&prefill_layouts) {
    // whichever kernel the existing per-item eligibility logic already selects
    // (decoupled6 / ws4 / flash_attn2), invoked per-item instead of requiring
    // the WHOLE batch to be exactly one item.
    kern.attn_prefill_dispatch(..., layout, offset)?;
    offset += item.tokens.len();
}
```

`offset` determines which slice of the shared `Activations` buffers (`act.q`, `act.attn`, etc.) each call reads/writes. GDN layer processing (`pool.set_gdn_layout`/`spans`, ~line 1870) and FFN/sampling, downstream of the 64-layer attention loop, see the same complete flat buffer as today regardless of how many internal calls produced it — they require no changes, since they only depend on every token's row being correctly populated by the time they run, not on how attention was internally dispatched. This must be independently re-verified during implementation (a prior same-session implementation attempt did not get far enough to confirm it, though a research pass claimed it).

`INFERO_SPLIT_MIXED_BATCH` (default on) gates this entire path; set to `0`, the code takes today's existing `single_seq_run`/`attn_decode`-fallback path unchanged, and `attn_partial` reverts to the full `batch_tokens`-wide allocation. Both paths coexist in the binary — rollback is an env var flip, not a redeploy of a different build.

## Error Handling

- Scheduler-ordering invariant: `debug_assert!` as shown above. If this ever fires in a debug build, treat it as a real bug requiring investigation before shipping, not a check to relax.
- `INFERO_SPLIT_MIXED_BATCH=0` is the designated escape hatch for any unexpected production issue — no new failure mode should require a rebuild to recover from.
- TP: `debug_assert!(rank.tp_size <= 1, "mixed-batch split not yet TP-aware")` at the point `WorkMsg` broadcast would otherwise need the same `kind` distinction propagated to followers. This fails loud in debug builds rather than silently producing wrong results under TP; release builds should not attempt to enable this path under `tp_size > 1` (implementation should gate `split_mixed_batch_enabled` itself on `tp_size <= 1`, not merely assert).

## Testing

1. **Real token-level output diffing**, fixed seed, before vs. after, for: (a) a pure single-sequence prefill+decode conversation, (b) an engineered mixed decode+prefill batch (one sequence finishes its prompt and starts decoding while another is still mid-chunked-prefill), (c) an engineered two-simultaneous-prefill batch. All three must match exactly token-for-token (or within a documented, justified floating-point tolerance if `decoupled6` and `attn_decode` are only expected to agree approximately for the same logical computation — this must be checked, not assumed, and any unexpected divergence treated as a real bug).
2. `compute-sanitizer --tool memcheck` and `--tool racecheck` against scenarios (b) and (c) above — the per-item buffer-offset slicing is exactly the kind of change where an off-by-one could produce a race or out-of-bounds access.
3. Full `scripts/server_stress_test.py --passes 5` — all 13 categories STABLE (existing known, unrelated `multiturn_2` model-sampling variance from commit `238f2e1` is not a regression signal).
4. Real before/after VRAM (`nvidia-smi --query-compute-apps=pid,used_memory --format=csv`) confirming `attn_partial`'s new size matches the computed ~9 MiB and total VRAM drops accordingly.
5. Real performance measurement for mixed-batch and multi-prefill-batch scenarios specifically, `INFERO_SPLIT_MIXED_BATCH=1` vs `=0`, confirming no regression (not assumed neutral).
6. Production rollout: deploy with the flag on; keep the `=0` fallback verified working as the rollback path.

## Out of Scope

- Tensor-parallel support for this dispatch split (deferred; see Scope).
- Any change to GDN, FFN, or sampling code (not needed per this design; must be verified, not just assumed, during implementation).
- Shrinking any buffer other than `attn_partial` (the separate ~3.4GB Activations/Scratch gap is tracked and partially addressed elsewhere this session, commit `23662aa`).
