# TurboQuant Prefill Dequant-Dispatch — Design

## Problem

TurboQuant (`kv_quant != KvCacheQuant::F16`) is real, working, correctness-tested KV-cache quantization (`crates/kernels/src/cu/turboquant.cu`, `crates/model/src/cache.rs`'s `Storage::TurboQuant`) but is **not enabled in production** (`kv_quant=f16` confirmed in live server logs) because of a real, measured performance problem, previously mischaracterized this session as a client "crash" before being root-caused as a 120s client timeout hitting genuinely slow prefill.

`crates/model/src/lib.rs:4872-4946`: the TQ attention branch dispatches **every** attention call — real decode (`n`=1 per sequence) and prefill chunks of any length alike — through the same `tq_attn_decode` kernel (or its unfused three-kernel fallback, `tq_attn_scores`/`attn_softmax`/`tq_attn_output`). This kernel grids one block per `(kv_head, query_token)` and re-unpacks every prior quantized key/value **per query token**, an `O(n_tokens × kv_len)` cost. The dense F16 path's tile kernels (`decoupled6`/`ws4`/`flash_attn2`) unpack each key once per tile instead — this is the entire gap.

The TQ branch is also, deliberately, not part of the mixed-batch-dispatch-split shipped `2026-09-05`: `lib.rs:4873-4882`'s comment records that it keeps one whole-batch call and still reads `attn_partial` at the batch's full `n`, guarded by `ensure_partial_fits`. This is unrelated to the present problem (that guard is about buffer *sizing*, not dispatch *cost*) and this design does not change it: `wide_prefill_avoids_attn_partial` (`lib.rs:179`) already returns `false` whenever `kv_quant != F16`, so enabling TQ always reverts `attn_partial` to its full `batch_tokens` width regardless of anything in this design — `tq_attn_decode`'s existing whole-batch call is already safe today and stays safe.

### What vLLM does, and why infero's answer differs

vLLM ships a real, same-named TurboQuant KV-cache dtype (`turboquant_3bit_nc`/`turboquant_4bit_nc`/`turboquant_k3v4_nc`/`turboquant_k8v4`, from the TurboQuant paper, Zandieh et al., ICLR 2026 — the same underlying algorithm as infero's, not a competing technique). Read directly from `/home/jeff/vllm312/lib/python3.12/site-packages/vllm/v1/attention/backends/turboquant_attn.py` on `bw`: vLLM has **no native tiled-quantized-KV prefill kernel either**. It dispatches three ways: (1) a request's first chunk (no prior cached KV) runs standard flash_attn on the *raw, unquantized* K/V, quantizing only on write; (2) a continuation chunk with `q_len <= 128` (`_CONTINUATION_DECODE_THRESHOLD`) reuses the decode kernel per query token — the same cost profile infero already pays, but capped small; (3) a continuation chunk above that threshold dequantizes the *entire* cached span once (`_tq_full_dequant_kv`, a Triton kernel) into a reusable scratch buffer, concatenates the current chunk's raw K/V, and calls the existing dense flash_attn.

vLLM needs three tiers because its dense flash_attn call is not gather-aware — it needs literal, contiguous, unpaged K/V, so a first chunk (nothing cached yet) can skip quantization/dequantization entirely by never touching the paged cache. infero's dense tile kernels are different: `decoupled6`/`ws4`/`flash_attn2` already read through `BatchLayout`/`slot_table` (`lib.rs:4499-4504`), a gather exactly analogous to vLLM's own `block_table`, which is *why* they already handle "some history, some new tokens" transparently and don't distinguish first-chunk from continuation at all. There is nothing for infero's version of tier (1) to buy — every prefill call under TQ already goes through a quantize-then-read round trip today regardless of whether the sequence is new, so skipping quantization for a first chunk would need a second, parallel raw-KV code path with no reader today. **Two tiers, not three**, covers the same ground:

| run length | path |
|---|---|
| ≤ threshold (decode, and short prefill remainders) | today's `tq_attn_decode` — already the right tool for small `n` |
| > threshold | one-shot dequantize the run's `0..kv_len` span into a dense f16 scratch buffer, then call the **existing, unmodified** `decoupled6`/`ws4`/`flash_attn2` against it |

The extra quantize→dequantize round trip a first-time long prompt pays under this design (vLLM's tier 1 would skip it) is real but small next to the fix: going from *re-unpacking the whole history once per query token* to *unpacking it once, period* is the order-of-magnitude change, and the further optimization of skipping quantization for brand-new sequences is a separate, smaller win this design does not chase (YAGNI — no reader exists for a "raw KV, not yet quantized" code path today, and building one to serve only first-chunk prefill is disproportionate to what it saves).

### A second infero-specific simplification: no inverse rotation in the new kernel

TurboQuant's quantization first projects Q, K, and V into a rotated basis (`tq_matvec` against `tq.tables.rotation`, `lib.rs:4790-4816`, producing `tq.q_rot`/`k_rot`/`v_rot`) — **both K and V**, unlike vLLM's TQ (which rotates keys only; values are plain uniform-quantized in their original basis). Every existing TQ kernel — `tq_attn_decode`, the unfused three-kernel fallback — operates entirely in this rotated basis and produces a rotated accumulator (`tq.acc_rot`); the *only* place the basis is ever undone is a single final `tq_matvec` call on the finished output (`lib.rs:4947-4949`), after attention is fully computed. This is valid because the rotation is orthonormal: `(Rq)·(Rk) = q·k` (dot products, hence attention scores, are unchanged), and because a weighted sum of rotated V vectors equals the rotation of the weighted sum (linearity), so the *un-rotate-the-output-once-at-the-end* trick already in the codebase covers a rotated V too.

The new dequantize kernel this design adds therefore does **not** need to inverse-rotate anything — unlike vLLM's `_tq_full_dequant_kv`, which does inverse-rotate keys back to their natural basis because vLLM's dense flash_attn needs to match an *unrotated* Q. Here, the dequantized K/V scratch stays in the rotated basis, and gets paired with the already-computed, already-rotated `tq.q_rot` — the reused `decoupled6`/`ws4`/`flash_attn2` kernels compute the same dot-product/weighted-sum attention regardless of which orthonormal basis their inputs happen to live in, so they need zero awareness that TQ is involved. The output lands in `tq.acc_rot`, exactly where `tq_attn_decode`'s output lands today, and the existing final un-rotate step handles it unchanged.

## Goal

Any TurboQuant prefill run longer than a measured threshold stops paying `O(n_tokens × kv_len)` redundant unpacking and instead pays one `O(kv_len)` unpack plus whatever the existing dense tile kernels already cost the F16 path for the same shape. Runs at or below the threshold are unaffected — they keep using `tq_attn_decode` exactly as today. No existing kernel (`decoupled6`, `ws4`, `flash_attn2`, `tq_attn_decode` itself) is modified; only a new dequantize kernel and new dispatch logic are added.

## Scope

- **TurboQuant only.** The F16 dense path, its `AttnDispatch`/`plan_attn_dispatch`/`run_attn` machinery from the mixed-batch-dispatch-split plan, and `attn_partial` sizing are untouched by this design. This is a separate lever inside the `Some(tq)` branch of `attention()` (`lib.rs:4785` onward), not an extension of that plan's dispatch abstraction — see Data Model for why reusing `AttnDispatch` directly does not fit.
- **Zero performance regression for runs at or below the threshold**, and no *accuracy* change for any run (the dequantize kernel must exactly invert `tq_store_k`/`tq_store_v`'s packing — same MSE/uniform dequantization math already implemented for `tq_attn_decode`'s own internal unpacking, just materialized into a buffer instead of consumed register-side).
- **Feature-flag gated**, matching this codebase's precedent (`INFERO_SPLIT_MIXED_BATCH`, `INFERO_PREFILL_T6`): `INFERO_TQ_PREFILL_DEQUANT`, default-on once measured to be a real, non-regressing win at the chosen threshold; settable to `0` to force every run back through today's `tq_attn_decode`-only behavior without a rebuild.
- **Not a production rollout gate.** TurboQuant is not enabled in production (`kv_quant=f16`), so this work is validated in isolation (a TQ-quantized load, real token diffing, real timing) rather than against a live model. Actually turning TQ on in production is a separate, later decision this design does not make.

## Data Model

`TqBuffers` (`lib.rs:1159` region, alongside `q_rot`/`k_rot`/`v_rot`/`acc_rot`) gains two new scratch fields sized once at load, analogous to `Scratch`'s existing pattern (`w16`/`q8_1`/`xq_e4m3`, `lib.rs:1175`):

```rust
/// Dense, rotated-basis KV for one sequence's dequantized prefill span.
/// Sized to `max_seq` once at load and reused across layers and calls --
/// the same "resolved once, reused everywhere" pattern as `Scratch`'s
/// other buffers. Only touched when `INFERO_TQ_PREFILL_DEQUANT` selects
/// the long-run path for at least one item in a batch.
dequant_k: Buf<f16>,  // [n_kv_heads, max_seq, d_head], rotated basis
dequant_v: Buf<f16>,  // [n_kv_heads, max_seq, d_head], rotated basis
```

`n_kv_heads * max_seq * d_head * 2 (f16) * 2 (k and v)` bytes, allocated once — at this checkpoint's real shape this is small next to the pool itself (single-sequence span, not `n_slots`-wide).

A new kernel, `tq_dequant_kv` (`crates/kernels/src/cu/turboquant.cu`, alongside `tq_store_k`/`tq_store_v`/`tq_attn_decode`):

```rust
pub fn tq_dequant_kv(
    &self,
    dequant_k: &mut ViewMut<f16>,   // [n_kv_heads, kv_len, d_head] dense, rotated
    dequant_v: &mut ViewMut<f16>,   // [n_kv_heads, kv_len, d_head] dense, rotated
    k_codes: &View<u8>, k_signs: &View<u8>, k_scale: &View<f16>, k_gamma: &View<f16>,
    v_codes: &View<u8>, v_scale: &View<f16>,
    seq_slots: &View<i32>,           // this sequence's own slot list, `kv_len` entries, logical order
    k_levels: &View<f16>, k_bits: u32,
    v_levels: &View<f16>, v_bits: u32,
    d_head: usize,
    n_kv_heads: usize,
    kv_len: usize,
) -> Result<()>
```

Grid: one thread block per `(kv_head, position)` pair (`kv_len * n_kv_heads` blocks total) — exactly the unpack work `tq_attn_decode` already does per query token, just done once instead of `n_tokens` times. `seq_slots` is this one sequence's own physical-slot list in logical order — already what `KvPool` tracks per sequence (`SeqState::slots`, `cache.rs:58`). This design adds one new pool-side accessor, `KvPool::seq_slots(&self, id: SeqId) -> &[i32]`, returning that `Vec<i32>` as a slice — no change to `Storage::TurboQuant`'s layout itself. (Verify at implementation time whether the batch-layout-construction code one level up in `attention()` already has an equivalent read available and can pass it straight through instead of calling this accessor a second time — either is correct, it's purely a question of avoiding a redundant lookup.)

**Why not extend `AttnDispatch` to cover this instead of a new TQ-local partition:** `AttnDispatch`/`plan_attn_dispatch` exists to solve *`attn_partial` sizing under a shrunk buffer* — its whole reason to route decode and prefill items separately is so `attn_partial` can be sized off the small decode ceiling instead of the full batch width (`plan_attn_dispatch`'s doc comment, mixed-batch-dispatch-split spec §Data Model). That concern doesn't exist here: `attn_partial` is always full-width whenever TQ is active (`wide_prefill_avoids_attn_partial` returns `false` for any non-F16 `kv_quant`), so there is nothing this design needs `AttnDispatch` *for*. What it does need — per-item run boundaries within a batch — is already available as `BatchItemKind`-tagged `items` (the same data the mixed-batch-dispatch-split plan added to `BatchItem`, `lib.rs:277`) at the point `attention()` runs; this design reads that tagging but adds its own short/long partition over `Prefill`-kind items, local to the `Some(tq)` branch, rather than threading a second concern through the F16 path's own dispatch plan.

## Dispatch Flow

Inside the `Some(tq)` branch (`lib.rs:4785` onward). The **write** side is untouched by this design and stays exactly as today: `tq_matvec` (rotate) and `tq_store_k`/`tq_store_v` (quantize + store) still run once, unconditionally, over the whole batch's `n` tokens — every token gets quantized and written to the pool regardless of which read path serves this call, because future decode/continuation calls need all of it in the pool. Only the **read** (the attention computation itself) gets partitioned, after storage:

```rust
let (short_items, long_items): (Vec<_>, Vec<_>) = items
    .iter()
    .partition(|it| it.kind == BatchItemKind::Decode || it.tokens.len() <= TQ_DEQUANT_THRESHOLD);

// Unchanged: today's tq_attn_decode call, now scoped to read only the
// short/decode items' token ranges (still one call covering all of them
// together) instead of the whole batch.
if !short_items.is_empty() {
    // ... existing tq_attn_decode, run over exactly the short_items'
    // token ranges. No change to tq_matvec/tq_store_k/tq_store_v above.
}

for item in &long_items {
    // The write side (tq_matvec/tq_store_k/tq_store_v, unconditional above
    // this point) already extended the pool with this call's own tokens, so
    // the pool's own tracked length already IS this sequence's real,
    // post-extension history -- no separate arithmetic needed.
    let kv_len = pool.len(item.seq); // KvPool::len(&self, id: SeqId) -> usize, cache.rs:537
    self.kern.tq_dequant_kv(
        &mut tq.dequant_k.slice_mut(..n_kv_heads * kv_len * d_head),
        &mut tq.dequant_v.slice_mut(..n_kv_heads * kv_len * d_head),
        /* this item's k_codes/k_signs/k_scale/k_gamma/v_codes/v_scale, seq_slots */
        ...
    )?;
    // A trivial, single-sequence, identity-mapped BatchLayout over the
    // compact scratch buffer -- physical slot i == logical position i,
    // table_stride == kv_len -- constructed fresh per item, distinct from
    // the real pool's `slot_table`/`table_stride` used elsewhere. This is
    // the only reason a synthetic BatchLayout is needed: the real pool's
    // slot_table indexes the FULL pool width, not this compact per-item buffer.
    let synth_batch = BatchLayout { /* identity mapping over dequant_k/v */ };
    // Existing kernels, called unmodified, against the rotated-basis
    // dequantized scratch and the already-rotated tq.q_rot -- output lands
    // in tq.acc_rot at this item's token offset, same as tq_attn_decode's
    // output would have.
    kern.attn_prefill_ws4(/* or decoupled6 / flash_attn2, same eligibility
                             logic the F16 path already uses */)?;
}
```

**Amended after Task 3 (implementation).** The sketch's `/* or decoupled6 / flash_attn2, same eligibility logic the F16 path already uses */` is not what shipped: the long path calls `attn_prefill_ws4` and only `attn_prefill_ws4`. `ws4` is the one tile kernel eligible across every shape this plan's own tests can exercise — the `d_head=64` test checkpoint included, since `ws4`'s `prefill_attention` gate has no `d_head == 256` requirement where `decoupled6`/flash_attn2 do — and it is the kernel `tq_dequant_threshold` already consults before classifying an item as long, so the gate and the call cannot disagree. Reproducing the F16 path's real three-way dispatch (its `INFERO_PREFILL_T6` handling and flash_attn2's `#[cfg(feature = "flash_attn2")]` gate) is separable follow-up work that Out of Scope never promised; if a later measurement shows `ws4` alone leaves real performance on the table at production's `d_head=256`, that is a follow-up, not a gap. See the plan's Task 3 Step 3 for the same reasoning at the point it was taken.

`TQ_DEQUANT_THRESHOLD` is a new constant, placed alongside `MIN_PREFILL_RUN` (`lib.rs:94`) — its real value is chosen by measurement during implementation (Testing §1 below), not copied from vLLM's `128` (different kernels, different hardware-generation cost ratios; vLLM's threshold reflects the cost profile of *their* decode kernel and *their* dequant kernel, not infero's).

`INFERO_TQ_PREFILL_DEQUANT=0` collapses `long_items` to always empty (i.e. `TQ_DEQUANT_THRESHOLD = usize::MAX`), restoring today's exact one-call-per-batch behavior — the rollback path, no rebuild required.

## Error Handling

- `tq_dequant_kv`'s dequantization math must be checked against `tq_attn_decode`'s own internal unpacking bit-for-bit (same MSE level table, same uniform-quant scale/zero, same sign/gamma handling) — a silent mismatch here is a silent *accuracy* bug, not a crash, and would be far harder to notice than a performance regression. Real token-level diffing (Testing §2) is the guard, not a runtime assertion.
- `dequant_k`/`dequant_v` are sized to `max_seq` at load; any `kv_len` beyond that is already an existing, separately-enforced invariant (`extend()`'s own `start + n <= self.max_seq` check, `cache.rs:561`) — this design adds no new bound to enforce.
- If `tq_dequant_kv` or the reused dense kernels are ever asked to run under a `kv_quant` this design didn't validate (only the checkpoint-relevant `k_bits`/`v_bits` combinations tested in Testing §2), fail loud rather than silently: an `ensure!` on supported bit-widths at dispatch time, mirroring `TqBuffers::new`'s existing "no kernels on this backend yet" loud failure (`lib.rs:1924`).

  **Amended after Task 3.5 (implementation) — this bullet is inverted by a deliberate later ruling.** The `ensure!` shipped first and was then measured to be actively harmful: `k8v4` is this repo's own quality sweep's recommended allocation and a preset two `crates/model/tests/turboquant.rs` tests already use, and a hard error made *every* prefill wider than `TQ_DEQUANT_THRESHOLD` fail outright under `INFERO_ATTN_MMA=1` — a 500 on a configuration that works at any length today, only slowly. What shipped instead is a **graceful gate, not an error**: the bit-width check is one of the three inputs to `tq_dequant_threshold`, which returns `usize::MAX` when it fails, so `plan_tq_dispatch` classifies nothing as long and the pass is bit-for-bit the one-call-per-batch dispatch this branch made before the path existed. "Unvalidated" therefore means *slower*, never broken. Nothing runs unverified dequantization math either way — that requirement is met, by exclusion rather than by erroring. The `ensure!`'s intent survives as a `debug_assert!` at the dequant call site, tying the two places together so a future edit that plans a long run without consulting the threshold trips in a debug build.

## Testing

1. **Real threshold measurement.** Before picking `TQ_DEQUANT_THRESHOLD`'s value, benchmark `tq_attn_decode` vs. `tq_dequant_kv`-then-`ws4`/`decoupled6` across a sweep of prefill-run lengths (mirroring the F16 `FA2_ROW_THRESHOLD` investigation's method: real wall-clock, fixed seed, unique-nonce prompts so the prefix cache can't interfere) on a TQ-quantized load of the real checkpoint or a small stand-in. Pick the crossover point empirically; do not assume it mirrors vLLM's `128` or infero's `MIN_PREFILL_RUN=8`.
2. **Real token-level output diffing**, fixed seed, TQ-quantized load, comparing generation before vs. after this change for: (a) a single long first-time prefill (no prior cache) that lands above threshold, (b) a continuation prefill chunk above threshold on a sequence with real prior TQ-cached history, (c) a batch mixing a short decode item with a long prefill item in the same call, (d) a run at exactly the threshold boundary on both sides. All must match token-for-token (TQ already has documented, bounded quantization error vs. F16 — the comparison here is against *TQ's own* pre-change output, not against F16, so this must be an exact match, not a tolerance).
3. `compute-sanitizer --tool memcheck` and `--tool racecheck` against scenarios (b) and (c) — the new synthetic single-sequence `BatchLayout` and the compact scratch buffer's indexing are exactly the kind of new addressing where an off-by-one produces a race or out-of-bounds read.
4. Real wall-clock comparison end-to-end on a TQ-quantized load, long-prompt scenario, `INFERO_TQ_PREFILL_DEQUANT=1` vs `=0` — confirming the fix actually resolves the original client-timeout-inducing slowness, not just a kernel microbenchmark.
5. Full `scripts/server_stress_test.py` run against a TQ-quantized load, confirming no regression in the categories that exercise prefill.

## Out of Scope

- Turning TurboQuant on in production. This design only fixes the performance blocker; the separate decision of whether/when to enable `kv_quant != F16` in the real server is not made here.
- vLLM's tier-1 optimization (skip quantization entirely for a brand-new sequence's first chunk). Flagged in Problem as a real, smaller, not-pursued win — no code in this design reads or writes a "not yet quantized" KV representation.
- Any change to `tq_attn_decode`, `tq_attn_scores`, `tq_attn_output`, `tq_store_k`, `tq_store_v`, or the mixed-batch-dispatch-split's `AttnDispatch`/`plan_attn_dispatch`/`run_attn` machinery.
- Multi-GPU/TP interaction (TurboQuant + tensor parallelism is untouched, deferred territory per the existing TP design's own scope).
