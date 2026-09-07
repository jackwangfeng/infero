# VRAM investigation — real finding, real fix, committed but NOT deployed

**Directive:** user felt production's VRAM capacity "still isn't right" before going to sleep; investigate with real data, fix if found, don't restart production without them.

## The real, confirmed finding

Production's startup log (`/tmp/infero_27b_live.log` on `bw`) has always shown two identical log lines back to back:

```
kv pool allocated quant=f16 mib=32768 n_slots=131072 max_seqs=2 max_seq=65536
recurrent state allocated linear_layers=48 mib=299 per_seq_mib=149
kv pool allocated quant=f16 mib=32768 n_slots=131072 max_seqs=2 max_seq=65536
recurrent state allocated linear_layers=48 mib=299 per_seq_mib=149
```

Earlier this session's memory attributed this to "MTP needs a second, shadow KV pool for rollback." **That explanation was never actually verified against the code, and it's wrong.** The real cause: `crates/server/src/scheduler.rs`'s `make_pool` bisects the pool's slot count by calling a `fits(n)` closure that constructs a full trial `KvPool` just to read its byte count, then discards it. Whenever the very first check (`fits(hi)`, the full requested concurrency) already fits — the common case whenever the card has enough free VRAM, which it does here — the function immediately built an **identical second pool** to actually return. Confirmed by reading the only two call sites of `KvPool::new` (`crates/model/src/cache.rs:203`) and `make_pool`'s real logic (`scheduler.rs:1936-1988` before this fix) — there is no separate MTP-shadow-pool code path anywhere; both log lines came from the same `make_pool` call.

## The fix

`crates/server/src/scheduler.rs`, commit `5cdf17e`: `fits` now returns the constructed `KvPool` itself (`Option<KvPool>`) instead of a bare `bool`, so the fast path and the bisection loop both reuse the pool that already proved it fits, rather than throwing it away and rebuilding an identical one. Same `lo`/`hi`/`mid` arithmetic throughout — this changes only how many times `KvPool::new` runs, never which pool size gets chosen.

**Verified with a real RED/GREEN test**, not just code reading: `crates/server/tests/make_pool_allocation_count.rs` calls `make_pool` once and reads the resulting pool's `id()` (a process-wide counter incremented on every real `KvPool::new`, including discarded trials), calls it again, and asserts the `id()` delta is exactly 1. Ran it against the code as found (git-stashed the fix): **real failure, delta=2**. Restored the fix: **real pass, delta=1**. Full `infero-server` suite (45 lib tests + `pool_admission.rs`'s 2 integration tests) still green — this is a pure allocation-count change, not a behavior change.

## What this does NOT fully resolve — honest accounting

Freed GPU memory is not necessarily handed back to the driver immediately by CUDA's allocator, so this bug was a real, confirmed extra allocation on every real startup — but I have **not restarted production to measure the real before/after total**, per the explicit instruction not to touch port 8301 while the user is asleep. So I cannot yet tell you the fix's real steady-state VRAM impact; that requires an actual restart, which is the user's call to make when they're back (`kill <pid>` then relaunch with the exact same command logged at `/tmp/infero_27b_live.log`'s own startup lines — same recipe this session used twice already tonight for the FA2/vLLM comparison work).

**The arithmetic still has an unexplained gap even accounting for this fix.** Summing every component the startup log actually reports (weights 28307 + vision tower 884 + vision scratch 340 + kv pool 32768 + recurrent state 299 + mtp head 739 + speculative journal ~25 + attn_partial 7 ≈ **63369 MiB**) against the real `nvidia-smi` total (**69426 MiB**) leaves **~6057 MiB unaccounted for** by anything the log names. Candidates I did not have budget to run down: CUDA context overhead (typically hundreds of MiB, not usually multiple GB, so probably not the whole gap on its own), the allocator's own caching/fragmentation behavior, or real `Activations`/`Scratch` buffers (`crates/model/src/lib.rs`) that are sized and allocated at load time but never get their own `tracing::info!` line — i.e., real VRAM the log simply doesn't itemize, not evidence of a leak. I did not find a second bug explaining this gap; I'm reporting it honestly rather than papering over it with the one fix I did confirm.

## The CUDA-graph-growth question (a separate, earlier-flagged, still-open concern)

Checked real `nvidia-smi` for the current production process across every point I touched it tonight (multiple times, across the FA2-followup work and this investigation): **69426 MiB every single time, no growth observed.** This doesn't rule out growth under different/heavier real traffic than tonight's, but within tonight's own session, production's VRAM has been flat, not climbing.

## Bottom line

- Real bug found, real fix committed (`5cdf17e`), real RED/GREEN test, full suite green — safe by construction (pure allocation-count change).
- **Not deployed.** Production is still running the old binary; this fix takes effect on the next restart, which I deliberately left for the user to trigger (or explicitly authorize) rather than doing myself overnight.
- The earlier "MTP shadow pool" theory in long-term memory should be corrected/removed — this investigation found the real mechanism and it isn't that.
- A real ~6GB gap between logged components and the real total remains open; not fixed, not fully explained, flagged honestly rather than guessed at.
- No evidence of runaway growth from CUDA graph accumulation tonight specifically, though that question was only checked over this session's own timeframe, not a longer real-traffic window.
