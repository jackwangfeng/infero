# Mixed-Batch Attention Dispatch Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Shrink infero's `attn_partial` GPU buffer from 6.05GB to roughly 9MB by making mixed decode+prefill batches dispatch each prefill item through its own single-sequence tile kernel instead of collapsing the whole batch into the generic `attn_decode` kernel.

**Architecture:** Add a `kind` tag to `BatchItem` so decode and prefill items stay distinguishable inside `Model::forward_batch_device`. Partition a batch's items into a decode subgroup (still one `attn_decode` call, since that kernel already supports multiple sequences) and a list of prefill items (each dispatched separately through the existing per-item-eligible tile kernel: `flash_attn2`/`decoupled6`/`ws4`). This relies on the scheduler already emitting decode items before prefill items, so subgroups are contiguous slices of the flat per-token buffers — no gather/scatter. Everything downstream of attention (GDN, FFN, sampling) is untouched: it only cares that every token's row is correctly populated by the time it runs, not how many internal attention calls produced it.

**Tech Stack:** Rust, CUDA (NVRTC-JIT kernels via `infero_kernels`), the existing `infero` workspace (`crates/model`, `crates/server`, `crates/kernels`).

**Spec:** `docs/superpowers/specs/2026-09-05-mixed-batch-attention-dispatch-split-design.md`

## Global Constraints

- Single-GPU only for this plan. Tensor-parallel (`--tensor-parallel-size > 1` / `rank.tp_size > 1`) must not silently run the new path — gate it off and assert loudly instead.
- Feature-flag gated: a new env var `INFERO_SPLIT_MIXED_BATCH`, default **on** (new behavior). Setting it to `"0"` must reproduce today's exact behavior (the existing `single_seq_run`/`attn_decode`-fallback path), unchanged, as a rollback path requiring no rebuild.
- Zero performance regression is a hard requirement for mixed-batch and multi-prefill-batch scenarios — this must be measured (before vs. after, flag on vs. off), never assumed neutral.
- Short prefill remainders (chunks shorter than `MIN_PREFILL_RUN=8` tokens, `crates/model/src/lib.rs:2004`) must stay folded into the decode-style call, exactly as today — do not force every prefill item through its own tile-kernel call regardless of length.
- A wrong buffer-offset or wrong per-item `kv_len` in this change is silent data corruption, not a crash. Every task that touches dispatch logic must be verified with real token-level output diffing and `compute-sanitizer` (`memcheck` + `racecheck`), not just "the server responds."
- Production infero (`bw`, `ssh bw`, port 8301, `--model /home/jeff/models/qwen38-27b-fp8 --ctx 65536 --max-seqs 2`, env `INFERO_FUSE_FFN=0 INFERO_FP8_UNIFIED=1 INFERO_ATTN_MMA=1`) must stay healthy throughout — if you stop it for testing, always relaunch this exact config afterward and verify `/health/live` returns 200 plus a real chat completion before moving on. Git operations happen in the local repo (`/home/jeffwang/work/infero`) and get synced to `bw`'s `/home/jeff/infero` (which has no `.git` of its own) — never attempt `git` commands on the `bw` copy.
- Never `git checkout`/`restore`/`reset` over uncommitted work. Check `git status` before any such command.

---

### Task 1: Capture the pre-change output-diffing baseline

This must run first: once later tasks start changing dispatch code, there's no clean way to get a "before" reference from the *current* binary. This task produces three saved transcripts that Task 7 diffs the post-change binary against.

**Files:**
- Create: `scripts/mixed_batch_baseline.py` (a one-off repro/capture script, not a permanent test — it is read by Task 7, so keep it, don't delete after use)
- Create: `docs/superpowers/plans/mixed_batch_baseline/` (output directory for saved transcripts — gitignored is fine, or commit small JSON files, your call, but keep them reachable by Task 7)

**Interfaces:**
- Produces: three saved JSON transcripts (`scenario_a_single_seq.json`, `scenario_b_mixed_decode_prefill.json`, `scenario_c_two_prefills.json`), each containing the exact request(s) sent, the sampling seed used, and the exact response token IDs (not just text) and `finish_reason` returned by the CURRENT (pre-change) production binary.

- [ ] **Step 1: Confirm production is on the current, unmodified binary**

```bash
ssh bw 'ps aux | grep "[i]nfero"; curl -s -m 5 http://127.0.0.1:8301/health/live -w "\n%{http_code}\n"'
```

Expected: one infero process running `--model /home/jeff/models/qwen38-27b-fp8 --ctx 65536 --max-seqs 2`, health check returns `200`.

- [ ] **Step 2: Write the capture script**

```python
#!/usr/bin/env python3
"""Capture token-level baseline transcripts for the mixed-batch dispatch
split (docs/superpowers/plans/2026-09-05-mixed-batch-attention-dispatch-split.md,
Task 1). Re-run this same script, unmodified, against the POST-change binary
in Task 7 and diff the saved JSON files token-for-token."""
import json
import sys
import time
import threading
import urllib.request

BASE = "http://127.0.0.1:8301"
SEED = 42424242
OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "docs/superpowers/plans/mixed_batch_baseline"

def chat(messages, max_tokens=64, seed=SEED):
    body = json.dumps({
        "model": "qwen38-27b-fp8",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": seed,
        "logprobs": False,
    }).encode()
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())

def save(name, obj):
    import os
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(f"{OUT_DIR}/{name}.json", "w") as f:
        json.dump(obj, f, indent=2)
    print(f"saved {OUT_DIR}/{name}.json")

# Scenario A: pure single-sequence prefill+decode -- exercises the existing
# single_seq_run path, must be untouched by this change.
resp_a = chat([{"role": "user", "content": "Count from 1 to 20, one number per line."}], max_tokens=80)
save("scenario_a_single_seq", resp_a)

# Scenario B: engineered mixed decode+prefill batch. Fire a long prompt
# (forces multi-chunk prefill, several batch_tokens=8192-sized chunks) on one
# connection, and a fraction of a second later fire a short prompt on a
# second connection whose first decode step should land in the SAME
# scheduler batch as one of the long prompt's later prefill chunks.
long_prompt = "Please summarize this list in detail, one sentence per item: " + \
    ", ".join(f"item number {i} is about topic {i%7}" for i in range(4000))
results = {}
def run_long():
    results["long"] = chat([{"role": "user", "content": long_prompt}], max_tokens=40)
def run_short():
    time.sleep(0.05)  # let the long prompt's first chunk get scheduled first
    results["short"] = chat([{"role": "user", "content": "What is 2+2? Answer with just the number."}], max_tokens=8)
t1 = threading.Thread(target=run_long)
t2 = threading.Thread(target=run_short)
t1.start(); t2.start()
t1.join(); t2.join()
save("scenario_b_mixed_decode_prefill", results)

# Scenario C: two simultaneous prefills (two prompts admitted close together,
# both still chunk-prefilling in the same batch).
prompt_x = "Explain photosynthesis in exactly three sentences."
prompt_y = "Explain how a car engine works in exactly three sentences."
results_c = {}
def run_x():
    results_c["x"] = chat([{"role": "user", "content": prompt_x}], max_tokens=60)
def run_y():
    results_c["y"] = chat([{"role": "user", "content": prompt_y}], max_tokens=60)
tx = threading.Thread(target=run_x)
ty = threading.Thread(target=run_y)
tx.start(); ty.start()
tx.join(); ty.join()
save("scenario_c_two_prefills", results_c)

print("done")
```

- [ ] **Step 2: Run it against the current production binary**

```bash
ssh bw 'cd /home/jeff/infero && python3 scripts/mixed_batch_baseline.py docs/superpowers/plans/mixed_batch_baseline'
scp -r 'bw:/home/jeff/infero/docs/superpowers/plans/mixed_batch_baseline' docs/superpowers/plans/
```

Expected: three JSON files saved locally, each with real, non-empty `choices[0].message.content` and a token count consistent with `max_tokens`. Scenario B and C's threaded requests must both return `200` (if either request errors, retry with a slightly longer `time.sleep` in `run_short` until both scheduler slots are genuinely concurrent — check the production log for evidence both requests were in the running set at the same time, e.g. two distinct request IDs active in the same short window).

- [ ] **Step 3: Commit**

```bash
git add scripts/mixed_batch_baseline.py docs/superpowers/plans/mixed_batch_baseline
git commit -m "Capture pre-change baseline transcripts for the mixed-batch dispatch split"
```

---

### Task 2: Verify the real MTP/decode-item token-count ceiling

The spec assumed `attn_partial`'s new size should cover `max_seqs*(mtp_k+1) + MIN_PREFILL_RUN` query tokens. Reading `crates/server/src/scheduler.rs:989` during planning showed `Work::Decode` always constructs a `BatchItem` with **exactly one token** (`tokens: std::slice::from_ref(r.next.as_ref().unwrap())`) — there is no `*(k+1)` multiplier visible at that site. Speculative decoding (`crates/model/src/mtp.rs`) appears to run as a structurally separate pass with its own KV cache and buffers ("the drafter's own single-sequence KV cache", `mtp.rs:164`), which may mean the main model's `attn_decode`/`attn_partial` never sees more than one token per decoding sequence at all. This must be confirmed with real evidence before Task 3 picks a formula — do not carry the spec's `*(mtp_k+1)` assumption forward unverified.

**Files:**
- Read: `crates/server/src/scheduler.rs` (`Work::Decode` construction, `~line 989`, and wherever the scheduler decides how many decode-style items to admit per step, search `plan()` around `line 1662`)
- Read: `crates/model/src/mtp.rs` (`MtpHead`, its `run` method, and how/whether its output ever gets fed back into the MAIN model's `forward_batch_device` `items`/`BatchItem`s as extra tokens for a sequence already past its prompt)
- Read: `crates/model/src/lib.rs` around wherever `MtpHead` is invoked from `Model`'s own forward/step method (search `mtp` in `crates/model/src/lib.rs`)

**Interfaces:**
- Produces: a single constant/expression, written down for Task 3 to consume verbatim — e.g. either confirm "a decode-phase sequence always contributes exactly 1 token to the main model's `items`, regardless of MTP; the real ceiling for the decode subgroup's total token count is `max_seqs` (one per concurrently-decoding sequence), not `max_seqs*(k+1)`" — or, if evidence shows otherwise (e.g. MTP verification really does feed `k+1` tokens per sequence into the main model's batch at some step), write down the real formula and cite the exact code that proves it.

- [ ] **Step 1: Read the real code paths listed above**

Look specifically for: does anything ever construct more than one `BatchItem`, or a `BatchItem` with `tokens.len() > 1`, for a single decoding sequence in one scheduler step? Does `MtpHead::run` get called with its own separate KV pool/buffers entirely outside `Model::forward_batch_device`'s `items` mechanism (in which case it never touches `attn_partial` at all), or does its verification step route back through the main model's normal batch machinery?

- [ ] **Step 2: Write the finding down as a one-paragraph note**

Append it as a comment directly above the `attn_partial` allocation site (`crates/model/src/lib.rs:5231`, before Task 3 changes it) citing the exact evidence (file:line) for whichever formula turns out to be real. This is not a throwaway investigation — Task 3's correctness depends on it, and a future reader of the buffer-sizing code needs the same evidence trail this session already established for `attn_partial`'s original 6.05GB sizing.

- [ ] **Step 3: If genuinely ambiguous after a real reading effort**

Do not guess. Pick the SAFE direction — size for the larger of the two candidate formulas (`max_seqs*2` as a conservative stand-in for "some decode-adjacent step might carry 2 tokens") rather than the smaller — and write down explicitly in the code comment that this is a conservative choice pending further clarification, not a confirmed fact. Getting this wrong in the unsafe direction (too small) causes silent corruption; getting it wrong in the safe direction (slightly too large) merely costs a few extra KB, which is immaterial next to the 6GB this whole change recovers.

No commit for this task alone — its output (the formula + evidence) feeds directly into Task 3's implementation, commit together there.

---

### Task 3: Add `BatchItem::kind` and the `attn_partial` sizing formula

**Files:**
- Modify: `crates/model/src/lib.rs:277` (`BatchItem` struct)
- Modify: `crates/model/src/lib.rs:1267` (`Activations::new` call site) and `:5231-5232` (`attn_partial` allocation)
- Modify: `crates/server/src/scheduler.rs:962` (`Work::Prefill` → `BatchItem` construction) and `:989` (`Work::Decode` → `BatchItem` construction)
- Test: `crates/model/src/lib.rs` (a new `#[cfg(test)]` unit test near `BatchItem`, or `crates/model/tests/` if this crate has an existing integration-test convention — check for one first with `ls crates/model/tests/ 2>/dev/null` before creating a new file)

**Interfaces:**
- Consumes: the real ceiling formula from Task 2's finding.
- Produces: `pub enum BatchItemKind { Decode, Prefill }` (derives at minimum `Clone, Copy, PartialEq, Eq, Debug`), a new `pub kind: BatchItemKind` field on `BatchItem<'a>`, and a `fn attn_partial_bound(max_seqs: usize, /* whatever Task 2 found */) -> usize` helper (name it accurately once Task 2's formula is known) that Task 4 will also read when constructing subgroup calls, to confirm no subgroup call ever asks for more `n_tokens` than this bound allows.

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn attn_partial_bound_matches_config() {
    // Replace the right-hand side with Task 2's real, evidence-backed formula.
    let bound = attn_partial_bound(/* max_seqs */ 2, /* whatever else Task 2 found */);
    assert!(bound >= 2, "decode subgroup ceiling must cover every concurrently-decoding sequence");
    assert!(bound < 64, "if this is anywhere near batch_tokens-scale, the formula is wrong -- \
        this defeats the entire point of the change");
}
```

- [ ] **Step 2: Run it to confirm it fails to compile** (the function doesn't exist yet)

```bash
cargo test -p infero-model attn_partial_bound_matches_config
```

Expected: compile error, `cannot find function attn_partial_bound`.

- [ ] **Step 3: Add `BatchItemKind` and the `kind` field**

At `crates/model/src/lib.rs:277`:

```rust
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BatchItemKind {
    Decode,
    Prefill,
}

pub struct BatchItem<'a> {
    pub seq: SeqId,
    pub kind: BatchItemKind,
    pub tokens: &'a [u32],
    pub wants_logits: bool,
    pub vision: Option<&'a VisionFeatures>, // match the real existing field type here
    pub vision_row_offset: usize,
    pub mrope: Option<&'a [i32]>,
    pub mrope_delta: i32,
}
```

(Check the real types of `vision`/`mrope` fields you're not otherwise touching by reading the current struct before editing — copy them verbatim, do not guess.)

- [ ] **Step 4: Wire the two scheduler construction sites**

At `crates/server/src/scheduler.rs:962`:

```rust
Work::Prefill { from, len, last } => BatchItem {
    seq: r.seq,
    kind: BatchItemKind::Prefill,
    tokens: &r.prompt[*from..*from + *len],
    // ...rest unchanged
```

At `crates/server/src/scheduler.rs:989`:

```rust
Work::Decode => BatchItem {
    seq: r.seq,
    kind: BatchItemKind::Decode,
    tokens: std::slice::from_ref(r.next.as_ref().unwrap()),
    // ...rest unchanged
```

Add `use infero_model::BatchItemKind;` (or the real crate-relative path — check how `scheduler.rs` currently imports `BatchItem`) at the top of `scheduler.rs`.

- [ ] **Step 5: Implement `attn_partial_bound` and wire `Activations::new`**

At `crates/model/src/lib.rs`, near `attn_partial_floats`'s existing call site (`~1267`):

```rust
fn attn_partial_bound(max_seqs: usize /*, whatever Task 2 found */) -> usize {
    // Task 2's real, evidence-backed formula goes here, plus MIN_PREFILL_RUN
    // headroom for a short prefill remainder folded into the decode call.
    max_seqs /* * whatever Task 2 found */ + MIN_PREFILL_RUN
}

let split_mixed_batch = !std::env::var("INFERO_SPLIT_MIXED_BATCH").is_ok_and(|v| v == "0");
let partial_n_tokens = if split_mixed_batch {
    attn_partial_bound(cfg.max_seqs /* real field/method name -- check Config */)
} else {
    batch_tokens
};
```

Update the `attn_partial: alloc_f32(...)` call (`~5231-5232`) to use `partial_n_tokens` instead of `chunk`. Store `split_mixed_batch` as a field on `Model` (next to `attn_backend_name`/`batch_tokens`, same "resolved once at load" pattern) so Task 4 can read it.

Also update the existing log line (`crates/model/src/lib.rs:1263-1264`, the one that already logs `partial_mib`) so it reflects the real new size at startup — this is how tonight's earlier fixes were verified against production logs; keep that verification path alive.

- [ ] **Step 6: Run the test, confirm it passes**

```bash
cargo test -p infero-model attn_partial_bound_matches_config
```

- [ ] **Step 7: Build the full workspace, confirm nothing else broke**

```bash
cargo build --release --features cutlass,flash_attn2,nccl
```

(`BatchItem` gained a required field — every other construction site in the workspace, e.g. any test fixture in `crates/model/tests/` or `crates/server/tests/`, must also set `kind` now. Fix each compile error by setting the obviously-correct `kind` for that call site, not by making the field `Option`/defaulted — an ambiguous default here is exactly the kind of silent-corruption risk this plan exists to avoid.)

- [ ] **Step 8: Commit**

```bash
git add crates/model/src/lib.rs crates/server/src/scheduler.rs
git commit -m "Add BatchItem::kind and size attn_partial off the real decode-only ceiling"
```

---

### Task 4: Verify GDN/FFN/sampling need zero changes

Do this BEFORE Task 5's dispatch rewrite, not after — if this task finds GDN/FFN *does* depend on attention having been dispatched as one unified call, that changes Task 5's scope materially, and Task 5 should not be attempted blind.

**Files:**
- Read: `crates/model/src/lib.rs` (`pool.set_gdn_layout`/`spans`, `~line 1870-1900`; the GDN forward computation inside the 64-layer loop; the FFN/sampling code downstream of the attention branch, `~line 3800` onward until the layer loop's end)

**Interfaces:**
- Produces: a written finding (as a code comment near the GDN dispatch site, and a note added to this plan's Task 5 section if anything changes) confirming — with a specific mechanism cited, not "looks fine" — that GDN/FFN/sampling read `self.act.x`/`self.act.attn`/token-to-sequence maps (`seq_of`, `slots`) purely by absolute row position, with no dependency on how many attention kernel calls populated those rows or in what order.

- [ ] **Step 1: Trace what GDN reads that attention wrote**

Find where, after the attention branch (`~3763` onward), the code proceeds to `l.attn().wo` (output projection) and then, for GDN layers, the `GdnActs` buffers. Confirm the output-projection matmul and everything after it reads `attn_out`/`self.act.attn` purely as `[chunk, d_attn]` rows indexed by absolute batch position — i.e., row `i` is token `i` of the flat batch, full stop, regardless of which attention kernel call wrote it.

- [ ] **Step 2: Check `pool.set_gdn_layout`/`spans` specifically**

Read what `spans` (`vec![(0i32, 0i32); pool.max_seqs()]`, `~line 1875`) represents and confirm it's derived from `items`/`seq_of` directly (i.e., independent of the attention dispatch mechanism) rather than from anything attention-branch-specific.

- [ ] **Step 3: Write the finding down**

If confirmed (expected outcome): add a one-line comment at the top of the GDN forward block, e.g. `// Independent of how attention was internally dispatched (see docs/superpowers/specs/2026-09-05-mixed-batch-attention-dispatch-split-design.md) -- reads self.act.attn purely by absolute row position.` This is cheap insurance for the next person who touches this code.

If NOT confirmed: STOP. Do not proceed to Task 5. Write up exactly what GDN/FFN/sampling actually depends on, and treat this as new information requiring the spec/plan to be revised (this is real, valuable negative information — report it rather than forcing Task 5 through against a false premise, matching this session's established discipline).

No commit needed unless Step 3 adds the comment (small, can fold into Task 5's commit).

---

### Task 5: Implement the dispatch-flow partition and loop

**Files:**
- Modify: `crates/model/src/lib.rs` (`~1811` area, the `kv_len` computation loop; `~3571-3763`, the `BatchLayout` construction and attention dispatch branch, inside the 64-layer loop)
- Test: same test file/location as Task 3

**Interfaces:**
- Consumes: `BatchItem::kind` (Task 3), `split_mixed_batch` field on `Model` (Task 3), the confirmed GDN/FFN independence (Task 4).
- Produces: the actual restructured dispatch — decode items batched into one `attn_decode` call, each prefill item dispatched via its own tile-kernel call — gated by `split_mixed_batch`.

- [ ] **Step 1: Compute per-item real `kv_len`, not just the batch-wide max**

The existing loop (`~1771-1818`) computes ONE shared `kv_len = kv_len.max(start + item.tokens.len())` across every item in the batch. This value is correct for `attn_decode` (which already handles heterogeneous sequences sharing one call) but is WRONG to reuse as-is for a per-item prefill call once items are split apart: item A's own real KV extent is `starts[A.seq.0].1 + A.tokens.len()`, which may be smaller than the batch-wide max if some OTHER item in the batch (e.g. a longer-running decode sequence) has a larger position. Add, alongside the existing `starts: Vec<(usize, usize)>` (flat-batch start offset, prior sequence length):

```rust
// Per-item real kv_len: this item's own sequence extent, NOT the batch-wide
// max computed just below for `attn_decode`'s shared use. Needed once
// prefill items are dispatched individually (see docs/superpowers/specs/
// 2026-09-05-mixed-batch-attention-dispatch-split-design.md).
let mut item_kv_len: Vec<usize> = Vec::with_capacity(items.len());
```

populated inside the same per-item loop (`~1777-1818`) as `item_kv_len.push(start + item.tokens.len())` right where the shared `kv_len` is updated.

- [ ] **Step 2: Write a targeted correctness test proving the per-item kv_len is used correctly**

Before touching the dispatch branch itself, write a test that would fail under the OLD "reuse the shared max `kv_len` for every subgroup" approach and pass under the correct per-item approach. The cleanest real test: construct a batch with one decode item for a sequence at a LARGE position (e.g. a sequence that's generated 500 tokens already) and one prefill item for a DIFFERENT, freshly-started sequence (position 0, prefilling its first 64 tokens). Run a real forward pass and confirm the prefill item's output does not depend on the unrelated decode sequence's position/length — e.g. compare its output against the SAME prefill run in isolation (no concurrent decode item at all). They must match exactly. (This is a real integration-level test; check `crates/model/tests/` for the existing convention for constructing a `Model` + `KvPool` fixture before writing a new one from scratch.)

- [ ] **Step 3: Run the test, confirm it fails against the current (pre-split) dispatch** — expected, since the split doesn't exist yet; this establishes the test is actually exercising the real hazard, not vacuously passing.

- [ ] **Step 4: Partition items and dispatch, gated by `split_mixed_batch`**

Replace the single dispatch branch (`~3661-3763`, inside the 64-layer loop) with, when `self.split_mixed_batch` is true:

```rust
if self.split_mixed_batch {
    let decode_count: usize = items.iter().filter(|i| i.kind == BatchItemKind::Decode).count();
    // Decode items are always the contiguous prefix (scheduler.rs's plan()
    // fills decode items before prefill items) -- verify this invariant
    // rather than trust it silently:
    debug_assert!(
        items.iter().take(decode_count).all(|i| i.kind == BatchItemKind::Decode)
            && items.iter().skip(decode_count).all(|i| i.kind == BatchItemKind::Prefill),
        "scheduler ordering invariant violated: decode items must precede prefill items"
    );
    let decode_tokens: usize = items.iter().take(decode_count).map(|i| i.tokens.len()).sum();

    if decode_count > 0 {
        let decode_kv_len = item_kv_len[..decode_count].iter().copied().max().unwrap();
        let decode_batch = BatchLayout {
            seq_of: &seq_of[..decode_tokens],
            positions: &batch_positions[..decode_tokens],
            slot_table: &table,
            table_stride,
        };
        attn_f16 = self.kern.attn_decode(
            &mut attn_out.slice_mut(..decode_tokens * da),
            wo_f16.then_some(&mut h16.slice_mut(..decode_tokens * da)),
            &self.act.q.slice(..decode_tokens * da),
            &pool.dense(layer).0.as_view(),
            &pool.dense(layer).1.as_view(),
            decode_batch,
            dims,
            decode_kv_len,
            attn_scale,
            &mut partial.as_view_mut(),
        )?;
    }

    let mut offset = decode_tokens;
    for (idx, item) in items.iter().enumerate().skip(decode_count) {
        let item_tokens = item.tokens.len();
        let item_batch = BatchLayout {
            seq_of: &seq_of[offset..offset + item_tokens],
            positions: &batch_positions[offset..offset + item_tokens],
            slot_table: &table,
            table_stride,
        };
        let this_kv_len = item_kv_len[idx];
        // Reuse the SAME per-item eligibility logic the old single_seq_run
        // path used (flash_attn2 contiguity check, d_head==256 for
        // decoupled6, INFERO_PREFILL_T6 rollback, ws4 fallback) -- apply it
        // per item here instead of once for the whole (necessarily
        // single-item) batch. Do not duplicate that eligibility logic by
        // hand; factor it into a small helper both the old and new paths
        // call, so a future change to the eligibility rules can't silently
        // diverge between them.
        self.dispatch_single_item_prefill(
            &mut attn_out.slice_mut(offset * da..(offset + item_tokens) * da),
            &self.act.q.slice(offset * da..(offset + item_tokens) * da),
            item_batch,
            dims,
            item_tokens,
            this_kv_len,
            attn_scale,
            &mut partial.as_view_mut(),
            layer,
        )?;
        offset += item_tokens;
    }
} else {
    // ...existing single-dispatch code, unchanged, exactly as it reads today
}
```

Write `dispatch_single_item_prefill` as a small private method on `Model` that contains exactly the `vendor_backend_run`/`prefill_run`-style eligibility chain already at `~3645-3763` today, parameterized by the item's own `run_tokens`/`kv_len`/`batch` instead of the batch-wide ones — refactor the OLD single-item path (the `else` branch above) to also call this same helper, so there is exactly one copy of the eligibility logic, not two that can drift apart.

(The exact parameter list above is illustrative — match it to the real signatures of `attn_decode`/`flash_attn2_backend.prefill`/`attn_prefill_decoupled6_f16acc`/`attn_prefill_ws4` as they exist in the current source; read them fresh before writing this, don't copy the illustrative snippet verbatim if a real field/type differs.)

- [ ] **Step 5: Run the Step 2 test, confirm it now passes**

- [ ] **Step 6: Add the `INFERO_SPLIT_MIXED_BATCH=0` rollback check**

Add a test (or extend the Step 2 test) that runs the SAME scenario with `INFERO_SPLIT_MIXED_BATCH=0` set and confirms it reproduces the OLD behavior bit-for-bit against a captured pre-change reference (this can reuse Task 1's harness pattern against a locally-built test binary rather than production).

- [ ] **Step 7: Build the full workspace**

```bash
cargo build --release --features cutlass,flash_attn2,nccl
```

- [ ] **Step 8: Commit**

```bash
git add crates/model/src/lib.rs
git commit -m "Split mixed-batch attention dispatch: batched decode + per-item prefill calls"
```

---

### Task 6: TP gating

**Files:**
- Modify: `crates/model/src/lib.rs` (wherever `split_mixed_batch` is resolved at load, Task 3's site)
- Modify: `crates/model/src/tp.rs` or wherever `RankId`/`rank.tp_size` is available at model-construction time (check `crates/model/src/lib.rs`'s `Model::new`/`Model::load`-equivalent constructor for where TP rank info is already threaded through, e.g. near `shard_for_tp`'s call site)

**Interfaces:**
- Consumes: `split_mixed_batch` (Task 3).
- Produces: `split_mixed_batch` forced to `false` whenever `tp_size > 1`, regardless of the env var.

- [ ] **Step 1: Write the failing test**

```rust
#[test]
#[cfg(feature = "nccl")]
fn split_mixed_batch_disabled_under_tp() {
    // Construct (or fake) a RankId with tp_size=2, confirm the resolved
    // split_mixed_batch is false even with INFERO_SPLIT_MIXED_BATCH unset/"1".
    // Match this to however Model's constructor actually threads tp_size --
    // read the real signature before writing this test, don't guess it.
}
```

- [ ] **Step 2: Run it, confirm it fails** (the gating doesn't exist yet — either a compile error if the test references something not yet built, or a logical failure).

- [ ] **Step 3: Implement the gating**

At Task 3's resolution site:

```rust
let split_mixed_batch = !std::env::var("INFERO_SPLIT_MIXED_BATCH").is_ok_and(|v| v == "0")
    && rank.tp_size <= 1; // see docs/superpowers/specs/2026-09-05-mixed-batch-attention-dispatch-split-design.md's Scope section -- v1 is single-GPU only
```

(Match `rank`'s real name/availability at this point in the constructor — it may already be a parameter, or need threading in from wherever `shard_for_tp` is called.)

- [ ] **Step 4: Run the test, confirm it passes**

- [ ] **Step 5: Build**

```bash
cargo build --release --features cutlass,flash_attn2,nccl
```

- [ ] **Step 6: Commit**

```bash
git add crates/model/src/lib.rs
git commit -m "Gate the mixed-batch dispatch split off under tensor parallelism (v1: single-GPU only)"
```

---

### Task 7: Full verification against production

**Files:**
- Read/run only — no source changes expected in this task (unless verification surfaces a real bug, in which case fix it, re-run this task's steps from the top, and note the fix in the commit message).

**Interfaces:**
- Consumes: Task 1's baseline transcripts, the built binary from Tasks 3/5/6.

- [ ] **Step 1: Build the release binary on `bw`**

```bash
ssh bw 'cd /home/jeff/infero && cargo build --release --features cutlass,flash_attn2,nccl 2>&1 | tail -50'
```

Expected: clean build, no warnings about `BatchItem` field mismatches anywhere.

- [ ] **Step 2: Token-level output diff against Task 1's baseline**

Stop the current production instance (note its exact launch command first), start a fresh instance on a free test port (e.g. 8302) with the new binary, `INFERO_SPLIT_MIXED_BATCH=1` (default), same model/config. Run `scripts/mixed_batch_baseline.py` against port 8302, diff its three output JSON files against Task 1's saved ones field-by-field (`choices[*].message.content`, and specifically the underlying token IDs if the API surfaces them — check whether `logprobs`/token IDs are exposed, and if not, temporarily enable whatever debug flag/log line gives you real token IDs rather than accepting text-only comparison for this one verification pass).

Expected: scenario A (single-seq) matches exactly — this path is unchanged when `items.len()==1` degenerates to the same one-item case Task 5's loop handles trivially. Scenarios B and C must also match exactly (or, if any kernel-to-kernel numerical difference is expected and justified — check whether `attn_decode` and `decoupled6` are documented to agree only approximately for equivalent inputs — document the exact tolerance and why it's acceptable; do not accept an unexplained divergence).

- [ ] **Step 3: `compute-sanitizer` on the mixed-batch scenarios**

```bash
ssh bw 'cd /home/jeff/infero && compute-sanitizer --tool memcheck ./target/release/infero --model /home/jeff/models/qwen38-27b-fp8 --host 127.0.0.1:8303 --ctx 65536 --max-seqs 2 &'
# then run scenario B and C's requests against port 8303, then check the sanitizer output for errors
ssh bw 'cd /home/jeff/infero && compute-sanitizer --tool racecheck ./target/release/infero --model /home/jeff/models/qwen38-27b-fp8 --host 127.0.0.1:8304 --ctx 65536 --max-seqs 2 &'
# same requests against port 8304
```

Expected: zero errors from both tools on both scenarios. If anything appears, STOP — do not proceed to production rollout with an unresolved sanitizer finding, per this session's established `feedback_stack_overflow_evades_memcheck`/`feedback_run_sanitizer_before_math_rederivation` lessons (a clean sanitizer run is necessary but not sufficient on its own — also sanity-check that `item_kv_len`/offset arithmetic by hand for the specific scenario B/C shapes you ran).

- [ ] **Step 4: Full stability stress test**

```bash
ssh bw 'cd /home/jeff/infero && python3 scripts/server_stress_test.py --base-url http://127.0.0.1:8302 --passes 5'
```

Expected: all 13 categories STABLE (the previously-tracked `multiturn_2` model-sampling variance from commit `238f2e1` is a known, unrelated exception — anything else flaky is a new regression to investigate, not to wave off).

- [ ] **Step 5: Real before/after VRAM**

```bash
ssh bw 'nvidia-smi --query-compute-apps=pid,used_memory --format=csv'
```

Expected: the new instance's `partial_mib` (check its startup log) reads Task 2/3's computed small value (roughly single-digit MiB), and total per-process VRAM is reduced by roughly 6GB versus a same-config instance built from the pre-change binary.

- [ ] **Step 6: Real performance A/B**

Run a mixed-batch-heavy load pattern (reuse scenario B's concurrent-request shape, repeated many times, or extend `scripts/server_stress_test.py`'s `concurrent_at_capacity`/`concurrent_over_capacity` categories) against: (a) the new binary with `INFERO_SPLIT_MIXED_BATCH=1`, (b) the new binary with `INFERO_SPLIT_MIXED_BATCH=0`. Measure real end-to-end latency/throughput for both. Confirm (a) is not measurably slower than (b) — if it is, this is a real regression against the Global Constraints' hard requirement; do not ship until resolved (either optimize the per-item dispatch loop, or if truly unresolvable within reasonable effort, report this back rather than shipping a regression).

- [ ] **Step 7: Restore production to the verified new binary**

```bash
ssh bw 'pkill -f "target/release/infero.*8301" || true; sleep 2'
ssh bw 'cd /home/jeff/infero && CUDA_VISIBLE_DEVICES=3 INFERO_FP8_UNIFIED=1 INFERO_ATTN_MMA=1 INFERO_FUSE_FFN=0 setsid nohup ./target/release/infero --model /home/jeff/models/qwen38-27b-fp8 --host 127.0.0.1:8301 --ctx 65536 --max-seqs 2 > /tmp/infero_27b_live.log 2>&1 < /dev/null & disown; sleep 3'
ssh bw 'curl -s -m 5 http://127.0.0.1:8301/health/live -w "\n%{http_code}\n"'
ssh bw 'curl -s -m 30 http://127.0.0.1:8301/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"qwen38-27b-fp8\",\"messages\":[{\"role\":\"user\",\"content\":\"用一句话介绍一下你自己\"}]}"'
```

Expected: health check `200`, real coherent chat completion. Clean up any leftover test instances on ports 8302/8303/8304.

- [ ] **Step 8: Final commit (if Step 2-6 surfaced any fix) and summary**

```bash
git status
git log --oneline -10
```

If any fix was needed during verification, it should already be committed as part of whichever task's step it belongs to — this step just confirms `git status` is clean and the log tells a coherent story. No new commit needed if nothing changed here.
