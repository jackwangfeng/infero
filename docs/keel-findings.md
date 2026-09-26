# What Keel found while switching onto infero

Companion to [`keel-integration.md`](./keel-integration.md), which said what Keel
needed. This one says what happened when Keel actually ran on it.

**The switch succeeded.** Keel's embedding path now runs entirely on infero:
7.6 ms single-text (median of 15, warm) against 62 ms on the Python/CPU service
it replaced, 1.47 ms/text at batch 64, and a 1-second cold start against 75
seconds. Its three acceptance criteria — vectors come back normalized, semantic
margin clears a floor, a dead engine raises rather than silently returning zeros
— all pass.

Five defects turned up on the way. One of them is serious and is currently
worked around on the Keel side; the rest are smaller but two of them quietly
degrade a signal Keel depends on. All of them were measured, not inferred.

---

## 1. The tokenizer ignores the checkpoint's own `post_processor`

**Severity: high. This silently degrades embedding quality.**

`Qwen3-Embedding-0.6B/tokenizer.json` carries a `post_processor` that appends
`<|endoftext|>` (id 151643) to every sequence:

```json
{"type": "Sequence", "processors": [
  {"type": "ByteLevel", ...},
  {"type": "TemplateProcessing",
   "single": [{"Sequence": {"id": "A", "type_id": 0}},
              {"SpecialToken": {"id": "<|endoftext|>", "type_id": 0}}],
   "special_tokens": {"<|endoftext|>": {"ids": [151643], ...}}}]}
```

That trailing EOS is not decoration. Qwen3-Embedding pools **the last token's**
hidden state, and the checkpoint was trained with EOS occupying that position.

infero has no concept of a post-processor at all: `crates/tokenizer/src/lib.rs`'s
`encode(&self, text, add_bos, parse_special)` does BPE plus explicit special-token
parsing and stops there, and `grep -rn post_processor --include=*.rs` returns
nothing across the whole repository. So pooling reads the hidden state of the
last *content* character instead.

### Measured

Three Chinese probes chosen so that the near pair and the far pair **share no
characters** — the comparison is purely semantic. Deterministic: three runs,
spread 0.00.

| | near-pair cos | far-pair cos | margin | |
|---|---|---|---|---|
| A · as-is (infero today) | 0.3380 | 0.2419 | **+0.0960** | below Keel's 0.10 floor |
| B · instruction prefix only | 0.3283 | 0.2326 | **+0.0958** | prefix hypothesis eliminated |
| C · EOS sentinel only | 0.4802 | 0.1884 | **+0.2917** | passes |
| D · prefix + sentinel | 0.4686 | 0.2348 | +0.2338 | worse than C |

The margin **triples** when the EOS token is present. B is indistinguishable
from A, which rules out the obvious competing explanation (that Qwen3-Embedding
needs an `Instruct:`/`Query:` prefix). D is worse than C because these probes
are document-vs-document comparisons, where a query-side prefix does not belong.

### Why infero's own test suite does not catch this

`crates/model/tests/embed.rs` calls the same `tokenizer.encode(t, None, true)`,
so it is subject to the same bug — but its three probes are
`"a red running shoe for men"`, `"...for women"`, `"a bag of coffee beans"`.
The first two differ by one word. Run through the same four variants, **the
as-is case already scores +0.5319**: lexical overlap carries the signal whether
or not pooling reads the right position. The probe has no discriminating power
for this defect, and the 0.53 quoted in the docs was measured with the bug
present.

A probe that can see this needs pairs whose semantic relationship is *not*
mirrored by surface overlap.

### Current workaround, and what has to happen when this is fixed

Keel appends `<|endoftext|>` to every text itself, in `postBatch` in
`internal/inference/client.go` — one function, so the index side and the query
side share a single concatenation and cannot drift apart. The constant is named
`PoolingSentinel` and its doc comment states the deletion condition.

**When infero starts running the post-processor, Keel's sentinel must be
deleted in the same change**, or every sequence gets two EOS tokens. Worth
naming `PoolingSentinel` explicitly in whatever issue tracks this.

---

## 2. `derive_model_id` truncates model names at the first dot

**Severity: medium. Wrong data, already persisted.**

`crates/server/src/engine.rs:761`:

```rust
fn derive_model_id(path: &str, name: &str) -> String {
    std::path::Path::new(path).file_stem()
```

`file_stem()` on `/mnt/data/tuili-models/Qwen3-Embedding-0.6B` treats `.6B` as
an extension and yields **`Qwen3-Embedding-0`**. The reranker is reported as
`Qwen3-Reranker-0` for the same reason.

Version-number-bearing directory names are the norm for HF checkpoints
(`-0.6B`, `-1.5B`, `-2.7B`, `Qwen2.5-...`), so this misfires on most of them.

Keel records whatever the engine actually says, so `product_text_vectors.
model_name` is `Qwen3-Embedding-0` in the live database today — a gate rejects
writing a prettier name than the engine reported. **When infero fixes this,
that gate will go red immediately and point at the row**, which is the correct
outcome: `model_name` really did change, and the stored vectors really do need
recomputing.

---

## 3. `model_version` collapses to the model id for hand-placed checkpoints

**Severity: medium. Not a bug — a documented fallback — but Keel loses a signal.**

`derive_model_version` returns the HF snapshot sha when `--model` resolves
inside a `snapshots/<sha>/` path, and otherwise falls back to the id. The code
says so, and says why. But a checkpoint placed by hand (which is how this one is
deployed) has no sha, so `model_version == model_id`.

The consequence downstream: `product_text_vectors.model_version` **cannot answer
"the weights changed under the same name, do I need to recompute?"** Swap the
files in place and every stored row still looks current. Keel's staleness
decision therefore rests on `model_name` alone.

A cheap improvement that would work for any layout: hash the weight files'
sizes and mtimes, or the safetensors header. Not a sha, but it changes when the
weights change, which is the whole job.

---

## 4. `--help` describes `--model` as a GGUF file; it also takes a directory

**Severity: low. Documentation.**

`--help` says "Path to a GGUF model file". `main.rs:165` loads a directory as
safetensors, which is the only path that works for this model.

Related, and worth a line in the docs: **the official
`Qwen3-Embedding-0.6B` GGUF does not load.** infero rejects it with
`d_head 128 * n_heads 16 != d_model 1024`. So for this model safetensors is not
merely supported, it is mandatory — the opposite of what `--help` implies.

---

## 5. The embedding batch cap is `2 × --max-seqs`, not a constant 64

**Severity: low. Documentation, with a real failure mode behind it.**

`MAX_BATCH` is documented as 64, but the effective ceiling is the scheduler's,
measured at twice `--max-seqs`. Running with `--max-seqs 8` and submitting 64
texts fails with `64 sequences want logits, the limit is 16`.

So the documented 64 only holds at `--max-seqs >= 32`. Keel pins the value in
its launch script and asserts the ceiling, but a caller reading `MAX_BATCH`
would get a runtime error instead of a 400.

---

## What Keel did *not* find wrong

Worth recording, since the list above reads bleaker than the experience was.

- **`/v1/embeddings` matches the contract exactly** — batched arrays in,
  `dim` + `model` + `model_version` out, hard 400 above the cap.
- **Normalization is real.** Recomputing `sqrt(sum(x*x))` from Postgres after a
  full round-trip gives 0.999999997 / 0.999999998 / 0.999999999.
- **Failures fail loudly.** Killing the port produces an error and zero vectors
  written — no silent zero-fill. This was confirmed accidentally as well as
  deliberately: a stale container still calling `/v1/embed` got a 404, and not
  one zero vector reached the database.
- **`/v1/rerank` scores track the query, not the document.** "夏天穿的连衣裙"
  → dress 0.9931, coffee pot 0.000021, tyre 0.000014; "手冲咖啡用的器具"
  → coffee pot 0.9988 first. Keel does not consume it until M5.
- **Cold start is 1 second**, against 75 for the Python service it replaced.

---

## Suggested order

1. **§1**, the post-processor. It is the only one that degrades output quality,
   and Keel is carrying a workaround for it. Coordinate the fix with removing
   `PoolingSentinel`.
2. **§2**, `file_stem()`. One line, and it is writing wrong values into a
   downstream database right now.
3. **§3**, a weight fingerprint for non-HF-cache layouts.
4. **§4 and §5**, documentation.
