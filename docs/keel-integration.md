# What Keel needs from infero

[Keel](https://github.com/jackwangfeng/keel) is a multi-tenant e-commerce system
whose AI layer talks to exactly two inference endpoints. Today those are served by
a throwaway Python service (FastAPI + sentence-transformers + torch CPU, 1.39 GB
image) running BGE-M3. This file records what infero would have to grow for Keel
to drop that service, and — more usefully — **which of those things are worth
building for infero's own sake and which are not.**

Written from the Keel side at the end of its M3 milestone. Numbers here are
measured, not estimated; where something is an inference it says so.

## The contract between them is two endpoints

Keel's design pins this down as "the only coupling point between the AI layer and
the engine" — swap the engine, business code does not move:

```
POST /v1/embed
  { "model": "...", "texts": ["..."], "normalize": true }
  → { "embeddings": [[...]], "dim": 1024, "model": "...", "model_version": "..." }

POST /v1/rerank
  { "model": "...", "query": "...", "documents": ["...", "..."], "top_k": 30 }
  → { "results": [{ "index": 3, "score": 0.91 }, ...] }
```

The path names are negotiable (an OpenAI-shaped `/v1/embeddings` is fine, Keel
adds a thin adapter). Two things are not: **requests are batched arrays**, and
**responses carry the model version**.

---

## 1. `/v1/embeddings` — the one that unblocks a real swap

**Implemented.** `POST /v1/embeddings` takes exactly the shape this file
specifies — `{"texts": [...], "model": "...", "normalize": true}` in,
`{"embeddings": [[...]], "dim": ..., "model": "...", "model_version": "..."}`
out — with a hard 400 above the 64-text batch cap
(`infero_model::embed::MAX_BATCH`). `normalize` is accepted for shape
compatibility but not honoured as a toggle: see "L2 normalization owned by
the server" under "What this needed" below for why.

Real, measured against `Qwen/Qwen3-Embedding-0.6B` (the checkpoint this
section's dimension argument is about, loaded as a plain safetensors
directory the same way an AWQ/FP8 checkpoint is):

- Every returned row is `dim`-wide and L2-normalized to within `1e-3` of 1.0
  (checked, not assumed — see below).
- This file's own acceptance test #2, run for real rather than described:
  `"a red running shoe for men"` / `"...for women"` / `"a bag of coffee
  beans"` scored a near–near cosine of 0.826 against near–far cosines of
  0.294 and 0.286 — a margin of **0.53**, more than 5x this file's own 0.10
  bar.
- The same text embedded alone and as part of a 3-text batch agrees to
  4-5 significant figures, not bit-for-bit — the same batch-width-dependent
  kernel selection this repo's own README documents for logits (`README.md`'s
  "Batch invariance, precisely" note): one row takes the integer mat-vec,
  more take the tensor-core GEMM, and the two sum over `k` in different
  orders. Real, expected, and not something this endpoint's contract
  promises against — batch invariance is asserted for chat completions,
  not claimed here.

Test coverage: `crates/model/tests/embed.rs`, gated on
`INFERO_TEST_EMBEDDING_SAFETENSORS` pointing at a real checkpoint directory
(skips cleanly when unset, same convention every other real-checkpoint test
in this repo uses).

### Why the dimension matters more than anything else

Keel stores vectors in `product_text_vectors.embedding vector(1024)`. In pgvector
the dimension is part of the **column type**, so changing models means changing
DDL, writing a migration, and re-embedding the whole catalog.

**Qwen3-Embedding-0.6B is natively 1024-dim and is a Qwen3 dense decoder** — the
architecture family infero already runs, and inside what the Metal backend covers
(dense decode + GQA). So the single most expensive part of switching engines is
free.

**One real compatibility gap this surfaced, not anticipated by the paragraph
above:** Qwen3-Embedding-0.6B's only published checkpoint is repackaged for the
`sentence-transformers` library, not left in `AutoModelForCausalLM`'s own
layout — its real tensor names are `embed_tokens.weight` / `layers.0...` /
`norm.weight`, with no `model.` prefix at all (confirmed against the real
safetensors header, not assumed), because that export strips the wrapper
module and the `lm_head` an embedding model has no use for. infero's loader
originally recognized only `model.embed_tokens.weight` and
`model.language_model.embed_tokens.weight`; it now also tries a bare
`embed_tokens.weight` (`crates/model/src/weights.rs`'s `stem_join`/`stem`
probe). Anyone loading a *different* embedding checkpoint should check which
layout it actually ships before assuming this "just works" — it worked here
because this specific probe was added for this specific real file, not
because embedding checkpoints in general are guaranteed to look like this.

### What this needed (done)

Embedding is not generation. The forward pass already existed; this added:

- **Exposing the final hidden state without going through `lm_head`, and
  without sampling.** `Model::hidden_states_host` reads `act.xb` — the exact
  post-`output_norm` vector the `lm_head` dispatch chain would otherwise
  consume — instead of running that projection at all.
- **Last-token pooling** (what Qwen3-Embedding uses) — not new plumbing:
  `BatchItem::wants_logits`'s existing per-item last-token row selection
  already computes this, so pooling fell out of reusing it rather than a
  masking/padding scheme of its own. No left padding needed, either: infero's
  batching is already ragged (each item carries its own token slice), so
  every item's true last token is just `tokens.len() - 1` with nothing to
  mask.
- **L2 normalization owned by the server, with a self-check.** See below.

That last point is not pedantry — it comes from a real bug on the Keel side.
BGE-M3's `modules.json` already contains a `Normalize` module, which means the
`normalize: true` request parameter **does nothing**: flipping it to `false` and
rebuilding still produced unit vectors (measured: 0.99999999 / 1.00000003 /
1.00000004). A guarantee that lives in some upstream repo's config **fails
silently** the day you change models. So `infero_model::embed::normalize`
divides by the norm itself and asserts the result is within `1e-3` of 1.0,
rather than trusting the model card — and the HTTP layer does not expose a
way to turn that off (see the endpoint's own request-shape note above).

### Operational requirements (done)

- **Batch**, 32–64 per request, with a hard error above the cap. Callers must not
  loop single texts. Implemented as a fixed 64-text cap
  (`infero_model::embed::MAX_BATCH`), checked in the HTTP handler before the
  request ever reaches the model.
- **`model` and `model_version` in every response.** Keel writes both into
  `product_text_vectors` and uses them to answer "does this row need
  recomputing". **Correction to this file's own earlier claim:** infero's
  health endpoint is `/health`, not `/healthz`, and at the time this
  paragraph was first written it returned no `model_version` field at all —
  that guarantee did not exist yet. It does now: `/health` and
  `/v1/embeddings` both carry `model_version`, derived from a real HF-cache
  snapshot commit sha when the checkpoint's own path resolves to one
  (`~/.cache/huggingface/hub/models--org--repo/snapshots/<sha>/...`, the
  layout `snapshot_download`/`from_pretrained` actually produce), falling
  back to the same `model` id otherwise — a weaker signal for a checkpoint
  placed by hand rather than resolved from the HF cache, but a real,
  never-fabricated one either way (`crates/server/src/engine.rs`'s
  `derive_model_version`/`hf_snapshot_sha`, unit-tested against both cases).
- **A server-side time budget that returns immediately when exceeded**, so the
  caller can take its degradation path. **Not implemented.** Keel's search
  falls back to keyword-only recall and still answers 200 when the engine is
  unreachable; this endpoint currently has no request-level timeout of its
  own, and blocks behind whatever else is queued on the same single
  inference worker thread (see the note on concurrency just below). Worth
  picking up before this endpoint carries real traffic.

**One real, honest limitation this implementation carries:** infero is one
GPU, one model, one worker thread (`crates/server/src/engine.rs`'s own doc
comment). An embedding batch runs to completion on that same thread rather
than through the scheduler's continuous batching, so it blocks any
`/v1/chat/completions` request already queued behind it for the duration of
the forward pass. Acceptable for M7's own sequencing (embeddings arrive
before schema-constrained generation does, so nothing sends both kinds of
traffic at once yet) but not something to carry forward silently once that
changes.

---

## 2. Schema-constrained decoding — the one worth building regardless

Keel's roadmap has a `generate` endpoint with **schema constraints** (attribute
extraction, review attribution, return-reason attribution). The requirement is
not "valid JSON" — it is "output conforming to *this* JSON Schema".

**This is where a hand-written engine beats a hosted one.** Most services stop at
a JSON mode. Constraining the sampling space token-by-token against an arbitrary
schema is something you can only do with access to the sampler, which infero has
and an HTTP client does not.

Keel does not need this until a later milestone — it was built anyway, per this
section's own original framing: this is the capability that makes infero
not-substitutable, not merely sufficient, and the milestone gap was time to build
it before it is on a critical path.

**Implemented.** OpenAI's own `response_format` field on
`POST /v1/chat/completions` — `{"type": "json_schema", "json_schema": {"schema":
{...}}}` — diverts to a real token-by-token constrained decode loop instead of
this server's normal continuous-batched path. Real, working
(`crates/model/src/json_grammar.rs`, `crates/model/src/constrained.rs`), not a
best-effort JSON mode: a byte-level cursor built from the schema rejects any
sampled token whose bytes would not keep the output a valid prefix of some
schema-conforming completion, so what comes back is not merely likely to be
valid JSON — it cannot be anything else.

**What it supports, real and independently unit-tested rather than exercised
only end to end:** objects with required/optional properties (keys restricted
to the declared `properties`, closing `}` refused until every required one has
appeared), arrays of a given item schema, strings (free or `enum`-constrained,
with live prefix narrowing — closing the string early on a value that is a
*prefix* of an enum member but not itself a member is refused), numbers,
integers, and booleans, arbitrarily nested. **What it deliberately does not
support**, to keep this a correct, testable subset rather than a partial
reimplementation of the JSON Schema spec: `oneOf`/`anyOf`/`allOf`, `pattern`,
`minLength`/`maxLength`/`minimum`/`maximum`, number exponents, `null` as a
value for a non-null-typed field, escape sequences inside an
`enum`-constrained string, and `additionalProperties` (unrecognized
properties are never accepted, only the declared ones). A field with no
recognized `type` falls back to a free string rather than an open-ended JSON
value.

Real, measured against Qwen2.5-0.5B-Instruct (Q8_0, this repo's own
already-present test fixture) — three schemas run end to end, not just
byte-fed to the cursor in isolation:

- Extraction (`{name: string, age: integer, city: string}`, all required)
  against a real sentence produced `{"name": "John", "age": 34, "city":
  "Paris"}` — every field present, every type correct, parsed with
  `serde_json::from_str`, not eyeballed.
- An `enum`-constrained `sentiment` field plus a free-form `keywords` array
  produced a value from the exact four allowed strings and a non-empty array
  of strings, against a real (mixed-sentiment, genuinely ambiguous) product
  review.
- All four primitive types composed in one schema (`name`/`legs`/`can_fly`/
  `diet`, the last `enum`-constrained) produced a fully valid, fully typed
  object in one pass.

**The real, not-yet-optimized cost, stated rather than hidden:** every decode
step tests every vocabulary token's bytes against a disposable clone of the
cursor before masking logits — a full vocabulary pass (151,936 entries on
this checkpoint) per generated token. Measured on this card: on the order of
tens of milliseconds per token including that scan, not the sub-10ms a
completion this short would otherwise take. Real and workable for the
attribute-extraction/classification shapes Keel's own roadmap names, not
something to reach for on a latency-critical hot path without narrowing the
per-step candidate set first (an obvious, real, unimplemented next step: a
byte-trie over the vocabulary would let a rejected prefix prune every token
that shares it in one step, instead of testing each one to its own first
rejected byte independently).

**Not implemented:** streaming (`response_format` always returns the whole
completion at once, even if the request also set `stream: true` — see
`crates/server/src/routes.rs`'s `generate_constrained` for why streaming
partial, unparseable JSON is not obviously better than waiting for the
close), and continuous-batching integration (like `/v1/embeddings` and
`/v1/rerank`, this runs to completion on its own throwaway session on the
single inference worker thread, blocking any ordinary chat request already
queued behind it — the same real, honest limitation `/v1/embeddings`'s own
section already names, for the same reason: nothing sends both kinds of
traffic at once yet).

Test coverage: `crates/model/src/json_grammar.rs`'s own unit tests (13 cases,
byte by byte, including a real bug this session's own testing caught: an
array's first item was being consumed by the array frame itself instead of
retried against the pushed item frame, rejecting every non-empty array until
fixed) plus `crates/model/tests/constrained.rs`'s three real end-to-end
cases above, gated on `INFERO_TEST_GGUF` with the same skip-when-unset
convention as this repo's other real-checkpoint tests.

---

## 3. `/v1/rerank` — feasibility confirmed, implemented (not with Keel's own model)

Keel's reranker is `bge-reranker-v2-m3`, a **cross-encoder** — bidirectional
attention plus a classification head over `[CLS]`, an architecture infero's
decoder-only forward pass genuinely cannot run, and building true encoder
support (a new attention mask shape, a new head, a new test surface) would
have been a real architecture investment, not "add a route." That part of
this section's own worry was right.

**What made it moot: a real, decoder-based reranker exists in the same
family this file's `/v1/embeddings` section already relies on.**
`Qwen/Qwen3-Reranker-0.6B` (`architectures: ["Qwen3ForCausalLM"]`,
`model_type: "qwen3"` — checked against its real config, not assumed) scores
a (query, document) pair by formatting a fixed instruction prompt and reading
the logits of the literal tokens `"yes"`/`"no"` at the position right after
it, then taking `sigmoid(yes_logit - no_logit)`. This is Qwen's own
published recipe (`Qwen/Qwen3-Reranker-0.6B`'s model card), not something
reverse-engineered — `infero_model::rerank`'s `PREFIX`/`SUFFIX` constants are
that recipe's exact prompt strings, byte for byte.

**Implemented.** `POST /v1/rerank` takes this file's own documented shape —
`{"query": "...", "documents": ["...", ...], "top_k": 30}` in, `{"results":
[{"index": 3, "score": 0.91}, ...]}` out, sorted by score descending and
truncated to `top_k` — plus an optional `instruction` field Qwen3-Reranker's
own recipe supports and defaults sensibly without (see
`infero_model::rerank::DEFAULT_INSTRUCTION`).

Real, measured against the model card's own published example: querying
`"What is the capital of China?"` against `"The capital of China is
Beijing."` and an unrelated sentence about gravity scored 0.9995 and
5.1e-6 — matching the model card's own reported behavior (raw logits 7.625
vs -11.375, i.e. a sigmoid pinned near 1) and its own direction. A second
test swaps which document is relevant by changing the query alone (same two
documents, "what is the capital" vs "explain gravity") and confirms the
score tracks the query, not a fixed document.

**The substitution this section makes is real, not free, and worth being
explicit about:** this is not `bge-reranker-v2-m3`. It is a different model,
from a different training run, with its own real strengths and blind spots,
and Keel's own eventual reranking quality on its real catalog is an empirical
question this section does not answer — only that the *capability* (a
working, decoder-native reranker infero can run without new architecture
work) is real. Confirming Qwen3-Reranker's reranking quality against Keel's
own real queries, on Keel's own real catalog, is Keel's call to make, not
infero's to assume.

Test coverage: `crates/model/tests/rerank.rs`, gated on
`INFERO_TEST_RERANKER_SAFETENSORS`, same skip-when-unset convention as
`/v1/embeddings`'s own test file.

---

## 4. A CPU backend — was "not required", now the one thing left

**This section used to say the opposite, and it is worth recording why it
flipped rather than quietly editing it.**

The original argument was: Keel's CI runs on GPU-less runners, but that is
solved on Keel's side — a deterministic stand-in behind a build tag
(`keel_fake_embedder`) sits in the default gates, and a separate real-engine job
runs the genuine article without blocking PRs. So infero could stay GPU-only and
still be adopted.

Two things happened after the switch actually landed.

**The separate real-engine job does not exist any more.** It was deleted rather
than left as `continue-on-error` or pinned to a self-hosted label that nobody
runs: a job that is permanently queued or permanently skipped looks, in the
checks list, like it is still guarding something. So the three acceptance
criteria (normalization verified by reading back from Postgres, semantic margin
above a floor, a dead engine raising rather than zero-filling) now run **only
when a human runs them**, on a machine with an NVIDIA GPU. That is a real
regression in coverage and Keel's CI file says so in plain words.

**And the fallback argument does not survive contact with users.** Keel is about
to go public. The stand-in covers *Keel's own tests*; it does nothing for a
person who clones the repository on a laptop. For them, no GPU means no
`KEEL_EMBED_ENDPOINT`, which means search degrades to keyword-only — which works,
is deliberate, and has an executor, but it is also the single headline feature
of the milestone being switched off for most of the audience.

Keel briefly considered keeping its old Python/CPU service alive as a second leg
behind a configuration switch. That was rejected: two engines means two vector
spaces, two sets of model metadata in `product_text_vectors`, and a second code
path that nothing exercises. **One engine, with a CPU backend, is the smaller
system.**

### What "CPU backend" has to mean here, concretely

Not fast — *correct and present*. The bar is the one Keel already measures:

- **Same numbers as the GPU path**, within floating-point noise. The three
  acceptance criteria are the test: normalization to 1.0, the semantic margin
  above 0.10, and errors that are errors. If CPU and CUDA disagree on the margin,
  the CPU path is not a backend, it is a second model.
- **The same `model` and `model_version` strings.** Keel keys staleness off them.
  A CPU run that reports a different model id would invalidate every stored
  vector.
- Speed is explicitly not a requirement. The service it replaced took 62 ms per
  text on CPU and that was acceptable for a single-machine deployment; anything
  in that neighbourhood is fine. Batch throughput matters even less — the
  index-side batch of 64 can take seconds.

### What it unlocks on Keel's side

- CI gets its executor back. The three criteria go from "a human runs them on a
  GPU box" to a job that goes red on a PR.
- Deployment shape A in Keel's architecture document — "runs on one machine,
  no GPU" — becomes true with semantic search intact, instead of true only with
  search degraded.
- `services/inference/` gets deleted. It is already unreachable (the client pins
  infero's path, model name and pooling sentinel as constants) and it is the last
  reason anyone would keep it.

---

## 5. `/v1/systemone` — Kev decision-model scoring, implemented and real-hardware verified

Keel asked for a third endpoint: typed decisions (yes/no, multiple choice, a
rating scale) with calibrated probabilities, not generated text — the shape
[Kev](https://github.com/jaredpalmer/kev) (Apache-2.0, an open-weights
alternative to TypeSafe's "Jev") and its upstream TypeSafe adapter both call
`/v1/systemone`. Keel's own stated priority: small checkpoints (0.8B/4B) for
the online path, sharing a GPU with embeddings at tens of milliseconds a
call; a larger one (27B) for offline batch labeling is fine to run slower.

**The mechanism, not a rewrite.** Kev's real backbone is the same
Qwen3.5/3.8 hybrid (GatedDeltaNet + attention) checkpoint family infero
already runs — the only new things this needed were a classification head in
place of the vocab projection, and a way to score many questions against one
shared block of text without recomputing it per question. Both turned out to
be small once put next to what already existed:

- **State sharing is `KvPool::fork`**, a primitive built for (and already
  tested by) MTP's speculative tree-draft verification, previously with zero
  production call sites. Score the shared text once, fork its KV + recurrent
  GatedDeltaNet state onto a fresh sequence per question (a real device-to-device
  copy of the recurrent state, ~151 MiB on the 27B per the primitive's own
  measurement; negligible at 0.8B), run each question as an ordinary causal
  continuation. No new attention masking, no new batching concept.
- **The one genuinely new piece**: a question needs the hidden state at its
  `<decide>` token *and* every option's `</opt>` token, scattered through the
  branch rather than trailing — not expressible through the existing
  `wants_logits`/spec-decode readout (which only ever means "the last N
  tokens of this item"), and that machinery's own buffers are sized for
  ordinary serving concurrency, not "however many options one question has".
  `Model::hidden_states_at` reads straight from the full per-pass residual
  stream into its own small scratch buffer instead — a few dozen lines, zero
  changes to the existing forward path.
- **The pointer head itself** (`kev.model.PointerHead`: two small `Linear(d,
  256)` projections and a scaled dot product) runs on the CPU in
  `infero_model::kev::PointerHead` — cheap enough at this scale that a CUDA
  kernel would only add surface area, not speed.
- **LoRA merging is an offline step**, not engine code: `scripts/merge_kev_lora.py`
  folds a Kev checkpoint's adapter into its base in fp32 (the same arithmetic
  Kev's own `LoadOptions.merge` does) and writes a plain safetensors
  checkpoint infero's existing loader reads completely unmodified, plus a
  `kev_head.safetensors`/`kev_head.json` sidecar for the pointer head
  (`head.pt` itself is a raw torch pickle this crate does not parse). A
  checkpoint directory carrying that sidecar *is* a Kev checkpoint — picked
  up automatically at load time, no CLI flag.

**Real, not simulated**: `jaredpalmer/kev-0.8b`'s real LoRA adapter (372
tensors) merged onto a real downloaded `Qwen/Qwen3.5-0.8B-Base` — 186 of 186
targets matched. Loaded into a real running `infero` server and hit over real
HTTP:

```
POST /v1/systemone  state: "This product exceeded my expectations...buy it again."
  noul "Is this review positive?"            -> 0.9622
  choice "What is this review about?"        -> quality (0.76), shipping (0.14), price (0.10)

POST /v1/systemone  state: "This product broke after one day...refund immediately."
  noul (same question)                        -> 0.0186 (i.e. 0.9814 "no")
  score "Rate customer satisfaction" (5 levels) -> 0.37 (70% mass on "very dissatisfied")
```

Positive/negative text correctly discriminated on a question the model's
instructions never hardcode a keyword for, a 3-option `choice` question
ranked sensibly against text that only clearly supports one of the three, and
a `score` question's expected value and probability mass both land where a
human reading the same text would put them. Not cherry-picked from a larger
run — these are the only two records tried.

**What this does not yet do**: `/v1/systemone` runs the same way
`/v1/embeddings`/`/v1/rerank` already do — to completion on the worker
thread, blocking any `Generate` request queued behind it, no cross-request
batching across separate HTTP calls (one record's own questions *do* batch
against each other, just not with another request's). A choice question's
`criteria` object is read back in sorted-key order, not request order (this
workspace's `serde_json` is not built with `preserve_order` — does not
affect correctness, since every probability is still keyed by its own option
name). A 0.8B/4B LoRA checkpoint is exactly what Keel asked for first; the
27B full-weight path was this project's own original guess at where to
start and needs no new code, just a download, since the loader never knew
the difference between a merged-LoRA checkpoint and a native full-weight one.

Test coverage: `crates/model/src/kev.rs`'s own 9 unit tests (packing layout,
pointer-head math against hand-derived expected values, error cases) plus
`crates/model/examples/kev_decide_smoke.rs`, a real-checkpoint smoke test in
the same spirit as this repo's other `INFERO_TEST_*`-gated real-checkpoint
tests.

---

## What Keel will hold it to

The Python service had to pass these before it was allowed into the index path.
infero would face the same three, unchanged:

1. **Normalization, verified from storage.** Read the vector back out of
   PostgreSQL and recompute the L2 norm there — not just assert on the slice the
   client received. (Measured on the current engine: min 1.000000000, max
   1.000000358 across the corpus.)

2. **A semantic margin test with a falsifying control.** Three texts — two
   related, one unrelated — and the cosine gap between (near, near) and
   (near, far) must exceed 0.10.

   | engine | near–near | near–far | margin |
   |---|---|---|---|
   | real BGE-M3 | 0.6234 | 0.4327 | **+0.19** |
   | hash-based fake, shape-identical | −0.0204 | 0.0344 | **−0.05** |

   The fake produces 1024 dims, unit norm, and the right `model` string. Every
   shape assertion passes on it. **Only the margin test fails.** That is the
   point: a test that a fake engine can satisfy is not testing the engine.

3. **Failure must be an error, never a zero vector.** A zero vector passes shape
   checks, gets written to the database, and then quietly poisons every search
   result that comes near it. The client must surface an explicit
   "engine unavailable" so the caller can degrade. infero's own normalization
   step (`infero_model::embed::normalize`) refuses to divide by a degenerate
   norm and returns an error instead — real, not assumed: it is what a
   near-zero or non-finite norm hits before ever reaching a caller.

**Stole #2 for infero's own test suite**, per this section's own suggestion:
`crates/model/tests/embed.rs`'s `near_texts_score_a_real_margin_above_far_ones`
runs this exact three-text setup against a real Qwen3-Embedding-0.6B and
measured a 0.53 margin (see "1. `/v1/embeddings`" above for the full numbers)
— comfortably clearing the 0.10 bar, and a genuinely harder bar than "the
server responded".

---

## Where this sits on Keel's roadmap

**This section predicted M7 and was wrong; the switch happened at M4.** Left
here corrected rather than deleted, because the reason it was wrong is the
useful part.

The prediction was: M3–M6 need embeddings and reranking, which an off-the-shelf
encoder process serves cheaply; the direction only changes from M7, where
conversational shopping, the schema-constrained `generate` endpoint (M9) and
Text-to-SQL (M10) all need real LLM generation — which is what infero is. So
keep the Python service through M6 and adopt infero at M7, wiring both endpoints
in one pass.

What that reasoning missed is that **"cheaply served" was doing a lot of work.**
The Python service was 62 ms per text on CPU against 7.6 ms here, a 75-second
cold start against 1, and a 1.39 GB image. None of that is fatal on its own. But
Keel's own latency budget for the search path is 15 ms, and the Python leg did
not fit inside it — so the thing being deferred to M7 was not "a nicer engine",
it was "the milestone's headline feature meeting its own stated budget".

Actual state, as of Keel's M4:

| | |
|---|---|
| `/v1/embeddings` | **in production use.** Every product vector in Keel's database came out of infero. |
| `/v1/rerank` | implemented and confirmed working, **not yet consumed** — Keel wires it at M5. |
| schema-constrained `generate` | implemented, **not yet consumed** — Keel needs it from M9 (attribute extraction, review attribution). |
| CPU backend | **not implemented; now the blocking item.** See §4. |

So the sequencing that actually held was the reverse of the prediction: the
encoder work drove adoption, and the LLM work it was supposed to wait for is
still ahead. The remaining gap is not a capability — it is that the engine only
runs where there is a GPU.
