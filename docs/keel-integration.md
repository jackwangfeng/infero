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

### Why the dimension matters more than anything else

Keel stores vectors in `product_text_vectors.embedding vector(1024)`. In pgvector
the dimension is part of the **column type**, so changing models means changing
DDL, writing a migration, and re-embedding the whole catalog.

**Qwen3-Embedding-0.6B is natively 1024-dim and is a Qwen3 dense decoder** — the
architecture family infero already runs, and inside what the Metal backend covers
(dense decode + GQA). So the single most expensive part of switching engines is
free.

### What is actually missing

Embedding is not generation. The forward pass already exists; what does not:

- **Expose the final hidden state without going through `lm_head`, and without
  sampling.** A prefill-only path that returns hidden states.
- **Last-token pooling** (what Qwen3-Embedding uses), with left padding.
- **L2 normalization owned by the server, with a self-check.**

That last point is not pedantry — it comes from a real bug on the Keel side.
BGE-M3's `modules.json` already contains a `Normalize` module, which means the
`normalize: true` request parameter **does nothing**: flipping it to `false` and
rebuilding still produced unit vectors (measured: 0.99999999 / 1.00000003 /
1.00000004). A guarantee that lives in some upstream repo's config **fails
silently** the day you change models. So the server divides by the norm itself
and asserts the result, rather than trusting the model card.

### Operational requirements

- **Batch**, 32–64 per request, with a hard error above the cap. Callers must not
  loop single texts.
- **`model` and `model_version` in every response.** Keel writes both into
  `product_text_vectors` and uses them to answer "does this row need recomputing".
  infero's `/healthz` already returns the HF snapshot commit sha as
  `model_version` — that is exactly the right shape; keep it.
- **A server-side time budget that returns immediately when exceeded**, so the
  caller can take its degradation path. Keel's search falls back to
  keyword-only recall and still answers 200 when the engine is unreachable.

---

## 2. Schema-constrained decoding — the one worth building regardless

Keel's roadmap has a `generate` endpoint with **schema constraints** (attribute
extraction, review attribution, return-reason attribution). The requirement is
not "valid JSON" — it is "output conforming to *this* JSON Schema".

**This is where a hand-written engine beats a hosted one.** Most services stop at
a JSON mode. Constraining the sampling space token-by-token against an arbitrary
schema is something you can only do with access to the sampler, which infero has
and an HTTP client does not.

Keel does not need this until a later milestone. It is listed second because it is
the capability that would make infero not-substitutable, whereas embeddings make
it merely sufficient.

---

## 3. `/v1/rerank` — check feasibility before scheduling

Keel's reranker is `bge-reranker-v2-m3`, a **cross-encoder**. infero is a decoder
engine end to end, so this is probably not "add a route" — it may need encoder
support that does not exist yet.

**Confirm the cost before committing to it.** Keel is fine running the two
endpoints on two different engines; the coupling point is defined precisely so
that this is allowed.

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
   "engine unavailable" so the caller can degrade.

Worth stealing #2 for infero's own test suite — it is a much harder bar than
"the server responded".

---

## What is explicitly *not* required

**A CPU backend.** Keel's CI has to run the whole chain on GPU-less runners, but
that is already solved on Keel's side: a deterministic stand-in behind a build tag
(`keel_fake_embedder`) sits in the default gates, and the real-engine job runs
separately without blocking PRs. infero can stay GPU-only (CUDA + Metal) and still
be adopted.

---

## Where this sits on Keel's roadmap

M3–M6 need embeddings and reranking — encoder work, cheaply served by an
off-the-shelf process. From M7 the direction changes: conversational shopping
(M7), the schema-constrained `generate` endpoint (M9), and Text-to-SQL (M10) all
need **LLM generation**, which is what infero is.

So the natural sequencing is: keep the Python service through M6, adopt infero at
M7, and add `/v1/embeddings` at the same time — by then the LLM path has to use it
anyway, and wiring two endpoints in one pass beats wiring one twice.
