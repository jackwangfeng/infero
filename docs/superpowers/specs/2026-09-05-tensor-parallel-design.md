# Tensor-parallel inference across multiple GPUs

Goal: split a model's weights and KV cache across N GPUs on one node so
infero can serve checkpoints too large for one card, and lay the groundwork
(interfaces only, not implementations) for pipeline parallelism and
cross-node deployment later. Validated end-to-end on a real, tiny checkpoint
(`qwen2.5-0.5b-instruct-q8_0.gguf`, 645MB) split across `bw`'s free GPU
headroom before ever touching a real large checkpoint.

## Decisions already made (brainstorming, this session — don't re-litigate)

- **Tensor parallel now; pipeline-parallel interfaces reserved, not built.**
- **NCCL** for cross-GPU communication — same "pure C ABI, no libtorch/Python"
  integration shape as the existing CUTLASS/FA2 FFI work (`INFERO_NCCL_DIR`-style
  build gate, mirroring `INFERO_CUTLASS_DIR`).
- **Single-node now; cross-node interfaces reserved.** Concretely: rank
  bootstrap (agreeing on an NCCL unique ID) is its own small trait with a
  local-file-based implementation today; a future cross-node implementation
  swaps only that piece.
- **SPMD: one OS process per GPU**, not one process driving multiple device
  contexts — matches vLLM/Megatron, and is the only shape that survives the
  jump to cross-node later without a rewrite.
- **Each rank reads only its own weight shard from disk** — never loads the
  full checkpoint and slices locally. Required for real large checkpoints
  (172GB `Qwen3.8-Flash-Next-FP8` at TP=4 would otherwise mean 688GB of
  redundant disk reads); the small validation checkpoint works either way,
  so this is a real requirement being validated early, not deferred.
- **Configurable N**, not fixed at 2. `bw` validates with N=2 (needs only two
  GPUs' free headroom, more of it available than a 4-way split) and N=4.
- **GDN and standard attention shard the same way: by head**, dividing
  `n_heads`/`n_kv_heads` (attention) and `num_k_heads`/`num_v_heads` (GDN) by
  `tp_size`. This is not a guess — verified against vLLM's real production
  source this session: `vllm/model_executor/models/qwen3_next.py` shards
  standard attention heads by `tp_size` (`self.num_heads = self.total_num_heads
  // tp_size`, with KV-head replication when `total_num_kv_heads < tp_size`);
  `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` shards GDN
  identically (`self.num_k_heads // self.tp_size`, `self.num_v_heads //
  self.tp_size`). infero's own GDN state is a per-head `[DK, DV]` matrix
  (established earlier this session) — sharding by head means each rank's
  GDN recurrence is *fully local*, no cross-rank communication mid-recurrence,
  only at the final output projection, exactly like attention.

## Out of scope, explicitly

- **MoE tensor-parallel.** `qwen3-30b-moe-awq` exists on `bw` but the
  validation target is the dense 0.5B checkpoint. MoE-TP (expert-parallel
  sharding, `total_num_experts` vs `tp_size` checks) is real, different work —
  flagged as a follow-up, not attempted here. Nothing in this design should
  make MoE-TP harder to add later, but nothing here implements it.
- **Fault tolerance.** If any rank errors or panics, the whole TP group fails
  fast (NCCL collectives will hang/timeout on a dead rank regardless — this
  is the same behavior vLLM has without its own health-check layer, not a
  corner being deliberately cut relative to the ecosystem). Real fault
  tolerance is a cross-node-era concern.
- **Pipeline parallelism and cross-node itself** — interfaces reserved per
  above, neither implemented.

## Process model and rank bootstrap

One OS process per GPU. A thin supervisor (a new small binary or a shell
script, decide during planning — not an architectural question) reads
`--tensor-parallel-size N`, spawns N worker processes with
`CUDA_VISIBLE_DEVICES` pinned one-per-rank and `INFERO_TP_RANK`/
`INFERO_TP_WORLD_SIZE` env vars set. Rank 0 generates an NCCL unique ID and
writes it to a local file at a path derived from a shared run ID (env var);
every rank (including rank 0) reads that file and calls `ncclCommInitRank`.
This file-based handoff is the whole content of a `RankBootstrap` trait —
`fn broadcast_unique_id(rank: usize, world_size: usize, run_id: &str) ->
Result<NcclUniqueId>` — so a future cross-node implementation (e.g. reading
the ID from a coordination service instead of a local file) only replaces
this one function, nothing downstream.

`(pp_rank, tp_rank)` is the real rank identity type from day one, even though
`pp_rank` is always `0` and `pp_size` is always `1` right now — every place
that currently would say "the rank" says "the tp_rank within this pp_rank's
layer range" instead, so adding real PP later is "change what layer range a
`pp_rank` owns," not a rank-identity rewrite.

## Weight loading: per-rank shard reads

`crates/safetensors`/`crates/gguf`'s existing loaders read a whole tensor's
bytes given its header-declared offset/shape. Sharded loading adds, per
tensor that gets column- or row-sharded (attention/GDN Q/K/V/gate projections
column-sharded, O/down projections row-sharded — standard Megatron-style
placement), a byte-range computation: given the tensor's full shape, `tp_size`,
and `tp_rank`, compute the sub-range of rows or columns this rank owns and
read only those bytes from the file (safetensors' flat row-major layout makes
row-slicing a contiguous byte range; column-slicing needs a strided read —
one read per row instead of one contiguous read, real but mechanical). This
lives as a new sharding-aware wrapper around the existing loader's byte-range
read primitive, not a rewrite of the loader itself. Non-sharded tensors
(layernorm weights, embeddings — replicated across all ranks) are read in
full by every rank, unchanged.

## Forward pass: where sharding and all-reduce land

Standard Megatron placement, applied identically to attention and GDN given
they now share the same head-sharding story:

- **Column-parallel** (each rank computes its own slice, no communication
  needed yet): Q/K/V projections (attention), the combined QKVZBA-style input
  projection (GDN, matching `qwen_gdn_linear_attn.py`'s own layout), FFN
  gate/up projections.
- **Row-parallel + all-reduce** (each rank's partial output must be summed
  across ranks before the next layer can proceed): the output projection
  (attention's `o_proj`, GDN's own output projection), FFN down projection.
  One `ncclAllReduce` (sum) per occurrence, on the stream already in flight —
  no host sync forced by the collective itself.
- Layernorms, RoPE tables, and anything per-token-scalar (not per-hidden-dim)
  are replicated identically on every rank — cheap, and avoids a
  synchronization point that buys nothing (every rank computes the same
  answer from data it already fully has).

## KV pool under TP

Each rank's KV pool is sized for `n_kv_heads / tp_size` (attention) and
`num_k_heads / tp_size`/`num_v_heads / tp_size` (GDN state) — a real,
proportional memory reduction per rank, not just a compute split. The pool's
existing slot-table/allocation logic (`crates/model`) is parameterized by
head count already (it has to be, to serve different model shapes) — TP
support here is "construct the pool with the sharded head count," not new
pool logic.

## Server/scheduler: rank 0 drives, others follow

`crates/server`'s existing `scheduler.rs` makes real scheduling decisions
(which requests, what batch, prefix-cache hits) that must be **identical**
across every rank each step — a TP group only stays coherent if every rank
runs the exact same forward pass shape at the exact same moment (an NCCL
collective with mismatched participants across ranks hangs or corrupts,
it does not degrade gracefully). So: **only `tp_rank == 0` runs the real
scheduler** (owns the HTTP API, the request queue, prefix cache, tokenizer)
and, once it decides a step's batch, broadcasts the decision (token ids,
sequence ids, slot assignments — already a compact, serializable struct
today) to ranks `1..N` via one `ncclBroadcast` (or a lightweight side-channel
if that's simpler for non-tensor metadata — decide during planning, not
architectural). Ranks `1..N` run a stripped-down loop: wait for a broadcast
batch, run the identical forward pass, discard/don't compute the final
logits-to-token sampling step (rank 0 does that, since it owns the actual
response stream) — only rank 0's output leaves the TP group.

## Testing and validation

1. Unit-level: the byte-range shard-loading math (given shape/tp_size/tp_rank,
   is the computed byte range correct?) is pure, host-side logic — test it
   directly against known shapes, no GPU needed.
2. NCCL plumbing: a minimal 2-rank (or N-rank) smoke test that does a
   real `ncclAllReduce` on a known tensor and checks the result — proves the
   FFI binding and rank bootstrap work before any model code depends on them.
3. Real end-to-end correctness: run `qwen2.5-0.5b-instruct-q8_0.gguf` two
   ways on `bw` — today's existing single-GPU path (the reference), and the
   new TP=2 (then TP=4, using GPU headroom already confirmed available: GPU0
   ~8.6GB free, GPU1 ~9.2GB, GPU2 ~7.4GB, GPU3 ~66GB free — none of it
   belongs to other users' allocations) path — same prompt, same sampling
   seed, and diff the generated tokens/logits. This is the real acceptance
   test for the whole design, not a unit test standing in for it.
4. memcheck/racecheck on any new kernel-adjacent code (the shard byte-range
   reader touches no CUDA kernels directly, so this mainly applies if any new
   device-side reduction/copy code is written beyond bare NCCL calls).

## What this design does not do

Does not touch MoE-TP, fault tolerance, pipeline parallelism itself, or
cross-node itself — all explicitly out of scope above, with interfaces
placed so none of them require revisiting the decisions in this document.
Does not change anything about the existing single-GPU path — TP is an
additive mode (`--tensor-parallel-size 1`, the default, is exactly today's
existing single-process behavior, unchanged).
