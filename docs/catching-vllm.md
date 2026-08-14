# Where the gap to vLLM is, and what has already been tried

State at the end of the session that produced this file: **4876 tok/s against
vLLM's 5248 at a batch of 32 — 1.076x behind** on the distinct-prompt load
(5398 and 1.107x on the shared-prompt one; see the fairness section), on a Blackwell RTX PRO 6000 with
Llama-3.1-8B AWQ. The session before it read 4218.9 against 5454.2; the load
generator is noisy to about ±5% and both engines were re-measured today, back to
back, so those are the numbers to compare against.

The session started at 3974 and 1.36x. Three changes landed, all measured on the
served engine rather than on a kernel:

| change | tok/s |
|---|---|
| baseline, re-measured | 3974 |
| fused `qkv` and `gate_up` on by default | 4150 (+4.4%) |
| `attn_decode_gqa_f32` replacing the three attention kernels | 4344 (+4.7%) |
| its tile cut from 32 keys to 16 | 4392 (+1.1%) |
| the residual add folded into the norm that follows it | inside the noise, 32 fewer launches a step |
| a k-tile of weights prefetched into registers | 4449 (+0.7%) |
| the grid constant re-swept after it: 2 blocks an SM, not 4 | 4599 (+3.4%) |
| the vocab projection's scales split out of its quants (Q8_0S) | 4697 (+2.1%) |
| `down_proj`'s f16 activation written by the SwiGLU kernel | 4724 (+0.6%) |
| `o_proj`'s written by the attention combine — `f32_to_f16` gone | 4765 (+0.9%) |
| the FFN residual added by the next layer's attention norm | 4772 (+0.15%) |
| one row-group width for every projection, not two | 4849 (+1.6%) |
| q/k/v read where the stacked projection wrote them | **4876 (+0.6%)** |

`layers_ms` at the end, as an interval average between two cumulative samples
rather than the run's mean: **5.749 ms**, against 6.095 where this segment
started. GPU time a step is 6.417 ms of a 6.56 ms step, so there is no idle left
to find — the 1.23 ms the trace below reports was node-level tracing overhead,
and that entry stands corrected.

Both engines were re-measured at the end, on the same box, minutes apart:
tuili 4599 / 4603 / 4593 / 4604 / 4594, vLLM 5387 / 5477 / 5391. Take the
*plateau*, not the first run: throughput climbs over the first two or three
benches — 4518, 4545, 4599 — because a decode graph is captured per
`(tokens, kv bucket)` pair and a cold cache pays for the captures.

This is an index, not a record. Every measurement below lives in a comment next
to the code it is about; the point of this file is to say where those comments
are and which direction has been paying, so that the next attempt does not
re-run the experiments that already failed.

## Measure it the way the other engine is measured

The previous edition of this file pointed at two targets and both were
artifacts of how they were measured. Before trusting any number here, know
which of these it is:

* **`TUILI_PROFILE` puts a CUDA event pair around every launch and serializes
  the stream.** A one-thread, one-store kernel measures **3.44 us** under it.
  So every small kernel in that profile carries about 3.4 us that a graphed step
  does not pay, and the *shares* it reports are right while its microseconds,
  for anything under about 10 us, are not.
* **A warm L2 flatters a microbenchmark.** This card has 128 MB of it. A
  layer's K and V at batch 32 with 512 of history are 67 MB, so fifty
  back-to-back launches over one cache measure something the engine never sees;
  `bwidth_attn.rs` cycles four caches to defeat it, and `bwidth.rs`'s weight
  probes do not, which is why they report 5543 GB/s on a 1792 GB/s card.
* **The other engine's harness has overhead too.**
  `scripts/flash_attn_bandwidth.py` reports 58 us for vLLM's decode attention at
  *every* history from 128 to 512, because ~28 us of it is the Python call. Its
  kernel takes 29.6 us inside its own engine.

* **Confirm the binary is the one you built.** A remote `cargo build` whose
  output was piped through `grep -E "^error"` printed nothing and looked clean;
  `cargo` was not on the `PATH` of a non-interactive shell, and the filter ate
  the message. Two A/B runs then measured a binary from before the change and
  agreed to three digits — which is what a null result looks like, and was the
  only reason I checked the timestamp. Compare `ls -l target/release/tuili`
  against the source mtime before believing a remote number.

The measurement that settles arguments is `nsys` on both engines under the same
load, and then the kernel-summary tail of the trace. Kernel durations there are
device-side and comparable; the *gaps* between them are inflated by node-level
graph tracing (the same server steps at 7.71 ms without `nsys` and 8.72 with).

## Where a step actually goes

Both engines, `~/bench.py` at 32 clients, 512 tokens a request, traced with
`nsys profile -t cuda --cuda-graph-trace=node` and summarized per decode step:

| per step | tuili | vLLM |
|---|---:|---:|
| layer GEMMs | 3.617 ms | 3.290 ms |
| attention | ~1.50 | 1.065 |
| vocab projection | 0.699 -> 0.489 | 0.705 |
| sampling | 0.176 | 0.097 |
| everything else | ~0.60 | ~0.48 |
| **GPU busy** | **6.95** | **5.63** |
| GPU idle | 1.23 | 0.51 |
| kernels a step | 556 | 433 |

The three deficits are the GEMM, attention, and a long tail of small kernels —
in that order. There is no single thing to fix.

Against the machine rather than against vLLM: a step reads 4.24 GB of weights
and about 1.7 GB of KV. The measured ceiling for the weight read is **1440
GB/s** (`attn_kv_probe`, and `bwidth.rs`'s 16-byte probe agrees once L2 is
discounted). tuili's GEMMs run at 955 GB/s of weights, vLLM's Marlin at 1063.
**Both engines are two thirds of the way to the memory wall**, which is where
the remaining 1.5 ms a step is, and neither is close to taking it.

## What has been paying

**Compute each kernel's achieved bandwidth and look for loads that are not in
flight** — but compute it from a *slope*, not from one shape, and against a
measured ceiling rather than the datasheet.

Landed, all verified against the unquantized reference or against the host
implementation they replace, and all confirmed to leave a real model's greedy
output byte-identical:

| change | effect |
|---|---|
| sampling moved to the device (`cu/sample.cu`) | +47% |
| `attn_output` read 16 bytes a thread, not 2 | kernel 2.4x |
| `q`/`k`/`v` and `gate`/`up` fused into one matmul each | +4.4% |
| `attn_scores_gqa` given two keys a warp | kernel +22% |
| `f32_to_f16` and `add_assign` four elements a thread | ~0.16 ms a step |

The fused projections are now the default (`weights.rs`), enabled whenever the
stacked copies fit in a third of free VRAM; `TUILI_FUSE_FFN=0` restores the
three narrow matmuls. They cost 2 GiB on an 8B model because the originals are
kept for the batch-1 mat-vec. Dropping them means teaching the mat-vec to read a
column range of a stacked matrix, whose scales live past all of its quants —
two disjoint byte ranges, not one. Worth doing, not yet done.

## The vocab projection was an encoding, not a bandwidth

It read 558 MB a step at 90 GB/s on an A4000 and 780 GB/s on the Blackwell, and
the reflex reading — a big matrix, so a bandwidth wall — was wrong. A packed
Q8_0 block is 34 bytes: two of scale in the *middle* of every 32 quants. So a
thread's run of weights is never wider than 32 bytes and, past the first block,
never 16-byte aligned. `mmq_load_w_q8_0` reads it two bytes at a time because
that is all the layout allows.

`WeightType::Q8_0S` holds the same numbers with the scales moved to a table
after the quants. Same 558 MB, so nothing about the bytes changes; a row becomes
contiguous and the tile load becomes one `uint4`:

| | A4000 | Blackwell |
|---|---:|---:|
| packed Q8_0, isolated | 5619 us, 99 GB/s | 716 us, 780 GB/s |
| split Q8_0S, isolated | 3142 us, 178 GB/s | 470 us, 1186 GB/s |
| the phase, served | 6.17 -> 3.53 ms | 0.728 -> 0.489 ms |
| throughput | 956 -> 1059 (+11%) | 4529 -> 4697 (+3.7%) |

Both A/Bs are one binary against `TUILI_LM_HEAD=packed`, which still selects the
old form. The A4000 pays four times more for the same mistake because its L2 is
4 MB against 128, which is the general shape of this: **the small card exposes
layout, the big card hides it.** Two of the three landed wins this session were
found on the 1.5x-slower machine.

Batch 1 keeps the packed matrix. A single row reads the same bytes whatever the
layout, so `has_mmvq` excludes Q8_0S deliberately rather than for want of a
kernel, and it is never an embedding table, so the warm-up skips its row gather
the way `Q4G128T`'s is skipped.

Worth noting what the *other* engine spends here: vLLM keeps `lm_head` in f16
and reads 1.05 GB a step against this path's 532 MiB. The vocab projection is a
phase tuili wins on bytes and was losing on layout, and the remaining 1186 vs
1265 GB/s against `q4_g128t` is the last of it. Going further would mean a
4-bit head, which trades quality for a number — not the same kind of win.

### A 23% improvement that was a wrong answer

Sweeping all eight instantiated shapes of the integer GEMM over this matrix,
`mmqw8` measured 213 GB/s against the default shape's 173. It was computing
NaNs. `mmq_variant` built the grid from `MMQ_MAX_ROWS / 8` warps for any
explicitly named plain shape, so an eight-warp kernel was launched with 128
threads, and the bandwidth harness reads zero-filled buffers, where a kernel
that indexes wrongly still finishes fast and reports beautifully. The one- and
two-warp shapes had been failing to launch at all for the same reason, which I
first read as "unsupported at this shape".

With the launch fixed the same kernel is 177 GB/s against 178 — the 23% *was*
the bug. `the_split_q8_0_layout_matches_the_packed_one` now checks all sixteen
(shape, encoding) pairs against the default shape's output and demands exact
equality, which is the only assertion that would have caught it: peak-relative
closeness survives a scale table off by one block.

The two lessons are the same lesson. A timing harness cannot see correctness,
and a name-selected shape carries a launch configuration that has to be read
from the name. Any sweep in this file that selected a shape by name before this
commit is suspect for the same reason.

## The small-kernel tail, and what fusion is worth

Half the remaining per-layer deficit was the tail of small kernels — nine of
them, 19.6 us, against vLLM's four and ~12 — so this is where the last six
percent came from. What landed, per layer:

| | before | after |
|---|---:|---:|
| `f32_to_f16`, twice | 2.4 us | gone |
| `split_qkv` | 1.1 | gone |
| `add_assign` | 1.3 | 0.85 of it gone |
| the kernels that absorbed them | | +0.03 |

Three rules came out of it, and they are the useful part:

**Write the f16 copy from the register that holds the f32.** The fused norm
already did this for `qkv` and `gate_up`; the two activations no norm produces —
attention's output and the SwiGLU product — were being written as f32, read back
and converted. `silu_mul_split_f16_f32` averages 2397 ns against the f32-only
form's 2367, and `attn_flash_reduce_f16_f32` 2286 against 2298. So 0.080 ms a
step is *removed*, not moved, and `f32_to_f16` is absent from the trace.

**Price the kernel that absorbs the work, not the launch that disappears.**
Folding the FFN residual into the next layer's attention norm removes a 1.28 us
`add_assign`, and is worth 0.45: the fused add-and-norm reads and writes the
residual stream, so it runs 3.60 us against the plain norm's 2.77. Measured
0.009 ms a step against a predicted 0.014. Taking the removed launch's cost as
the saving would have overstated it threefold.

**A copy that exists to simplify indexing is not a copy worth making.**
`split_qkv` scattered the stacked projection's output row into three buffers so
`rope_qk` and `store_kv2` could index without a stride. Both now read the packed
row; `v` never moves at all.

What is left of the tail is two norms (6.4), rope (2.3), the attention combine
(2.3), `store_kv2` (1.5) and the residual adds at the two ends of the stack.

## Is the load generator fair? Yes, to within 3%

`bench.py` sends all 32 clients the same prompt at temperature 0, so all 32
sequences generate the identical continuation. vLLM hashes KV in 16-token blocks
and shares them, which looked like it could hand the other engine an L2-resident
cache and most of the remaining gap. Measured, one prompt per client against one
shared prompt, both engines, back to back:

| | same prompt | distinct prompts |
|---|---:|---:|
| tuili | 4691 | 4691 |
| vLLM | 5398 | 5248 (-2.8%) |

So the sharing is worth 2.8% to vLLM and nothing to tuili, and the gap on the
fairer load is **1.12x**. The reason the effect is small is that concurrent
sequences do not dedupe mid-flight: a block is looked up when a *new* request
prefills, not while thirty-two sequences are already running. Worth knowing
before reading anything else here — I spent a while on the assumption that this
was the answer.

## Attention: 81% of a pure-read ceiling on this card

`bwidth_attn.rs` on the Blackwell, batch 32, history 512, 32q/8kv x 128:

| | us a layer | GB/s of KV |
|---|---:|---:|
| the three kernels | 58.3 | 1152 |
| `attn_decode_gqa_f32` | 57.4 | 1169 |
| `attn_kv_probe_f32` — read the bytes and do nothing | 46.6 | 1440 |

A perfect kernel that only reads KV would save 10.8 us a layer, 0.35 ms a step,
**5%**. That is the whole envelope for attention on this card, and it prices
every idea about it before the idea is tried.

`attn_decode_mma_f32`, the tensor-core version, is *worth 7% on an A4000 and
costs 7.4% here* (4342/4361 tok/s against 4697, layers 6.56 ms against 6.06). It
stays opt-in behind `TUILI_ATTN_MMA=1`, now for two reasons rather than one — it
also breaks batch invariance, since P is f16 for the tensor core. Which way the
arithmetic trade goes evidently depends on the card, so the entry that priced
`m16n8k16` at 3% was pricing it on the wrong machine.

## What has not been paying

Ranked by how much time went into it. All of these are recorded with numbers at
the sites listed above; do not re-run them.

* **The three "worst kernels in the profile" have no bandwidth to win.**
  `store_kv`, `rope_qk` and `attn_softmax` measured 114, 225 and 700 GB/s and
  were estimated at 0.4 ms a step. Their marginal bandwidth — hold the shape,
  grow the batch to 512 tokens, take the slope — is 1375, 1962 and 2689 GB/s,
  and the fixed cost hiding in each of them is 4.6 us of event pair. Together
  they are 1.4% of a batch-32 step and most of that 1.4% is the measurement.
  See `crates/kernels/tests/bwidth_ops.rs`.
* **Fused decode attention: the eighth shape of it works.** The seven in the
  comment above `attn_flash_split` still stand, and `attn_decode_gqa_f32` is not
  one of them — see `Where to go next`'s entry on it below for what it does
  differently. What did *not* work, on the way: double-buffering the tiles in
  shared costs more occupancy than it buys (19.5 KB a block to 36.9 drops an SM
  from five resident blocks to two, and the kernel from 56.9 us to 76.6);
  prefetching through registers instead is no better; reading V straight from
  global rather than staging it loses 5%, because the group's four warps each
  fetch the row the tile load would have fetched once; and an 8-key tile is
  worse than a 16-key one at every history.
* **The ceiling says there is 23% left in attention and neither path takes it.**
  `attn_kv_probe` reads exactly what attention reads over the same grid and
  discards it: 36.5 us a layer at a history of 384, against 47.6 for the three
  kernels. It is *insensitive to the split count* — 256 blocks of 128 threads
  already saturate — so the missing 23% is not parallelism. It is that both
  paths interleave arithmetic with their loads and the probe does not.
* **Porting more of Marlin.** tuili's GEMM is faster than `marlin_gemm` on five
  of the six shapes a layer uses *in a warm-L2 microbenchmark*; in the engine,
  cold, it is 10% slower than Marlin overall. The microbenchmark ranking is not
  the engine's.
* **Re-fitting the GEMM's constants on Blackwell.** The comment in `mmq.cu`
  says the shape constants were fitted on an sm_86 card with 100 KiB of shared
  memory and are re-openable at 228. They were re-opened, in the engine, at
  batch 32: sixteen warps (`mmqy1w16s2`, newly instantiated) run a step's
  matmuls in 96.1 ms against the default mix's 96.5 — noise. Wider row blocks at
  sixteen warps are far worse (`mmqy2w16s2` 135.4, `mmqy4w16s2` 123.0). The
  striped partition's blocks-per-SM constant was swept from 2 to 48 and the
  fitted 4 is still best. The vocab projection's warp count was swept and 8 is
  still best.
* **CUDA graph instantiation.** `TUILI_GRAPH_MODE` prices it: `autofree` 8.46 ms
  a step, `plain` plus an explicit `upload()` 8.60, and
  `INSTANTIATE_FLAG_UPLOAD` is rejected by the driver. Dropping the graph costs
  0.8 ms a step, so it is paying; the 721 us gap that prompted the experiment is
  mostly the node-level tracing that measured it.
* **`mmq`'s batch scaling.** Understood, not fixed; the four candidate fixes
  ruled out by measurement are unchanged from the last edition.
* **Compressing the KV cache.** It is a third of a step's traffic and tuili
  has TurboQuant built in, so `--kv-quant tq8` looked like half a millisecond
  for free. It measures **2300 tok/s against 4392**: the rotation and the decode
  cost far more than the bandwidth saves. Not a lever, at any precision.
* **Aligning the GEMM's partition to row groups.** The striped partition's runs
  straddle row-group boundaries, and a straddling run has to `atomicAdd` into an
  output that therefore has to be zeroed first — 128 memsets and 170 MB a step,
  none of which appears in a kernel profile. Sizing the grid to the row groups
  makes every run whole and removes both. It loses 15% (`TUILI_MMQ_ALIGNED=1`
  re-runs it): `gate_up` goes from 752 blocks to 224, and the block count is
  worth more than the atomics and the memset together.
* **Q8_1 activations in the new pipeline** (`mmqe_*`, already in the tree) are
  30% down. The activation tile is the larger half of a block's traffic at 32
  tokens — 16 KB against 8.7 of weights — and 1.125 bytes an element instead of
  2 does not pay for the ten instructions an int8 A-fragment costs. Sixteen
  warps, which halves the same traffic by covering twice the rows, is level.
* **The KV cache's page size.** vLLM pages sixteen tokens at a time and tuili
  hands out one slot per token, which looked like it would matter for a
  256-byte row. `TUILI_ATTN_PAGE` prices it: 1-token and 16-token pages are
  within noise of each other at every history. Only a fully contiguous history
  helps, and only the fused kernel, and only at short histories.

## Where to go next

Ranked by size, with what each is worth a step. The four small ones together do
not close the gap; the first one does, and it is the hard one.

* **The GEMM is now level with Marlin, and the last 0.9 ms is territory
  neither engine occupies.** It reads 4.24 GB of weights at **1037 GB/s**
  against Marlin's 1063 and the card's 1440. Two changes got it there from 955,
  and the second only worked because of the first:

  1. **A k-tile of weights prefetched into registers.** The loads sat between
     `mmq_bar_arrive` and `mmq_bar_sync` and the MMAs needed them the
     instruction after — one barrier of cover for a DRAM read. This is level two
     of Marlin's pipeline, which `vendor/marlin/README.md` explicitly deferred,
     and the reason it was thought not to fit was `mmqb_*`'s 5-8% loss at depth
     four. That kernel is bound by registers (75 of them at four warps and
     static shared); this one is bound by its 34 KB of *dynamic* shared at 2.9
     blocks an SM, where 50 registers would allow 5.1. The twelve registers the
     prefetch costs are inside that gap. Worth 0.7% end to end on its own.
  2. **Then the grid constant, re-swept.** With a k-tile in flight a block that
     covers more of the flattened partition amortizes what it used to only
     lengthen, and the optimum moved from 4 blocks an SM to 2: 91.15 ms a step
     of matmuls against 96.39, reproducible to 0.01 ms, and better at every
     batch width. **+3.4%.**

  The interaction is the lesson: the constant was fitted correctly for the
  kernel that existed, and re-fitting it was worth five times the change that
  moved it. Every other knob was re-swept at the new point and none of them
  moved — 16 warps 107.7, `NBLK`=2 95.8, `NBLK`=4 124.3, four stages 100.8, the
  aligned partition 102.5, depth three does not build inside the register
  budget.

  **And the last 20% is unattributed.** `mmq_bw_probe_s16` reads everything the
  kernel reads — quants and scales, four buffers cycled so 248 MB defeats L2 —
  and gets 1343 GB/s where the kernel gets 1037. Each of these was isolated and
  none of them is it:

  | isolated by | worth |
  |---|---|
  | `mmqnm_*`, the MMAs deleted | 0% |
  | stubbing the A-fragment shared read | 0.8% |
  | stubbing both barriers | 2.2% |
  | the same bytes coalesced (`_c16`) | 0-5% |
  | block-major scales (`_sc16`) | 0% |
  | running the probe at the kernel's own 2.9 blocks an SM | 0% |
  | a write stream beside the read one (`_rw16`) | 1% |
  | Marlin's `cp.async` weight ring (`mmqc1w8s2`) | **-7.5%** |

  The occupancy row is the surprising one: the probe reads at 1341-1406 GB/s at
  1504 blocks and at 376, so the 34 KB activation ring's cost in resident blocks
  buys nothing back — which also means the weight ring's 16 KB was never the
  reason it should lose, and it loses anyway. It also kills the fp8-activation
  idea this file suggested a revision ago: the whole activation read is worth
  0.8%, so halving it is worth nothing.
  
  The probe is not itself warm — 1345 GB/s at four buffers, eight and sixteen,
  which is a 944 MB working set against 128 MB of L2.

  So about 9% of the 23% is accounted for and 14% is not. Attention sits in
  exactly the same place — 1091 GB/s of KV against its own probe's 1390, 78% —
  which suggests one cause rather than two, and nothing on this list is it.

  What neither engine does, and what could therefore beat Marlin rather than
  match it: this card has TMA (`cp.async.bulk.tensor`) and warp specialization
  with `mbarrier`, which is how every CUTLASS 3.x kernel feeds a tensor core on
  Hopper and later. Marlin is an Ampere-generation design and so is this port.
  That is the one direction left with a mechanism behind it rather than a
  constant to sweep, and it is a rewrite, not a session.

  (`mmqc_*` was unreachable from the model until this session: `mmq_f16_variant_for`
  did not accept the prefix, so asking for it silently ran the integer fallback
  at 4500 launches instead of 2580. Any earlier measurement of that family
  measured something else.)

  `tests/regcount.rs` says why nine structural changes measured zero.
  `mmqy1w8s2_2_q4_g128` is **50 registers and 19.5 KB of dynamic shared, and
  holds five blocks to an SM** — which is 1280 threads at 50 registers against
  the SM's 65536, and 97.5 KB of shared against its 100. *Both* limits are at the
  wall simultaneously. So every scheme for putting more weight loads in flight
  trades a resident block for them, one for one:

  | prefetch | costs | blocks/SM |
  |---|---|---|
  | one k-tile of weights ahead, in registers | +12 regs | 5 -> 4 |
  | Marlin's `cp.async` weight ring, 2 deep | +16 KB shared | 5 -> 2 |
  | four activation stages instead of two | +15 KB shared | 5 -> 2 |

  Which is the general form of `mmqb_*`'s 5-8% loss and of the note in `mmq.cu`
  that memory-level parallelism cannot be bought with registers here. The
  frontier moves only by making an *operand* smaller: fp8 activations would halve
  the ring and free the room, and Blackwell's `m16n8k32` takes them directly.
  That is a precision change, so it is a decision and not a tuning step.
* **The vocab projection, 0.34 ms.** 717 us to read 532 MiB is 780 GB/s, and
  the reason is in the comment on `mmq_load_w_q8_0`: a Q8_0 block is 34 bytes,
  so its quants are only ever halfword-aligned and the kernel reads them **two
  bytes at a time** — the exact defect that cost `attn_output` 2.4x. tuili
  quantizes this matrix itself at load, so the layout is ours to choose: split
  the quants from the scales the way `Q4_G128T` does and the loads widen to
  sixteen bytes. It needs a weight type, a repack, a `mmq_load_w_q8_0s`, and an
  `mmvq` path for batch one.
* **Attention, 0.3 ms, and it is the arithmetic.** `attn_decode_gqa_f32` is
  46.1 us a layer against the probe's 36.2. Deleting its arithmetic and keeping
  the tile loads and the barriers measures **38.8 us** — so the shared staging
  and the two barriers a tile are worth 2.6 us and the dot products are worth
  **7.3**. That is the opposite of what this kernel's history suggests and it
  points somewhere specific: `m16n8k16` takes f16 operands and accumulates in
  f32, so it removes both the FMAs *and* the `__half22float2` on every one of
  them. Sixteen MMA instructions a tile against about two hundred and thirty.
  A 4-of-16 row utilization wastes three quarters of the tensor core, which
  `mmqnm_*` established is free anyway.
* **The step's own idle, ~0.4 ms.** tuili's GPU is idle 5.6% of a served step
  against vLLM's 1.7%. vLLM prepares step *n+1* while the GPU runs step *n*;
  tuili synchronizes on the sampled tokens every step.
* **Sampling, 0.08 ms.** 176 us to read 16 MB of logits is 94 GB/s.

## What the arithmetic says about catching up

The GEMM now matches Marlin. What is left, priced against a 6.97 ms step:

| available | worth |
|---|---|
| attention's arithmetic, via `m16n8k16` | 3% |
| the GEMM's memsets, via Marlin's locks | 1.9% |
| ~~the vocab projection at 128 rows a block~~ | taken: 3.7%, see below |
| the remaining small kernels | 1.4% |
| **all of it** | **8%, to about 4970** |

The gap is 18%. So the identified work does not close it, and neither does the
one lever big enough to matter on its own: an fp8 KV cache would take attention
from 1.5 ms to 0.75 and is worth 10% — but vLLM has `--kv-cache-dtype fp8` too
and would take the same 10%, so it moves both engines and closes nothing.

**Beating vLLM at this batch needs a GEMM that beats Marlin**, and the 14% this
kernel cannot account for is where that would come from. Everything cheap has
been measured; see the elimination table above. What is left is the architecture:
TMA and warp specialization, which is a different kernel, not a tuned one.

### The last mechanism, and it does not hold

`mmqy1w8s2d3` is the depth-three weight prefetch — two k-tiles in flight instead
of one — and it exists because the 14% has one mechanism left that fits the
numbers. Count what a warp has outstanding: four 16-byte loads across a loop body
of about 250 instructions, at 23 warps an SM, is roughly 276 KB in flight per SM,
where 1345 GB/s at ~700 ns of latency wants 940 KB. The kernel is short of
memory-level parallelism, and depth three is the only way left to buy it that the
register budget allows — 101 registers against 66, two resident blocks against
2.9, which the occupancy probe says costs nothing.

**It loses on both cards** — 91.6-93.4 ms of matmuls against depth two's 90.7 on
the Blackwell, 2% down at every shape on an A4000. Prefetching only the quants
and reading the scale at its use point is worse still (93.2-94.2): the scale load
lands on the critical path in front of the MMAs, and it does not save the
registers it was meant to. Parameterizing the depth at all costs the default 34
registers — the third set is declared whether or not it is written — which is why
the shape is a comment in `mmq.cu` and not a variant.

The operating point was then confirmed two-dimensionally, since the one
interaction that paid was `bps` against the prefetch: variant against `bps`,
2580 launches each, `mmqy1w8s2` at 96.6 / **91.9** / 99.0 / 95.8 / 97.7 for
bps 1/2/3/4/6 and `mmqy2w8s2` at 96.8 / 97.5 / 100.5 / 99.4 / 98.6. The default
is the best cell in the grid. So the last mechanism that fit
the numbers does not hold either, and the sharpest way to put what is left is
this: the probe reaches 1345 GB/s at two blocks an SM, this kernel reaches 1037
at 2.9, and eleven separate isolations do not say why.

## Running it

Remote box, GPU 3. tuili lives in `~/tuili`, the load generator is `~/bench.py`,
vLLM starts with `~/run_vllm3.sh` on port 8232. `nsys` is at
`/usr/local/cuda/bin/nsys`; `ncu` is there too and refuses to run without
`ERR_NVGPUCTRPERM` cleared, so every kernel-level finding here comes from
timing, not counters.

Three traps, all of which cost measurement rounds:

* **`TUILI_PROFILE` disables CUDA graph capture**, because per-kernel events
  cannot coexist with it. A throughput number measured under it is ~40% low.
  Use `TUILI_STEP_TIMING` for host-side step timing and `TUILI_PROFILE` only
  for per-kernel shares — and subtract 3.4 us a launch before believing one.
* **The end-to-end harness is noisy to about ±5%.** `batch_bench` is worse: it
  could not resolve the 4.4% the fused projections are worth. Anything below
  that is only visible per-kernel.
* **vLLM v1 runs the model in a child process.** `nsys` needs
  `--trace-fork-before-exec=true` or it traces an idle parent.
