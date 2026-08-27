# Marlin, vendored for reference

Sources fetched from `vllm-project/vllm` at `v0.10.0`, `csrc/quantization/gptq_marlin/`
plus `csrc/core/scalar_type.hpp`. Apache 2.0; see `../LICENSE.marlin`.

Marlin is a separate project — `IST-DASLab/marlin`, Frantar, Castro, Chen,
Hoefler and Alistarh, arXiv:2408.11743 — that vLLM carries in-tree and has
extended considerably (AWQ, fp8, nvfp4, MoE, act-order). Nothing here is
compiled; these files are the reference the port in
`crates/kernels/src/cu/mmq.cu` is being written against.

## What was established by reading them

The paper's subject is exactly the regime `infero` is stuck in. Four-bit kernels
before Marlin — AWQ's and GPTQ's own, and llama.cpp's MMQ, which this repo's
`mmq.cu` follows — help at a batch of one, where the step is bandwidth-bound,
and stop helping as the batch grows. Marlin's claim is a near-ideal 4x out to a
batch of 16-32. That llama.cpp also measures ~500 tok/s at 32 clients here is
the same fact from the other side: both engines are on the pre-Marlin design.

### Launch shape

`gptq_marlin.cu`, `small_batch_thread_configs` / `large_batch_thread_configs`:

| | thread_k | thread_n | threads |
| --- | --- | --- | --- |
| m <= 16 | 128 | 128 | 256 |
| larger | 64 | 256 | 256 |

`thread_k` and `thread_n` size one *iteration tile*, not the block's whole
share. `max_thread_m_blocks` is 4, so a block holds up to 64 tokens.

### Work partition (`marlin_template.h`, around line 355)

```c
int k_tiles = prob_k / 16 / thread_k_blocks;
int n_tiles = prob_n / 16 / thread_n_blocks;
int iters   = div_ceil(k_tiles * n_tiles * parallel, gridDim.x);
```

`gridDim.x` comes from the device, not the matrix. The (n slice, k chunk) pairs
are flattened k-major and block `b` takes `[iters*b, iters*(b+1))`, continuing
across n-slice boundaries. Most slices therefore come out of one block and store
directly; only straddling runs reduce, coordinated by `locks`, `slice_idx` and a
fp32 `C_tmp`, with `use_atomic_add` / `use_fp32_reduce` as alternatives.

This repo ported the partition (`mmqs_*` in `mmq.cu`) and measured it level with
the cruder `gridDim.z` split it was meant to replace. The reduction traffic was
never the constraint; the block count was, and the crude split already supplied
it.

### The pipeline (`marlin_template.h`, 1516-1580)

This is the part `mmq.cu` has no equivalent of.

```c
start_pipes:                                 // prime stages-1 async copies
  for i in 0..stages-1: fetch_to_shared(i, i, i < slice_iters);
  wait_for_stage(); fetch_to_registers(0, 0);

while (slice_iters) {
  for (pipe = 0; pipe < stages;) {
    for (k = 0; k < b_sh_wr_iters; k++) {
      fetch_to_registers(k + 1, pipe % stages);        // one k ahead
      if (k == b_sh_wr_iters - 2) {
        fetch_to_shared((pipe + stages - 1) % stages,  // issue the next stage
                        pipe, slice_iters >= stages);
        pipe++; wait_for_stage();
      }
      matmul(k);
    }
    slice_iters--;
  }
}
```

Two levels: global to shared is `stages` deep (four) through `cp.async`, drained
with `cp_async_wait<stages - 2>`; shared to registers runs one k-step ahead of
the MMA. The inner loop contains no `__syncthreads`. `cp_async4` is raw inline
PTX in `marlin.cuh`, so none of this needs `cuda_pipeline.h` — it is reachable
from NVRTC.

### Operands

`dequant.h` turns four bits into f16 with `lop3`: place the nibble in the
mantissa of f16 1024.0 (`EX = 0x64006400`), then `hsub2` against `0x64086408`
for the low half and `hfma2` with `0x2c002c00` / `0xd480d480` for the high one.
Two values per instruction, landing directly in an `mma.m16n8k16.f16` fragment.
Activations stay f16 throughout, so there is no activation quantization, no
per-group activation scale, no stored sum and no zero-point-times-sum term.

### Why the port has to be whole

Measured here, each of these alone is worth nothing:

| tried alone | result |
| --- | --- |
| weight fragments straight from global, no shared expansion | 0% |
| four groups under one scale, one s32 accumulation | 0% |
| a 32x32 or 128x32 register tile per warp | 0% |
| the striped partition above | 0% |

They are not independent optimizations. A wide register tile gives the MMAs
something to be overlapped *with*; `cp.async` gives the overlap; the f16 operand
path removes the bookkeeping that would otherwise sit between them. Taking one
without the others trades a win on one axis for a loss on another and nets zero,
which is what all four measurements say.

The one change that did land — splitting k twelve ways rather than three, worth
22% — is orthogonal to all of it: it is about having enough blocks resident, not
about what happens inside one.

## Porting order

The pieces are interdependent, so the order matters — and each step has to be
correct on its own before the next, or a wrong answer somewhere in the middle is
unattributable.

1. ~~**`cp.async` ring buffer onto the wide tile.**~~ Done — `mmqa_*` and
   `mmqr_*` in `mmq.cu`. The description above undercounts the work: the
   pipeline is *two* levels, and only the first is what "give it four shared
   stages" asks for. Both landed.

   | 256 tokens of history, tok/s | batch 8 | 16 | 32 |
   | --- | --- | --- | --- |
   | `mmqd`, the default | 329 | 565 | 527 |
   | `mmqx4w4`, the wide tile alone | 200 | 362 | 505 |
   | `mmqa4w4s4`, + global→shared | 204 | 380 | 579 |
   | `mmqa2w4s4` | 212 | 395 | 623 |
   | `mmqr2w2s4`, + shared→register | 258 | 472 | 635 |
   | `mmqr2w4s4` | **242** | **447** | **702** |

   The two levels behave differently and it is worth keeping them apart:

   - **Level one** (`cp.async` ring, `mmqa_*`) pays only at batch 32: +18%
     there, nothing below. It hides the weight stream, and how much of that
     there is to hide per block scales with the tokens.
   - **Level two** (`fetch_to_registers` one k-substep ahead, `mmqr_*`) pays
     +14% at *every* batch width. It hides an operand load behind the preceding
     MMA, and that ratio does not move with the batch.

   Four things came out of writing it that the reading above did not predict:

   - `cp.async` is a copy, not a transform, so the shared tile has to hold the
     `block_q8_1` bytes verbatim — 36 per group, scale interleaved with quants.
     That deletes the whole staging pass, and it also costs `ldmatrix`, which
     cannot address a 36-byte stride. Marlin does not hit this because its
     activations are plain f16.
   - The weight scales had to move to registers in the same change. In shared
     they need a second `__syncthreads` to publish, and a barrier between the
     `cp.async` issue and the MMAs is exactly the overlap being bought.
   - Marlin's mid-tile `cp.async` issue (`marlin_template.h:1566`, at
     `k == b_sh_wr_iters - 2`) is not a scheduling nicety, it is what makes the
     overwrite safe once level two exists. The buffer being overwritten had its
     last read at that same position one stage earlier, with a barrier
     immediately after. Level one can afford to issue at the top of the tile
     because it has no reads outstanding; level two cannot.
   - Stage depth matters least of the three: two stages measured 612 against
     four stages' 623. Tile width and the second level are the substance.

   That left a 21% loss at 16 tokens and 26% at 8, all of it the block count.
   Not registers — the driver reports every one of these kernels limited by its
   `__launch_bounds__` rather than below it.

   **The striped partition's null result did not survive the rewrite.** `mmqs_*`
   measured level with the `gridDim.z` split it replaced, and the note in
   `mmq.cu` explains why that was unsurprising. Against `mmqr_*` it is worth
   30% at four tokens and 16-30% through the middle — exactly the range the port
   had been losing in. `mmqsr_*` is that partition around the register-pipelined
   loop:

   | 256 tokens of history, tok/s | batch 4 | 8 | 16 | 32 |
   | --- | --- | --- | --- | --- |
   | `mmqd`, the default | **177** | **329** | 565 | 527 |
   | `mmqr2w4s4` | 123 | 244 | 447 | 713 |
   | `mmqsr2w4s4` | 168 | 323 | **580** | **741** |

   At or above the default from eight tokens up, and 41% above it at 32. Two
   findings from the blocks-per-SM sweep behind that row:

   - The two ends want opposite grids — 16 tokens keeps rising to 48 blocks per
     SM, 32 tokens peaks at 8 and falls off. The split is the token-tile count,
     so the default keys off `tiles` rather than off a constant.
   - 4 blocks per SM, which is `mmqs`'s default, is the *worst* point in the
     sweep, below 2 at every width. The curve is not monotonic at the low end
     and nothing here explains that; it stays a measurement.

   One caution on reading small differences: three cold runs of the same binary
   spread 0.8% at batch 32 (617, 623, 613), not the 0.2% claimed below.

   The default still dispatches `mmqd`. Switching it is a real decision — it
   would change every caller of the crate, it is still 5% down at batch 4, and
   these numbers are one matrix shape on one card.
2. **Keep the weights on the direct path** (`mmqd_*`'s global-to-register
   fragments) for the first version. Marlin stages them packed in shared, but
   that is an optimization on top, not a prerequisite.
3. ~~**f16 operands last.**~~ Done — `mmqf_*`, and **it is worth nothing**.
   The prediction above was that this step was the largest of the three; it
   measures 3% up at four tokens and 7% down at thirty-two against `mmqsr_*`.

   Everything the paragraph claimed it would remove, it removes: the activation
   quantization, the per-group scale, the stored sum, the zero-point term, the
   whole epilogue. The f32 accumulator spans all of k, so the store *is* the
   epilogue. It buys nothing, because none of that was the constraint.

   What it costs is legible in one comparison. Two pipeline stages beat four
   here by 23% at 32 tokens, where in `mmqa_*`/`mmqr_*` four beat two — and the
   only difference between them is that a half is 2 bytes against Q8_1's 1.125,
   so four stages cost 34.8 KB of shared memory rather than 19.5, and the SM
   holds fewer blocks. The step trades the thing that was never binding for the
   thing that is.

   It did need the model dispatch change and a second reference, both as
   described. The reference is worth keeping for a different reason than
   expected: this path never quantizes the activation, so its test asserts 0.2%
   against the dequantized weights where the integer kernels need 2%.

### What the six negative results have in common

Nine structural changes have now been measured on this kernel. They sort
cleanly, and not by how clever they are:

| what the change did | measured |
| --- | --- |
| removed the weight staging (`mmqd`) | 0% |
| collapsed four groups under one scale (`mmqp`) | 0% |
| widened the register tile alone (`mmqx`) | 0% |
| striped partition, against the old narrow kernel | 0% |
| removed the entire epilogue (`mmqf`) | 0% |
| synchronous double buffering | 0% |
| split k twelve ways instead of three | +22% |
| `cp.async` ring on the wide tile (`mmqa`) | +18% |
| register double buffering (`mmqr`) | +14% |
| striped partition, against the pipelined kernel (`mmqsr`) | +30% |

Every change that removed *instructions* measured zero. Every change that put
more *weight loads in flight at once* measured large. Six and four, with no
overlap. This kernel is latency-bound on the weight stream and has been the
whole time, and the reading of Marlin at the top of this file — which ranks the
register tile first and the operand path fourth — has the ranking upside down
for this machine.

Which says what is left. The activations are now fully pipelined: `cp.async`
into a shared ring, then one k-substep ahead into registers. **The weights are
still read synchronously, global straight to register, four bytes at a time.**

The obvious next move is to prefetch them further ahead, and it does not work.
`mmqb_*` runs the weight ring at depth 4 instead of 2 and measures 5-8% *down*
at every batch width. `Kernels::kernel_registers` says why: 168 registers a
thread against 128, which is three resident blocks per SM against four.

That is worth stating as a rule, because it rules out a family of ideas rather
than one: **memory-level parallelism cannot be bought with registers here.**
Registers are what buy occupancy, and occupancy is where the parallelism was
coming from. Any deeper software pipeline over this operand hits the same wall.

So the only remaining way to put more weight bytes in flight is to make each
load carry more of them, and that is a layout question, not a scheduling one:

| | this port | Marlin |
| --- | --- | --- |
| weight matrix type | `block_q4_g128*`, read as `uint32_t` | `const int4*` (`marlin_template.h:59`) |
| bytes per weight load | 4, twice per fragment | 16, once (`cp_async4`, :757) |
| global to shared | none — straight to register | `cp.async`, four stages |
| B fragment in registers | masked from two words | one `I4` (:689) |
| what makes that possible | — | `gptq_marlin_repack` at load time |

**And that turned out to be worth nothing.** `mmqfp_*` in `mmq.cu` is the f16
kernel with the repacked layout assumed — sixteen-byte loads, no `prmt` in the
dequantization — computing wrong answers with the right traffic, so the repack
could be priced before the loader, `unpack_row`, the mat-vec, the float path
and every test pinning them were touched:

| GB/s of weights | 8 tokens | 32 tokens |
| --- | --- | --- |
| `mmqf1w8s2`, as built | **330** | **214** |
| repacked, one 16-byte request | 306 | 210 |
| repacked, two 8-byte requests | 303 | 180 |
| repacked, four 4-byte requests | 292 | 184 |

Level to 7% down, over three different request shapes on the same layout. So
the piece of Marlin this port skipped twice is a piece it did not need, and the
skips were right for a reason nobody had established at the time.

Two artefacts had to come out of that probe before it said anything, and both
are worth remembering because each was the size of the effect:

- `ldmatrix` on the A side measured 10% slower. `MMQ_XF_STRIDE` is 544, chosen
  so an 8-byte gather at `8 * (lane % 4)` is bank-conflict-free — and 544 is 8
  words mod 32, exactly the stride that two-ways `ldmatrix`. Whether `ldmatrix`
  pays is a question about the *activation* tile and does not belong in a probe
  about the weight layout.
- `bsrc[j] - wq` is a pointer difference on a 68-byte struct, so it compiles to
  a division by a non-power-of-two in the inner loop. That alone cost 8%.

### The probe that turned this around

Before repacking anything, price the load width: `mmq_bw_probe_w4` and
`mmq_bw_probe_w16` in `mmq.cu` walk a weight matrix with this kernel's exact
access pattern and nothing else. On `ffn_gate` (4096 by 14336):

| | achieved |
| --- | --- |
| four-byte loads, the pattern this kernel uses | 340 GB/s |
| sixteen-byte loads, what a repack would allow | 388 GB/s |

**The first version of this probe was wrong and said 294**, because it folded
every load into one `acc ^= ...` and measured its own dependency chain. The
tell was the real kernel beating it: `mmqne1w4s2d2` reached 337 GB/s against a
claimed 294 ceiling. Four independent accumulators fixed it. Worth recording
because the wrong number made the weight repack look like a 33% lever when it
is a 14% one, and that number was already written down here as settled.

Measured per GEMM rather than per decode step (`tests/bwidth.rs`):

| GB/s of weights | 8 tokens | 32 tokens |
| --- | --- | --- |
| the four-byte ceiling | 340 | 340 |
| `mmqd`, the default | 240 | 93 |
| `mmqsr2w4s4`, the best of the port so far | 238 | 150 |
| `mmql1w4s2d2` | 300 | 187 |
| **`mmqf1w8s2`** | **331** | **215** |
| `mmqne1w4s2d2`, integer with the epilogue stubbed out | 337 | 218 |

The last 8% of that came from sweeping blocks-per-SM for this kernel instead of
inheriting `mmql_*`'s: it holds eight warps to four and twice the shared memory
a stage, and wants 24 blocks per SM at one token tile and 4 at two, against the
48 and 8 it had been given. Interleaved three ways because a single sweep put
the spread at 8% and the ordering at noise.

**At eight tokens the kernel is done** — 331 against a 341 ceiling, with the
epilogue-free probe at 337. **At thirty-two it is at 63%**,
and that deficit is not the load width, because the same loads reach the
ceiling at eight tokens. It is the work that scales with the token tiles: the
A-fragment shared loads, the MMAs and the epilogue.

**And `NBLK=1` wins — no wide register tile at all.** That contradicts the
premise this port was built on: the reading at the top of this file ranks the
64x64 register tile first, on the argument that a warp should own more output
so each operand load amortizes further. On this machine the opposite holds. A
wide tile costs registers, registers cost resident blocks — 158 against 124 at
two token tiles, three blocks per SM against four — and this kernel is waiting
on memory, so what it wants is warps, not amortization.

### And the f16 result was wrong, for the same reason

`mmqf_*` was recorded above as a clean negative — 3% up at four tokens, 7% down
at thirty-two — and that measurement was taken at NBLK=2, because that was the
only shape this kernel had at the time. Re-measured at the narrow shape it
reverses:

| GB/s of weights on `ffn_gate` | 8 tokens | 32 tokens |
| --- | --- | --- |
| `mmql1w4s2d2`, the best integer shape | 299 | 187 |
| `mmqf2w4s2`, the shape f16 was first judged on | 234 | 132 |
| **`mmqf1w8s2`** | **308** | **204** |
| `mmqne1w4s2d2`, integer with the epilogue stubbed out | 338 | 218 |

The epilogue *was* worth removing — the probe puts it at 17% — and the first
cut of the f16 path spent more than that on a register tile it did not need.
Eight warps rather than four, because halving k per MMA doubles the MMAs and
the shape wants its issue slots back.

The lesson is about method, not about f16: an unswept shape parameter turned a
17% win into a 7% loss, and the wrong conclusion sat in this file as settled
until someone swept it.

End to end, against the same `batch_bench` every other row here used:

| tok/s | batch 4 | 8 | 16 | 32 |
| --- | --- | --- | --- | --- |
| `mmqd`, the default | 177 | 329 | 565 | 527 |
| `mmqsr2w4s4` | 168 | 323 | 580 | 741 |
| `mmqb1w4s2d2` | 188 | 362 | 625 | 770 |
| `mmql1w4s2d2` | 190 | 366 | 635 | 786 |
| `mmqf1w8s2` | 211 | 394 | 681 | 827 |
| **`mmqf1w8s2` + the fused f16 norm, now the default** | **220** | **407** | **699** | **838** |

The first shapes in this effort that beat the default at *every* batch width —
by 25% at four tokens and 59% at thirty-two — which puts the gap to vLLM at
2.1x where it started at 3.4x.

`mmqf1w8s2` is now what `Kernels::mmq()` picks for Q4_G128 without being asked,
validated on every shape a Llama-3.1-8B step touches rather than on the one it
was tuned against:

| GB/s of weights, `mmqd` -> now | 8 tokens | 32 tokens |
| --- | --- | --- |
| `ffn_gate`, 4096x14336 | 237 -> 331 | 93 -> 214 |
| `ffn_down`, 14336x4096 | 159 -> 341 | 75 -> 227 |
| `attn_q`, 4096x4096 | 180 -> 260 | 87 -> 172 |
| `attn_k`, 4096x1024 | 167 -> 179 | 72 -> 115 |

The last 4% came from the profile rather than the kernel. With the f16 GEMM in
place, `rms_norm_q8_1` was still producing a Q8_1 activation nothing read
(1.8% of a batch-32 step) and every projection then paid its own `to_f16` over
the same row (3.6%). `rms_norm_f16_f32` writes both forms in the pass that
already has the row in registers, and the projections sharing an input — q/k/v,
gate/up — share it, which is what `norm_for_group` already did for Q8_1.

### What is left, and where

`INFERO_PROFILE=1` at batch 32 puts the GEMM at 72% of the step, attention at
20%, everything else at 7% — so the GEMM is still the thing.

At 32 tokens it runs 185 GB/s against the 294 it reaches at 8. Forcing one
token tile there gives 150 GB/s of *counted* weights, but that shape reads the
weights twice, so the real traffic is 300 GB/s — the ceiling again. **Two token
tiles is not short of memory bandwidth; it is short of everything else.** Per
k-tile per warp it issues 8 weight loads, 64 A-fragment shared loads and 64
accumulator updates, and the last two scale with the token tiles while the
first does not. That points at `ldmatrix` for the A side and at the epilogue —
neither of which the raw Q8_1 activation ring can have, because `cp.async` is a
copy and Q8_1's 36-byte group stride is not 16-byte aligned.

Priced against the machine. A batch-32 step reads 5.03 GB of weights:

| | ms/step | effective weight bandwidth |
| --- | --- | --- |
| `mmqd`, where this started | 60.7 | 83 GB/s |
| `mmqsr2w4s4` | 42.5 | 118 GB/s |
| `mmqf1w8s2`, now | 38.7 | **130 GB/s** |
| vLLM, same card and checkpoint | 18.0 | 279 GB/s |
| this card's own mat-vec | — | 375 GB/s |
| pure streaming read | — | 405 GB/s |

Note how little the end-to-end number moved against how much the GEMM did:
the isolated `ffn_gate` went 150 to 185 GB/s while the step went 118 to 121.
Most of a decode step's weight volume is not in the matrices this kernel is
being tuned on.

Keep `INFERO_MMQ_VARIANT` as the switch and add each stage as a new name, so
every step stays A/B-able against the ones before it.

## How to measure

Two instruments, and picking the wrong one wasted a day here.

- `cargo run --release -p infero-model --example batch_bench -- <awq-dir> 256`
  reproduces to 0.2% and is what a kernel change should be judged on.
- The server benchmark varies by **±18%** at 32 clients — identical code has
  measured 434 and 518 — because the batch width changes as sequences finish.
  Use it for the headline number against vLLM and Ollama, never for an A/B.

Let the card fall below 55 C between runs. A back-to-back sweep drops SM clocks
enough to invert an ordering: the same ctx-256 step measured 14.77 ms cold and
16.77 ms hot.

Two traps that cost real time:

- Selecting the two arms of an A/B through *different* switches. `direct` looked
  worth 10% that way and is worth 0% when both arms go through
  `INFERO_MMQ_VARIANT`.
- Concluding from a mechanism rather than a measurement. Five structural changes
  here were argued for convincingly and measured at zero. The one that worked
  differed from a failed attempt only in a constant.

## Where this stands, and what is ruled out

838 tok/s at batch 32 against vLLM's 1775 on the same card and checkpoint:
2.12x, from 3.37x. The decode step splits 71% GEMM, 22% attention, 7% the rest.

**Attention is not inefficient.** At batch 32 and 256 tokens of context the KV
cache is 1.2 GB a step against the weights' 5.03 — 24% of the traffic for 22%
of the time, which is bandwidth-proportional. The K row cannot be shared across
tokens the way it is shared across a GQA group, because a decode batch is 32
*different sequences* and each indexes its own pages. TurboQuant is implemented
here and would cut those bytes fourfold; measured, `INFERO_KV=tq4` runs 537 tok/s
against F16's 838, so the dequantization costs more than the bandwidth saves.

**The GEMM is done at small batch and unexplained at large.** It reaches 97% of
its access pattern's ceiling at 8 tokens and 63% at 32, and at 32 tokens it is:

| ruled out | by | worth |
| --- | --- | --- |
| the tensor cores | `mmqnm_*`, MMAs deleted | 0% |
| weight load width | `mmqfp_*`, repacked layout assumed | 0% |
| the A-fragment order | `mmqm_*`, `ldmatrix` + permuted activations | -25% |
| the activation stream | `mmqnh_*`, half the copies | 7% |
| more weight rows a block | `mmqf2w8s2`, `mmqf1w16s2`, `mmqf4w8s2` | -30% |
| eight-bit activations | `mmqe_*`, Q8_1 ring under the f16 MMA | -30% |
| the KV cache | `INFERO_KV=tq4` | -36% |

The occupancy story is what is left, and `mmqe_*` is what killed it. At two
token tiles the f16 ring is 34.8 KB and the SM holds two blocks of eight warps;
a Q8_1 ring is 19.5 KB and would hold four. `mmqe_*` builds exactly that — Q8_1
activations widened to f16 in registers, so the f32 accumulator and the absent
epilogue both survive — and it measures 30% *down*, by as much at eight tokens,
where the ring was already small enough, as at thirty-two.

So the ten extra instructions an A fragment costs outweigh doubling the
resident warps. Which corrects the reading of `mmqnm_*`: the tensor cores being
free means the MMA pipe is idle, not that the issue slots are. On the operand
path this kernel has no ALU to spare at all.

That leaves 32 tokens at 63% of its ceiling with every constructible change
measured and negative, and five subtractive probes unusable because deleting
work from this kernel changes its schedule and the schedule is what is being
measured. `mmqnm_*` is the exception and the reason to trust it — it measured
identical, which an artefact cannot fake.

## The server was measuring something else

Every number above is `batch_bench`, a fixed batch with nothing else running.
The comparison against vLLM is the load generator, and there the same build
measured 368 tok/s at 32 clients while the fixed batch measured 831.

`--max-seqs` defaulted to **8**. However many clients connected, the scheduler
had eight sequences to batch, so the GEMM ran at a batch of eight and the 32-
client row was measuring a shape none of this work targets — while vLLM was
being given `--max-num-seqs 64`. Raising it to 32 doubles the server: 368 to
725, without touching a kernel.

It was 8 because the KV pool defaulted to `max_seqs * ctx` slots, and 32
sequences of 4096 tokens is 17 GB on an 8B model — so asking for more
concurrency made the server refuse to start. `make_pool` now bisects on the
pool's own byte count against the VRAM left after the weights, and the
scheduler already refuses a prompt that will not fit, so oversubscribing is
safe.

| infero AWQ, 32 clients | tok/s | behind vLLM |
| --- | --- | --- |
| as recorded in the top-level README | 515 | 3.45x |
| this session's kernels, `--max-seqs 8` | 368 | 4.82x |
| this session's kernels, defaults fixed | **722** | **2.46x** |

The middle row is the one worth keeping: the kernel work is worth 58% on a
fixed batch and *negative* on the server, because at a batch of eight the
shapes it tuned for do not appear. A kernel benchmark and a serving benchmark
disagreed by a factor of two and the disagreement was a command-line default.

## The roofline, and why 2.5x is not one more kernel

Priced per decode step at a batch of eight, where `INFERO_PROFILE=1` puts the
GEMM at 78% and attention at 13%:

| | ms | note |
| --- | --- | --- |
| `mmq` | 14.7 | 4.27 GB of weights, so 290 GB/s |
| attention | 2.7 | its own bandwidth bound is 0.75 |
| norms, rope, silu, adds, `to_f16` | 1.4 | eight small elementwise kernels |
| **infero** | **18.8** | 425 tok/s |
| **vLLM** | **14.2** | 564 tok/s |

**This kernel alone takes as long as vLLM's whole step.** Set attention to its
bandwidth bound and delete every elementwise kernel and the step is 15.4 ms —
519 tok/s, still short of 564. There is no arrangement of the rest of the
engine that closes this; the GEMM has to get faster.

And the GEMM is at its access pattern's ceiling: `mmq_bw_probe_w4` reaches 340
GB/s reading these weights and nothing else, `mmqf1w8s2` reaches 331. The only
higher ceiling on this card is the sixteen-byte one at 388, which needs the
weights repacked — and `mmqfp_*`, which assumes exactly that layout, measures
level to 7% down.

So the honest position is that the remaining 2.5x is not a missing
optimization. It is that vLLM's Marlin moves these bytes through a layout this
port declined to adopt, and adopting it measures as worth nothing *in this
kernel* — which means the layout and the kernel around it are one thing, and
half of it is not portable. That is the same sentence the top of this file
opens with, arrived at from the other end.

Two things left that are worth doing and are not this:

- **Attention is 3.3x off its bandwidth bound** (2.7 ms against 0.75) and the
  value side still reads V once per query head rather than once per KV head.
  `attn_output_gqa_f32` exists and is switched off because it costs the grid;
  a split version of it does not exist yet. Worth ~10% of the step.
- **Eight elementwise kernels for 1.4 ms.** They are launch- and
  latency-bound, not bandwidth-bound, and most of them read and write the same
  activation.

### Attention, priced

Two shapes were built for the value side and both lose at these sizes, which is
worth recording because the arithmetic argues for both.

`attn_output_gqa_split_f32` reads each V row once for the whole query group —
a quarter of the traffic at 32 heads over 8 — and splits the key range to buy
back the grid that grouping costs. Wired up so it actually runs at a batch of
eight, it measures **387 tok/s against 406**. The traffic it saves is not DRAM:
a layer's V cache at 256 tokens of context is 590 KB and lives in L2, so the
four reads are L2 reads, while the partial sums the split writes and the reduce
that consumes them are new traffic and a third launch.

The fused path (`attn_flash_f32`) is gated to shapes with too few blocks, and
the guess was that its other two properties — the score matrix never reaching
HBM, two launches a layer instead of three — would pay at any batch.
`INFERO_FLASH_WIDE=1` measures **379 against 405** at a batch of eight and **730
against 852** at thirty-two. The fused block holds a chunk of scores in shared
memory and that caps its occupancy; the unfused one does not.

So attention sits at 2.7 ms against a 0.75 ms bandwidth bound, and the two
obvious ways to close it are both slower. What that bound assumes is that the
KV cache is read from DRAM once, and at this context length it is not read from
DRAM at all.

## The repack, done for real

`mmqfp_*` assumed the repacked layout and measured level, and that reading
stood in this file as "the layout and the kernel around it are one thing, half
of it is not portable". It was wrong, and wrong because the probe was a probe:
an address forced into alignment, a dequantization pairing that did not match,
a pointer difference on a 68-byte struct compiling to a division in the inner
loop. Built as the layout itself — `awq::transpose_words`, `mmqz_*`,
`WeightType::Q4G128T` — it measures:

| GB/s of weights | 8 tokens | 32 tokens |
| --- | --- | --- |
| `ffn_gate` 4096x14336 | 332 -> 338 | 215 -> **252** |
| `ffn_down` 14336x4096 | 342 -> 328 | 227 -> 229 |
| `attn_q` 4096x4096 | 261 -> 267 | 173 -> **196** |
| `attn_k` 4096x1024 | 179 -> **201** | 116 -> 123 |

Weighted by the bytes each shape contributes to a layer, 11% on the GEMM and
5.8% on the decode step. The transform is a transpose of the 4x4 matrix of
4-byte words inside each 64-byte nibble run — which puts the four words a lane
fetches side by side — plus moving the scales out of the blocks, because `qs`
sits four bytes into a 68-byte block and no offset inside it is ever 16-byte
aligned. The split is per row, not global, so a kernel needs only the row base
it already computes; that is what lets the mat-vec macros take the new layout
with nothing but a different dot product. It costs `k % 512 == 0`, which every
real projection width satisfies.

### And it shipped wrong first

`batch_bench` measured the transposed path at 916 tok/s and the workspace tests
passed. Both were true and the model emitted `( ( ( ( (`.

The model's f16 branch picked its kernel from `Kernels::mmq_f16_variant()`,
which returns `mmqf1w8s2` — the kernel for the *packed* layout — so transposed
weights went through a reader that misinterprets every byte. Nothing failed:
`batch_bench` times steps and never looks at what comes out of them, and the
unit tests exercised `mmqz` directly rather than through the dispatch. The
selector is now `mmq_f16_variant_for(ty)`, and the lesson is that a timing
harness is not a test.

### Where the scales live

The transposed layout was built twice. The first put each row's scales after
its own quants, so a kernel needs only the row base it already computes and the
mat-vec macros take the layout with nothing but a different dot product. The
second puts all the quants first and all the scales after, which costs the
mat-vec a matrix width it was not being handed — `wn`, threaded through five
macros, picked alongside the matrix in the fused ones.

| GB/s of weights | 8 tokens | 32 tokens |
| --- | --- | --- |
| per row, `ffn_gate` | **344** | 224 |
| global, `ffn_gate` | 338 | **251** |
| per row, `ffn_down` | **342** | 224 |
| global, `ffn_down` | 328 | **229** |

Each wins at one batch width, and `batch_bench` settles it: 79.2 / 429.8 /
904.7 tok/s at batches 1 / 8 / 32 against the per-row layout's 79.3 / 412.8 /
894.7. Global, by 4% at eight tokens and 1% at thirty-two.

The load generator disagreed — 758 against 782 at 32 clients — and it is wrong
rather than measuring something else: three consecutive runs of the same binary
gave 768.0, 756.9 and 744.2 as the card warmed. That is the ±18% this file
warns about, and it is why the layout question was decided on the fixed-batch
bench and only the headline taken from the server.

One artefact cost 13% on the way and is worth remembering: writing the row's
*index* into the register that indexes weights, and multiplying by the row
stride inside the k-loop, puts a 64-bit multiply on every weight address.
Storing the byte offset instead recovered 326 GB/s to 344.

## The barrier was the thing

Two token tiles crossed one `__syncthreads` per k-tile and one token tile
crossed one per k-tile too, so the barrier count looked like a constant — but
`mmqg_*`, which crosses two and has *more* resident warps, measured 6% slower,
and that was the only surviving explanation for 32 tokens running at 65% of the
weight-read ceiling where 8 runs at 89%.

`__syncthreads()` is `bar.sync 0` — arrive and wait in one instruction, with
nothing able to happen between them. PTX's named barriers split the two, so a
block can announce it has finished reading the ring, load the k-tile's weights
(global, and needing nothing the barrier publishes), and only then wait.
`mmqy_*` is that, and it wins on every shape and both batch widths:

| GB/s of weights | 8 tokens | 32 tokens |
| --- | --- | --- |
| `ffn_gate` | 338 -> 347 | 251 -> **273** |
| `ffn_down` | 328 -> **355** | 229 -> **283** |
| `attn_q` | 266 -> **287** | 196 -> **209** |
| `attn_k` | 201 -> 202 | 119 -> **128** |

End to end: 79.3 / 445.9 / 933.5 tok/s at batches 1 / 8 / 32, against 79.2 /
429.8 / 904.7 before. On the load generator, 78.1 / 405 / 782 at 1 / 8 / 32
clients.

The count is twice the thread count, and getting it wrong hangs rather than
fails: `bar.sync` arrives as well as waits, so the pair contributes two
arrivals per thread and a barrier told to expect one per thread releases when
half of them have reached the arrive. The first version deadlocked the test
runner.

Hoisting the *dequantization* inside the barrier as well — it is registers in
and registers out and touches no shared memory — is the obvious next step and
is not done: four attempts at it by editing a 200-line macro left the braces
unbalanced, and a broken kernel in the tree is worse than a missing one.
