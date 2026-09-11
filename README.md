# infero

[中文](README.zh-CN.md)

An inference engine written in Rust for GGUF, AWQ, native FP8 (W8A8), and
native NVFP4 (W4A4) checkpoints, with hand-written kernels for both NVIDIA
(CUDA) and Apple Silicon (Metal) GPUs — no PyTorch, no `libtorch`, no ggml.
CUDA is the primary, most complete backend; Metal covers dense decode, GQA
attention and GatedDeltaNet today, with MoE, the vision tower, and the
INT4/FP8/NVFP4 tensor-core GEMM paths still CUDA-only (see
[Metal](#also-runs-on-apple-gpu-metal) below). The whole path from a model
checkpoint on disk to an OpenAI-compatible HTTP response is in this
repository.

<p align="center"><img src="docs/images/demo.png" width="700" alt="infero serving a GGUF model over an OpenAI-compatible endpoint"></p>

The official `openai` Python SDK works against it unmodified, streaming included.

## Status

Runs Qwen2, Llama-family, Qwen3-MoE, and Qwen3.5-style hybrid
attention/GatedDeltaNet (linear-attention) checkpoints, in GGUF, AWQ, native
FP8 (W8A8), and native NVFP4 (W4A4) form, on one GPU or sharded across several
with tensor parallelism. Correctness is checked against the reference
implementations rather than eyeballed: the tokenizer is compared
token-for-token against Hugging Face, the quantized decoders against the F16
build of the same checkpoint, and the forward pass against `transformers`
logits.

The KV cache can be compressed with TurboQuant; see below for what that
actually buys on this model.

Requests are served with continuous batching over a paged KV cache, CUDA
graphs replay a decode step's ~500-700 kernel launches as one, layers can be
offloaded to host memory to fit a model into less VRAM, and completed prompt
prefixes are cached across requests so a shared system prompt or a multi-turn
conversation only pays for its new tokens (prefix caching is off for models
with recurrent GatedDeltaNet layers, whose state a shared prefix cannot
reconstruct).

**Beyond plain text decode:**

- **MoE.** Sparse-FFN architectures (Qwen3-MoE and similar) load their experts
  individually — AWQ or FP8 per expert — and route through a dedicated top-k
  kernel at decode and a counting-sort-then-per-expert-GEMM path at prefill.
- **Hybrid linear attention (GatedDeltaNet).** Qwen3.5-style checkpoints
  interleave ordinary GQA attention blocks with GatedDeltaNet ones — a
  fixed-size per-sequence recurrent state, overwritten every step rather than
  grown, updated by the gated delta rule (`crates/model/src/qwen35.rs`,
  `crates/model/src/gdn_state.rs`). The paged KV pool grows a matching
  per-slot GDN state array alongside the ordinary attention pages.
- **Native NVFP4 (W4A4).** A checkpoint quantized end to end — attention,
  GatedDeltaNet and FFN projections, not just the FFN — loads and runs off
  its own real `hf_quant_config.json` (both the `quantized_layers` allowlist
  shape and the newer `quant_algo` + `exclude_modules` denylist shape) with a
  dedicated CUTLASS FP4 GEMM path, no dequantize-then-requantize step
  (`crates/model/src/weights.rs`'s `Fp4Targets`).
- **Vision and video.** Qwen3.5-VL-style checkpoints take `image_url` and
  `video_url` content parts over the same chat-completions endpoint, with
  M-RoPE for the resulting 3-axis position ids, chunked prefill of the vision
  placeholder tokens, and content-aware token pruning for long clips
  (`crates/model/src/qwen35_vision*.rs`, `crates/server/src/video.rs`).
- **Speculative decoding.** A GGUF's embedded or sidecar MTP head drafts `k`
  tokens ahead of the main model, with a device-resident Gumbel-max draft
  path available alongside the host one; `INFERO_SPEC_K` controls the draft
  depth and `0` turns it off (`crates/model/src/spec.rs`, `crates/model/src/mtp.rs`).
- **Tensor parallelism.** `--tensor-parallel-size N` shards a model across `N`
  GPUs over NCCL, one process a rank (`crates/model/examples/tp_generate.rs`,
  `docs/superpowers/specs/2026-09-05-tensor-parallel-design.md`). Vision/video
  requests, M-RoPE, and speculative decoding are not supported yet at `N > 1`
  — the server refuses them outright rather than silently mishandling them.
- **GPU-side sampling.** Penalty bookkeeping, top-k/top-p and the draw itself
  run in a device kernel for the batch shapes it covers, falling back to the
  host path — never a different distribution — outside them
  (`Kernels::sample_rows`/`sample_rows_split`/`sample_rows_greedy`).
- **Tool calls.** OpenAI-style `tools`/`tool_choice`, with `<tool_call>` tags
  scanned out of the model's own output and returned as structured
  `tool_calls`, streaming included (`crates/server/src/tool_call.rs`).

**Not yet:** split GGUF files (`*-00001-of-0000N.gguf`), native GPTQ
checkpoints, vision/video/speculative-decoding requests under tensor
parallelism.

**Batch invariance, precisely.** Two properties hold exactly, and are asserted
in the tests rather than assumed:

- A request's logits do not depend on *which other requests* share its batch
  (`a_batch_does_not_leak_between_its_members`).
- The tensor-core GEMM gives bit-identical results at any batch width, so the
  vocab projection — which uses it at every row count — is invariant
  (`tensor_core_gemm_gives_the_same_answer_at_any_batch_size`).

What does not hold: the layer projections switch kernel between a one-token step
and a many-token step, because at one token the integer mat-vec is meaningfully
faster than the tensor-core GEMM. Unifying them would cost that much on
single-request latency, which is the case this engine exists to serve, so the
switch stays. The two kernels sum over `k` in different orders, so greedy
decoding can eventually pick the other side of a near-tie. Seeded sampling at
temperature is reproducible against a fixed batch width, not across widths.

## Quick start

```bash
./scripts/setup-cuda.sh                     # links a CUDA userspace into vendor/
mkdir -p models && cd models
curl -LO https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q8_0.gguf
cd ..

cargo run --release -p infero-server -- --model models/qwen2.5-0.5b-instruct-q8_0.gguf
```

A terminal client comes with it:

```bash
cargo run --release -p infero-tui -- --host 127.0.0.1:8080
```

Streams tokens as they arrive, shows tok/s per reply, and `esc` cancels a
generation mid-flight — which drops the connection, and the scheduler retires
that sequence from the batch on its next step rather than finishing into the
void. It speaks plain OpenAI SSE, so it works against anything with that API.

There is also a CLI for a single generation, which is what to reach for when
something looks wrong:

```bash
cargo run --release -p infero-model --example generate -- \
    models/qwen2.5-0.5b-instruct-q8_0.gguf "Explain RoPE in one sentence." --greedy
```

and a GGUF inspector:

```bash
cargo run -p infero-gguf --example info -- models/qwen2.5-0.5b-instruct-q8_0.gguf --tensors
```

Sharding a larger model across GPUs:

```bash
# one process a rank, CUDA_VISIBLE_DEVICES pinned to a distinct physical GPU each
cargo run --release -p infero-server --features nccl -- \
    --model models/big-model.gguf --tensor-parallel-size 2
```

### Also runs on Apple GPU (Metal)

`infero-gpu` is a thin device-layer trait that either `infero-cuda` or
`infero-metal` implements; whichever is compiled in is the only one linked,
picked by feature flag rather than a runtime branch:

```bash
cargo run --release -p infero-server --no-default-features --features metal -- \
    --model models/qwen2.5-0.5b-instruct-q8_0.gguf
```

The Metal backend was built to match `cudarc`'s own shapes — `Buf`, `View`,
`ViewMut`, `LaunchConfig`, method names, argument order — so the launch sites
in `infero-kernels` needed no change at all to compile against it; only the
device layer underneath differs. Kernels are ported file for file: `ops.cu`
→ `ops.metal`, `quant.cu` → `quant.metal`, `gdn.cu` → `gdn.metal`, `mmvq.cu` →
`mmvq.metal`, and so on, with `unimplemented.metal` standing in for what has
not been ported yet, so a missing kernel fails at pipeline-build time rather
than silently returning garbage.

On Metal today: F16 and Q8_0 decode, the integer mat-vec, a fused GQA decode
attention kernel, GatedDeltaNet, host-side sampling, and M-RoPE all run and are
checked against the same CPU references and logits fixtures the CUDA path
uses. Not yet ported: the tensor-core-style integer GEMM (`mmq.cu` and
`vendor/marlin` have no MSL twin — Apple GPUs have no equivalent
matrix-multiply instruction shape to target), MoE, the vision tower, FP8/NVFP4
(Apple GPUs have no FP8/FP4 matrix unit), and TurboQuant KV compression. Those
stay behind `#[cfg(feature = "cuda")]` rather than being faked on Metal. Design
notes and the measured starting point are in
`docs/superpowers/specs/2026-08-23-infero-metal-port-design.md`.

### No CUDA toolkit required

There is no `nvcc` here and no `/usr/local/cuda` — only the driver, for the
default build. Most kernels are compiled at runtime by NVRTC, and
`scripts/setup-cuda.sh` links `vendor/cuda` at the CUDA userspace shipped
inside the pip `nvidia-*` wheels that PyTorch already pulls in. Set
`CUDA_HOME` to use a real toolkit instead. The `cutlass` and `flash_attn2`
Cargo features are the exception — they AOT-compile a CUTLASS-based FP8 GEMM
and a vendored FlashAttention2 shim with `nvcc` at build time, and need a full
toolkit (`INFERO_NVCC`, `INFERO_CUTLASS_DIR`); everything else needs neither.

Because those libraries are not on the system search path, `infero-cuda` opens
them by absolute path with `RTLD_GLOBAL` at startup; `dlopen` dedupes by soname,
so cudarc's later lookup by bare name finds them. That trick is what makes
`libnvrtc-builtins.so` resolve without `LD_LIBRARY_PATH`.

## Layout

| crate | what it does |
| --- | --- |
| `infero-gguf` | GGUF container: header, metadata, tensor index. mmap'd, zero-copy. |
| `infero-gpu` | The device-layer trait `infero-cuda` and `infero-metal` both implement; exactly one is linked in. |
| `infero-cuda` | The NVIDIA half: device, stream, cuBLAS handle, NVRTC compilation with a PTX disk cache, NCCL for tensor parallelism. |
| `infero-metal` | The Apple half: device, buffers, MSL compilation, dispatch — built to the same shapes as `infero-cuda`. |
| `infero-kernels` | The `.cu`/`.metal` sources and their launch wrappers. |
| `infero-tokenizer` | Byte-level BPE built from the GGUF vocab, plus the chat template. |
| `infero-model` | Config, weight upload, the forward pass, KV cache, sampling. |
| `infero-server` | Continuous-batching scheduler and the OpenAI-compatible HTTP API. |
| `infero-tui` | Terminal chat client. Hand-rolled HTTP so no proxy env var can redirect a loopback request. |

One ordinary decoder block:

```
x ──► rms_norm ──► q,k,v = W·x + b ──► rope ──► store kv
│                                        │
│                            attention over the cache
│                                        │
└────────────────► + ◄── W_o · attn ─────┘
                   │
                   ├──► rms_norm ──► silu(W_g·x) * (W_u·x) ──► W_d·
                   │                                            │
                   └──────────────────► + ◄──────────────────────┘
```

A Qwen3.5-style hybrid model interleaves blocks shaped like this one with
GatedDeltaNet blocks, in which the attention-over-the-cache stage above is
replaced by a fixed-size recurrent state updated by the gated delta rule —
one state per sequence, overwritten every step, never grown with context
length the way a KV cache is.

### Design notes

**Weights are never dequantized on the device during decode.** They stay in
their GGUF block encoding (or their AWQ/FP8/NVFP4 layout) and are consumed in
place. That is the whole reason a quantized model is smaller in VRAM and not
just on disk.

**Decode goes through integers.** The activation row is quantized to Q8_1 and
dotted against the packed weights with `__dp4a`, four weights and four
activations per instruction, never materializing a float. The per-type dot
products are ported from llama.cpp's `vecdotq.cuh` (MIT — see
`vendor/LICENSE.ggml`); the launcher and the activation quantizer are ours.
This is worth borrowing rather than deriving: it was a 9x difference on
Llama-3.1-8B, and three rounds of guessing at the float kernel had bought 1.8x.

**Batches go through the integer tensor cores.** A batched projection is a
GEMM, and `mmq` runs it straight off the quantized weights: `mma.m16n8k32.s8`
per 32-element quantization group, with the block scales folded in afterwards in
float. K=32 is not a tuning choice — every ggml block is 32 elements wide, so
one MMA consumes exactly one block and a scale never straddles an accumulator.
Structure follows llama.cpp's `mmq.cu`, which vLLM also carries as its GGUF
path. Q6_K needs a scale every *sixteen* elements, which one MMA cannot span;
the fragment layout happens to put registers 0/1 in `k ∈ [0,16)` and 2/3 in
`[16,32)`, so zeroing half of the B operand isolates one scale group.

The fragment layouts are pinned by a test (`crates/kernels/tests/mma.rs`) that
checks a one-hot MMA against an integer reference, because an index off by one
there yields a matrix product that looks plausible in a cosine test and ruins
generation.

**FP8 and NVFP4 batches go through CUTLASS.** A dedicated small-M tile
(operand-swapped, ported from vLLM's own real technique with two real bugs
found and fixed along the way) covers the narrow-batch decode shapes; the FFN
gate/up projection is fused into one GEMM call the way vLLM's
`MergedColumnParallelLinear` does. Both are real, measured, shipped wins —
see `crates/kernels/src/cu/mmq.cu`'s design note and `vendor/marlin/README.md`
for the GGUF-side equivalent.

**Which kernel runs when.** One token: integer mat-vec (`mmvq`). Two to 96
tokens: tensor-core GEMM (`mmq`/CUTLASS). Above that, `mmq` re-reads the
weights once per token tile often enough that dequantizing to an f16 scratch
and calling cuBLAS wins instead. A matrix whose type has a mat-vec but no GEMM
repeats the mat-vec per token up to twelve tokens — the float `gemv` decodes
one weight per thread and runs an order of magnitude below the memory bound,
so even a dozen repeated passes beat it once. Thresholds are measured per
device, not derived; `INFERO_MMQ_TILES` and `INFERO_NO_MMQ` exist to
re-measure them.

The vocab projection uses the tensor-core path at *every* row count including
one. That looks like a throughput sacrifice and is the opposite: it is what
makes the logits independent of batch width, and the profile had the float
mat-vec it replaced at 59% of a batch-32 decode step on the model this was
first measured on.

**Activations are f32, the KV cache is f16.** Keeping activations wide costs
bandwidth a llama.cpp-style engine would rather spend elsewhere, but it makes
every intermediate directly comparable against a CPU reference — which is what
finding a wrong RoPE convention actually requires.

**Sampling can run on the host or the device.** The device path exists
because reconstructing the sampling distribution on the host for speculative
verification means copying `n * vocab` floats and walking the whole
vocabulary a row — real, measured overhead on a wide model. Both paths draw
from the same per-sequence `StdRng`, so which one ran is not observable in the
output.

### Continuous batching

Requests share the GPU. Each step assembles one batch from everything in
flight and runs a single forward pass; a sequence that finishes leaves at the
end of that step and a waiting request takes its place at the start of the
next, with nothing else pausing.

```bash
infero --model model.gguf --max-seqs 32 --kv-slots 32768
```

Two rules shape a batch. **Decodes go first** — they cost one token each, and a
running sequence starved by someone else's prompt is a stall the client feels.
**Prefill fills what is left, and may be split** across steps, which is what
keeps one 4000-token prompt from freezing everyone else.

**The KV cache is paged**, at a page size of one token. Sequences draw slots
from a shared pool and keep a table mapping logical positions onto physical
slots, so lengths can differ wildly, a finished sequence returns its slots
immediately, and admitting a new one costs a table write rather than an
allocation. Page size one means no internal fragmentation at all; the table
costs four bytes per cached token, against roughly 24 KB per token for the
tokens themselves on this model. Larger pages would buy the attention loop
better locality and are the obvious next step.

**A CUDA graph replays a decode step's launches as one.** A step issues
several hundred kernels; capturing them once and replaying the graph removes
essentially all of that launch overhead from the hot path. `INFERO_NO_GRAPH`
disables capture for debugging a kernel that graphs would otherwise hide
inside one opaque replay.

Batching is a scheduling decision, not a numerical one, and the tests hold it
to that: four sequences decoded together produce token-for-token the same
output as each decoded alone, and a sequence joining a batch already in flight
is unaffected by who else is in it.

### CPU offload

`--gpu-layers N` keeps `N` blocks in VRAM and moves the rest to page-locked
host memory, streamed back in a layer at a time:

```bash
infero --model model.gguf --gpu-layers 12       # 12 blocks resident, rest streamed
infero --model model.gguf --gpu-layers 0        # only embeddings and the vocab head stay
```

**Compute never leaves the GPU.** This is not llama.cpp's `-ngl`, which runs
the offloaded layers on the CPU and needs a second set of kernels for every
quantization format. Here the weights travel and the arithmetic stays put, so
offload trades PCIe bandwidth for VRAM rather than GPU throughput for CPU
throughput — and there is exactly one implementation of every kernel.

A layer's seven big matrices are packed into a single page-locked blob, so
staging a layer is one contiguous DMA rather than seven. Two staging slots
alternate by layer parity: while the compute stream reads slot `L % 2`, the
copy stream fills slot `(L+1) % 2`, with events in both directions —
`ready[s]` gates compute on the transfer landing, `consumed[s]` gates the next
transfer on compute finishing. Norms and biases stay resident; they are
kilobytes, and streaming them would add descriptors without saving anything.

Because only the route changes, the result does not: `cargo test -p infero-model
--test offload` asserts the logits are **bit-for-bit identical** to a fully
resident run at 0, 1, 12 and 23 resident layers.

### KV cache: TurboQuant

The cache can be compressed with [TurboQuant](https://arxiv.org/abs/2504.19874)
(Zandieh et al., Google Research, ICLR 2026), implemented from the paper:

- **Algorithm 1, `TurboQuant_mse`** — a random rotation `Π` makes an arbitrary
  unit vector uniform on the sphere, so its coordinates follow the *known*
  density `f_X(x) ∝ (1-x²)^((d-3)/2)` no matter what came in. That is what
  lets an optimal scalar quantizer be solved once, offline, with no calibration
  data. `crates/kernels/src/turboquant.rs` solves Eq. (4) numerically per head
  dimension; the resulting distortions reproduce Max's Lloyd-Max table to four
  figures (0.3634 / 0.1175 / 0.03454 / 0.009497 for b = 1..4), which is what
  Theorem 1 quotes rounded.
- **Algorithm 2, `TurboQuant_prod`** — an MSE-optimal quantizer *shrinks*
  inner products, so keys get `b-1` bits of MSE codes plus a 1-bit QJL sign on
  the residual, which makes the attention logit unbiased. Measured on the
  kernels: the MSE-only estimator regresses onto the truth with slope 0.885,
  the two-stage one with slope 1.003.

Keys use Algorithm 2 and values Algorithm 1 — a key feeds an inner product,
a value a weighted average.

**Everything stays in the rotated basis.** `Π` is orthogonal and `S` is
i.i.d. Gaussian, so `S' = S·Πᵀ` is too, and the estimator becomes

```
<q, x~> = <Πq, y~> + (sqrt(pi/2)/d) · gamma · <S'(Πq), qjl>
```

The query is rotated once per token and **no cached vector is ever rotated
back**. For values the same substitution moves the inverse rotation from once
per cached vector to once per (head, token), after the weighted sum. Without
this the scheme would not be worth running.

Not implemented: the paper's outlier-channel split, which is where its
non-integer 2.5 and 3.5 bit rates come from (32 channels at 3 bits, 96 at 2,
over `d = 128`). Widths here are 2, 4 and 8 so codes pack into bytes.

```bash
infero --model model.gguf --kv-quant k8v4     # keys 8-bit, values 4-bit
infero --model model.gguf --kv-quant tq4      # the paper's symmetric 4-bit
```

Presets `tq2` / `tq4` / `tq8` are symmetric with QJL, `tq2-mse` / `tq4-mse`
drop the QJL stage, and `k<bits>v<bits>[+qjl]` sets the two sides
independently.

### Supported weight encodings

GGUF: `F32`, `F16`, `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q4_K`, `Q5_K`,
`Q6_K`. Plus native AWQ (`Q4_G128`), native FP8 W8A8, and native NVFP4 W4A4
Hugging Face checkpoints.

| | integer mat-vec | tensor-core / CUTLASS GEMM |
| --- | --- | --- |
| `Q8_0` | yes | yes |
| `Q4_K` | yes | yes, rows a multiple of 256 |
| `Q5_K` | yes | falls back to dequant + cuBLAS (see below) |
| `Q6_K` | yes | yes, rows a multiple of 256 |
| `Q4_G128` (AWQ) | yes | yes |
| FP8 (W8A8) | n/a (native precision) | yes, CUTLASS |
| NVFP4 (W4A4) | n/a (native precision) | yes, CUTLASS |
| others | no | no |

The rest fall back to the float mat-vec or to dequant + cuBLAS. `Q5_K` is
`Q4_K` plus a 5th bit per weight packed in a separate `qh` array — llama.cpp's
own `Q4_K_M`/`Q4_K_L` strategies use it for a checkpoint's "more sensitive"
tensors (`attn_k`/`attn_v`) rather than quantizing every tensor at the same
width, so a real `Q4_K_M` file the K-quant path did not expect to hold Q5_K
tensors is not unusual. It has a mat-vec (measured within single-digit
percent of `Q4_K` at the same shape, once the benchmark itself synchronizes
correctly — see `crates/kernels/examples/q5k_vs_q4k_bench.rs`) but, like
`Q4_G128`, no dedicated tensor-core GEMM yet; the generic dequant-to-f16 +
cuBLAS path covers it at wide batch. Adding a mat-vec for a new type means
porting its `vec_dot_*_q8_1`; adding a GEMM means a staging function that
expands its blocks into an int8 tile plus one scale per 16- or 32-element
group.

### Architectures

Rotary pairing follows the architecture, and it is not recorded in the file:
llama-family conversions permute Q and K so the *interleaved* pairing
reproduces Hugging Face's rotate-half, while Qwen2/Qwen3-family models want
NeoX. Getting it wrong gives fluent output that drifts with position rather
than an error — which is how it was found. Llama 3.1 additionally ships
`rope_freqs.weight`, a per-dimension frequency divisor for its 128k context,
and its chat template emits `{{ bos_token }}` itself.

A "Q4_K_M" file is a mixture. Qwen2.5-0.5B's hidden size of 896 is not a
multiple of the 256-element K-quant super-block, so most of its rows fall back
to `Q5_0`, and a model with a `d_model` that *is* a multiple of 256 can still
carry a handful of real `Q5_K` tensors on top of the bulk `Q4_K` — which is
why the legacy block-32 quants, and now `Q5_K`, are not optional.

## Correctness

`cargo test` runs several hundred tests across the workspace's crates. Those
needing a model or a real multi-GPU setup skip cleanly when the fixture or the
second device is absent.

| what | how it's checked |
| --- | --- |
| Tokenizer | Token-for-token against `AutoTokenizer` on 25 cases (CJK, emoji, code, whitespace runs). Chat template output compared byte-for-byte. |
| Quantized decoders | Each encoding's mat-vec against the same tensor from the F16 build, and (for the newest ones) against a from-scratch Rust port of the real reference dequantization/dot-product logic, not a second copy of the same CUDA source. |
| NVFP4 / FP8 | A real, fully-quantized checkpoint (attention + GatedDeltaNet + FFN, not just the FFN) loads and generates against its own real `hf_quant_config.json`, both known on-disk schema shapes. |
| TurboQuant | Codebook distortion against Max's Lloyd-Max table to four figures; measured distortion on quantized data against the codebook's prediction; the MSE-only estimator shown to shrink inner products and the two-stage one shown not to. |
| CPU offload | Logits bit-for-bit identical to a resident run at 0, 1, 12 and 23 resident layers, batched and token-at-a-time; one transfer per offloaded layer per pass. |
| Continuous batching | Four sequences prefilled together produce logits identical to each prefilled alone; a request's logits are bit-for-bit unchanged by swapping its batchmates; a sequence joining mid-flight is unaffected; recycled pool slots carry no history from their previous tenant. |
| Tensor-core / CUTLASS GEMM | `mma.m16n8k32.s8` fragment layouts pinned against an integer reference, including one-hot inputs that localize a mis-mapped index to one cell. Per-tensor cosine against the float mat-vec at several token counts, the ragged widths on purpose, since an edge slip in the token tile is what they catch. Bit-identical output across batch widths. |
| Tensor parallelism | A sharded run's logits checked against the same model run on one GPU. |
| TUI | SSE frames reassembled across chunk boundaries; wrapping never overflows a line, counting CJK as two cells. |
| Rotary variants | Both pairings preserve norms and differ from each other; a doubled frequency factor matches halving the position. |
| Kernels | RMSNorm, RoPE, SwiGLU, GQA attention with causal masking, GatedDeltaNet's delta rule, all against CPU references. |
| Forward pass | Argmax, top-10 set and logit spread against `transformers` f32 logits on four prompts. |
| KV cache | Token-at-a-time decode must land in the same state as batch prefill. |
| HTTP | Streaming chunks must reassemble into the non-streaming response; stop sequences, seeds, usage accounting, error shapes. |
| `compute-sanitizer` | New integer/dp4a kernels are run under `--tool memcheck` and `--tool racecheck` before being wired into the dispatch tables. |

Fixtures are regenerated with `scripts/make_tokenizer_fixtures.py` and
`scripts/make_logits_fixtures.py`; neither runs during `cargo test`.

## Performance

### Kernel-level parity with vLLM

The two kernels that dominate a decode step — the FFN GEMM and the
GatedDeltaNet delta-rule kernel — were benchmarked head-to-head against
vLLM's own real, compiled kernels on the same GPU, at the same real shapes, not
inferred from reading vLLM's source. At the FFN GEMM's real shape,
infero's own small-M CUTLASS tile ran *faster* than vLLM's own
`cutlass_scaled_mm` at that shape; the GatedDeltaNet kernels landed
statistically tied once both benchmarks used the same non-cache-flattering
methodology (rotating state buffers rather than one reused buffer that
happens to fit in L2). Neither of the two largest per-step kernels is where a
remaining end-to-end gap lives.

Two real, shipped levers got there:

- **An operand-swapped small-M CUTLASS FP8 GEMM tile**, vLLM's own real
  technique, ported (not vLLM's own M≤64 threshold — a real sweep on this
  card found a 32-token crossover instead): a measured ~10% real win on
  batch=16 no-speculation decode throughput.
- **The FFN gate/up projection fused into one GEMM** the way vLLM's
  `MergedColumnParallelLinear` does, after fixing a real 3x-VRAM-redundancy
  bug in the fusion path: a further real ~3% at the same shape.

Every other kernel-level candidate checked at that point — narrow-N matvec
over CUTLASS, stream-K decomposition, CTA-rasterization swizzle, deeper
weight prefetch — either regressed, measured inside noise, or (stream-K)
won on some individual shapes but not enough of a decode step's mix to move
the end-to-end number. What is left of the gap to vLLM's own best numbers is
not currently attributed to any single kernel; async host-side scheduling,
GatedDeltaNet decode parallelism, and CUDA-graph launch granularity were each
checked against vLLM's real source and found to already match its design.

### Speculative decoding: real, and not free

An MTP (multi-token prediction) draft head is supported end to end, including
a from-scratch device-resident draft loop (Gumbel-max sampling, one host
sync a round instead of several) built specifically to remove host-sync
stalls from the draft path. Measured rather than assumed: that device path is
a real, reproducible **net loss** against the simpler host-driven one on this
card, because Gumbel-max's own lower acceptance rate (no repetition-penalty or
top-p term) outweighs the sync savings — so the host-driven draft loop is
what ships, and the device-resident kernels stay in the tree as tested,
unused infrastructure rather than a default. Speculative decoding's own real
payoff is workload-dependent: it does not help every batch shape, and gating
it off rather than forcing it on is a deliberate, measured choice
(`crates/model/src/spec.rs`).

### CPU offload

Qwen2.5-0.5B-Instruct, Q8_0, 41-token prompt, 150 tokens generated:

| `--gpu-layers` | VRAM (MiB) | offloaded (MiB) | prefill | decode |
| --- | --- | --- | --- | --- |
| 24 (all) | 639 | 0 | 745 tok/s | 235 tok/s |
| 18 | 578 | 91 | 712 tok/s | 108 tok/s |
| 12 | 488 | 181 | 645 tok/s | 62 tok/s |
| 6 | 397 | 272 | 596 tok/s | 44 tok/s |
| 0 | 306 | 363 | 557 tok/s | 34 tok/s |

**Prefill barely notices, decode pays in full.** Prefill amortizes each weight
read over a whole chunk of tokens, so at zero resident layers it still runs at
75% of the resident rate. Decode reads every weight once per token, so it lands
straight on the PCIe bus: 363 MiB per token at 34 tok/s is 12.2 GB/s, against
the 13.2 GB/s this machine reaches on a pinned host-to-device copy
(`cargo run --release -p infero-kernels --example launch_overhead`). At 92% of
the link's ceiling there is nothing left to win in the transfer path — the
prefetch is fully hiding the compute, and the remaining lever is moving fewer
bytes, not moving them faster.

That is also why the pinned allocation matters: the same benchmark measures
9.8 GB/s for pageable memory, so page-locking is worth 35% here.

### AWQ, FP8, and NVFP4 checkpoints

`--model` takes a Hugging Face checkpoint directory as well as a GGUF file.
The loader inspects each tensor rather than the checkpoint as a whole:
`.qweight`/`.qzeros`/`.scales` means AWQ, `.weight`/`.weight_scale_inv` means a
native FP8 (W8A8) checkpoint, a real `hf_quant_config.json` with NVFP4 targets
means native NVFP4 (W4A4), and any of these can be mixed in one file — an MoE
checkpoint that ships some experts AWQ and others FP8 loads either way with no
extra flag. AWQ's quantized projections are transposed and repacked into
`Q4_G128` on the way in — 128 weights per block, an `f16` scale and zero,
output-major, so the existing mat-vec and tensor-core GEMM read them
unchanged. vLLM's `awq_marlin` repacks for the same reason. FP8 and NVFP4
tensors are repacked into the block layouts their own CUTLASS paths read
directly, at native precision — no dequantize-then-requantize step.

**AWQ is not fewer bytes.** Its layers are 13% smaller than a Q4_K_M file's —
4.25 bits against 4.83 — but it ships `lm_head` as `f16`, which more than
cancels it unless the head is separately quantized (see below). The format
wins on *decode cost*, not volume: a Q4_K dot product unpacks a 6-bit scale
and a 6-bit minimum from a packed twelve-byte field every 32 weights;
`Q4_G128` reads one `half2` every 128.

**The vocabulary projection is worth quantizing.** Left as `f16` it is a fifth
of a decode step and the float mat-vec reads it well below the memory bound;
quantized to Q8_0 at load, that cost drops by roughly 6-7x. Eight bits is not
a meaningful loss for a projection whose output is fed to an argmax over a
100k+ vocabulary, and the existing tensor-core-at-every-width property (see
Design notes above) is what lets it be quantized without giving up batch
invariance.

### Against vLLM and llama.cpp

Same RTX A4000-class hardware, one load generator against every engine's
OpenAI endpoint, temperature 0. Two comparisons worth separating, because
they answer different questions:

**Against the engine reading the exact same bytes.** llama.cpp on the same
GGUF file, no format difference to hide behind: single-stream throughput
lands within single-digit percent either way, and infero's continuous
batching pulls ahead at moderate client counts before the gap narrows again
at high concurrency — the same shape of result whether the measured gap that
particular week was single digits or larger, because it is the kernels and
the scheduler being compared, not the file format.

**Against vLLM reading a different, better-suited format (AWQ/FP8).**
This is the harder comparison, and the honest state of it: vLLM's own
GEMM tiling for its native formats is more mature than the GGUF K-quant path
either engine here uses, and closing that gap is a format and kernel-tiling
investment rather than a scheduling one. What is proven, not estimated, is
that infero's *own* CUTLASS-based FP8/NVFP4 kernels are not the reason for
whatever gap remains at a given point in time — see "Kernel-level parity
with vLLM" above. Re-run `cargo run --release -p infero-model --example
batch_bench` and the server's own `INFERO_PROFILE=1` output for a
current, honest number on your own hardware rather than trusting a number
measured on someone else's card on a different day; both this project's own
history and this section's own past revisions are proof that a stale
benchmark is worse than none.

### Where the time goes

`INFERO_PROFILE=1` times every kernel with CUDA events and prints a table
sorted by share (it serializes the stream, so absolute numbers are inflated
and only the split is meaningful). `INFERO_STEP_TIMING` gives host-side phase
timing with CUDA graphs left capturing, since graphs and per-kernel profiling
cannot coexist. `cargo run --release -p infero-model --example decode_floor`
replays exactly the mat-vecs a decode step performs, nothing else, as the
floor a step cannot beat.

A long list of plausible-sounding optimizations have been built and measured
here and found to buy nothing — narrower launch grids, fusing kernels a CUDA
graph had already made free to launch, `ldmatrix`-free operand loads,
alternate register tile shapes, Marlin's own load-balanced k-split ported
onto a cruder split that had already supplied the thing it optimizes for. The
ones that did land shared one property: they attacked a real, measured
bottleneck (a scale-lookup path that was 22% of a kernel, a k-split that
under-supplied blocks per SM, a per-row read repeated three times when it
could be held in registers once) rather than a plausible-sounding guess.
`crates/kernels/src/cu/mmq.cu`'s design note and `crates/model/examples/
gemm_bench.rs` are where this history and its methodology live in full; it is
kept in the source rather than duplicated here because it is long, dated, and
specific to hardware this README does not assume you have.

## Requirements

- NVIDIA GPU, compute capability 7.0+ (tested on sm_86), driver supporting
  CUDA 12 or 13, and a CUDA userspace from pip wheels or a toolkit install
- NCCL (`libnccl.so`) for tensor parallelism across more than one GPU, via the
  `nccl` Cargo feature
- Or: Apple Silicon GPU (Metal 3+), macOS — see
  [Metal](#also-runs-on-apple-gpu-metal) above for what runs there today
- Rust 1.90+

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=jackwangfeng/infero&type=Date)](https://star-history.com/#jackwangfeng/infero&Date)
