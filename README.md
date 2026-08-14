# tuili

A CUDA inference engine for GGUF models, written in Rust. Hand-written kernels,
no PyTorch, no `libtorch`, no ggml — the whole path from a `.gguf` file on disk
to an OpenAI-compatible HTTP response is in this repository.

```
$ tuili --model models/qwen2.5-0.5b-instruct-q8_0.gguf
  qwen2.5-0.5b-instruct (qwen2) 24 layers, d_model 896, 14 heads / 2 kv, ...
  model ready quant=Q8_0 weights_mib=638
  listening on http://127.0.0.1:8080
```

```console
$ curl http://127.0.0.1:8080/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"What is the capital of France?"}]}'
{"choices":[{"message":{"role":"assistant","content":"The capital of France is Paris."}}], ...}
```

The official `openai` Python SDK works against it unmodified, streaming included.

## Status

Runs Qwen2 and Llama-family GGUF models on a single CUDA GPU. Correctness is
checked against the reference implementations rather than eyeballed: the
tokenizer is compared token-for-token against Hugging Face, the quantized
decoders against the F16 build of the same checkpoint, and the forward pass
against `transformers` logits.

The KV cache can be compressed with TurboQuant; see below for what that
actually buys on this model.

Requests are served with continuous batching over a paged KV cache, and layers
can be offloaded to host memory to fit a model into less VRAM.

**Not yet:** split GGUF files (`*-00001-of-0000N.gguf`), prefix caching,
multi-GPU, tool calls, GPU-side sampling.

**Batch invariance, precisely.** Two properties hold exactly, and are asserted
in the tests rather than assumed:

- A request's logits do not depend on *which other requests* share its batch
  (`a_batch_does_not_leak_between_its_members`).
- The tensor-core GEMM gives bit-identical results at any batch width, so the
  vocab projection — which uses it at every row count — is invariant
  (`tensor_core_gemm_gives_the_same_answer_at_any_batch_size`).

What does not hold: the layer projections switch kernel between a one-token step
and a many-token step, because at one token the integer mat-vec is 1.9x faster
on Q4_K and 3.2x faster on Q6_K than the tensor-core GEMM. Unifying them would
cost that much on single-request latency, which is the case this engine exists
to serve, so the switch stays. The two kernels sum over `k` in different orders,
so greedy decoding can eventually pick the other side of a near-tie — measured
at one token in eight across four prompts. Seeded sampling at temperature is
reproducible against a fixed batch width, not across widths.

## Quick start

```bash
./scripts/setup-cuda.sh                     # links a CUDA userspace into vendor/
mkdir -p models && cd models
curl -LO https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q8_0.gguf
cd ..

cargo run --release -p tuili-server -- --model models/qwen2.5-0.5b-instruct-q8_0.gguf
```

A terminal client comes with it:

```bash
cargo run --release -p tuili-tui -- --host 127.0.0.1:8080
```

Streams tokens as they arrive, shows tok/s per reply, and `esc` cancels a
generation mid-flight — which drops the connection, and the scheduler retires
that sequence from the batch on its next step rather than finishing into the
void. It speaks plain OpenAI SSE, so it works against anything with that API.

There is also a CLI for a single generation, which is what to reach for when
something looks wrong:

```bash
cargo run --release -p tuili-model --example generate -- \
    models/qwen2.5-0.5b-instruct-q8_0.gguf "Explain RoPE in one sentence." --greedy
```

and a GGUF inspector:

```bash
cargo run -p tuili-gguf --example info -- models/qwen2.5-0.5b-instruct-q8_0.gguf --tensors
```

### No CUDA toolkit required

There is no `nvcc` here and no `/usr/local/cuda` — only the driver. Kernels are
compiled at runtime by NVRTC, and `scripts/setup-cuda.sh` links `vendor/cuda`
at the CUDA userspace shipped inside the pip `nvidia-*` wheels that PyTorch
already pulls in. Set `CUDA_HOME` to use a real toolkit instead.

Because those libraries are not on the system search path, `tuili-cuda` opens
them by absolute path with `RTLD_GLOBAL` at startup; `dlopen` dedupes by soname,
so cudarc's later lookup by bare name finds them. That trick is what makes
`libnvrtc-builtins.so` resolve without `LD_LIBRARY_PATH`.

## Layout

| crate | what it does |
| --- | --- |
| `tuili-gguf` | GGUF container: header, metadata, tensor index. mmap'd, zero-copy. |
| `tuili-cuda` | Device, stream, cuBLAS handle, NVRTC compilation with a PTX disk cache. |
| `tuili-kernels` | The `.cu` sources and their launch wrappers. |
| `tuili-tokenizer` | Byte-level BPE built from the GGUF vocab, plus the chat template. |
| `tuili-model` | Config, weight upload, the forward pass, KV cache, sampling. |
| `tuili-server` | Continuous-batching scheduler and the OpenAI-compatible HTTP API. |
| `tuili-tui` | Terminal chat client. Hand-rolled HTTP so no proxy env var can redirect a loopback request. |

One decoder block:

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

### Design notes

**Weights are never dequantized on the device during decode.** They stay in
their GGUF block encoding and are consumed in place. That is the whole reason a
quantized model is smaller in VRAM and not just on disk.

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

**Which kernel runs when.** One token: integer mat-vec (`mmvq`). Two to 96
tokens: tensor-core GEMM (`mmq`). Above that, `mmq` re-reads the weights once
per token tile often enough that dequantizing to an f16 scratch and calling
cuBLAS wins instead. A matrix whose type has a mat-vec but no GEMM repeats the
mat-vec per token up to twelve tokens — the float `gemv` decodes one weight per
thread and runs an order of magnitude below the memory bound, so even a dozen
repeated passes beat it once. Both thresholds were measured on an A4000, not
derived; `TUILI_MMQ_TILES` and `TUILI_NO_MMQ` exist to re-measure them.

The vocab projection uses `mmq` at *every* row count including one. That looks
like a throughput sacrifice and is the opposite: it is what makes the logits
independent of batch width, and the profile had the float mat-vec it replaced at
59% of a batch-32 decode step.

**Activations are f32, the KV cache is f16.** Keeping activations wide costs
bandwidth a llama.cpp-style engine would rather spend elsewhere, but it makes
every intermediate directly comparable against a CPU reference — which is what
finding a wrong RoPE convention actually requires.

**Sampling runs on the host.** One 600 KB logit transfer per token is a
rounding error next to the forward pass, and it keeps the penalty bookkeeping
in ordinary Rust.

### Continuous batching

Requests share the GPU. Each step assembles one batch from everything in
flight and runs a single forward pass; a sequence that finishes leaves at the
end of that step and a waiting request takes its place at the start of the
next, with nothing else pausing.

```bash
tuili --model model.gguf --max-seqs 32 --kv-slots 32768
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

Batching is a scheduling decision, not a numerical one, and the tests hold it
to that: four sequences decoded together produce token-for-token the same
output as each decoded alone, and a sequence joining a batch already in flight
is unaffected by who else is in it.

### CPU offload

`--gpu-layers N` keeps `N` blocks in VRAM and moves the rest to page-locked
host memory, streamed back in a layer at a time:

```bash
tuili --model model.gguf --gpu-layers 12       # 12 blocks resident, rest streamed
tuili --model model.gguf --gpu-layers 0        # only embeddings and the vocab head stay
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

Because only the route changes, the result does not: `cargo test -p tuili-model
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
tuili --model model.gguf --kv-quant k8v4     # keys 8-bit, values 4-bit
tuili --model model.gguf --kv-quant tq4      # the paper's symmetric 4-bit
```

Presets `tq2` / `tq4` / `tq8` are symmetric with QJL, `tq2-mse` / `tq4-mse`
drop the QJL stage, and `k<bits>v<bits>[+qjl]` sets the two sides
independently.

### Supported weight encodings

`F32`, `F16`, `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q4_K`, `Q6_K`.

| | integer mat-vec | tensor-core GEMM |
| --- | --- | --- |
| `Q8_0` | yes | yes |
| `Q4_K` | yes | yes, rows a multiple of 256 |
| `Q6_K` | yes | yes, rows a multiple of 256 |
| others | no | no |

The rest fall back to the float mat-vec or to dequant + cuBLAS. Adding one to
the mat-vec means porting its `vec_dot_*_q8_1`; adding one to the GEMM means a
staging function that expands its blocks into an int8 tile plus one scale per
16- or 32-element group.

### Architectures

Rotary pairing follows the architecture, and it is not recorded in the file:
llama-family conversions permute Q and K so the *interleaved* pairing
reproduces Hugging Face's rotate-half, while Qwen2 wants NeoX. Getting it wrong
gives fluent output that drifts with position rather than an error — which is
how it was found. Llama 3.1 additionally ships `rope_freqs.weight`, a
per-dimension frequency divisor for its 128k context, and its chat template
emits `{{ bos_token }}` itself.

A "Q4_K_M" file is a mixture. Qwen2.5-0.5B's hidden size of 896 is not a
multiple of the 256-element K-quant super-block, so most of its rows fall back
to `Q5_0` — which is why the legacy block-32 quants are not optional.

## Correctness

`cargo test` runs 139 tests. Those needing a model skip cleanly when `models/`
is empty.

| what | how it's checked |
| --- | --- |
| Tokenizer | Token-for-token against `AutoTokenizer` on 25 cases (CJK, emoji, code, whitespace runs). Chat template output compared byte-for-byte. |
| Quantized decoders | Each encoding's mat-vec against the same tensor from the F16 build. Cosine ≥ 0.997 for Q4_K, 0.99998 for Q8_0. |
| TurboQuant | Codebook distortion against Max's Lloyd-Max table to four figures; measured distortion on quantized data against the codebook's prediction; the MSE-only estimator shown to shrink inner products and the two-stage one shown not to. |
| CPU offload | Logits bit-for-bit identical to a resident run at 0, 1, 12 and 23 resident layers, batched and token-at-a-time; one transfer per offloaded layer per pass. |
| Continuous batching | Four sequences prefilled together produce logits identical to each prefilled alone; a request's logits are bit-for-bit unchanged by swapping its batchmates; a sequence joining mid-flight is unaffected; recycled pool slots carry no history from their previous tenant. Greedy decode is required to track solo decode for at least five of eight steps — see the batch-invariance note above for why not all eight. |
| Tensor-core GEMM | `mma.m16n8k32.s8` fragment layouts pinned against an integer reference, including one-hot inputs that localize a mis-mapped index to one cell. Per-tensor cosine ≥ 0.99993 against the float mat-vec for Q8_0, Q4_K and Q6_K at 1, 5, 16, 19, 33 and 64 tokens — the ragged widths on purpose, since an edge slip in the token tile is what they catch. Bit-identical output across batch widths 1, 5, 16, 17 and 64. |
| TUI | SSE frames reassembled across chunk boundaries; wrapping never overflows a line, counting CJK as two cells. |
| Integer mat-vec | Per-tensor cosine 0.999994 against the float path for Q8_0, Q4_K and Q6_K; end-to-end decode cosine 0.99982 against a float-only run of the same model (`TUILI_NO_MMVQ=1`). |
| Rotary variants | Both pairings preserve norms and differ from each other; a doubled frequency factor matches halving the position. |
| Kernels | RMSNorm, RoPE, SwiGLU, GQA attention with causal masking, all against CPU references. |
| Forward pass | Argmax, top-10 set and logit spread against `transformers` f32 logits on four prompts. |
| KV cache | Token-at-a-time decode must land in the same state as batch prefill. |
| HTTP | Streaming chunks must reassemble into the non-streaming response; stop sequences, seeds, usage accounting, error shapes. |

Fixtures are regenerated with `scripts/make_tokenizer_fixtures.py` and
`scripts/make_logits_fixtures.py`; neither runs during `cargo test`.

## Performance

### KV cache compression

Qwen2.5-0.5B-Instruct, 16 prompts, KL divergence of the next-token
distribution against the dense f16 cache (lower is better), and how often the
predicted token is unchanged:

| setting | bits/channel | argmax kept | KL (nats) |
| --- | --- | --- | --- |
| f16 | 16.00 | 16/16 | 0 |
| tq8 | 8.88 | 15/16 | 0.105 |
| **k8v4** | 6.25 | 13/16 | 0.229 |
| k8v2 | 5.25 | 11/16 | 0.501 |
| tq4 | 4.88 | 6/16 | 1.914 |
| tq4-mse | 4.25 | 6/16 | 2.354 |
| k2v8 | 5.25 | 3/16 | 4.353 |
| tq2 | 2.88 | 2/16 | 5.884 |

Two things this measures that are worth stating plainly.

**Keys are not values.** `k8v2` and `k2v8` spend the same 5.25 bits per
channel; the first keeps 11 of 16 predictions, the second 3, and their KL
differs by 8.7×. A key's error is amplified through the softmax, a value's is
averaged away. Bits belong on the keys.

**The paper's operating points do not transfer to this model.** TurboQuant
reports quality neutrality at 3.5 bits/channel on Llama-3.1-8B; here 4.88 bits
already changes 10 of 16 predictions. That is a difference in the model, not
the algorithm — Qwen2.5-0.5B has 64-wide heads and only 2 KV heads, so there
is neither the per-channel amortization of the norms nor the averaging across
heads that an 8B model with `d = 128` and 8 KV heads gets. The useful setting
here is `k8v4`: 2.6× smaller cache for a fifth of a nat.

**The QJL stage is not asserted either way.** It helps at 4-bit keys
(KL 1.914 with, 2.354 without) and hurts at 2-bit keys (5.884 vs 4.073), for
0.63 extra bits per channel. The mechanism is visible at the kernel level: it
trades a multiplicative bias, which a softmax mostly absorbs as a temperature
change, for variance, which a softmax does not. After removing each
estimator's own best-fit slope the residual error is 0.362 for MSE-only and
0.424 for the two-stage version.

Cache size at 4096 positions: f16 48.0 MiB, `tq4` 14.6 MiB (3.3×),
`tq2` 8.6 MiB (5.6×).

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
(`cargo run --release -p tuili-kernels --example launch_overhead`). At 92% of
the link's ceiling there is nothing left to win in the transfer path — the
prefetch is fully hiding the compute, and the remaining lever is moving fewer
bytes, not moving them faster.

That is also why the pinned allocation matters: the same benchmark measures
9.8 GB/s for pageable memory, so page-locking is worth 35% here.

### Continuous batching

Decode steps with 512 tokens of history per sequence, on an RTX A4000
(`cargo run --release -p tuili-model --example batch_bench`). `TUILI_NO_MMQ=1`
is the same engine with the tensor-core GEMM disabled, so the column isolates
what it bought:

Qwen2.5-0.5B Q8_0:

| batch | ms/step | tokens/s | no mmq | speedup |
| --- | --- | --- | --- | --- |
| 1 | 4.83 | 207 | 162 | 1.28x |
| 4 | 10.47 | 382 | 329 | 1.16x |
| 8 | 11.22 | 713 | 389 | 1.84x |
| 16 | 13.44 | 1190 | 688 | 1.73x |
| 32 | 19.25 | 1662 | 837 | 1.99x |

Llama-3.1-8B Q4_K_M:

| batch | ms/step | tokens/s | no mmq | speedup |
| --- | --- | --- | --- | --- |
| 1 | 19.4 | 52 | 40 | 1.30x |
| 4 | 37.8 | 106 | 11 | 9.6x |
| 8 | 42.2 | 190 | 52 | 3.65x |
| 16 | 52.9 | 302 | 91 | 3.32x |
| 32 | 93.7 | 342 | 144 | 2.37x |

Qwen2.5-14B Q4_K_M: 28.2 tok/s at batch 1 and 164 at batch 32, against 21.7 and
81 without the GEMM.

The 9.6x at batch 4 is not the GEMM being brilliant; it is the float mat-vec
being terrible on the one Q6_K matrix per layer that a Q4_K_M build contains.
That case now has two ways out — the GEMM, or repeating the integer mat-vec per
token — and either beats the float path by roughly an order of magnitude.

Batch 2 still costs about twice batch 1 on these models, and that is the
dispatch boundary rather than a bug: one token takes the mat-vec, two take the
GEMM, and the GEMM's pass over the weights is 1.9x to 3.2x more expensive than
the mat-vec's. Closing it means making the GEMM's tile staging overlap its
tensor-core work, which is the next thing to do here (see below).

End to end over HTTP, N clients each asking for 128 tokens at temperature 0:

| clients | 0.5B Q8_0 | Llama-3.1-8B Q4_K_M |
| --- | --- | --- |
| 1 | 240 tok/s | 55 tok/s |
| 8 | 421 tok/s | 120 tok/s |
| 32 | 934 tok/s | 297 tok/s |

Two things had to be fixed before batching paid for itself, and both are worth
recording because neither was in the batching code:

- **The sampler sorted the whole vocabulary for every token.** 150k entries,
  O(V log V), per sequence per step — at a batch of 32 that was more CPU time
  than the entire forward pass took on the GPU. Partitioning to the top-k with
  `select_nth_unstable` instead took the HTTP numbers from 279 to 844 tok/s at
  32 clients, and single-stream decode up 43%.
- **The vocab projection was pinned to the float mat-vec** to keep logits
  independent of batch width. Per-kernel timing put it at 59% of a batch-32
  decode step: 21 ms per step, 145 MB of weights at an effective 15 GB/s. The
  tensor-core GEMM is invariant across batch widths by construction, so it
  replaced the mat-vec there without giving up the property the mat-vec was
  there to protect.

### AWQ

`--model` takes a Hugging Face checkpoint directory as well as a GGUF file. The
quantized projections are transposed and repacked into `Q4_G128` on the way in —
128 weights per block, an `f16` scale and zero, output-major, so the existing
mat-vec and tensor-core GEMM read them unchanged. vLLM's `awq_marlin` repacks
for the same reason.

Two things this is worth stating precisely, because the obvious version of both
is wrong.

**AWQ is not fewer bytes.** Its layers are 13% smaller than a Q4_K_M file's —
4.25 bits against 4.83 — but it ships `lm_head` as `f16`, 1.05 GB against a
Q4_K_M's 0.43 GB of Q6_K, and that more than cancels it. Per decode step: 4.68 GB
for AWQ, 4.62 GB for Q4_K_M. The format wins on *decode cost*, not volume. A
Q4_K dot product unpacks a 6-bit scale and a 6-bit minimum from a packed
twelve-byte field every 32 weights; `Q4_G128` reads one `half2` every 128. On the
same card the layers move at 366 GB/s against 300.

**The vocabulary projection is worth quantizing.** Left as `f16` it is a fifth of
the step and the float mat-vec reads it at 141 GB/s, so it costs 7.47 ms of a
17 ms step. Quantized to Q8_0 at load it costs 1.17 ms. Eight bits is not a
meaningful loss for a projection whose output is fed to an argmax over 128k
logits; vLLM leaves it alone, and this does not.

Together those take the weights-only floor from 15.19 ms per token to 11.16 —
382 GB/s, 94% of what this card's streaming read achieves at all.

| | ms per token | GB/s |
| --- | --- | --- |
| GGUF Q4_K_M | 15.19 | 304 |
| AWQ, `f16` head | 17.61 | 270 |
| AWQ, Q8_0 head | **11.16** | **382** |

The nibble order inside an AWQ `i32` is `[0, 2, 4, 6, 1, 3, 5, 7]`, and getting
it wrong is invisible from inside the file: every weight still decodes to a
plausible value, only attributed to the wrong output channel. So
`tests/awq_order.rs` recovers the permutation from the data instead of asserting
it, by correlating each nibble position against each output-channel offset of the
same model quantized independently as GGUF — 0.76 to 0.84 on the diagonal against
0.05 elsewhere. Two traps it had to learn: compare at a fixed *input* channel,
because AWQ scales each input channel by a factor chosen to protect the salient
ones and correlating along `k` measures that envelope instead (it reads 0.89
whether the order is right or not); and never compare against `attn_q` or
`attn_k`, whose rows llama.cpp permutes during GGUF conversion to suit its
interleaved rotary convention.

### Against vLLM and llama.cpp

Same RTX A4000, one load generator against every engine's OpenAI endpoint, 200
tokens per request at temperature 0, GPU allowed to cool to 62 C before each run
(sustained benchmarking drops this card's clocks to 74% and is worth 5% of the
reading). llama.cpp runs *the same GGUF file*, which is what separates engine
quality from quantization format:

| clients | vLLM 0.27.1 (AWQ) | llama.cpp (GGUF) | tuili (GGUF) | tuili (AWQ) |
| --- | --- | --- | --- | --- |
| 1 | 76.1 tok/s | 66.6 | 63.2 | **78.1** |
| 8 | 564.5 | 167.3 | 199 | **405** |
| 32 | 1774.9 | 500.6 | 497 | **782** |

**The 32-client row used to read 515, and half of that was a default.**
`--max-seqs` was 8, so the scheduler never had more than eight sequences to
batch however many clients connected — the same run measures 368 tok/s at
eight and 725 at thirty-two, and vLLM was being given `--max-num-seqs 64`. The
default is now 32 and the KV pool sizes itself from the VRAM left after the
weights rather than from `max_seqs * ctx`, which is what forced the low number
in the first place: 32 sequences of 4096 tokens is 17 GB on this model.

The rest of the gain is the tensor-core GEMM; see `vendor/marlin/README.md`.

Reading the same AWQ checkpoint, single-stream is level with vLLM and 15% ahead
of Ollama, which is llama.cpp behind a Go server and measures 66.5 here. Batch
throughput is 2.5x behind, down from 3.4x, and that gap is the tensor-core GEMM
rather than the format — see the design note at the top of
`crates/kernels/src/cu/mmq.cu` for what reading Marlin established about it,
and `vendor/marlin/README.md` for what porting it measured.

Against the engine reading the same bytes, tuili is 7-11% behind at one token,
**ahead by 15% at eight**, and 4.7% behind at 32. Against vLLM it is 3.7x behind
at 32 — and llama.cpp is 3.5x behind there too.

That gap is the quantization format, not the kernels. A Q4_K_M file keeps
`attn_v` and `ffn_down` in Q6_K, so both GGUF engines move 4.87 GiB per token
where AWQ's uniform 4 bits moves 4.68 GiB, and more importantly AWQ's layout is
what Marlin was built to consume. Two independent implementations of the K-quant
path land within 5% of each other and neither goes near vLLM: 500 tok/s is
roughly where this format sits on this card.

Which reframes what is worth doing. Closing on vLLM means supporting AWQ or
GPTQ, a format decision.

Ollama, which is llama.cpp behind a Go server, measures 66.5 tok/s at one
client on the same file against tuili's 63. That 5% is the subject of the
next section, and it is smaller than it looks.

### How much of a decode step is left to win

`cargo run --release -p tuili-model --example decode_floor` replays exactly the
mat-vecs a decode step performs — the same tensors in the same order inside one
CUDA graph — and nothing else. It is the floor: a step has to read every weight
once, and on this class of card that read is the job.

Measured on Llama-3.1-8B Q4_K_M, 4.62 GB of weights, against the server's own
windowed per-step average over 200 decode steps:

| | ms per token |
| --- | --- |
| the mat-vecs alone, sustained | **14.93** (309 GB/s) |
| tuili's full forward pass | 15.75 |
| Ollama's whole token, HTTP included | 15.04 |

So the mat-vecs are 95% of a step, and everything else tuili does — attention,
normalization, RoPE, the KV writes, the residual adds, sampling, streaming — is
0.82 ms. Ollama's entire token costs less than tuili's mat-vecs do, which puts
llama.cpp's own mat-vec at 323 GB/s or better against tuili's 309: 4% apart, on
a card whose pure streaming read tops out at 405.

Run the floor with `TUILI_FLOOR_REPS=220` rather than the default 20. Twenty
steps finish before the clocks drop and report a floor no server will ever see;
it is the difference between 14.27 ms and 14.93.

### Where the time goes

`TUILI_PROFILE=1` times every kernel with CUDA events and prints a table sorted
by share. It serializes the stream, so absolute numbers are inflated and only
the split is meaningful — that is the point. Adding it was the first step of the
tensor-core work, after three rounds of guessing at a different kernel had
bought 1.8x and one look at the actual algorithm had bought 9x.

Four hypotheses have been killed by that instrumentation, each of them
plausible enough to have been worth building without it:

| guess | what it predicted | what it measured |
| --- | --- | --- |
| the grid is too small | narrower blocks give 2-4x the blocks | 27.9 → 27.8 → 27.8 us; no change |
| the barriers block overlap | double-buffered staging overlaps them | 38.8 → 38.4 us, and worse at batch |
| `ldmatrix` for the A operands | fragment gathers dominate shared traffic | removing them entirely saves 12% |
| the scale path is minor | not worth touching | 22% of the kernel; hoisting bought 17% |

A second round went after the decode step's launch count, on the theory that
~300 kernels per step is what stands between tuili and llama.cpp. Every one of
them measured level, and for one reason: the CUDA graph had already removed the
launch cost, so merging kernels only merged their work.

| guess | what it measured |
| --- | --- |
| fuse attention's three kernels into flash-decoding | 0% |
| one launch for Q's and K's rotary embeddings | 0% |
| one launch for the K and V cache writes | 0% |
| one block per KV head, so V is read once per group | 0% |
| more attention chunks for a wider grid | slightly worse |
| a warp per mat-vec row instead of a block | 16.20 vs 16.33 ms; inside the noise |

A third round went after the tensor-core GEMM at batch, where the gap to vLLM
is 3.4x. `TUILI_PROFILE` had attributed 68% of that kernel to filling shared
memory and 17% to the tensor cores, so two candidates followed from it directly,
and both were built and measured:

| guess | what it measured |
| --- | --- |
| read weight fragments straight from global, no shared tile at all | 263 vs 263 tok/s at 8 tokens, 457 vs 457 at 16 |
| collapse the four 32-groups under one Q4_G128 scale into one s32 accumulation | no faster, marginally slower |
| a 32x32 or 128x32 register tile per warp instead of 8x16 | 262 vs 263, 458 vs 457 |
| slice k three ways for more blocks | 262 vs 263, 455 vs 456 |
| **slice k twelve ways** | **320 vs 263 at 8 tokens, 562 vs 456 at 16** |

The last two are the same change, and the difference between them is the whole
lesson. A 4096-row projection at 64 rows per block makes 64 blocks — 1.3 per SM.
Asking for `sm_count * 4` blocks yields three slices and nothing; asking for
`sm_count * 16` yields twelve and is worth 22%. The device does not want enough
blocks to be *busy*, it wants enough concurrent weight loads to cover their
latency — the same reason the mat-vec, at one block per output row, reads the
same bytes three times faster than this kernel did.

A fifth followed from reading Marlin, which sizes its grid from the device and
then partitions the flattened (row group, k chunk) list across it, so that k is
split only as much as the balance requires and only boundary runs need reducing.
Ported onto the same inner loop it measures level with the cruder split — 328.8
against 327.3 tok/s at eight tokens — because the reduction traffic it saves was
never the constraint either. The block count was, and the cruder split had
already supplied it.

Four restructurings measured nothing before one measured 22%, and all four were
changing things inside a kernel that was waiting on memory. What is, is visible in one subtraction: a batch of one
costs 12.6 ms and a batch of sixteen 34.9, so sixteen tokens add 22.3 ms of
arithmetic — 223 GFLOP at 10 TOPS against roughly 153 TOPS of int8 tensor-core
throughput. The tensor cores are busy 6.5% of the time, and per 32-weight group
this kernel issues one MMA against about fifteen other instructions. Reading
Marlin says the same thing from the other side: its register tile is 64x64 per
warp where this one is 8x16, so each weight fragment feeds four to sixteen MMAs
instead of one. Everything else — `cp.async` staging, keeping the shared tile
packed at four bits, dequantizing to f16 with `lop3` — is downstream of having
enough work attached to each MMA to be worth overlapping. The design note at the
top of `crates/kernels/src/cu/mmq.cu` records the whole comparison.

The one that did land changed a kernel's *duration* rather than its launch
count. `rms_norm_q8_1` read its row from global memory three times — once for
the sum of squares, once to scale it, once more to quantize — and the block-wide
reduction it needs confines it to a single block, so each of those passes is the
full latency again. Holding the row in registers across all three phases took it
from 19.4 us to 8.9, worth 2.8% of a token. The Q8_1 groups turn out to line up
exactly with the registers a strided load produces — group `b` lands in warp
`b % warps` at register `32b / blockDim.x` — so the quantization needs neither
shared memory nor a barrier.

Graphs and per-kernel profiling cannot coexist: timing records CUDA events, and
that is illegal on a capturing stream. `TUILI_PROFILE` therefore turns capture
off, and `TUILI_STEP_TIMING` exists for the other question — host-side phase
timing with the graphs left alone.

At 32 tokens the cost turned out to be spread almost evenly — staging 36%,
MMA and B operands 28%, scale lookups 22%, A operands 14% — which is why every
single-target fix returned a tenth and no more.

`cargo run --release -p tuili-model --example gemm_bench` isolates one real GGUF
tensor across kernels and token counts, so a change takes seconds to evaluate
instead of a full model run. The `no-A`, `no-scale` and `stage` columns are
variants of the real kernel with one part stubbed out; that is how the table
above was produced.

### Single-stream throughput

Qwen2.5-0.5B-Instruct on an RTX A4000 (16 GB, sm_86), 41-token prompt,
200 tokens generated, batch size 1, fully resident:

| build | prefill | decode |
| --- | --- | --- |
| F16 | 801 tok/s | 180 tok/s |
| Q8_0 | 789 tok/s | 243 tok/s |
| Q4_K_M | 736 tok/s | 156 tok/s |

Decode was 97 tok/s before two fixes worth recording:

- The kernel cache hashed the entire `.cu` source on every launch. A decode
  step issues ~500 kernels, so that was ~7 ms per token of pure CPU. Keying the
  hot lookup by module label instead took per-launch cost from 13.4 µs to
  1.45 µs.
- The quantized mat-vec gave each thread a whole quant block. A 896-element row
  is only 28 Q8_0 blocks, so a 256-thread block ran at 11% occupancy and still
  paid for a block-wide reduction. Eight elements per thread instead.

`cargo run --release -p tuili-kernels --example launch_overhead` reports the
per-launch floor on your machine.

Q4_K_M trailing Q8_0 is expected here: that file is mostly `Q5_0`, whose
decoder is per-element rather than per-block, and 896-wide rows are not a
multiple of the 256-element super-block that the tensor-core GEMM needs for a
K-quant. The larger models do not have this problem — Llama-3.1-8B decodes at
57.7 tok/s and Qwen2.5-14B at 32.4.

Llama-3.1-8B Q4_K_M single-stream decode went 54.5 → 62.0 tok/s in three steps,
each of them measured before it was written:

- **The vocab projection moved back to the mat-vec at one row.** It had been
  pinned to the tensor-core GEMM to keep logits independent of batch width, but
  `matmul` already switches kernels at the same boundary, so holding one matrix
  invariant bought nothing end to end. The GEMM fills 16 token slots and at one
  row fifteen are zeros: 171 GB/s against the mat-vec's 369 on the same weights.
  1.36 ms per token.
- **Split-K attention output.** One block per (head, token) is 32 blocks on a
  48-SM device at batch 1, two thirds of it idle. The same kernel at batch 32
  has 1024 blocks and is 10x more efficient per token — the work was never the
  problem, the grid was. Chunking the KV range and reducing afterwards took it
  from 1.68 ms to 0.55 ms, and the plain path still runs when the grid is
  already long.
- **Row scales hoisted out of the token-tile loop** in the tensor-core GEMM,
  worth 17% of that kernel at 32 tokens.

**Block width came from llama.cpp's tuning table, not from reasoning.** Their
`mmq-config-ampere.cuh` is 35 KB of `CASE(type, nthreads, occupancy, I, J, ...)`
lines — the distilled result of tuning this kernel per architecture. For Q4_K it
asks for 256 threads, 128 output rows per block, and `occupancy = 1`. This
kernel had 128 threads and 32 rows, and the instinct behind that was to keep
blocks small so the grid stays long; the table says the opposite, and says it
for every batch width. Widening to 64 rows (128 does not fit the 48 KB static
shared-memory limit here) is worth 10% at one token on a 14336-row projection.

It is not free everywhere: on a 1024-row projection a wide block halves an
already-short grid, so the width is chosen per matrix. Flat 8 warps gained 2% at
batch 32 and lost 5% at batch 8; choosing by row count keeps both.

Still on the list, in the order the measurements rank them:

- **The tensor-core GEMM still moves bytes at a quarter of the mat-vec's rate**
  (89 GB/s against 375 at batch 32, on the same weights). That single ratio is
  the batch gap. Closing it means the tile layout llama.cpp actually uses:
  128 rows per block on dynamic shared memory, `ldmatrix` for the operands, and
  stream-K decomposition. That is a port, not a patch — their MMQ is ~300 KB of
  templated CUDA across four files plus a per-architecture config table.
- **CUDA Graphs.** A step issues ~700 launches; vLLM issues one.
- **One Q8_1 quantization shared** across the projections that take the same
  input (q/k/v, and gate/up), which would retire 40% of those launches.

## Requirements

- NVIDIA GPU, compute capability 7.0+ (tested on sm_86)
- Driver supporting CUDA 12 or 13
- Rust 1.90+
- A CUDA userspace from pip wheels or a toolkit install
