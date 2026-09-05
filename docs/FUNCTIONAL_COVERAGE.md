# Functional coverage matrix

What's actually exercised by a real test versus assumed to work, across this
system's real feature dimensions. Built from this session's own history —
every "real bug found here" line below is a genuine incident from this
session's own commits, not a hypothetical.

Status key: **tested** (a real, named, repeatable automated test) /
**verified-once** (a human/session manually confirmed it once, real evidence
exists, but no repeatable test guards it) / **untested** (assumed, no
evidence either way) / **unclear** (couldn't determine from the code in the
time this pass had).

## Checkpoint format × quantization

| | F16 KV | TurboQuant Tq2/Tq4 | FP8 weights (unified layout) | FP8 weights (legacy layout) |
|---|---|---|---|---|
| GGUF (Q8_0) | tested (`gguf` sharding tests, real TP validation) | unclear | n/a (GGUF checkpoints in this repo are integer-block quantized, not FP8) | n/a |
| GGUF (Q4_K/Q6_K mixed) | tested (llama-3.1-8b TP validation, real bug found+fixed: `w_kv` fusion reading full width under sharding) | unclear | n/a | n/a |
| safetensors/AWQ (FP8) | tested for non-sharded load; TP sharding tested for attention/FFN+GDN, bit-exact | untested (no safetensors+TurboQuant combination has been run this session) | tested (default path since the priority flip; real bug found+fixed: single-threaded `pad_rows` losing to `repack_rows`) | tested (original path, real production baseline) |

## Attention backend × model architecture

| | Dense (pure GQA attention) | GDN-hybrid |
|---|---|---|
| `InferoHandRolled` (`ws4`/`decoupled6`) | tested (this session's whole earlier investigation, `reg128` GDN kernel work) | tested (deployed default until the FA2 priority flip) |
| `FlashAttn2Ffi` | **tested for a single fresh prefill; untested/known-broken for non-contiguous KV slots** — this is the live incident: a real production 500 (`KV slots ... not physically contiguous`) surfaced only under real multi-turn/concurrent traffic, never caught by this session's own single-shot `prefill_profile`-style validation. A concurrent fix is in flight; a follow-on stress test found the failure mode is worse than first reported — one violation appears to wedge ALL subsequent requests, not just the triggering one. **This is this matrix's single highest-priority open item.** | untested combination — FA2 is gated to standard attention only (GDN never routes through it), so this cell is really "n/a, GDN always uses `InferoHandRolled`" rather than a real gap, but worth stating explicitly rather than leaving implicit. |

## Tensor parallelism × architecture × format (the cross-product this session's own history flags hardest)

| | TP off | TP=2 | TP=4 |
|---|---|---|---|
| Dense, GGUF | tested (baseline) | tested, bit-exact (qwen2.5-0.5b) | tested, bit-exact (llama-3.1-8b, after a real fix: value-head-tiling bug) |
| GDN-hybrid, GGUF | tested (baseline) | tested, bit-exact (real 27B Q8_0 model, after two real fixes: tiled-value-head sharding bug, then a CUDA-graph+NCCL interaction bug — the "wedged after decode step 2" failure) | **untested** — no GDN-hybrid checkpoint with a head count divisible by 4 has been tried at TP=4 |
| Dense, safetensors/AWQ | untested — this session's TP work never validated a plain dense (non-GDN) model in safetensors format specifically | untested (same reason) | untested |
| GDN-hybrid, safetensors/FP8 (the real production checkpoint) | tested (baseline, this is the normal deployed path) | **partially tested, NOT bit-exact as of the last real attempt** — loads correctly, produces coherent output, but diverges from TP=1 starting at the first token; root cause not found. **Second-highest-priority open item in this matrix.** | untested |

Every real cross-cutting bug this session found (bias-vector sharding, `w_kv`
fusion, GDN value-head tiling, the CUDA-graph/NCCL interaction) was found
specifically in a cell of this table that had never been exercised before —
this table's own gaps are the most credible place to expect the next one,
not a generic "test more" instinct.

## Serving path (single running server, real HTTP traffic)

| | Automated test | Note |
|---|---|---|
| Single-turn request | tested (`scripts/server_stress_test.py`) | passes reliably |
| Multi-turn conversation (2/5/10 turns) | tested, **real quality gap found**: the harness's own semantic-recall check (does the model correctly recall a fact planted in turn 1) failed across all three turn-counts in the one real run so far — not a crash, a real correctness/quality question worth its own follow-up, separate from the contiguity crash | not yet root-caused |
| Concurrent requests (at and over configured `--max-seqs`) | tested | passed in the one clean run available; needs re-running after the contiguity fix to confirm under real repeated load |
| Sequence retirement + slot reuse | tested, **hit the contiguity bug immediately** | this is the scenario that most directly produces non-contiguous slot allocation — expect this to be the first thing to re-check once the fix lands |
| Tool-calling | tested, **also hit the contiguity bug** | same as above |
| Vision/video requests | untested this session (vision tower loads and is wired in, but no real vision/video request was sent through the stress harness or otherwise this session) | real gap |
| Speculative decoding on/off | tested implicitly (it's on by default and every real generation this session used it) but never explicitly A/B'd for correctness (does turning it off change output beyond the expected "no drafting" latency difference) | untested as its own axis |

## CUDA architecture

| | Verification level |
|---|---|
| sm_120a (this session's real hardware) | execution-tested: correctness, memcheck, racecheck, real benchmarks, all real |
| sm_90a (Hopper) | compile-tested only — a real, separate CUTLASS kernel body exists and the fatbinary contains a real, non-stub cubin (`cuobjdump`-verified), but no H100 was available to run it |
| sm_100a (Blackwell datacenter) | compile-tested only, same caveat as sm_90a |
| sm_89 (Ada) | **not implemented at all** — no matching CUTLASS blockwise-FP8 config exists in the vendored checkout; falls back to the legacy NVRTC kernel (which IS real, portable, and already execution-tested on sm_120a, just slower) |
| sm_80 (Ampere) | not targeted at all (no FP8 tensor-core hardware exists on this architecture) — legacy kernel fallback only, same as sm_89 |

## Highest-priority real gaps, ranked

1. **`FlashAttn2Ffi` under non-contiguous KV slots, and the "one failure wedges the whole server" escalation.** Live incident, actively being fixed as of this writing. Re-run `scripts/server_stress_test.py` (5 passes) once the fix lands — this is the real acceptance bar, not a fresh benchmark.
2. **GDN-hybrid + safetensors/FP8 + TP=2 is not bit-exact.** The real production checkpoint's actual TP story is unverified at the correctness level that every OTHER cell in the TP table has met. This is the single largest correctness-risk gap in the whole matrix given it's the real deployed model.
3. **GDN-hybrid + TP=4** (any format) — never attempted; TP=4 has only been proven on dense architectures.
4. **Multi-turn semantic-recall quality** — not a crash, but a real, measured miss worth its own investigation independent of the contiguity bug.
5. **Vision/video requests** — wired in, loaded, never exercised by any test this session.
6. **Dense architecture + safetensors format under TP** — an untested combination that's structurally simple (no GDN sharding complexity) and thus likely low-risk, but genuinely zero evidence either way.
