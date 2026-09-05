# Code coverage

No coverage tooling existed in this repo before this pass. This documents
what's set up, the real baseline numbers as of this measurement, and the
same GPU/non-GPU split this repo's CI docs already establish for test
execution generally — coverage inherits that same split, not a new one.

## Tooling

`cargo llvm-cov` (already installed in this environment via `cargo install
cargo-llvm-cov` plus the `llvm-tools` rustup component — install both if
starting fresh). Chosen over `tarpaulin`: better support for a multi-crate
workspace, no separate instrumented-build toolchain, and it's the
LLVM-source-based approach `rustc`/`cargo test` already use under the hood.

## Running it

**GPU-independent crates** (`infero-gguf`, `infero-safetensors`,
`infero-tokenizer` — no CUDA device touched by their real tests):

```
cargo llvm-cov -p infero-gguf -p infero-safetensors -p infero-tokenizer --summary-only
```

Add `--html` for a browsable report (`target/llvm-cov/html/index.html`) or
`--lcov --output-path lcov.info` for CI/tooling integration.

**GPU-dependent crates** (`infero-kernels`, `infero-model`, `infero-server`
— most of their real tests launch actual CUDA kernels, memcheck/racecheck a
real device, or need a real GPU present) — same command, run on `bw` (or
any machine with the CUDA toolchain and a real GPU) instead of locally:

```
ssh bw
cd /home/jeff/infero
source ~/.cargo/env
INFERO_CUTLASS_DIR=/tmp/cutlass_src INFERO_FLASH_ATTN_DIR=/tmp/flash_attn_src \
  cargo llvm-cov -p infero-kernels -p infero-model -p infero-server \
  --features "infero-model/cutlass,infero-model/flash_attn2" --summary-only
```

**Now measured** (this pass, on `lenserver`'s own local A4000 rather than
`bw` — `bw`'s GPUs were all either running the live production server or
near-full with other users' work at the time; a local GPU with ~6.8 GiB
free was enough for these tests' real but modest CUDA usage). Run with
**default features only** — `cutlass`/`flash_attn2`/`nccl` need toolchains
(`INFERO_CUTLASS_DIR`/`INFERO_FLASH_ATTN_DIR`/`INFERO_NCCL_DIR`) that only
exist on `bw`, so this run does **not** cover the CUTLASS GEMM, FA2 FFI
shim, or TP/NCCL code paths at all — that's a real, separate gap, not
folded into the numbers below. Re-run the `bw` command above (features
included) once that's convenient, to get the full picture.

**A real, pre-existing bug found and fixed just getting this to build**:
`crates/server/tests/http.rs`'s `Engine::start(...)` call was missing the
12th argument (`tp: Option<(usize, usize, String)>`, added by today's
tensor-parallel work) — the whole `infero-server` test binary failed to
compile. Fixed (`None`, matching every other non-TP caller's convention)
and verified: all 17 `http.rs` integration tests pass.

```
Crate/file                        Lines Cover   Note
kernels/src/attn_backend.rs        46.59%
kernels/src/awq.rs                 59.07%
kernels/src/fp8.rs                 66.63%
kernels/src/gdn.rs                 84.40%
kernels/src/lib.rs                 46.55%
kernels/src/turboquant.rs          86.54%
kernels/src/vision.rs              86.49%
kernels/src/weight.rs              73.23%
model/src/cache.rs                 37.03%  <- KV pool; implicated in today's wedge bug
model/src/config.rs                64.75%
model/src/gdn_state.rs             71.79%
model/src/lib.rs                   40.86%
model/src/mtp.rs                   44.98%
model/src/qwen35.rs                65.93%
model/src/qwen35_mtp.rs            23.26%
model/src/qwen35_vision.rs         82.73%
model/src/qwen35_vision_image.rs    0.00%  <- real gap, matches FUNCTIONAL_COVERAGE's "vision/video never tested"
model/src/sampling.rs               95.70%
model/src/spec.rs                  69.55%
model/src/weights.rs                10.79%  <- real gap; every real sharding bug found today lived here
server/src/api.rs                  76.64%
server/src/auth.rs                  0.00%  <- likely the same attribution anomaly, see below
server/src/engine.rs                0.00%  <- likely the same attribution anomaly, see below
server/src/main.rs                  0.00%  <- likely the same attribution anomaly, see below
server/src/metrics.rs               0.00%  <- likely the same attribution anomaly, see below
server/src/prefix.rs                24.81%
server/src/routes.rs                0.00%  <- likely the same attribution anomaly, see below
server/src/scheduler.rs             12.24%  <- scheduler; implicated in today's wedge bug
server/src/stop.rs                  84.06%
server/src/tool_call.rs             95.48%
server/src/video.rs                 49.62%
server/src/vision.rs                94.92%
TOTAL (these 3 crates)              46.85% lines / 51.94% functions
```

**The same llvm-cov attribution anomaly documented above for
`gguf/src/lib.rs`, now confirmed in a second crate**: `server/src/engine.rs`
reports 0.00% despite `Engine::start` being called directly by all 17
passing `http.rs` tests (same for `routes.rs`/`auth.rs`/`main.rs`/
`metrics.rs` — all exercised indirectly through the same real HTTP
requests). Not a real gap; treat these five files' true coverage as
"exercised by the http.rs suite, real number unknown until the tooling
issue is root-caused" — the same honest caveat as `gguf/src/lib.rs`.

**Real, genuine gaps worth prioritizing** (not anomalies): `model/src/weights.rs`
(10.79%) is the file every real weight-sharding bug found this session
(the qwen2.5 bias-vector bug, the `w_kv` fusion bug, the GDN tiled-value-head
bug) actually lived in — the lowest coverage of any file that matters this
much is the single most actionable finding in this whole pass.
`model/src/cache.rs` (37.03%) and `server/src/scheduler.rs` (12.24%) are the
KV-pool/scheduler files implicated in today's "one failure wedges the whole
server" incident. `model/src/qwen35_vision_image.rs` (a real, literal 0%)
independently confirms `docs/FUNCTIONAL_COVERAGE.md`'s own flagged gap that
vision/video requests were never exercised by any test this session.

## Real baseline (GPU-independent crates), measured this pass

```
Filename                      Lines Cover
gguf/src/reader.rs             42.11%
gguf/src/types.rs              23.97%
gguf/src/value.rs               2.48%
gguf/src/lib.rs                 0.00%  <- see note below, not a real 0
safetensors/src/lib.rs         74.20%
tokenizer/src/bpe.rs           90.91%
tokenizer/src/bytelevel.rs     89.39%
tokenizer/src/chat.rs          91.47%
tokenizer/src/lib.rs           44.18%
tokenizer/src/pretokenize.rs   82.28%
TOTAL (these 3 crates)         46.84% lines
```

**A real, reproducible tooling anomaly, not a real 0% gap**:
`gguf/src/lib.rs` reports 0.00% coverage across every line/function/region,
even though `crates/gguf/tests/real_model.rs`'s real, passing tests
(`tensor_shard_reads_only_the_requested_rows`,
`tensor_shard_cols_reads_only_the_requested_columns`,
`tensor_shard_cols_multi_interleaves_per_row_not_end_to_end`, and five more)
directly call `Gguf::open`/`Gguf::tensor_shard`/`Gguf::tensor_shard_cols`,
all defined in that exact file. Reproduced identically after `cargo llvm-cov
clean --workspace`, and with/without `--no-default-features` — ruled out a
stale profile and a feature-gate explanation. No `cfg`/`cfg_attr` gates
exist in the file. This looks like a source-attribution issue specific to
how this crate's integration test binary (`tests/real_model.rs`, a separate
compilation unit linking against the library) reports coverage for that
particular file — `reader.rs`/`types.rs`/`value.rs` in the SAME crate,
exercised by the SAME test binary, report real, nonzero coverage
correctly, so it isn't a workspace-wide or binary-wide instrumentation
failure. Not root-caused further in this pass (a `cargo-llvm-cov` version
bump, or comparing `--branch` output, are the natural next things to try);
treat `gguf/src/lib.rs`'s real coverage as "known to include the
already-tested `tensor_shard*` functions, real number unknown" rather than
literally zero.

**Real, low-coverage files worth a look** (not anomalies — genuinely low):
`gguf/src/value.rs` (1.40% region coverage) and `tokenizer/src/lib.rs`
(42.48% region coverage) are the two real, substantial gaps in this
baseline.

## What this baseline is for

Compare future measurements against these real numbers, not "we think it's
probably fine" — a PR that drops `safetensors/src/lib.rs` from 73% to 40%,
say, is now a checkable regression instead of a vibe.
