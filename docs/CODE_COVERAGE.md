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

**Not attempted in this pass**: the GPU-dependent run above, deliberately —
at the time of this measurement, `bw`'s shared working tree had a different,
concurrent, uncommitted fix in flight in exactly the files an instrumented
build would touch (`crates/kernels/src/flash_attn2.rs`,
`crates/model/src/lib.rs`). Building an instrumented binary against a
mid-edit tree risked either a spurious build failure or interference with
that fork's own live testing on the same machine. Run the command above
once that tree is back to a clean, committed state — it's a real gap in
this baseline, not a limitation of the tooling.

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
