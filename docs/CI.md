# CI

`.github/workflows/ci.yml` runs two jobs on every push/PR to `main`.

## `host-tests`

Plain `ubuntu-latest`, no GPU. Covers the three crates with zero CUDA
dependency: `infero-gguf`, `infero-safetensors`, `infero-tokenizer`
(`--no-default-features`, confirmed this doesn't need `vendor/cuda` at all).
Runs real `clippy` and `cargo test` — these actually execute, not just
compile.

## `cuda-build-check`

Compiles the full default-feature workspace (every CUDA-gated crate
included: `infero-cuda`, `infero-gpu`, `infero-kernels`, `infero-model`,
`infero-server`) against real CUDA headers and stub `.so` files, vendored
from pip `nvidia-*-cu13` wheels the same way `scripts/setup-cuda.sh` already
does for dev boxes with no system CUDA install. `build.rs` (`crates/cuda`)
only checks that `vendor/cuda/{include,lib}` exist with real files in
them — it never touches a device, so this works with no physical GPU
present on the runner.

**This is `cargo build`, not `cargo test`.** It proves the code compiles
against real CUDA bindings — a real, useful check, since most of this
session's work touches these exact crates — but it cannot run anything that
launches a kernel or opens a device (`Device::new(0)` and everything
downstream of it will fail at runtime with no GPU). None of the GPU-gated
`#[test]` functions run here.

## What's not covered, and why

- **No GPU test execution anywhere in CI** — `compute-sanitizer`
  (`--tool memcheck`/`--tool racecheck`), real kernel launches, register
  counts, and actual benchmark numbers all need a real GPU. This whole
  session's own verification discipline (every kernel change checked against
  a reference, memcheck, racecheck, a real benchmark) has no CI equivalent
  yet — it's still entirely manual/session-based. Closing this gap needs a
  **self-hosted runner with a real GPU attached**, registered as a GitHub
  Actions runner on hardware like `bw`. That's a real, separate
  infrastructure decision (who owns that box, how it's secured, whether it's
  shared with interactive dev work) — flagged here as the next real step,
  not attempted in this pass.
- **No `cargo fmt --check` gate.** This codebase has never been run through
  `cargo fmt` and has substantial, pre-existing drift from default rustfmt
  across files this pass didn't touch (checked: even the CUDA-free crates
  fail `cargo fmt --check` today, in files unrelated to any of this
  session's own work). Reformatting the whole tree is a separate, one-time
  cleanup that should happen in its own PR when nothing else is mid-flight
  (several concurrent changes were touching many of the same files while
  this CI setup was being done), not as a side effect of adding CI. Once
  that cleanup lands, add `cargo fmt --check` back as a real gate.
- **`clippy` on the CUDA-gated crates** isn't run in CI at all yet (only the
  three host-only crates get it) — `cuda-build-check` only builds, doesn't
  lint, those crates. Worth adding once the fmt cleanup above also gives a
  natural point to sort out any `-D warnings`-blocking lints across the
  whole workspace at once.
- **The `cuda-build-check` job's pip-wheel CUDA vendoring step was verified
  by local reasoning against this repo's own `scripts/setup-cuda.sh`
  pattern and by confirming the underlying `build.rs` gate only needs files
  present, not a live device — it was NOT verified against a real GitHub
  Actions run.** Concurrent work from other in-flight tasks was actively
  modifying files in this same working tree while this was being set up,
  which made pushing a throwaway test branch (switching branches in a
  shared checkout while other work is mid-commit) a real risk of clobbering
  someone else's commit target — so that verification step was skipped
  rather than risk it. The exact pip package internal directory layout
  (`nvidia/cuda_runtime/`, `nvidia/cuda_nvrtc/`, `nvidia/cublas/` vs. a
  unified `nvidia/cu13/`) is the most likely thing to need a real-run fix on
  the first actual push to `main`.
