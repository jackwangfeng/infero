# Tensor-Parallel Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run one model across N GPUs on one node (tensor parallel), validated end-to-end on a real tiny checkpoint (`qwen2.5-0.5b-instruct-q8_0.gguf`) split across `bw`'s free GPU headroom, with interfaces reserved (not implemented) for pipeline parallelism and cross-node.

**Architecture:** SPMD — one OS process per GPU, coordinated via NCCL. Rank 0 runs the real request scheduler and broadcasts each step's batch decision to the other ranks; every rank runs an identical forward pass over its own head-shard of the weights and KV cache, with `ncclAllReduce` after every row-parallel projection (attention/GDN output projections, FFN down projection). Sharding happens once, at `Config`/weight-load time — `Config.n_heads`/`n_kv_heads`/GDN head counts are divided by `tp_size` immediately after parsing, so `KvPool`, `AttnDims`, and the existing kernel dispatch code downstream need no further changes; they already parameterize on those fields.

**Tech Stack:** Rust, CUDA, NCCL (pure C ABI, no libtorch/Python — same FFI shape as the existing CUTLASS integration).

**Spec:** `docs/superpowers/specs/2026-09-05-tensor-parallel-design.md`

## Global Constraints

- Tensor-parallel only this pass. Pipeline-parallel and cross-node get interface seams (a `RankBootstrap` trait, a `(pp_rank, tp_rank)` rank-identity type with `pp_size` always `1` for now) but no implementation — do not build PP or multi-node logic.
- MoE tensor-parallel and fault tolerance are explicitly out of scope — do not touch `qwen3-30b-moe-awq`-related code paths, and do not add retry/health-check logic beyond what NCCL already does (hang/timeout on a dead rank).
- No libtorch/Python dependency, ever — NCCL is linked as a plain C library, same AOT-build-and-static-link shape as `crates/kernels/build.rs`'s existing `cutlass` feature block.
- Sharding strategy for both standard attention and GDN is by head count, dividing `n_heads`/`n_kv_heads` (attention) and `linear_num_key_heads`/`linear_num_value_heads` (GDN) by `tp_size` — verified against real vLLM source this session (`qwen3_next.py`, `qwen_gdn_linear_attn.py`), do not invent a different sharding axis.
- Every rank reads only its own weight shard from disk at load time — never load the full tensor and slice in memory.
- `--tensor-parallel-size 1` (the default) must remain byte-identical in behavior to today's existing single-GPU path — TP is additive, not a replacement of the existing code path.
- Validate on `bw`: `ssh bw` (ControlMaster active). Validation checkpoint: `/home/jeff/models/qwen2.5-0.5b-instruct-q8_0.gguf` (645MB). Use GPU headroom already confirmed free and NOT allocated to other users: GPU0 ~8.6GB, GPU1 ~9.2GB, GPU2 ~7.4GB, GPU3 ~66GB. `CUDA_VISIBLE_DEVICES` pins one rank to one physical GPU index.
- `ncu`/hardware profiler access needs `sudo` on `bw` (confirmed working this session) — not needed for this plan (no perf tuning here, correctness only), don't reach for it.
- Never commit to git unless a task step below explicitly says to; stage only the files that step names, never a blanket `git add -A`.

---

### Task 1: NCCL FFI wrapper + a real multi-rank all-reduce smoke test

Nothing else in this plan works if this doesn't. No model code yet — pure plumbing.

**Files:**
- Create: `crates/kernels/src/cu_vendor/nccl_shim.rs` — thin `extern "C"` bindings to the NCCL C API (not a compiled shim like `flash_attn2_shim.cu`; NCCL ships its own prebuilt shared library, so this is bindings-only, no AOT-compile step of our own code needed).
- Modify: `crates/kernels/Cargo.toml` — add an `nccl` feature (`nccl = ["cuda"]`, mirroring `cutlass = ["cuda"]` at line 38).
- Modify: `crates/kernels/build.rs` — add a link-search/link-lib block gated on `CARGO_FEATURE_NCCL`, pointing at `INFERO_NCCL_DIR` (env var, mirroring `INFERO_CUTLASS_DIR`'s `resolve_cutlass_dir` pattern at the bottom of `build.rs`) for NCCL's `lib/`+`include/` (a real NCCL install, e.g. from `apt install libnccl2 libnccl-dev` or NVIDIA's tarball — check what's already on `bw` first: `dpkg -l | grep nccl` or `find / -iname 'libnccl*'` before assuming it needs installing).
- Create: `crates/kernels/examples/nccl_allreduce_smoke.rs` — a standalone multi-process smoke test.
- Test: the smoke test above IS the test for this task (no unit test framework needed — this is inherently a multi-process integration check).

**Interfaces:**
- Produces: `pub struct NcclComm` (owns a `ncclComm_t` handle), `NcclComm::init_rank(unique_id: NcclUniqueId, rank: i32, world_size: i32, dev: &Device) -> Result<Self>`, `NcclComm::all_reduce_sum_f32(&self, buf: &mut ViewMut<'_, f32>, stream: &Stream) -> Result<()>`, `pub struct NcclUniqueId([u8; 128])` (NCCL's real unique-id size — confirm the exact byte count from `nccl.h`'s `NCCL_UNIQUE_ID_BYTES` constant rather than assuming 128; adjust if different), `NcclUniqueId::generate() -> Result<Self>` (wraps `ncclGetUniqueId`). Later tasks (2, 4) depend on exactly these four items.

- [ ] **Step 1: Confirm NCCL is available on `bw` and find its headers/libs**

Run: `ssh bw "find / -iname 'nccl.h' -o -iname 'libnccl.so*' 2>/dev/null | grep -v proc"`
If nothing is found, install it: `ssh bw "sudo apt-get install -y libnccl2 libnccl-dev"` (passwordless sudo confirmed working on `bw` this session) — then re-run the find command to get the real header/lib paths for `INFERO_NCCL_DIR`.

- [ ] **Step 2: Write the FFI bindings**

```rust
// crates/kernels/src/cu_vendor/nccl_shim.rs
use std::ffi::c_void;

pub const NCCL_UNIQUE_ID_BYTES: usize = 128; // confirmed against the real nccl.h in Step 1 before trusting this

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ncclUniqueId {
    pub internal: [u8; NCCL_UNIQUE_ID_BYTES],
}

pub type ncclComm_t = *mut c_void;

#[repr(i32)]
pub enum ncclDataType_t {
    ncclFloat32 = 7, // confirm against the real nccl.h enum order in Step 1's header, do not assume
}

#[repr(i32)]
pub enum ncclRedOp_t {
    ncclSum = 0,
}

unsafe extern "C" {
    pub fn ncclGetUniqueId(unique_id: *mut ncclUniqueId) -> i32;
    pub fn ncclCommInitRank(
        comm: *mut ncclComm_t,
        nranks: i32,
        comm_id: ncclUniqueId,
        rank: i32,
    ) -> i32;
    pub fn ncclCommDestroy(comm: ncclComm_t) -> i32;
    pub fn ncclAllReduce(
        sendbuff: *const c_void,
        recvbuff: *mut c_void,
        count: usize,
        datatype: ncclDataType_t,
        op: ncclRedOp_t,
        comm: ncclComm_t,
        stream: cudarc::driver::sys::CUstream,
    ) -> i32;
}
```

Read the real `nccl.h` found in Step 1 and correct `NCCL_UNIQUE_ID_BYTES` and the enum discriminant values against it before moving on — these must match the real ABI exactly or every call silently corrupts.

- [ ] **Step 3: Wire the build.rs link block**

```rust
// in crates/kernels/build.rs, alongside the existing cutlass block
if std::env::var_os("CARGO_FEATURE_NCCL").is_some() {
    println!("cargo:rerun-if-env-changed=INFERO_NCCL_DIR");
    let nccl_dir = resolve_nccl_dir();
    println!("cargo:rustc-link-search=native={}/lib", nccl_dir.display());
    println!("cargo:rustc-link-lib=dylib=nccl");
}

fn resolve_nccl_dir() -> PathBuf {
    if let Ok(p) = std::env::var("INFERO_NCCL_DIR") {
        return PathBuf::from(p);
    }
    panic!("the `nccl` feature needs INFERO_NCCL_DIR set to a directory containing lib/libnccl.so and include/nccl.h");
}
```

Adjust the exact `lib`/`include` subpath to match whatever Step 1 actually found (a system package install typically puts `libnccl.so` in `/usr/lib/x86_64-linux-gnu/` and `nccl.h` in `/usr/include/` directly, not under a single `INFERO_NCCL_DIR/lib` — if so, `resolve_nccl_dir` should return something usable directly as both search paths, or use two separate env vars; verify against Step 1's real find output rather than assuming the CUTLASS checkout's directory shape applies here too).

- [ ] **Step 4: Safe Rust wrapper**

```rust
// crates/kernels/src/attn_backend.rs already exists from the attention-backend work — 
// put NCCL wrapper types in a NEW file, crates/kernels/src/tp/mod.rs, not there (different concern)
use crate::cu_vendor::nccl_shim::*;
use anyhow::{ensure, Result};

pub struct NcclUniqueId(pub [u8; NCCL_UNIQUE_ID_BYTES]);

impl NcclUniqueId {
    pub fn generate() -> Result<Self> {
        let mut id = ncclUniqueId { internal: [0u8; NCCL_UNIQUE_ID_BYTES] };
        let rc = unsafe { ncclGetUniqueId(&mut id) };
        ensure!(rc == 0, "ncclGetUniqueId failed, code {rc}");
        Ok(Self(id.internal))
    }
}

pub struct NcclComm {
    handle: ncclComm_t,
}
unsafe impl Send for NcclComm {}
unsafe impl Sync for NcclComm {}

impl NcclComm {
    pub fn init_rank(unique_id: &NcclUniqueId, rank: i32, world_size: i32) -> Result<Self> {
        let mut handle: ncclComm_t = std::ptr::null_mut();
        let id = ncclUniqueId { internal: unique_id.0 };
        let rc = unsafe { ncclCommInitRank(&mut handle, world_size, id, rank) };
        ensure!(rc == 0, "ncclCommInitRank failed, code {rc}");
        Ok(Self { handle })
    }

    pub fn all_reduce_sum_f32(
        &self,
        buf: &mut infero_kernels::ViewMut<'_, f32>,
        stream: &infero_cuda::Stream,
    ) -> Result<()> {
        let n = buf.len(); // check ViewMut's real length-accessor method name, likely `.len()` — grep other `ViewMut` usages in this crate
        let (ptr, _guard) = buf.device_ptr_mut(stream); // matches the pattern already used in cutlass_fp8.rs for device pointers
        let rc = unsafe {
            ncclAllReduce(
                ptr as *const _, ptr as *mut _, n,
                ncclDataType_t::ncclFloat32, ncclRedOp_t::ncclSum,
                self.handle, stream.cu_stream(),
            )
        };
        ensure!(rc == 0, "ncclAllReduce failed, code {rc}");
        Ok(())
    }
}

impl Drop for NcclComm {
    fn drop(&mut self) {
        unsafe { ncclCommDestroy(self.handle) };
    }
}
```

Check the real `ViewMut::device_ptr_mut` signature (used at `cutlass_fp8.rs:351` per this session's own earlier reading — `let (d_ptr, _rd) = d_pad_view.device_ptr_mut(stream)`) and match it exactly rather than guessing.

- [ ] **Step 5: Write the 2-rank smoke test**

```rust
// crates/kernels/examples/nccl_allreduce_smoke.rs
//! Run as: for each of 2 terminals/processes:
//!   INFERO_NCCL_RANK=0 INFERO_NCCL_WORLD_SIZE=2 INFERO_NCCL_ID_FILE=/tmp/nccl_smoke_id CUDA_VISIBLE_DEVICES=0 cargo run --release --features nccl --example nccl_allreduce_smoke
//!   INFERO_NCCL_RANK=1 INFERO_NCCL_WORLD_SIZE=2 INFERO_NCCL_ID_FILE=/tmp/nccl_smoke_id CUDA_VISIBLE_DEVICES=1 cargo run --release --features nccl --example nccl_allreduce_smoke
use infero_kernels::tp::{NcclComm, NcclUniqueId};

fn main() -> anyhow::Result<()> {
    let rank: i32 = std::env::var("INFERO_NCCL_RANK")?.parse()?;
    let world_size: i32 = std::env::var("INFERO_NCCL_WORLD_SIZE")?.parse()?;
    let id_file = std::env::var("INFERO_NCCL_ID_FILE")?;

    let unique_id = if rank == 0 {
        let id = NcclUniqueId::generate()?;
        std::fs::write(&id_file, id.0)?;
        id
    } else {
        // crude polling wait -- fine for a one-shot smoke test, not for the
        // real RankBootstrap trait built in Task 2
        loop {
            if let Ok(bytes) = std::fs::read(&id_file) {
                if bytes.len() == infero_kernels::tp::NCCL_UNIQUE_ID_BYTES {
                    let mut arr = [0u8; infero_kernels::tp::NCCL_UNIQUE_ID_BYTES];
                    arr.copy_from_slice(&bytes);
                    break NcclUniqueId(arr);
                }
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
    };

    let dev = infero_cuda::Device::new(0)?; // CUDA_VISIBLE_DEVICES already remaps this to the right physical GPU
    let comm = NcclComm::init_rank(&unique_id, rank, world_size)?;
    let stream = dev.stream();

    let host_val = (rank + 1) as f32; // rank 0 contributes 1.0, rank 1 contributes 2.0
    let mut buf = stream.alloc_zeros::<f32>(4)?;
    stream.memcpy_htod_sync(&[host_val; 4], &mut buf)?; // check the real method name for a host->device sync copy in this crate, e.g. grep `memcpy_htod` usages elsewhere
    comm.all_reduce_sum_f32(&mut buf.as_view_mut(), &stream)?;
    stream.synchronize()?;
    let mut out = [0f32; 4];
    stream.memcpy_dtoh_sync(&buf, &mut out)?; // same, check the real dtoh method name
    println!("rank {rank}: all_reduce result = {out:?} (expected all 3.0 for a 2-rank sum of 1.0+2.0)");
    assert_eq!(out, [3.0f32; 4], "all_reduce produced the wrong sum");
    Ok(())
}
```

- [ ] **Step 6: Run it for real on `bw`, two processes, two GPUs**

```bash
ssh bw
cd /home/jeff/infero && source ~/.cargo/env
rm -f /tmp/nccl_smoke_id
INFERO_NCCL_DIR=<from Step 1> CUDA_VISIBLE_DEVICES=0 INFERO_NCCL_RANK=0 INFERO_NCCL_WORLD_SIZE=2 INFERO_NCCL_ID_FILE=/tmp/nccl_smoke_id \
  cargo run --release --features nccl --example nccl_allreduce_smoke &
INFERO_NCCL_DIR=<from Step 1> CUDA_VISIBLE_DEVICES=1 INFERO_NCCL_RANK=1 INFERO_NCCL_WORLD_SIZE=2 INFERO_NCCL_ID_FILE=/tmp/nccl_smoke_id \
  cargo run --release --features nccl --example nccl_allreduce_smoke &
wait
```

Use `CUDA_VISIBLE_DEVICES=0`/`=1` here as placeholders for whichever two of GPU0-3 currently have free headroom (re-check `nvidia-smi` at run time — the free-headroom GPUs may have shifted since this plan was written; GPU3 reliably has the most room).

Expected: both processes print `all_reduce result = [3.0, 3.0, 3.0, 3.0]` and neither asserts. If this hangs, the most likely cause is a mismatched `NCCL_UNIQUE_ID_BYTES` or enum value from Step 2 not matching the real installed NCCL's ABI — re-verify against the real header, don't just increase timeouts.

- [ ] **Step 7: Commit**

```bash
git add crates/kernels/src/cu_vendor/nccl_shim.rs crates/kernels/src/tp/mod.rs \
        crates/kernels/Cargo.toml crates/kernels/build.rs crates/kernels/src/lib.rs \
        crates/kernels/examples/nccl_allreduce_smoke.rs
git commit -m "Add NCCL FFI wrapper, verified with a real 2-GPU all-reduce smoke test"
```

---

### Task 2: `RankBootstrap` trait and rank identity

**Files:**
- Create: `crates/model/src/tp.rs`
- Modify: `crates/model/src/lib.rs` — `pub mod tp;`

**Interfaces:**
- Consumes: `NcclComm`, `NcclUniqueId` from Task 1 (`infero_kernels::tp`).
- Produces: `pub struct RankId { pub pp_rank: usize, pub pp_size: usize, pub tp_rank: usize, pub tp_size: usize }`, `pub trait RankBootstrap { fn broadcast_unique_id(&self, rank: &RankId) -> Result<NcclUniqueId>; }`, `pub struct LocalFileBootstrap { pub run_id: String }` implementing it. Task 5 (server wiring) and Task 6 (validation) construct a `RankId`/`LocalFileBootstrap` pair directly from CLI args/env vars.

- [ ] **Step 1: Write the failing test**

```rust
// crates/model/src/tp.rs, #[cfg(test)] mod tests
#[test]
fn local_file_bootstrap_round_trips_a_real_unique_id_across_two_calls() {
    let run_id = format!("test-{}", std::process::id());
    let bootstrap = LocalFileBootstrap { run_id: run_id.clone() };
    let rank0 = RankId { pp_rank: 0, pp_size: 1, tp_rank: 0, tp_size: 2 };
    let rank1 = RankId { pp_rank: 0, pp_size: 1, tp_rank: 1, tp_size: 2 };

    // rank 0 generates and publishes; rank 1 reads the same bytes back.
    let id0 = bootstrap.broadcast_unique_id(&rank0).expect("rank 0 bootstrap");
    let id1 = bootstrap.broadcast_unique_id(&rank1).expect("rank 1 bootstrap");
    assert_eq!(id0.0, id1.0, "both ranks must agree on the same NCCL unique id");
}
```

- [ ] **Step 2: Run it, confirm it fails to compile**

Run: `cargo test -p infero-model --lib tp::tests -- --nocapture`
Expected: FAIL — `RankId`/`LocalFileBootstrap`/`RankBootstrap` don't exist yet.

- [ ] **Step 3: Implement**

```rust
// crates/model/src/tp.rs
use anyhow::{Context, Result};
use infero_kernels::tp::NcclUniqueId;

#[derive(Debug, Clone, Copy)]
pub struct RankId {
    pub pp_rank: usize,
    pub pp_size: usize,
    pub tp_rank: usize,
    pub tp_size: usize,
}

pub trait RankBootstrap {
    /// Rank `tp_rank == 0` (within its `pp_rank`) generates the unique id;
    /// every other rank reads what rank 0 published. All ranks in the same
    /// `pp_rank`'s TP group call this — it blocks until the id is available.
    fn broadcast_unique_id(&self, rank: &RankId) -> Result<NcclUniqueId>;
}

pub struct LocalFileBootstrap {
    pub run_id: String,
}

impl LocalFileBootstrap {
    fn path_for(&self, rank: &RankId) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("infero_tp_{}_pp{}.id", self.run_id, rank.pp_rank))
    }
}

impl RankBootstrap for LocalFileBootstrap {
    fn broadcast_unique_id(&self, rank: &RankId) -> Result<NcclUniqueId> {
        let path = self.path_for(rank);
        if rank.tp_rank == 0 {
            let id = NcclUniqueId::generate().context("generating NCCL unique id")?;
            std::fs::write(&path, id.0).context("publishing NCCL unique id")?;
            Ok(id)
        } else {
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
            loop {
                if let Ok(bytes) = std::fs::read(&path) {
                    if bytes.len() == infero_kernels::tp::NCCL_UNIQUE_ID_BYTES {
                        let mut arr = [0u8; infero_kernels::tp::NCCL_UNIQUE_ID_BYTES];
                        arr.copy_from_slice(&bytes);
                        return Ok(NcclUniqueId(arr));
                    }
                }
                anyhow::ensure!(
                    std::time::Instant::now() < deadline,
                    "timed out waiting for rank 0 to publish the NCCL unique id at {}",
                    path.display()
                );
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
        }
    }
}
```

- [ ] **Step 4: Run the test, confirm it passes**

Run: `cargo test -p infero-model --lib tp::tests -- --nocapture`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/model/src/tp.rs crates/model/src/lib.rs
git commit -m "Add RankId and RankBootstrap trait with a local-file implementation"
```

---

### Task 3: Sharded `Config` + sharded GGUF weight loading

**Files:**
- Modify: `crates/model/src/config.rs` — add a `shard_for_tp(&mut self, rank: &RankId)` method.
- Modify: `crates/model/src/weights.rs` — the GGUF loading path (`Matrix::load` at line 513, and whatever it calls per-tensor for the sharded matrices — `describe` at line 1234, `pack_layer` at line 2071) needs a `rank: Option<&RankId>` threaded through, defaulting to `None`/`tp_size=1` (today's exact behavior) everywhere it isn't explicitly passed.
- Modify: `crates/gguf/src/lib.rs` — `Gguf` needs a byte-range tensor read that isn't "the whole tensor": add `pub fn tensor_shard(&self, t: &TensorInfo, dim0_range: std::ops::Range<usize>) -> Result<Vec<u8>>` alongside the existing `tensor_data`/`data` (lines 204-209) — reads only the rows in `dim0_range` from `self.data(t)`, computing the per-row byte stride from `t.shape()` and the tensor's dtype size (check `TensorInfo`'s dtype field, likely `GgmlType`, and its own byte-width — grep `GgmlType` in `crates/gguf/src/types.rs` for a size-in-bytes accessor, or add one if none exists).

**Interfaces:**
- Consumes: `RankId` from Task 2.
- Produces: `Config::shard_for_tp(&mut self, rank: &RankId)` — divides `n_heads`, `n_kv_heads`, and the GDN-specific `linear_num_key_heads`/`linear_num_value_heads` fields (read their real field names from `config.rs` around lines 625-627 first) by `rank.tp_size`, in place. `Gguf::tensor_shard(&self, t: &TensorInfo, dim0_range: Range<usize>) -> Result<Vec<u8>>`. Task 4 and Task 6 call `Config::shard_for_tp` once, right after parsing, before any weight loading happens.

- [ ] **Step 1: Write the failing test for `Gguf::tensor_shard`**

```rust
// crates/gguf/src/lib.rs, #[cfg(test)] mod tests (create if it doesn't exist -- check first)
#[test]
fn tensor_shard_reads_only_the_requested_rows() {
    let f = Gguf::open("tests/fixtures/tiny.gguf").expect("open fixture"); // use whatever real small
        // fixture this crate's existing tests already load -- grep `Gguf::open` in
        // crates/gguf/tests or crates/gguf/src for the real fixture path, don't invent one
    let t = f.tensor("some_2d_weight_name_from_the_real_fixture").expect("tensor exists");
    let full = f.tensor_data("some_2d_weight_name_from_the_real_fixture").expect("full read");
    let shape = t.shape(); // [rows, cols]
    let row_bytes = full.len() / shape[0] as usize;
    let half = shape[0] as usize / 2;

    let shard = f.tensor_shard(t, 0..half).expect("shard read");
    assert_eq!(shard.len(), half * row_bytes);
    assert_eq!(&shard[..], &full[..half * row_bytes], "shard must match the corresponding prefix of a full read");

    let shard2 = f.tensor_shard(t, half..shape[0] as usize).expect("second shard read");
    assert_eq!(&shard2[..], &full[half * row_bytes..], "second shard must match the corresponding suffix");
}
```

- [ ] **Step 2: Run it, confirm it fails**

Run: `cargo test -p infero-gguf tensor_shard_reads_only_the_requested_rows`
Expected: FAIL — `tensor_shard` doesn't exist.

- [ ] **Step 3: Implement `Gguf::tensor_shard`**

Read `TensorInfo`'s real fields (`crates/gguf/src/lib.rs:35`) and `GgmlType`'s byte-width first — this needs the per-element byte size to compute `row_bytes = shape[1..].iter().product::<u64>() as usize * elem_bytes`. Then:

```rust
pub fn tensor_shard(&self, t: &TensorInfo, dim0_range: std::ops::Range<usize>) -> Result<Vec<u8>> {
    let shape = t.shape();
    anyhow::ensure!(!shape.is_empty(), "cannot shard a scalar tensor");
    anyhow::ensure!(dim0_range.end <= shape[0] as usize, "shard range out of bounds");
    let elem_bytes = t.dtype.byte_width(); // confirm the real method/field name on GgmlType for this
    let row_elems: u64 = shape[1..].iter().product::<u64>().max(1);
    let row_bytes = (row_elems as usize) * elem_bytes;
    let full = self.data(t);
    let start = dim0_range.start * row_bytes;
    let end = dim0_range.end * row_bytes;
    anyhow::ensure!(end <= full.len(), "computed shard byte range exceeds tensor data");
    Ok(full[start..end].to_vec())
}
```

- [ ] **Step 4: Run the test, confirm it passes**

Run: `cargo test -p infero-gguf tensor_shard_reads_only_the_requested_rows`
Expected: PASS.

- [ ] **Step 5: Add `Config::shard_for_tp`**

```rust
// crates/model/src/config.rs
impl Config {
    pub fn shard_for_tp(&mut self, rank: &crate::tp::RankId) {
        if rank.tp_size <= 1 {
            return;
        }
        assert!(self.n_heads % rank.tp_size == 0, "n_heads must divide tp_size");
        assert!(self.n_kv_heads % rank.tp_size == 0, "n_kv_heads must divide tp_size");
        self.n_heads /= rank.tp_size;
        self.n_kv_heads /= rank.tp_size;
        // GDN head fields: use the real field names found around config.rs:625-627
        // (linear_num_key_heads / linear_num_value_heads or whatever this struct
        // actually calls them once read) -- only present when the model has GDN
        // layers, so this is conditional on however that's already represented
        // (an `Option<GdnConfig>` field or similar -- read the surrounding struct
        // before writing this part).
    }
}
```

Finish the GDN-field-sharding part of this function by actually reading `config.rs` around lines 590-660 (the struct these fields belong to) and mirroring its real shape — this plan intentionally doesn't guess that struct's name.

- [ ] **Step 6: Thread `Gguf::tensor_shard` into `Matrix::load` for the sharded weight categories**

At `crates/model/src/weights.rs:513` (`Matrix::load`) and its callees (`describe` at 1234, `pack_layer` at 2071): for tensors that are column-sharded (Q/K/V projections, GDN's combined input projection, FFN gate/up) or row-sharded (attention/GDN output projections, FFN down projection), call `f.tensor_shard(t, range)` instead of `f.tensor_data(name)`, where `range` is computed from `rank.tp_rank`/`rank.tp_size` against that tensor's sharded dimension. Everything else (layernorms, embeddings, anything not in those two categories) keeps calling the existing full-read path unchanged. Add a `rank: Option<&RankId>` parameter to `Matrix::load` (and thread it through its callees), defaulting call sites that don't pass one to `None` — with `None`, behavior must be byte-identical to today (this is the regression check for the whole task).

- [ ] **Step 7: Regression test — `None` rank still loads identically**

Run: `cargo test -p infero-model` (the full existing suite, unmodified) — confirms threading `Option<&RankId>` through didn't change default (`None`) behavior.
Expected: same pass/fail status as before this task (establish the baseline by running this once *before* Step 6, if not already known from this session's own earlier work).

- [ ] **Step 8: Commit**

```bash
git add crates/gguf/src/lib.rs crates/model/src/config.rs crates/model/src/weights.rs
git commit -m "Add sharded GGUF tensor reads and Config::shard_for_tp for tensor-parallel loading"
```

---

### Task 4: Forward-pass all-reduce after row-parallel projections

**Files:**
- Modify: `crates/model/src/lib.rs` — the standard-attention output-projection call site (search for where `o_proj`/the probe named `"o_proj_out"` at line 3708 is produced — the matmul immediately before that probe call is the row-parallel projection needing an all-reduce right after it), the FFN down-projection call site (grep `down_proj`/`down` matmul calls in the FFN forward function), and the GDN output-projection call site (`gw.out_proj` used at line 2722).
- Modify: `Model` struct definition — add `comm: Option<Arc<infero_kernels::tp::NcclComm>>` (feature-gated behind `nccl`, mirroring how `flash_attn2` fields are already feature-gated in this same struct from the attention-backend work).

**Interfaces:**
- Consumes: `NcclComm::all_reduce_sum_f32` from Task 1, `RankId`/`Config::shard_for_tp` from Tasks 2-3.
- Produces: `Model` gains real multi-GPU-correct forward passes when `self.comm.is_some()`; behavior is exactly today's when `self.comm.is_none()` (the default, `tp_size=1` case).

- [ ] **Step 1: Read the three real call sites in full before changing anything**

At each of the three locations named above, read the surrounding ~20 lines to find exactly which buffer holds the row-parallel projection's output right after the matmul that produces it and before anything else reads it — this is the buffer the all-reduce must run on, in place, before control proceeds.

- [ ] **Step 2: Write a test that exercises single-GPU (`comm: None`) with the new code paths present but inert**

```rust
// add to crates/model's existing test suite, wherever forward-pass correctness
// is already tested against a reference (this codebase has an established
// pattern of checking model output against a real `transformers` capture --
// find that existing test and add a case that constructs `Model` with
// `comm: None` explicitly, rather than leaving it implicit, to make the
// "TP off looks identical to today" claim an explicit, checked one)
#[test]
fn model_with_no_tp_comm_matches_the_existing_reference_output() {
    // reuse whatever this crate's existing reference-output test already does --
    // this test's only new content is asserting it still passes with the new
    // `comm` field explicitly set to `None`, not a new reference capture.
}
```

- [ ] **Step 3: Insert the all-reduce calls**

At each of the three sites, immediately after the row-parallel matmul and before the result is consumed:

```rust
if let Some(comm) = &self.comm {
    comm.all_reduce_sum_f32(&mut <the buffer identified in Step 1>, self.dev.stream())?;
}
```

- [ ] **Step 4: Run the existing full `infero-model` test suite**

Run: `cargo test -p infero-model`
Expected: identical pass/fail to before this task — `comm: None` is the only path any existing test exercises, and it must be unchanged.

- [ ] **Step 5: Commit**

```bash
git add crates/model/src/lib.rs
git commit -m "Insert NCCL all-reduce after row-parallel attention/GDN/FFN projections"
```

---

### Task 5: `crates/server` — rank 0 drives scheduling, other ranks follow

**Files:**
- Modify: `crates/server/src/scheduler.rs` — read its current structure first (this plan doesn't assume its exact shape); the real change is gating the parts that talk to the outside world (HTTP request intake, response streaming) behind `rank.tp_rank == 0`, and adding a step where rank 0's batch decision (whatever struct it already builds per step — find its real name) gets serialized and broadcast to ranks `1..tp_size` before every rank calls into `Model::forward_batch_device`.
- Modify: `crates/server/src/main.rs` (or wherever the process entry point/CLI parsing lives) — add `--tensor-parallel-size` and read `CUDA_VISIBLE_DEVICES`/rank env vars, construct a `RankId`, and for `tp_rank != 0`, skip standing up the HTTP listener entirely.

**Interfaces:**
- Consumes: `RankId`, `LocalFileBootstrap` from Task 2; `NcclComm` from Task 1; the sharded `Model` construction from Tasks 3-4.
- Produces: a runnable multi-process server — Task 6 launches `tp_size` copies of this binary.

- [ ] **Step 1: Read `crates/server/src/scheduler.rs` and `engine.rs` in full to find the real per-step batch-decision type**

This plan does not assume its name or shape — find it before writing Steps 2-4.

- [ ] **Step 2: Add a broadcast of that type to non-zero ranks**

Use `NcclComm`'s existing `all_reduce_sum_f32` is NOT the right primitive for this (it's for float tensors, not arbitrary metadata) — add a second method to `NcclComm` in Task 1's file if this task discovers it's needed: `pub fn broadcast_bytes(&self, buf: &mut [u8], root: i32) -> Result<()>` wrapping `ncclBroadcast` (same FFI-binding pattern as `ncclAllReduce`, add its declaration to `nccl_shim.rs` when this task starts — this was intentionally left out of Task 1 since Task 1 didn't yet know this call site's exact needs). Serialize the batch-decision struct (this codebase likely already derives `serde`/uses a simple encoding elsewhere for API request/response types — match whatever pattern already exists rather than introducing a new serialization dependency) to bytes, broadcast the byte length first (fixed-size, e.g. a `u64`), then the payload.

- [ ] **Step 3: Gate HTTP/API startup behind `tp_rank == 0`**

In the server's entry point, construct `RankId` from CLI/env, and only call whatever function starts the HTTP listener (`routes.rs`/`api.rs`, per this crate's existing structure) when `rank.tp_rank == 0`. Non-zero ranks enter a loop: receive a broadcast batch decision, call `Model::forward_batch_device`, discard the output (or, if the existing code path already needs to do *something* with logits even on a non-driving rank — check whether sampling reads anything rank-local before assuming it's fully skippable), loop.

- [ ] **Step 4: Manual integration check (no automated test for this step — it requires real multi-process launch, deferred to Task 6's real end-to-end run)**

Confirm this task's code at least compiles and the single-rank (`tp_size=1`) path is unaffected: `cargo build -p infero-server && cargo test -p infero-server` (or whatever this crate's real test invocation is — check for a `tests/` dir or inline tests first).

- [ ] **Step 5: Commit**

```bash
git add crates/server/src/scheduler.rs crates/server/src/main.rs crates/kernels/src/cu_vendor/nccl_shim.rs crates/kernels/src/tp/mod.rs
git commit -m "Gate server scheduling/API behind rank 0, broadcast batch decisions to other ranks"
```

---

### Task 6: Real end-to-end validation — TP=2, then TP=4, against the single-GPU reference

**Files:** none new — this task runs what Tasks 1-5 built.

- [ ] **Step 1: Capture the single-GPU reference output**

On `bw`: run the existing, unmodified single-GPU server (or a simpler existing example/CLI path if the server is heavier than needed) against `/home/jeff/models/qwen2.5-0.5b-instruct-q8_0.gguf` with a fixed prompt and a fixed/greedy sampling setting (temperature 0, or whatever this codebase's existing deterministic-sampling flag is called — check `crates/server`/`crates/model` for it). Record the generated token ids (and, if cheaply available, the raw logits for the first few tokens — a stronger check than token ids alone, which can hide near-tie logit differences).

- [ ] **Step 2: Build and run TP=2**

Using the smoke test's process-launch pattern from Task 1 Step 6, launch two copies of the (now TP-aware) server binary — `--tensor-parallel-size 2`, `CUDA_VISIBLE_DEVICES` pinned to two GPUs with free headroom (re-check `nvidia-smi` at run time), same shared `run_id` for `LocalFileBootstrap`. Send the identical prompt/sampling request to rank 0's HTTP endpoint.

- [ ] **Step 3: Diff**

Compare TP=2's generated token ids against Step 1's reference. Expected: identical token ids. If logits were captured in Step 1, compare those too — expect very close (not necessarily bit-identical, since all-reduce order and FP accumulation order differ from a single-GPU sum) but not meaningfully divergent; a large divergence points at a real sharding bug (wrong byte range in `tensor_shard`, wrong axis sharded, a missing all-reduce), not acceptable floating-point noise.

- [ ] **Step 4: Repeat at TP=4**

Same as Steps 2-3, `--tensor-parallel-size 4`, one rank per GPU (GPU3's ~66GB free is by far the most headroom of the four — the small checkpoint fits trivially on any single one of them, so which physical GPU maps to which rank doesn't matter for this size of model).

- [ ] **Step 5: Confirm the KV pool actually shrank per rank**

`KvPool::new` (`crates/model/src/cache.rs:92`) already sizes everything off `cfg.n_kv_heads` (`let per_head = cfg.n_kv_heads * n_slots;` at line 111) — since Task 3's `Config::shard_for_tp` divides `n_kv_heads` before the pool is ever constructed, no `cache.rs` code should need to change, but this is a claim to verify, not assume. Compare each rank's reported `KvPool::bytes()`-equivalent (or a memory query via `nvidia-smi` per rank) between the TP=1 reference run and a TP=2/TP=4 run — expect roughly a `1/tp_size` reduction in KV-pool bytes per rank (not the full model's memory, just the pool). If it did NOT shrink, `Config::shard_for_tp` isn't running before pool construction somewhere in the load path — find and fix that ordering bug before declaring this task done.

- [ ] **Step 6: Report honestly**

If TP=2 and TP=4 both match the reference: real, verified success — say so with the actual token ids/logit deltas, not just "it worked." If either diverges: do not proceed to declare victory or move on to a different checkpoint — root-cause the specific divergence (most likely culprits, in rough likelihood order: an off-by-one in `tensor_shard`'s byte-range math, a row/column-sharding axis mixed up for one specific projection, a missing all-reduce at one of the three sites from Task 4, or the GDN head-count sharding in `Config::shard_for_tp` using the wrong field names) — report exactly what you found, even if not fully resolved, rather than a vague "close enough."
