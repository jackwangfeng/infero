//! FFI binding for the Triton-AOT-compiled
//! `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` kernel -- vLLM/FLA's real
//! kernel-2 (the sequential per-chunk state-pass), AOT-compiled with
//! `python3 -m triton.tools.compile` at its real, empirically-confirmed
//! winning autotune config for this GPU and shape (`BV=64, num_warps=4,
//! num_stages=3`, `H=48, Hg=16, K=128, V=128, BT=64`), then linked into this
//! crate as a real compiled artifact -- not a hand-derived CUDA kernel that
//! merely mirrors Triton's structure (that path is `gdn.cu`'s own
//! `MmaTritonMatch` variant; this file calls the actual Triton-compiled
//! machine code instead).
//!
//! # How the linked artifact is produced (not by this crate's build)
//!
//! `build.rs`'s own `triton_aot` feature block expects `INFERO_TRITON_AOT_DIR`
//! to point at a directory already containing `triton.tools.compile`'s
//! output: a generated `gdn_h.<hash>.c` (embeds the cubin as a byte array,
//! plus `cuModuleLoadData`/`cuLaunchKernel` driver-API glue) and its
//! matching `.h`. That directory is **not** vendored into this repo, the
//! same way `INFERO_CUTLASS_DIR`/`INFERO_FLASH_ATTN_DIR` point at external
//! checkouts rather than a vendored copy -- except here the "external
//! checkout" is a build artifact of Triton's own AOT tool, not a git
//! checkout, so there is no upstream repo to point at; regenerating it is a
//! `triton.tools.compile` invocation, documented in that directory's own
//! `README_repro.sh` (produced on `bw` at
//! `/home/jeff/infero-gdn-bench/aot/README_repro.sh`).
//!
//! `build.rs` compiles that generated `.c` file with a **plain C compiler**
//! (not `nvcc`): unlike `cutlass_fp4.rs`'s `fp4_bw_gemm.cu`, there is no
//! device code to compile here -- Triton already emitted the cubin as a
//! byte array -- so the generated file is host-only C that calls the CUDA
//! **driver** API directly. It links against `-lcuda` (the driver stub),
//! not `-lcudart_static` the way `cutlass`/`flash_attn2`'s nvcc-compiled
//! objects do (though `archive_and_link` still links `cudart_static`
//! harmlessly alongside it, since this crate can have both features enabled
//! at once and that function is shared).
//!
//! # Calling convention (genuinely different from `cutlass_fp4.rs`)
//!
//! `cutlass_fp4.rs`'s FFI functions take a `cudaStream_t` and plain
//! pointers/`f32`s -- ordinary host functions nvcc emits for a `<<<...>>>`
//! launch it compiled itself. This kernel's launcher is Triton's own
//! generated wrapper around the CUDA **driver** API: it takes a `CUstream`
//! and `CUdeviceptr`s (both `u64`-sized, but a distinct vendor type from
//! `cudaStream_t`/raw pointers) and returns a `CUresult` (an `i32` status
//! code), not `void`. `cudarc::driver::DevicePtr::device_ptr` already
//! returns `sys::CUdeviceptr` directly (see that trait's own signature),
//! so call sites pass it straight through with no pointer cast, unlike
//! `cutlass_fp4.rs`'s `as *const c_void`.
//!
//! # Scope (matches this task's own brief)
//!
//! This binds the kernel as a **standalone call**, not as a `Kernels`
//! method: the compiled artifact is fixed to one shape (`H=48, Hg=16,
//! K=128, V=128, BT=64`, `N=1` i.e. one sequence per launch -- `IS_VARLEN=0`
//! is baked in, so `cu_seqlens`/`chunk_offsets` are always null) and to
//! FLA's own real `(B, T, Hg, K)`/`(B, T, H, V)` tensor layout, not infero's
//! real `[heads, 128, 128]` persistent-state layout used elsewhere in this
//! crate (`gdn.rs`/`gdn.cu`). Splicing this into the real 3-kernel pipeline
//! (reconciling both layouts, handling `N>1`/batched launches, and
//! parameterizing the grid's `N*H` dimension instead of the `(2, 48, 1)`
//! baked into this one compiled config) is explicitly out of this task's
//! scope -- a separate follow-up, if this phase's numbers justify it.

use anyhow::Result;
use cudarc::driver::sys::CUdeviceptr;
use infero_gpu::Stream;

/// Fixed shape this one compiled artifact supports (baked into the cubin at
/// AOT-compile time via `--signature`'s trailing constexprs, not a runtime
/// parameter). See this module's own doc comment.
pub const H: usize = 48;
pub const HG: usize = 16;
pub const K: usize = 128;
pub const V: usize = 128;
pub const BT: usize = 64;

mod ffi {
    use cudarc::driver::sys::{CUdeviceptr, CUstream};

    // The real symbol name Triton's AOT tool derived from the kernel name +
    // a signature hash (see `gdn_h.<hash>.h`, generated alongside the `.c`
    // this binds against) -- not something this crate chose. A recompile
    // with a different signature/config would change this hash and this
    // declaration would need updating to match (`nm libinfero_gdn_triton_aot.a`
    // is the fastest way to find the new name).
    unsafe extern "C" {
        pub fn gdn_h_6dd2187e_0d1d2d3d4d5d6d7d8d9d10d(
            stream: CUstream,
            k: CUdeviceptr,
            v: CUdeviceptr,
            w: CUdeviceptr,
            v_new: CUdeviceptr,
            g: CUdeviceptr,
            gk: CUdeviceptr,
            h: CUdeviceptr,
            h0: CUdeviceptr,
            ht: CUdeviceptr,
            cu_seqlens: CUdeviceptr,
            chunk_offsets: CUdeviceptr,
            t: i32,
        ) -> i32; // CUresult
    }
}

/// Launches the compiled kernel for one `N=1` sequence of `t` tokens.
///
/// All pointers are raw `CUdeviceptr`s (from
/// `cudarc::driver::DevicePtr::device_ptr`/`DevicePtrMut::device_ptr_mut`) in
/// FLA's own real layout -- see this module's own doc comment for exactly
/// which layout each argument needs, and the example this is written for
/// (`examples/gdn_triton_aot_probe.rs`) for how those buffers get built and
/// uploaded. `gk` (the per-key-head gate) and `cu_seqlens`/`chunk_offsets`
/// (varlen support) are always null here: this compiled artifact was built
/// with `USE_GK=0, IS_VARLEN=0` baked in, so passing anything else would be
/// read by a kernel body that assumes those paths are dead code.
///
/// # Safety
/// Every `CUdeviceptr` must point at device memory of at least the size its
/// own doc-commented shape implies, allocated on the same device `stream`
/// runs on, and must (per the AOT-compile phase's own finding -- see this
/// module's doc comment) be 16-byte aligned: this compiled artifact was
/// generated with `:16`-hinted pointer args and silently computes wrong
/// (not a crash) results for a misaligned pointer. `stream` must belong to
/// a CUDA context that has already loaded this device's primary context
/// (any prior CUDA call on it is enough).
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_chunk_gated_delta_rule_fwd_h(
    stream: &Stream,
    k: CUdeviceptr,
    v: CUdeviceptr,
    w: CUdeviceptr,
    v_new: CUdeviceptr,
    g: CUdeviceptr,
    h: CUdeviceptr,
    h0: CUdeviceptr,
    ht: CUdeviceptr,
    t: i32,
) -> Result<()> {
    let status = unsafe {
        ffi::gdn_h_6dd2187e_0d1d2d3d4d5d6d7d8d9d10d(
            stream.cu_stream(),
            k,
            v,
            w,
            v_new,
            g,
            0, // gk: USE_GK=0 baked in, always null
            h,
            h0,
            ht,
            0, // cu_seqlens: IS_VARLEN=0 baked in, always null
            0, // chunk_offsets: ditto
            t,
        )
    };
    anyhow::ensure!(
        status == 0,
        "Triton-AOT chunk_gated_delta_rule_fwd_h launch returned CUresult {status}"
    );
    Ok(())
}
