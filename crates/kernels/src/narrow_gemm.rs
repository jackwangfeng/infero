//! A dedicated mat-vec-batched kernel for GDN's narrow, unquantized
//! `in_proj_a`/`in_proj_b` gate weights (`[n=48, k=5120]`, plain resident
//! F16) against prefill-scale `M` -- see `cu/narrow_gemm.cu`'s own header
//! for the design (grid over M-tiles, weight staged into shared memory once
//! a block, `x` read exactly once from global) and why it inverts the
//! grid-over-N convention this crate's other narrow-output kernels
//! (`gemv_f16`/`gemv_f16_ksplit`, `mmv_f8_plain_*`) use -- those are sized
//! for decode, where `x` is the tiny operand; this is sized for prefill,
//! where `x` is the large one.
//!
//! Not wired into the real forward pass. `model/src/lib.rs`'s `matmul_pre`
//! dispatch still falls through to the generic `to_f16`+`gemm_f16` path for
//! this weight -- see `tests/narrow_gemm.rs` for the isolated
//! correctness/sanitizer work, `examples/narrow_gemm_bench.rs` for the
//! isolated microbenchmark, and `cu/narrow_gemm.cu`'s own header for the
//! honest result: correct and sanitizer-clean, but a real, meaningful LOSS
//! against the composition it was meant to replace at every shape measured
//! (~4-29x slower on this development box's GPU) -- a **no-go** for
//! integration as currently designed. Kept in the tree as a real, working,
//! tested artifact of this pass rather than deleted, per the task's own
//! "report honestly rather than integrate something that doesn't help"
//! instruction.

use anyhow::{Context, Result};
use infero_gpu::{KernelArg, LaunchConfig, View, ViewMut};

use crate::{Kernels, narrow_gemm_src};

/// Hard compile-time cap on `n` -- matches `THREADS_PER_ROW * COLS_PER_THREAD`
/// in the `.cu` source exactly (each lane owns up to `COLS_PER_THREAD`
/// columns). A caller asking for a larger `n` is declined (`Ok(false)`),
/// never silently truncated.
pub const NARROW_GEMM_MAX_N: usize = 64;

const THREADS_PER_ROW: u32 = 32;
const ROWS_PER_BLOCK: u32 = 32;
const NARROW_GEMM_BLOCK: u32 = THREADS_PER_ROW * ROWS_PER_BLOCK;
const NARROW_GEMM_KTILE: usize = 128;

impl Kernels {
    /// `out[m,n] = x[m,k] @ w[n,k]^T`, `w` plain resident F16 (not
    /// quantized), `x` read directly as F32 (no separate `to_f16` pass --
    /// see the module doc for why that's a deliberate, real numerical
    /// difference from the unfused `to_f16`+`gemm_f16` composition this is
    /// meant to replace, not a bug). `out` is NOT accumulated into; this
    /// always overwrites it.
    ///
    /// Declines (`Ok(false)`) when `n` exceeds [`NARROW_GEMM_MAX_N`] -- the
    /// `.cu` side gives each output column its own thread, so `n` can never
    /// exceed how many threads a block devotes to one row. `k`/`m` are
    /// unbounded by this kernel's own design (tiled over `k`, gridded over
    /// `m`); `m == 0` is a legal no-op.
    pub fn narrow_gemm_f16(
        &self,
        out: &mut ViewMut<'_, f32>,
        w: &View<'_, u8>,
        x: &View<'_, f32>,
        k: usize,
        n: usize,
        m: usize,
    ) -> Result<bool> {
        if n > NARROW_GEMM_MAX_N {
            return Ok(false);
        }
        debug_assert!(w.len() >= n * k * 2, "a [{n},{k}] F16 matrix wants {} bytes, the view holds {}", n * k * 2, w.len());
        debug_assert!(x.len() >= m * k, "x holds {} f32s, need {}", x.len(), m * k);
        debug_assert!(out.len() >= m * n, "out holds {} f32s, need {}", out.len(), m * n);
        if m == 0 {
            return Ok(true);
        }

        let f = self.dev.kernels().get("infero_narrow_gemm", narrow_gemm_src(), "narrow_gemm_f16_f32")?;
        let smem = (NARROW_GEMM_KTILE * n * 2) as u32;
        let cfg = LaunchConfig {
            grid_dim: (m.div_ceil(ROWS_PER_BLOCK as usize) as u32, 1, 1),
            block_dim: (NARROW_GEMM_BLOCK, 1, 1),
            shared_mem_bytes: smem,
        };
        let (ki, ni, mi) = (k as i32, n as i32, m as i32);
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(out).arg(w).arg(x).arg(&ki).arg(&ni).arg(&mi);
        self.dev.profile().time("narrow_gemm_f16", self.dev.stream(), || {
            unsafe { b.launch(cfg) }.context("narrow_gemm_f16_f32")?;
            Ok(())
        })?;
        Ok(true)
    }
}
