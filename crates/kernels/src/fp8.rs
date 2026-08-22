//! Launchers for block-scaled FP8 E4M3 weights.
//!
//! Why these exist rather than dequantizing at load, which is what tuili did
//! first and which is correct: a decode step reads every weight exactly once,
//! so it is bound by how many bytes the weights are. Expanding FP8 to f16 at
//! load doubles that, and the profiler put `gemm_f16` at 75% of a step on the
//! 27B — 13.2 tok/s against vLLM's 34 on the same checkpoint.
//!
//! Reading the FP8 bytes directly buys two things at once, and the second was
//! not the plan. Half the bytes, obviously. But it also replaces cuBLAS at
//! batch one: with the weights stored as f16 there is no quantized type for the
//! mat-vec path to match on, so every batch-1 projection went through an f16
//! GEMM with m = 1, at 86.8 us each. A mat-vec is the right shape for that and
//! cuBLAS's GEMM is not.
//!
//! The layout is described on [`crate::WeightType::F8E4M3`]. The one thing to
//! hold on to: a scale covers 128 rows *and* 128 columns, so it depends on the
//! output row as well as the position along k. Applying it per row, or once per
//! matrix, is not a rounding difference — it is a different matrix.

use anyhow::{Context, Result};
use cudarc::driver::{CudaView, CudaViewMut, LaunchConfig, PushKernelArg};
use half::f16;

use crate::{Kernels, fp8_src};

/// The scale grid's block size, in both directions.
pub const FP8_BLOCK: usize = 128;

/// How many bytes an `[n, k]` FP8 matrix occupies, quants plus scale grid.
pub fn fp8_bytes(k: usize, n: usize) -> usize {
    n * k + scale_grid(k, n) * std::mem::size_of::<f32>()
}

/// How many scales an `[n, k]` matrix's grid holds.
pub fn scale_grid(k: usize, n: usize) -> usize {
    n.div_ceil(FP8_BLOCK) * k.div_ceil(FP8_BLOCK)
}

impl Kernels {
    /// `out = W x`, with `W` in FP8 and its block scales, at one token.
    ///
    /// `w` is the whole buffer: quants then grid. `accum` adds into `out`
    /// instead of overwriting, which folds the residual add into the projection
    /// that feeds it — the same trick the other mat-vecs use.
    #[allow(clippy::too_many_arguments)]
    pub fn mmv_f8_block(
        &self,
        out: &mut CudaViewMut<'_, f32>,
        w: &CudaView<'_, u8>,
        x: &CudaView<'_, f32>,
        k: usize,
        n: usize,
        accum: bool,
    ) -> Result<()> {
        anyhow::ensure!(
            k.is_multiple_of(4),
            "the mat-vec reads a row four bytes at a time; k is {k}"
        );
        debug_assert!(out.len() >= n);
        debug_assert!(x.len() >= k);
        debug_assert!(
            w.len() >= fp8_bytes(k, n),
            "an [{n}, {k}] FP8 matrix wants {} bytes, the view holds {}",
            fp8_bytes(k, n),
            w.len()
        );

        let f = self
            .dev
            .kernels()
            .get("tuili_fp8", fp8_src(), "mmv_f8_block_f32")?;
        // Eight warps, so eight of the row's 128-wide slices are in flight and
        // each finishes with a shuffle rather than a barrier.
        const BLOCK: u32 = 256;
        let cfg = LaunchConfig {
            grid_dim: (n as u32, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let (ki, ni) = (k as i32, n as i32);
        let scols = k.div_ceil(FP8_BLOCK) as i32;
        let acc = i32::from(accum);
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(out)
            .arg(w)
            .arg(x)
            .arg(&ki)
            .arg(&ni)
            .arg(&scols)
            .arg(&acc);
        self.dev
            .profile()
            .time("mmv_f8_block", self.dev.stream(), || {
                unsafe { b.launch(cfg) }.context("mmv_f8_block")?;
                Ok(())
            })?;
        Ok(())
    }

    /// Expand an FP8 matrix to f16 on the device, for the batched path.
    ///
    /// Prefill still goes through the f16 GEMM, so the bytes have to be
    /// expandable where they are. This is the work that used to happen on the
    /// host and cost 22 GiB of resident memory.
    pub fn dequant_f8_block_to_f16(
        &self,
        out: &mut CudaViewMut<'_, f16>,
        w: &CudaView<'_, u8>,
        k: usize,
        n: usize,
    ) -> Result<()> {
        debug_assert!(out.len() >= k * n);
        debug_assert!(w.len() >= fp8_bytes(k, n));
        let f = self
            .dev
            .kernels()
            .get("tuili_fp8", fp8_src(), "dequant_f8_block_f16")?;
        let cfg = LaunchConfig {
            grid_dim: (k.div_ceil(FP8_BLOCK) as u32, n as u32, 1),
            block_dim: (FP8_BLOCK as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let (ki, ni) = (k as i32, n as i32);
        let scols = k.div_ceil(FP8_BLOCK) as i32;
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(out).arg(w).arg(&ki).arg(&ni).arg(&scols);
        self.dev
            .profile()
            .time("dequant_f8_block", self.dev.stream(), || {
                unsafe { b.launch(cfg) }.context("dequant_f8_block")?;
                Ok(())
            })?;
        Ok(())
    }
}
