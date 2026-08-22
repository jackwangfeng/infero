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

/// Above this many tokens, expand and call a GEMM instead of the batched
/// mat-vec.
///
/// The mat-vec holds one accumulator a token in registers, so its cost in
/// occupancy grows with the batch while a GEMM's does not. 32 is where the
/// second instantiation tops out; the crossover is measured rather than
/// assumed, and `TUILI_FP8_BATCH_MAX` moves it for an A/B.
pub const MAX_BATCH_TOKENS_FP8: usize = 32;

/// Where the batched mat-vec stops winning, measured.
///
/// The first guess was the register bound, 32, on the reasoning that reading
/// each weight once must beat expanding it. That is true of DRAM traffic and
/// stops being the binding constraint above a few tokens: at 32 tokens a thread
/// does 128 FMAs per four weight bytes loaded, and a SIMT f32 loop loses that to
/// cuBLAS's tensor cores. The second guess was 4, which was too low in the other
/// direction. Both were guesses, so here is the sweep, at the 27B's shape on an
/// RTX PRO 6000, milliseconds a decode step:
///
/// ```text
///   batch    mat-vec   expand+GEMM    which
///       1      28.30             —    mat-vec
///       2      34.69        131.55    mat-vec, 3.8x
///       4      49.08        133.65    mat-vec, 2.7x
///       8      80.41        137.39    mat-vec, 1.7x
///      16     ~136          139.85    level
///      32     ~251          140.66    expand, 1.8x
/// ```
///
/// The expansion path is nearly flat in the batch size, because its cost is one
/// whole-matrix expansion per projection whichever way the tokens fall — which
/// is also why it is catastrophic at two tokens, 4.6x slower than one token. The
/// mat-vec grows about 7.2 ms a token.
///
/// They cross at 16, and 16 is where this sits rather than 8. On time the two
/// are level there, so the tie is broken on what else the expansion path costs:
/// it needs `scratch.w16` to hold an entire projection — 17408 x 5120 halves,
/// 170 MiB for the widest one — and it writes those halves before reading them
/// back. The mat-vec needs neither. Level on time and cheaper in memory and
/// traffic is not a tie.
///
/// `TUILI_FP8_BATCH_MAX` moves it, which is how the table above was produced.
/// The real fix for the large-batch end is an FP8 GEMM that feeds tensor cores
/// directly rather than either of these.
pub fn batched_matvec_limit() -> usize {
    static LIMIT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *LIMIT.get_or_init(|| {
        std::env::var("TUILI_FP8_BATCH_MAX")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(16)
            .min(MAX_BATCH_TOKENS_FP8)
    })
}

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

    /// `out[t] = W x[t]` for a handful of tokens, reading each weight once.
    ///
    /// The case the expansion path was getting badly wrong: expanding to f16 and
    /// calling cuBLAS costs five bytes a weight against resident f16's two, and
    /// at a few tokens the weights still dominate — so batched decode ended up
    /// with 2.5x the traffic it had before FP8. Here the weight lands in
    /// registers and every token reuses it.
    ///
    /// `x` is `[n_tokens, k]`, `out` is `[n_tokens, n]`. Returns whether it ran:
    /// past `MAX_BATCH_TOKENS_FP8` a GEMM is the right answer and the caller
    /// should take the expansion path, where the cost amortizes over enough
    /// tokens to stop mattering.
    #[allow(clippy::too_many_arguments)]
    pub fn mmv_f8_block_batch(
        &self,
        out: &mut CudaViewMut<'_, f32>,
        w: &CudaView<'_, u8>,
        x: &CudaView<'_, f32>,
        k: usize,
        n: usize,
        n_tokens: usize,
        accum: bool,
    ) -> Result<bool> {
        if n_tokens > batched_matvec_limit() {
            return Ok(false);
        }
        anyhow::ensure!(
            k.is_multiple_of(4),
            "the mat-vec reads a row four bytes at a time; k is {k}"
        );
        debug_assert!(out.len() >= n_tokens * n);
        debug_assert!(x.len() >= n_tokens * k);
        debug_assert!(w.len() >= fp8_bytes(k, n));

        // Two instantiations rather than one at 32: the accumulators are
        // registers, so a block that only needs eight of them should not pay for
        // thirty-two. Which one runs is decided here, not by the kernel.
        let name = if n_tokens <= 8 {
            "mmv_f8_block_batch8_f32"
        } else {
            "mmv_f8_block_batch32_f32"
        };
        let f = self.dev.kernels().get("tuili_fp8", fp8_src(), name)?;
        const BLOCK: u32 = 256;
        let cfg = LaunchConfig {
            grid_dim: (n as u32, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let (ki, ni) = (k as i32, n as i32);
        let scols = k.div_ceil(FP8_BLOCK) as i32;
        let nt = n_tokens as i32;
        let acc = i32::from(accum);
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(out)
            .arg(w)
            .arg(x)
            .arg(&ki)
            .arg(&ni)
            .arg(&scols)
            .arg(&nt)
            .arg(&acc);
        self.dev
            .profile()
            .time("mmv_f8_block_batch", self.dev.stream(), || {
                unsafe { b.launch(cfg) }.context("mmv_f8_block_batch")?;
                Ok(())
            })?;
        Ok(true)
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
