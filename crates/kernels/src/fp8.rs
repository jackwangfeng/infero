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

/// `(token bound, kernel, rows per block)`, tightest bound first.
///
/// The third field must equal `FP8_MMV_ROWS*` in `fp8.cu`. The kernel decides
/// which rows it owns from `blockIdx.x * ROWS`; the launcher decides how many
/// blocks there are. If the two disagree downward, the output's tail is never
/// written — and that wrong answer is a plausible one, a projection whose last
/// rows are stale, which reads as slightly-off text rather than as an error.
/// `a_row_count_that_does_not_divide_the_row_tile_still_writes_every_row` in
/// `tests/fp8_matvec.rs` is the check.
///
/// The token count has to be the tightest available compile-time bound and not
/// just an upper one: `#pragma unroll` with a runtime `break` still allocates
/// every slot, so running two tokens through a sixteen-token kernel pays for
/// sixteen accumulators. That is why there are four instantiations.
///
/// **Rows per block is 1, and that is a measured result rather than the obvious
/// starting point.** A verification pass over `k + 1` rows costs 6.9 ms a row on
/// the 27B for zero extra weight bytes, and the arithmetic says why: with one
/// output row to a block, every block reads the whole activation, so activation
/// traffic is four bytes per *weight element* — 118 GB a token against 29.6 GB
/// of weights. It is L2 rather than DRAM, which is why it never showed up in a
/// bandwidth argument, and 118 GB at this card's L2 rate is the right size for
/// 6.9 ms. Neither the arithmetic (0.5 ms) nor the load count (16 us) is within
/// two orders of magnitude.
///
/// So handling several output rows a block should divide that traffic. Measured,
/// on the 27B, a two-row pass:
///
/// ```text
///   rows/block   accumulators        ms
///            1              2     34.77   <- shipped
///            8             64     95.65
///           16             32     47.29
/// ```
///
/// Both are worse, and the second rules out the first explanation. 64
/// accumulators plainly spilled; 32 does not, and 16 rows a block is still 36%
/// slower than one. What changes with the row count is the *weight* stream: a
/// block reading R rows at the same position along k has R concurrent streams
/// 5120 bytes apart instead of one contiguous run. The weight stream is the part
/// that is genuinely DRAM-bound — 29.6 GB at 1.8 TB/s is 16.4 ms of a 25.3 ms
/// step, so the step is already at 65% of the weight-read ceiling — and scattering
/// it costs more than the activation traffic saved.
///
/// Which leaves the structure that pays for the reuse without scattering the
/// loads: stage weight and activation tiles in shared memory with `cp.async`,
/// double-buffered, the way `mmq.cu` does for AWQ. That is the identified next
/// step, and `mmq.cu`'s own notes set the expectation — it lands at about 63% of
/// the weight-read ceiling and is then limited by its shared-memory footprint,
/// which for this shape would be ~26 ms flat in the row count against 34.8 and
/// climbing.
pub const BATCH_KERNELS: [(usize, &str, usize); 4] = [
    (2, "mmv_f8_block_batch2_f32", 1),
    (4, "mmv_f8_block_batch4_f32", 1),
    (8, "mmv_f8_block_batch8_f32", 1),
    (16, "mmv_f8_block_batch16_f32", 1),
];

/// Above this many tokens, expand and call a GEMM instead of the batched
/// mat-vec.
///
/// The mat-vec holds `ROWS * TOKENS` accumulators in registers, so its cost in
/// occupancy grows with the batch while a GEMM's does not. This is the widest
/// instantiation there is, derived from [`BATCH_KERNELS`] rather than repeated,
/// so that adding a kernel cannot leave the dispatch reaching for one that does
/// not exist. `TUILI_FP8_BATCH_MAX` moves the *crossover* for an A/B and is
/// clamped to this.
pub const MAX_BATCH_TOKENS_FP8: usize = BATCH_KERNELS[BATCH_KERNELS.len() - 1].0;

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
        let &(_, name, rows_per_block) = BATCH_KERNELS
            .iter()
            .find(|(max, _, _)| n_tokens <= *max)
            .with_context(|| {
                format!(
                    "{n_tokens} tokens is past the widest batched mat-vec, {}",
                    BATCH_KERNELS.last().unwrap().0
                )
            })?;
        let f = self.dev.kernels().get("tuili_fp8", fp8_src(), name)?;
        const BLOCK: u32 = 256;
        // One block per *group* of output rows, not per row. This has to track
        // `FP8_MMV_ROWS8` / `FP8_MMV_ROWS32` in the kernel: too few blocks and
        // the tail of the output is never written, which is a wrong answer that
        // looks like a plausible one.
        let cfg = LaunchConfig {
            grid_dim: (n.div_ceil(rows_per_block) as u32, 1, 1),
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
