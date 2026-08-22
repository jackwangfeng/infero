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

/// `(token bound, kernel)`, tightest bound first.
///
/// The token count has to be the tightest available compile-time bound and not
/// just an upper one: `#pragma unroll` with a runtime `break` still allocates
/// every slot, so running two tokens through a sixteen-token kernel pays for
/// sixteen accumulators. That is why there are four instantiations.
///
/// Every one of them reads four output rows a block, because the layout
/// interleaves four rows — see [`repack_rows`]. The road to that, in numbers,
/// because two intermediate answers were wrong in instructive ways.
///
/// A verification pass over `k + 1` rows used to cost 6.9 ms a row on the 27B
/// for zero extra weight bytes. The arithmetic ruled out the obvious suspects:
/// at two tokens a thread does eight FMAs per four weight bytes, four FLOP a
/// byte against this card's 64, which is 0.5 ms a step; the extra scalar loads
/// were 16 us. Both two orders of magnitude too small.
///
/// Handling several rows a block in the *row-major* layout should have divided
/// the activation traffic and instead made it worse — 95.7 ms at eight rows and
/// 47.3 at sixteen, against 34.8 at one. Sixteen rules out register spilling as
/// the explanation (32 accumulators, no spill, still 36% slower). What it does
/// is turn the weight stream into R runs 5120 bytes apart, and the weight stream
/// is the part that is genuinely DRAM-bound.
///
/// Reading the activation as one `float4` instead of four scalars then bought
/// 9.6% at two rows without touching a byte of traffic — which is the clue: the
/// limit is memory *requests*, not bytes. L1 serves a bounded number a cycle
/// whatever their width.
///
/// So: permute the weights at load so four rows are contiguous, and read all
/// four in one request. Milliseconds a decode pass, 27B shape:
///
/// ```text
///   rows      row-major   +float4    repacked
///      1          27.88     25.28       23.57
///      2          34.77     31.43       26.05
///      3          41.56     41.91       28.08
///      4          48.78     47.64       30.29
///   per row        ~6.9      ~6.2        ~2.2
/// ```
///
/// A three-row pass went from 41.6 ms to 28.1, and the marginal row from 6.9 ms
/// to 2.2. The remaining 2.2 is close to what the extra `lm_head` row actually
/// costs in bytes (1.29 GB at this step's 1049 GB/s is 1.2 ms), so the row is
/// nearly byte-limited now, which is what makes speculation's acceptance length
/// convert into throughput.
pub const BATCH_KERNELS: [(usize, &str); 4] = [
    (2, "mmv_f8_block_batch2_f32"),
    (4, "mmv_f8_block_batch4_f32"),
    (8, "mmv_f8_block_batch8_f32"),
    (16, "mmv_f8_block_batch16_f32"),
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

/// Output rows interleaved into one group by [`repack_rows`].
///
/// Four, so that a thread's four rows at four positions along k are one
/// 16-byte load. Must divide 128, or a group would straddle a scale-grid row.
pub const ROW_GROUP: usize = 4;

/// How many bytes an `[n, k]` FP8 matrix occupies, quants plus scale grid.
///
/// Rows are padded up to a whole [`ROW_GROUP`]. The padding is read by the
/// kernels and discarded by their per-row bounds check; leaving it out would
/// mean the last group's loads run off the end of the allocation.
pub fn fp8_bytes(k: usize, n: usize) -> usize {
    n.next_multiple_of(ROW_GROUP) * k + scale_grid(k, n) * std::mem::size_of::<f32>()
}

/// Interleave every [`ROW_GROUP`] rows so that a thread reads four rows at four
/// positions along k as one 16-byte load.
///
/// **This is the one definition of the quant layout.** The loader packs with it
/// and `tests/fp8_matvec.rs` packs with it, so a kernel that disagrees shows up
/// as wrong numbers against the host dequantizer rather than as a silent
/// mismatch between two hand-written copies of the same permutation.
///
/// Why at all: the mat-vec's cost above one token turned out to be *requests*
/// rather than bytes or arithmetic. Row-major, a thread reads one 4-byte weight
/// word and then one activation load a token — three requests a group at two
/// tokens — and L1 is limited by requests a cycle, not bytes. Reading four rows
/// at once makes it 0.25 weight requests and 0.25 activation requests a token
/// per group of four bytes, a quarter of the traffic in requests, and the FMAs
/// per request go from 8 to 32.
///
/// Two earlier attempts to get this reuse without repacking both failed, and
/// their numbers are on [`BATCH_KERNELS`]: handling several rows a block in
/// row-major order scatters the weight stream into R runs 5120 bytes apart, and
/// that costs more than the reuse saves. The permutation is what makes the four
/// rows contiguous.
///
/// Layout, for group `g` and k-chunk `c` of four:
///
/// ```text
///   new[g * ROW_GROUP * k + c * 16 + r * 4 + j]
///       = old[(g * ROW_GROUP + r) * k + c * 4 + j]
/// ```
///
/// `k` must be a multiple of four, which every projection in this checkpoint is
/// and which the mat-vec launchers already require.
pub fn repack_rows(quants: &[u8], k: usize, n: usize) -> Result<Vec<u8>> {
    anyhow::ensure!(
        k.is_multiple_of(4),
        "the repack moves four bytes at a time; k is {k}"
    );
    anyhow::ensure!(
        quants.len() == n * k,
        "{} quant bytes for an [{n}, {k}] matrix",
        quants.len()
    );
    let padded = n.next_multiple_of(ROW_GROUP);
    let mut out = vec![0u8; padded * k];
    let chunks = k / 4;
    // Groups are independent — each writes one `ROW_GROUP * k` window and reads
    // four source rows — so this runs across threads. On one core it is 28
    // seconds of a 63-second load on the 27B: 7.4e9 four-byte moves with writes
    // striding by sixteen, which no amount of loop tuning fixes because the work
    // really is that many moves.
    //
    // `Kernels::fp8_repack_rows` does the same thing on the device and is the
    // right answer for a loader that can upload row-major first; this stays
    // because the loader assembles a host buffer before it uploads, and because
    // the tests want a definition that needs no GPU.
    let threads = std::thread::available_parallelism()
        .map(|n| n.get().min(16))
        .unwrap_or(1);
    let per_group = ROW_GROUP * k;
    std::thread::scope(|scope| {
        // One chunk of consecutive groups per thread, so each thread's writes
        // stay in its own region and its reads walk forward through `quants`.
        let groups = padded / ROW_GROUP;
        let per_thread = groups.div_ceil(threads);
        for (t, window) in out.chunks_mut(per_thread * per_group).enumerate() {
            let quants = &quants;
            scope.spawn(move || {
                let g0 = t * per_thread;
                for (gi, group) in window.chunks_mut(per_group).enumerate() {
                    let g = g0 + gi;
                    for r in 0..ROW_GROUP {
                        let row = g * ROW_GROUP + r;
                        if row >= n {
                            break; // padding stays zero
                        }
                        let src = &quants[row * k..(row + 1) * k];
                        for c in 0..chunks {
                            let dst = c * (ROW_GROUP * 4) + r * 4;
                            group[dst..dst + 4].copy_from_slice(&src[c * 4..c * 4 + 4]);
                        }
                    }
                }
            });
        }
    });
    Ok(out)
}

/// How many scales an `[n, k]` matrix's grid holds.
pub fn scale_grid(k: usize, n: usize) -> usize {
    n.div_ceil(FP8_BLOCK) * k.div_ceil(FP8_BLOCK)
}

impl Kernels {
    /// [`repack_rows`], on the device.
    ///
    /// Same permutation, and it exists because doing it on the host costs 28
    /// seconds of a 63-second load on the 27B — 7.4e9 four-byte moves on one
    /// core, with writes striding by sixteen. `src` holds `n * k` row-major
    /// quants; `dst` must be `n.next_multiple_of(ROW_GROUP) * k` bytes and is
    /// left zero in the padding.
    pub fn fp8_repack_rows(
        &self,
        dst: &mut CudaViewMut<'_, u8>,
        src: &CudaView<'_, u8>,
        k: usize,
        n: usize,
    ) -> Result<()> {
        anyhow::ensure!(
            k.is_multiple_of(4),
            "the repack moves four bytes at a time; k is {k}"
        );
        debug_assert!(src.len() >= n * k);
        debug_assert!(dst.len() >= n.next_multiple_of(ROW_GROUP) * k);
        let f = self
            .dev
            .kernels()
            .get("tuili_fp8", fp8_src(), "fp8_repack_rows")?;
        let chunks = n * (k / 4);
        const BLOCK: u32 = 256;
        let cfg = LaunchConfig {
            grid_dim: (chunks.div_ceil(BLOCK as usize) as u32, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let (ki, ni) = (k as i32, n as i32);
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(dst).arg(src).arg(&ki).arg(&ni);
        self.dev
            .profile()
            .time("fp8_repack_rows", self.dev.stream(), || {
                unsafe { b.launch(cfg) }.context("fp8_repack_rows")?;
                Ok(())
            })?;
        Ok(())
    }

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
        // Eight warps, so eight of the group's 128-wide slices are in flight and
        // each finishes with a shuffle rather than a barrier. One block per
        // *group* of `ROW_GROUP` rows, which is the unit the layout interleaves.
        const BLOCK: u32 = 256;
        let cfg = LaunchConfig {
            grid_dim: (n.div_ceil(ROW_GROUP) as u32, 1, 1),
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
        let &(_, name) = BATCH_KERNELS
            .iter()
            .find(|(max, _)| n_tokens <= *max)
            .with_context(|| {
                format!(
                    "{n_tokens} tokens is past the widest batched mat-vec, {}",
                    BATCH_KERNELS.last().unwrap().0
                )
            })?;
        let f = self.dev.kernels().get("tuili_fp8", fp8_src(), name)?;
        const BLOCK: u32 = 256;
        // One block per *group* of output rows, not per row. This has to track
        // `FP8_ROW_GROUP` in the kernel: too few blocks and the tail of the
        // output is never written, which is a wrong answer that looks like a
        // plausible one.
        let cfg = LaunchConfig {
            grid_dim: (n.div_ceil(ROW_GROUP) as u32, 1, 1),
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
            grid_dim: (
                k.div_ceil(FP8_BLOCK) as u32,
                n.div_ceil(ROW_GROUP) as u32,
                1,
            ),
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
