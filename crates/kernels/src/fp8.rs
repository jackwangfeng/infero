//! Launchers for block-scaled FP8 E4M3 weights.
//!
//! Why these exist rather than dequantizing at load, which is what infero did
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
use infero_gpu::{View, ViewMut, LaunchConfig, KernelArg};
use half::f16;

use crate::{Kernels, fp8_src};

/// The scale grid's block size, in both directions.
pub const FP8_BLOCK: usize = 128;

/// `#define`s that take pieces out of the mat-vec, for
/// `examples/fp8_row_cost.rs`.
///
/// A marginal row costs 2.25 ms where its DRAM bytes are zero, and three
/// end-to-end explanations were all wrong — so the move left is to remove one
/// piece at a time and see which one the cost follows. `INFERO_FP8_STRIP` takes
/// `fma`, `reduce`, or `both`; anything else, including unset, is the real
/// kernel. These produce wrong answers by construction.
pub fn strip_flags() -> &'static str {
    static F: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    F.get_or_init(|| {
        let want = std::env::var("INFERO_FP8_STRIP").unwrap_or_default();
        let fma = want == "fma" || want == "both";
        let reduce = want == "reduce" || want == "both";
        if fma || reduce {
            tracing::warn!(
                strip = %want,
                "INFERO_FP8_STRIP is set: the FP8 mat-vec is computing the wrong answer \
                 on purpose"
            );
        }
        format!(
            "#define FP8_STRIP_FMA {}\n#define FP8_STRIP_REDUCE {}",
            i32::from(fma),
            i32::from(reduce)
        )
    })
}

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
/// to 2.2, which is what makes speculation's acceptance length convert into
/// throughput.
///
/// **What the remaining 2.2 ms a row is, is not known.** Three explanations have
/// been tried and measured, and all three are wrong:
///
/// * *Arithmetic.* A third token adds 32 FMAs per sixteen weight bytes, which
///   totals 1.0 ms a step against a measured 4.5.
/// * *Request count.* This is what the repack fixed, and it took the row from
///   6.9 to 2.2 — so whatever is left is not that.
/// * *Activation L1 bytes.* Four rows a block means the activation is read
///   `4 bytes * weight_elements / 4` a token, so 29.6 GB a marginal row on this
///   checkpoint, and 13 TB/s is about what L1 does — the arithmetic fits. But
///   halving it by reading the activation as f16 measured 28.32 ms against
///   28.09, slightly *worse*, and the draft step went 4.11 to 4.21. Reverted:
///   f16 activations are a couple of percent off on a cancelled element (see the
///   error model in `tests/fp8_matvec.rs`) and they bought nothing.
///
/// It is the per-token arithmetic, and getting to that answer took two
/// corrections to the probe that found it. `examples/fp8_row_cost.rs` removes
/// pieces of the kernel instead of reasoning about them, and its first two
/// versions were both measuring the wrong thing:
///
/// * **The weights were in L2.** One 85 MiB projection reused forty times is
///   resident in this card's 128 MB L2 after the first rep, so the probe reported
///   3597 GB/s with the reduction stripped — twice the card's DRAM bandwidth,
///   which should have been the tell. It now rotates through four copies.
/// * **`strip=reduce` was deleting the arithmetic too.** Reducing only
///   `acc[0][0]` leaves fifteen of the sixteen accumulator chains unread, so
///   ptxas eliminated them; the register count fell from 56 to 33 and said so.
///   It now consumes every accumulator with plain adds, which removes the
///   shuffles, the shared memory and the barrier at an *unchanged* register
///   count.
///
/// With both fixed, at three tokens and 56 registers either way:
///
/// ```text
///                        1 tok   3 tok   marginal/row
///   everything           0.070   0.096         0.013
///   no reduction         0.069   0.088        0.0095
///   no multiply-add      0.061   0.066        0.0025
///   weights only         0.068   0.061             -
/// ```
///
/// So of a 0.013 ms marginal row, the reduction is 27% and the per-token
/// multiply-accumulate is 81%. **The previous commit had this backwards**, on the
/// strength of the contaminated switch.
///
/// `weights only` is flat at 0.061 ms = 1466 GB/s, 81% of this card's DRAM
/// bandwidth, so the load floor is real and the kernel at one token is already
/// within 16% of it. What is left is arithmetic running at a seventh of the f32
/// FMA bound — 0.0105 ms a row against 0.0015 — because it is a scalar loop
/// issuing sixteen FMA instructions per chunk per token. One `mma.m16n8k16` does
/// 2048 MACs in one instruction, and even wasting thirteen of its sixteen rows at
/// three tokens that is twelve times fewer instructions.
///
/// Two attempts to make the reduction cheaper, both measured, both a wash:
///
/// * **Unroll the chains so they interleave.** Twelve five-step chains at three
///   tokens are independent of each other, and the `break` on a runtime bound
///   inside a `#pragma unroll` loop was sequencing them. Removing it changed
///   nothing — 0.086 ms at three tokens either way — so the shuffles are limited
///   by throughput, not dependency: eight warps times twelve chains times five
///   steps is 480 instructions a block whatever their order.
/// * **One warp a block**, which makes the reduction pure shuffle with no shared
///   memory and no barrier, 60 instructions instead of 480:
///
/// ```text
///   threads   1 tok   3 tok   marginal/row
///        32   0.077   0.086       +0.005
///        64   0.076   0.086       +0.005
///       128   0.064   0.086       +0.011
///       256   0.062   0.086       +0.014
/// ```
///
///   It halves the marginal row and costs 24% at one token, because 32 threads
///   cannot hide the weight loads' latency the way 256 can. At three tokens they
///   land on the same number, and the wide block is better for plain decode, so
///   256 stays.
///
/// Four attempts, and the conclusion is that this is a floor for a kernel of
/// this shape: `rows * tokens` partial sums spread over 256 threads have to be
/// combined, and combining them costs what it costs. Getting past it means not
/// having them — a tiled GEMM keeps its accumulator in registers across the whole
/// k loop and never reduces across threads at all.
/// Output rows a tensor-core block owns: the M of `mma.m16n8k16`.
pub const MMA_ROWS: usize = 16;

/// Tokens one fragment column holds: the N of `mma.m16n8k16`.
///
/// Eight is the instruction's own width, so one token and eight cost the same
/// number of MMAs.
pub const MMA_TOKENS: usize = 8;

/// Staging the *activation* tile beside the weights was tried and reverted.
///
/// The reasoning looked sound: an unstaged `B` is read once a warp a group, and a
/// lane's eight bytes come from a different token than its neighbour's, so one
/// fragment is eight 32-byte sectors instead of two. That is `14.8 GB * GROUPS`
/// of L1 traffic for the 27B's forward, and prefill measured 451 GB/s at four
/// groups against a decode step's 1420.
///
/// Staged — `[token][k]` in shared with the same `+8` row padding, read once a
/// block and perfectly coalesced — prefill got 13% *slower*: the server's
/// `queued_ms` went 121 ms to 137, and the decode round did not move at all
/// (correctly: at one group there is nothing to stage). So the extra pass through
/// shared memory and its barrier cost more than the L1 requests they saved, and
/// the traffic was not what prefill was waiting on. Whatever prefill's 121 ms is,
/// it is not this.
///
/// Fragment columns off one staged weight tile, and so the token counts covered
/// in a single pass over the weights: 8, 16, 32, 64.
///
/// Prefill is what this is for. A 66-token prompt used to fall to the expansion
/// path at five bytes a weight — 148 GB against 29.6 for the 27B's forward.
/// Each extra group is one more MMA and one more `B` fragment against a shared
/// tile, so the weights are still read once.
pub const MMA_GROUPS: [(usize, &str); 4] = [
    (1, "mma_f8_block_f32"),
    (2, "mma_f8_block_g2_f32"),
    (4, "mma_f8_block_g4_f32"),
    (8, "mma_f8_block_g8_f32"),
];

/// The same, with four times the warps a block — for matrices too narrow to fill
/// the machine at eight.
///
/// A block owns [`MMA_ROWS`] output rows whatever its warp count, so a matrix of
/// `n` rows offers `n / 16` blocks, and the rate follows warps an SM until it
/// saturates near 48. At eight warps, one row, measured:
///
/// ```text
///        n   blocks   warps/SM     GB/s
///     5120      320       13.6      934
///     6144      384       16.3     1054
///    10240      640       27.2     1271
///    17408     1088       46.3     1413
/// ```
///
/// Most of the 27B's projections are 5120 or 6144 wide, which is why a decode
/// step's 433 launches averaged 1281 GB/s while the widest matrix reads 1413.
/// Only two group counts are instantiated: the wide-warp kernel stages four
/// scale blocks at once, so its shared tile is four times as large, and past two
/// groups it would not fit.
pub const MMA_GROUPS_W32: [(usize, &str); 2] = [
    (1, "mma_f8_block_w32_f32"),
    (2, "mma_f8_block_w32_g2_f32"),
];

/// Force one warp count, for the A/B that set [`MMA_WARP_TARGET`].
///
/// `INFERO_FP8_MMA_WARPS=8` or `=32`. Unset picks by width.
fn mma_warp_override() -> Option<u32> {
    static V: std::sync::OnceLock<Option<u32>> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("INFERO_FP8_MMA_WARPS")
            .ok()
            .and_then(|v| v.parse().ok())
    })
}

/// Where thirty-two warps a block stops paying, in warps an SM at eight.
///
/// The A/B, one row, `INFERO_FP8_MMA_WARPS` forcing each:
///
/// ```text
///        n   blocks   warps/SM   8 warps   32 warps
///     5120      320       13.6       913       1158   +27%
///     6144      384       16.3      1038       1028    -1%
///    10240      640       27.2      1263       1237    -2%
///    17408     1088       46.3      1404       1353    -4%
/// ```
///
/// A narrow boundary, and worth having anyway: `n = 5120` is `o_proj`,
/// `down_proj` and the GDN `out_proj`, about 130 of a decode step's 433
/// launches. Above it the wider kernel's four-times-larger shared tile costs a
/// couple of percent and the extra warps buy nothing, since eight already put
/// 16 an SM in flight.
///
/// 15 rather than 14 or 16 because the two measured points are 13.6 and 16.3 and
/// there is nothing in between to distinguish. Re-run the sweep on another card
/// before trusting the number there.
const MMA_WARP_TARGET: usize = 15;

/// Streaming multiprocessors on the card this was tuned against. Only used to
/// choose between the two warp counts, so being wrong by a little costs a little.
const MMA_SMS: usize = 188;

pub const BATCH_KERNELS: [(usize, &str); 8] = [
    (2, "mmv_f8_block_batch2_f32"),
    (3, "mmv_f8_block_batch3_f32"),
    (4, "mmv_f8_block_batch4_f32"),
    (5, "mmv_f8_block_batch5_f32"),
    (6, "mmv_f8_block_batch6_f32"),
    (7, "mmv_f8_block_batch7_f32"),
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
/// not exist. `INFERO_FP8_BATCH_MAX` moves the *crossover* for an A/B and is
/// clamped to this.
pub const MAX_BATCH_TOKENS_FP8: usize = BATCH_KERNELS[BATCH_KERNELS.len() - 1].0;

/// Above this many tokens, [`Kernels::mma_f8_block`] and
/// [`Kernels::mma_e4m3_block`] decline the shape too (see their own
/// `n_tokens > 4 * <groups>.last().0 * MMA_TOKENS` gate), and the caller
/// falls all the way to the expand-then-dequantize-then-GEMM path those
/// functions' doc comments call out as "a performance bug the profiler
/// cannot see." A prefill chunk sized above this for an FP8-weighted model
/// pays that cost on every step rather than the tensor-core GEMM's.
///
/// `MMA_GROUPS` and `MMA_E4M3_GROUPS` share the same last group count, so one
/// constant covers both gates.
pub const MMA_MAX_TOKENS_FP8: usize = 4 * MMA_GROUPS[MMA_GROUPS.len() - 1].0 * MMA_TOKENS;

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
/// `INFERO_FP8_BATCH_MAX` moves it, which is how the table above was produced.
/// The real fix for the large-batch end is an FP8 GEMM that feeds tensor cores
/// directly rather than either of these.
pub fn batched_matvec_limit() -> usize {
    static LIMIT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *LIMIT.get_or_init(|| {
        std::env::var("INFERO_FP8_BATCH_MAX")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(16)
            .min(MAX_BATCH_TOKENS_FP8)
    })
}

/// Output rows interleaved into one group by [`repack_rows`].
///
/// Four: one 16-byte load a thread, covering four rows at four positions along
/// k, so one activation `float4` feeds sixteen products.
///
/// Eight was tried after the repack landed, on the theory that the repack had
/// removed what made wide groups lose. It had not: a three-row pass went 28.06
/// to 36.57 ms and even one row got slower, 23.59 to 24.30. The repack makes the
/// *loads* contiguous, but the unpacked values still need `ROW_GROUP * 4`
/// registers alongside `ROW_GROUP * TOKENS` accumulators, and at eight that is
/// 64 registers of weights — the same wall the pre-repack attempts hit for a
/// different reason.
///
/// Must divide 128, or a group would straddle a scale-grid row, and must be a
/// multiple of four, since the kernels read it in `uint4`s.
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

/// [`repack_rows`], with the identity permutation: `[n,k]` row-major in,
/// zero-padded up to a whole [`ROW_GROUP`] rows, same total size
/// `repack_rows` returns. For [`Kernels::mmv_f8_plain`] and the `cutlass`
/// feature's unified-format weight path, where nothing reads the
/// interleave and `[n,k]` row-major is what CUTLASS wants natively too —
/// see `cutlass/fp8_bw_gemm.cu`'s header.
pub fn pad_rows(quants: &[u8], k: usize, n: usize) -> Result<Vec<u8>> {
    anyhow::ensure!(
        quants.len() == n * k,
        "{} quant bytes for an [{n}, {k}] matrix",
        quants.len()
    );
    let padded = n.next_multiple_of(ROW_GROUP);
    let mut out = vec![0u8; padded * k];
    out[..quants.len()].copy_from_slice(quants);
    Ok(out)
}

/// How many scales an `[n, k]` matrix's grid holds.
pub fn scale_grid(k: usize, n: usize) -> usize {
    n.div_ceil(FP8_BLOCK) * k.div_ceil(FP8_BLOCK)
}

/// Stack three already-packed FP8 matrices — each the `fp8_bytes` layout
/// [`projection_bytes`][crate] produces — along the output dimension, the way
/// `q_proj`/`k_proj`/`v_proj` become one launch instead of three.
///
/// A packed matrix is quants (`n.next_multiple_of(ROW_GROUP) * k` bytes) then
/// its scale grid (`scale_grid(k, n)` `f32`s), which is two disjoint regions —
/// concatenating the three whole buffers end to end would interleave them
/// wrong, quants-a, scales-a, quants-b, scales-b, ..., where a reader wants
/// all the quants first and then all the scales. This reassembles them.
///
/// Requires every `n` to be a multiple of [`FP8_BLOCK`] as well as
/// `ROW_GROUP`, so no matrix's scale grid or row-group padding falls in the
/// middle of another's rows; the caller checks this before calling in and
/// takes the unfused path when it does not hold, the same as the transposed
/// AWQ stack does for its own alignment case.
pub fn concat3(a: &[u8], n_a: usize, b: &[u8], n_b: usize, c: &[u8], n_c: usize, k: usize) -> Vec<u8> {
    concat(&[(a, n_a), (b, n_b), (c, n_c)], k)
}

/// [`concat3`] for two matrices — GatedDeltaNet's `in_proj_qkv` and
/// `in_proj_z`, which share `k` the same way q/k/v do.
pub fn concat2(a: &[u8], n_a: usize, b: &[u8], n_b: usize, k: usize) -> Vec<u8> {
    concat(&[(a, n_a), (b, n_b)], k)
}

fn concat(parts: &[(&[u8], usize)], k: usize) -> Vec<u8> {
    let scale_cols = k.div_ceil(FP8_BLOCK);
    let quant_len = |n: usize| n * k;
    let scale_len = |n: usize| (n / FP8_BLOCK) * scale_cols;
    for &(bytes, n) in parts {
        debug_assert!(n.is_multiple_of(FP8_BLOCK));
        debug_assert_eq!(bytes.len(), quant_len(n) + scale_len(n) * 4);
    }
    let mut out = Vec::with_capacity(parts.iter().map(|(b, _)| b.len()).sum());
    for &(bytes, n) in parts {
        out.extend_from_slice(&bytes[..quant_len(n)]);
    }
    for &(bytes, n) in parts {
        out.extend_from_slice(&bytes[quant_len(n)..]);
        debug_assert_eq!(bytes.len() - quant_len(n), scale_len(n) * 4);
    }
    out
}

/// The activation quantizer's group width — must match `QUANT_GROUP` in
/// `fp8.cu`. Chosen to equal [`FP8_BLOCK`] rather than independently, so the
/// GEMM's own `it * SCALES` index into the weight's scale grid also indexes
/// the activation's, with no second stride to carry.
pub const ACT_QUANT_GROUP: usize = 128;

/// `MMA_GROUPS`' entries, for `mma_e4m3_block`. A separate table because the
/// kernel names differ, not because the group counts do — `m16n8k32`'s N is
/// the same 8 `m16n8k16`'s is, so up to `8 * groups` tokens a pass means the
/// same thing here it does there.
pub const MMA_E4M3_GROUPS: [(usize, &str); 4] = [
    (1, "mma_e4m3_block_f32"),
    (2, "mma_e4m3_block_g2_f32"),
    (4, "mma_e4m3_block_g4_f32"),
    (8, "mma_e4m3_block_g8_f32"),
];

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
        dst: &mut ViewMut<'_, u8>,
        src: &View<'_, u8>,
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
            .get("infero_fp8", fp8_src(), "fp8_repack_rows")?;
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

    /// Dynamic per-token, per-[`ACT_QUANT_GROUP`] e4m3 quantization of an
    /// activation, the operand `mma_e4m3` (`mma.cuh`) wants in place of the
    /// f32 `mma_f8_block` reads directly. `xq` is `[n_tokens, k]`, `xs` is
    /// `[n_tokens, k / ACT_QUANT_GROUP]`.
    ///
    /// `caps().fp8` gates this the same as the MMA itself — the conversion
    /// inside has no software fallback, unlike `e4m3_to_f32`'s decode side,
    /// because a quantizer with no hardware path to feed is not worth having.
    pub fn quantize_act_e4m3(
        &self,
        xq: &mut ViewMut<'_, u8>,
        xs: &mut ViewMut<'_, f32>,
        x: &View<'_, f32>,
        k: usize,
        n_tokens: usize,
    ) -> Result<()> {
        anyhow::ensure!(
            k.is_multiple_of(ACT_QUANT_GROUP),
            "the activation quantizer's groups are {ACT_QUANT_GROUP} wide; k is {k}"
        );
        debug_assert!(x.len() >= n_tokens * k);
        debug_assert!(xq.len() >= n_tokens * k);
        debug_assert!(xs.len() >= n_tokens * (k / ACT_QUANT_GROUP));
        let f = self
            .dev
            .kernels()
            .get("infero_fp8", fp8_src(), "quantize_act_e4m3_f32")?;
        let cfg = LaunchConfig {
            grid_dim: (n_tokens as u32, (k / ACT_QUANT_GROUP) as u32, 1),
            block_dim: (ACT_QUANT_GROUP as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let ki = k as i32;
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(xq).arg(xs).arg(x).arg(&ki);
        self.dev
            .profile()
            .time("quantize_act_e4m3", self.dev.stream(), || {
                unsafe { b.launch(cfg) }.context("quantize_act_e4m3")?;
                Ok(())
            })?;
        Ok(())
    }

    /// Same quantization as [`Self::quantize_act_e4m3`], but for callers with
    /// no other use for the natural `[n_tokens, groups]` scale layout: writes
    /// directly into `sfa_t`, the transposed-and-padded `[groups, m_pad]`
    /// layout `mma_e4m3_cutlass`'s CUTLASS kernel wants, folding what used to
    /// be a separate `transpose_pad_scale_a_f32` pass into this kernel's
    /// existing per-group reduction. `xq` is still `[n_tokens, k]` —
    /// unaffected, that padding happens elsewhere. See
    /// [`crate::cutlass_fp8::Kernels::mma_e4m3_cutlass_sfa`], which consumes
    /// `sfa_t` directly rather than computing it from `xs`.
    pub fn quantize_act_e4m3_cutlass(
        &self,
        xq: &mut ViewMut<'_, u8>,
        sfa_t: &mut ViewMut<'_, f32>,
        x: &View<'_, f32>,
        k: usize,
        n_tokens: usize,
        m_pad: usize,
    ) -> Result<()> {
        anyhow::ensure!(
            k.is_multiple_of(ACT_QUANT_GROUP),
            "the activation quantizer's groups are {ACT_QUANT_GROUP} wide; k is {k}"
        );
        anyhow::ensure!(m_pad >= n_tokens, "m_pad {m_pad} is narrower than n_tokens {n_tokens}");
        debug_assert!(x.len() >= n_tokens * k);
        debug_assert!(xq.len() >= n_tokens * k);
        debug_assert!(sfa_t.len() >= (k / ACT_QUANT_GROUP) * m_pad);
        let f = self
            .dev
            .kernels()
            .get("infero_fp8", fp8_src(), "quantize_act_e4m3_cutlass_f32")?;
        let cfg = LaunchConfig {
            grid_dim: (m_pad as u32, (k / ACT_QUANT_GROUP) as u32, 1),
            // One warp a block: see `quantize_act_e4m3_cutlass_f32`'s doc
            // comment for why this replaced the old one-thread-a-lane,
            // four-warp, shared-memory-reduced layout.
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        let (ki, nt, mp) = (k as i32, n_tokens as i32, m_pad as i32);
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(xq).arg(sfa_t).arg(x).arg(&ki).arg(&nt).arg(&mp);
        self.dev
            .profile()
            .time("quantize_act_e4m3_cutlass", self.dev.stream(), || {
                unsafe { b.launch(cfg) }.context("quantize_act_e4m3_cutlass")?;
                Ok(())
            })?;
        Ok(())
    }

    /// One thread a trial: quantizes `q[trial]`/`k[trial]` (`d_head` each) to
    /// e4m3 per 128-wide group and computes both the exact and the
    /// quantize-round-tripped dot product. See `e4m3_qk_dot_accuracy_probe`
    /// in `fp8.cu` for why this exists — the accuracy half of evaluating an
    /// e4m3 QK^T/PV attention kernel, the throughput half being
    /// `Self::mma_e4m3_throughput_probe`.
    pub fn e4m3_qk_dot_accuracy_probe(
        &self,
        exact_out: &mut ViewMut<'_, f32>,
        quant_out: &mut ViewMut<'_, f32>,
        q: &View<'_, f32>,
        k: &View<'_, f32>,
        trials: usize,
        d_head: usize,
        group: usize,
    ) -> Result<()> {
        let f = self
            .dev
            .kernels()
            .get("infero_fp8", fp8_src(), "e4m3_qk_dot_accuracy_probe")?;
        let cfg = LaunchConfig {
            grid_dim: (trials as u32, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        let (dh, g) = (d_head as i32, group as i32);
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(exact_out).arg(quant_out).arg(q).arg(k).arg(&dh).arg(&g);
        unsafe { b.launch(cfg) }.context("e4m3_qk_dot_accuracy_probe")?;
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
        out: &mut ViewMut<'_, f32>,
        w: &View<'_, u8>,
        x: &View<'_, f32>,
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
            .get("infero_fp8", fp8_src(), "mmv_f8_block_f32")?;
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

    /// [`Kernels::mmv_f8_block`] against plain `[n,k]` row-major quants
    /// ([`pad_rows`], not [`repack_rows`]) and an untransposed
    /// `[n/128,k/128]` scale grid -- CUTLASS's native weight layout. The
    /// `cutlass` feature's unified-format path (single-token decode) uses
    /// this instead of `mmv_f8_block`, at a measured 2-17% cost on an SM120
    /// card (`examples/cutlass_vs_block.rs`) against
    /// `repack_rows`-interleaved reads -- small enough, on this hardware, to
    /// not keep two copies of every FP8 weight over. See
    /// `cutlass/fp8_bw_gemm.cu`'s header for why this is CUTLASS's format
    /// too, so a matrix stored this way needs no extra copy for either
    /// kernel.
    pub fn mmv_f8_plain(
        &self,
        out: &mut ViewMut<'_, f32>,
        w: &View<'_, u8>,
        x: &View<'_, f32>,
        k: usize,
        n: usize,
        accum: bool,
    ) -> Result<()> {
        debug_assert!(
            w.len() >= fp8_bytes(k, n),
            "an [{n}, {k}] FP8 matrix wants {} bytes, the view holds {}",
            fp8_bytes(k, n),
            w.len()
        );
        let f = self.dev.kernels().get("infero_fp8", fp8_src(), "mmv_f8_plain_f32")?;
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
        b.arg(out).arg(w).arg(x).arg(&ki).arg(&ni).arg(&scols).arg(&acc);
        self.dev
            .profile()
            .time("mmv_f8_plain", self.dev.stream(), || {
                unsafe { b.launch(cfg) }.context("mmv_f8_plain")?;
                Ok(())
            })?;
        Ok(())
    }

    /// The same product on tensor cores: `mma.m16n8k16`, f16 operands, f32
    /// accumulator.
    ///
    /// Why this exists is on `mma_f8_block_f32` in the `.cu`, and in one line it
    /// is that the scalar path's marginal row is 81% per-token arithmetic issued
    /// sixteen FMA instructions at a time. `MMA_TOKENS` is the N of the fragment,
    /// so anything up to eight tokens costs the same as one — the columns past
    /// `n_tokens` are fed zeros and their accumulators are never read.
    ///
    /// Requires `k % 128 == 0`, which is the scale block and holds for every
    /// projection in the 27B. Returns whether it ran.
    #[allow(clippy::too_many_arguments)]
    pub fn mma_f8_block(
        &self,
        out: &mut ViewMut<'_, f32>,
        w: &View<'_, u8>,
        x: &View<'_, f32>,
        k: usize,
        n: usize,
        n_tokens: usize,
        accum: bool,
    ) -> Result<bool> {
        // Where this stops being the cheaper path, by bytes a weight. The
        // widest instantiation covers 64 tokens a pass, so it reads the matrix
        // `ceil(m / 64)` times at one byte a weight; expanding reads one and
        // writes two and reads two back, five bytes a weight, whatever `m` is.
        // Five passes is where they meet. Derived, not measured — the crossover
        // is worth a sweep once a long prompt is the thing being timed.
        if !k.is_multiple_of(FP8_BLOCK)
            || n_tokens > 4 * MMA_GROUPS.last().unwrap().0 * MMA_TOKENS
        {
            return Ok(false);
        }
        debug_assert!(out.len() >= n_tokens * n);
        debug_assert!(x.len() >= n_tokens * k);
        debug_assert!(w.len() >= fp8_bytes(k, n));

        // Two choices, in this order. First the warp count: with eight warps a
        // block, a narrow matrix cannot fill the machine, so take thirty-two
        // when eight would leave the SMs under the rate's saturation point and
        // the wider shared tile still divides `k`. Then the narrowest group
        // count that covers the tokens, so a single-token decode does not carry
        // eight accumulator columns.
        let blocks = n.div_ceil(MMA_ROWS);
        let want = n_tokens.div_ceil(MMA_TOKENS);
        let fits_wide =
            k.is_multiple_of(32 * 16) && want <= MMA_GROUPS_W32.last().unwrap().0;
        let wide = match mma_warp_override() {
            Some(32) => fits_wide,
            Some(_) => false,
            // Below the saturation point eight warps a block leaves the machine
            // idle; above it the wider kernel's four-times-larger shared tile is
            // a loss. `MMA_WARP_TARGET` is where the sweep puts the boundary.
            None => fits_wide && blocks * 8 < MMA_WARP_TARGET * MMA_SMS,
        };
        let table: &[(usize, &str)] = if wide { &MMA_GROUPS_W32 } else { &MMA_GROUPS };
        let &(groups, name) = table
            .iter()
            .find(|(g, _)| want <= *g)
            .unwrap_or(table.last().unwrap());
        let warps = if wide { 32u32 } else { 8u32 };
        let f = self.dev.kernels().get("infero_fp8", fp8_src(), name)?;
        // One block per 16-row tile. The 16 has to track the MMA's M and the
        // kernel's shared tile together; it is not a tuning knob.
        let k_tile = warps as usize * 16;
        // The tile and the cross-warp partials share one allocation — see the
        // kernel — so the request is whichever life needs more.
        let shared = (2 * MMA_ROWS * (k_tile + 8) * 2).max(warps as usize * groups * 128 * 4);
        let cfg = LaunchConfig {
            grid_dim: (
                blocks as u32,
                n_tokens.div_ceil(groups * MMA_TOKENS) as u32,
                1,
            ),
            block_dim: (warps * 32, 1, 1),
            shared_mem_bytes: shared as u32,
        };
        let (ki, ni) = (k as i32, n as i32);
        let scols = k.div_ceil(FP8_BLOCK) as i32;
        let toks = n_tokens as i32;
        let acc = i32::from(accum);
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(out)
            .arg(w)
            .arg(x)
            .arg(&ki)
            .arg(&ni)
            .arg(&scols)
            .arg(&toks)
            .arg(&acc);
        self.dev
            .profile()
            .time("mma_f8_block", self.dev.stream(), || {
                unsafe { b.launch(cfg) }.context("mma_f8_block")?;
                Ok(())
            })?;
        Ok(true)
    }

    /// [`Self::mma_f8_block`], with both operands in e4m3 instead of widening
    /// the weight to f16 — `mma_e4m3` (`mma.cuh`) against `mma_f16`. `xq`/`xs`
    /// are `quantize_act_e4m3`'s output, not `mma_f8_block`'s f32 `x`.
    ///
    /// Only the `WARPS = 8` tier: this model's actual shapes (`d_model`,
    /// `d_ff`, `value_dim`, `d_attn`'s widths) all clear the eligibility check
    /// below at that tier already, so the wide variant `mma_f8_block` adds for
    /// narrow matrices at low occupancy is not yet built for this path — the
    /// same graceful decline that stands in for it (falling back to
    /// `mma_f8_block`) is safe, only unoptimized, for whatever shape does
    /// need it.
    #[allow(clippy::too_many_arguments)]
    pub fn mma_e4m3_block(
        &self,
        out: &mut ViewMut<'_, f32>,
        w: &View<'_, u8>,
        xq: &View<'_, u8>,
        xs: &View<'_, f32>,
        k: usize,
        n: usize,
        n_tokens: usize,
        accum: bool,
    ) -> Result<bool> {
        const WARPS: usize = 8;
        const K_TILE: usize = WARPS * 32;
        if !self.dev.caps().fp8
            || !k.is_multiple_of(K_TILE)
            || n_tokens > 4 * MMA_E4M3_GROUPS.last().unwrap().0 * MMA_TOKENS
        {
            return Ok(false);
        }
        let scale_cols = k.div_ceil(FP8_BLOCK);
        debug_assert!(out.len() >= n_tokens * n);
        debug_assert!(xq.len() >= n_tokens * k);
        debug_assert!(xs.len() >= n_tokens * scale_cols);
        debug_assert!(w.len() >= fp8_bytes(k, n));

        let blocks = n.div_ceil(MMA_ROWS);
        let want = n_tokens.div_ceil(MMA_TOKENS);
        let &(groups, name) = MMA_E4M3_GROUPS
            .iter()
            .find(|(g, _)| want <= *g)
            .unwrap_or(MMA_E4M3_GROUPS.last().unwrap());
        let f = self.dev.kernels().get("infero_fp8", fp8_src(), name)?;
        // Byte tile, not halves — no unpack means no reason to double it —
        // and `+16` bytes of pad plays `mma_f8_block`'s `+8` halves' role:
        // both add one `int4`'s worth of stride so a 16-row gather's banks
        // permute instead of lining up.
        let bstride = K_TILE + 16;
        let shared = (2 * MMA_ROWS * bstride).max(WARPS * groups * 128 * 4);
        let cfg = LaunchConfig {
            grid_dim: (
                blocks as u32,
                n_tokens.div_ceil(groups * MMA_TOKENS) as u32,
                1,
            ),
            block_dim: (WARPS as u32 * 32, 1, 1),
            shared_mem_bytes: shared as u32,
        };
        let (ki, ni) = (k as i32, n as i32);
        let scols = scale_cols as i32;
        let toks = n_tokens as i32;
        let acc = i32::from(accum);
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(out)
            .arg(w)
            .arg(xq)
            .arg(xs)
            .arg(&ki)
            .arg(&ni)
            .arg(&scols)
            .arg(&toks)
            .arg(&acc);
        self.dev
            .profile()
            .time("mma_e4m3_block", self.dev.stream(), || {
                unsafe { b.launch(cfg) }.context("mma_e4m3_block")?;
                Ok(())
            })?;
        Ok(true)
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
        out: &mut ViewMut<'_, f32>,
        w: &View<'_, u8>,
        x: &View<'_, f32>,
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
        let f = self.dev.kernels().get("infero_fp8", fp8_src(), name)?;
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
        out: &mut ViewMut<'_, f16>,
        w: &View<'_, u8>,
        k: usize,
        n: usize,
    ) -> Result<()> {
        debug_assert!(out.len() >= k * n);
        debug_assert!(w.len() >= fp8_bytes(k, n));
        let f = self
            .dev
            .kernels()
            .get("infero_fp8", fp8_src(), "dequant_f8_block_f16")?;
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
