// A dedicated mat-vec-batched kernel for GDN's own real `in_proj_a`/
// `in_proj_b` shape: a narrow-N (production: N=48, K=5120) weight, tiny and
// reused identically across every row of a *large* M (prefill-scale, tens of
// thousands of activation rows a chunk) -- the opposite regime from this
// file's siblings (`gemv_f16`/`gemv_f16_ksplit` in `quant.cu`,
// `mmv_f8_plain_*` in `fp8.cu`), which all grid over N (one block an output
// row/group) because *their* real usage is decode-scale (`x` tiny, re-reading
// it once a output row costs nothing). At prefill scale `x` is the operand
// that's expensive to re-read (a chunk's activation is tens of MB; the
// weight here is well under 1MiB), so this kernel inverts that: grid over
// M-tiles, keep the *weight* resident (tiled into shared memory once a
// block, reused across every row the block owns), and read each `x` element
// out of global memory exactly once.
//
// Today's real, unfused prefill path for this exact weight is a separate
// `to_f16` pass (casts the f32 activation down to f16) feeding a generic
// `gemm_f16` (cuBLAS-style, tuned for roughly-square tiles -- a poor fit for
// N=48). This kernel reads `x` directly as f32 and skips the `to_f16` pass
// entirely, which is a deliberate, real numerical difference from that
// unfused composition, not a bug: `to_f16` rounds every activation element to
// half precision *before* the dot product, this kernel does not. See the
// isolated test (`tests/narrow_gemm.rs`) for the measured size of that gap
// and why the tolerance it checks against is wider than this file's
// bit-exact-composition kernels use.
//
// ---- real, measured false start, kept here rather than silently erased ----
// A first version of this kernel mapped one THREAD to one output ROW, each
// thread looping the whole N=48-wide column reduction in a fully-unrolled,
// compile-time-sized `acc[48]` register array (the `quant.cu`
// `GEMV_KSPLIT_TOKENS`-style fixed-array convention, to sidestep the
// dynamic-index register-spill trap -- see project memory,
// `feedback_dynamic_index_forces_register_spill.md`). It was correct (passed
// every isolated test bit-for-bit against the real `to_f16`+`gemm_f16`
// composition within the cited tolerance) and genuinely 4-100x SLOWER than
// that composition at every real shape tried, decode and prefill alike.
// `ptxas -v` ruled out the obvious suspect (0 spill, 64 regs/thread, no
// worse than this file's siblings) -- the real cause was memory coalescing,
// not registers: with "thread = row", 32 adjacent threads in a warp own 32
// *different* rows, so at a fixed loop step every thread reads a different
// `x` cache line (`row * k` apart, tens of KB) instead of one shared line --
// a ~32-way serialization of what should have been one coalesced transaction
// per warp-instruction. Caught by actually running the isolated
// microbenchmark before claiming a win, per this project's own established
// practice (`feedback_verify_surprising_benchmarks.md`) -- a kernel that is
// *correct* and *much slower* than the baseline it was meant to replace is
// evidence of a bug, not a data point to report as-is.
//
// ---- this version: "thread = column", not "thread = row" ----
// `THREADS_PER_ROW` threads (one output column each, rounded up to a whole
// number of warps so no warp ever straddles two different rows) cooperate
// on one row at a time; `ROWS_PER_BLOCK` such row-groups share one block (and
// therefore one resident weight tile). For a fixed `(row, k)` step every
// thread assigned to that row reads the *same* `x[row*k+k0+kk]` address --
// a broadcast, not a scatter, and the cheapest read pattern the hardware
// has -- while each thread's own `w_tile[kk*n+col]` read is a plain
// contiguous shared-memory access across threads (`col` is the fast index
// `w_tile` is laid out by). Net effect: `x` is still read exactly once
// (the real goal), but now coalesced/broadcast instead of scattered, and
// each thread's accumulator is a single scalar register -- there is no
// per-thread array at all in this version, so the dynamic-index/spill
// concern that shaped the first version does not apply here either way.
// `ROWS_PER_BLOCK` is how many rows share one resident weight-tile load --
// the whole point of staging the weight into shared memory at all. Pushed
// to the max a 1024-thread block allows at `THREADS_PER_ROW=64`
// (`1024/64=16`): a first version at `ROWS_PER_BLOCK=4` genuinely measured
// SLOWER than the "thread = row" false start it replaced (10.8ms vs 1.7ms
// at the real M=8192 shape) -- amortizing a 491KB weight load over only 4
// rows' worth of compute is a bad trade, confirmed by rerunning the real
// isolated benchmark rather than assuming the coalescing fix alone was
// enough.
// One full warp a row (`THREADS_PER_ROW=32`, never straddling two rows),
// each lane owning up to `COLS_PER_THREAD=2` output columns (`n<=64`) so a
// 1024-thread block fits `ROWS_PER_BLOCK=32` rows -- double `ROWS_PER_BLOCK`
// over a one-column-a-thread/`THREADS_PER_ROW=64` layout, which only reached
// 16 rows/block at the same 1024-thread cap. Doubling the rows sharing one
// resident weight-tile load halves how many times the real 491KB weight is
// re-read from global/L2 across a whole prefill chunk's grid -- measured,
// not assumed, to matter: see the isolated benchmark's own progression
// (10.8ms -> 7.1ms -> 6.1ms as reuse grew at the real M=8192 shape).
//
// ---- honest result at this point: still a real loss, not a win ----
// `examples/narrow_gemm_bench.rs`, real production shape (K=5120, N=48),
// on this development box's RTX A4000 (NOT the real bw/GPU3 Blackwell
// production card -- the absolute numbers do not transfer, see that
// example's own module doc for the real-hardware caveat):
//
//   M=8192 (a real prefill chunk): this kernel 4.40ms, real to_f16+gemm_f16
//     composition 0.88ms -- this kernel is ~5x SLOWER.
//   M=2719 (the real prefill tail chunk): 1.34ms vs 0.32ms, ~4x slower.
//   M=4 (real speculative-verify decode): 0.20ms vs 0.018ms, ~11x slower.
//   M=1 (plain decode): 0.18ms vs 0.0064ms, ~29x slower.
//
// Two real, diagnosed, fixed bugs got here from a 4-100x loss
// (`ROWS_PER_BLOCK=4`, the strided weight-load decomposition): both were
// genuine memory-coalescing mistakes, not reasoning errors about the
// overall approach, and a direct ablation (`NARROW_GEMM_NO_SMEM_DEBUG`,
// not shipped -- reading `w` straight from global every step with no
// shared-memory staging at all) measured 8x WORSE still (36ms at M=8192),
// which rules out "shared-memory tiling was a mistake, L2 does this for
// free" as the explanation. What's left unexplained: even at
// `ROWS_PER_BLOCK=32` the real weight-reload traffic across a whole
// chunk's grid (`(M/32) * 491KB` -- 120MB at M=8192) is real and almost
// certainly the dominant remaining cost, and this kernel's design has no
// further lever against it without a materially different structure: a
// true persistent-kernel redesign (grid sized to the SM count, not to
// `M/ROWS_PER_BLOCK`, with per-row accumulator state carried in shared
// memory across many row-tiles so each weight K-tile is read from global
// exactly once per *block*, not once per row-tile) was NOT attempted here
// -- a substantially larger, higher-risk rewrite than this pass's
// remaining scope covers honestly. See `tests/narrow_gemm.rs` and
// `examples/narrow_gemm_bench.rs` for the real, final isolated numbers and
// go/no-go.
#define THREADS_PER_ROW 32
#define COLS_PER_THREAD 2
#define ROWS_PER_BLOCK 32
#define NARROW_GEMM_BLOCK (THREADS_PER_ROW * ROWS_PER_BLOCK)

// K-tile width the weight is staged into shared memory at. 128 divides
// GDN's real K=5120 exactly (40 tiles, no partial-tile remainder on the real
// shape), but the remainder path below is real and tested anyway (an N or K
// that doesn't divide evenly is a correctness case the isolated test
// exercises, not just the production shape).
#define NARROW_GEMM_KTILE 128

// Hard compile-time cap on `n` -- each lane owns up to `COLS_PER_THREAD`
// columns, so this is `THREADS_PER_ROW * COLS_PER_THREAD`, not just
// `THREADS_PER_ROW` as a one-column-a-thread layout would cap it at. A
// caller asking for a larger `n` is declined by the host wrapper, not
// silently truncated or wrapped.
#define NARROW_GEMM_MAX_N (THREADS_PER_ROW * COLS_PER_THREAD)

// out:   [m, n]   f32, row-major (token-major -- matches `gemv_f16`'s own
//                 output layout: `out[token * n + row]`)
// w:     [n, k]   f16, row-major (GDN's real resident layout for this
//                 weight -- see `WeightType::F16`'s direct-transmute read in
//                 `model/src/lib.rs`, no per-call dequant)
// x:     [m, k]   f32, row-major (the real pre-`to_f16` activation -- this
//                 kernel is the thing that would replace both `to_f16` and
//                 `gemm_f16` for this weight, so it reads the activation
//                 `gemm_f16` never gets to see)
extern "C" __global__ void narrow_gemm_f16_f32(float* __restrict__ out,
                                                const void* __restrict__ w,
                                                const float* __restrict__ x,
                                                int k, int n, int m) {
    // [NARROW_GEMM_KTILE, n] half, kk-major -- loaded flat across the whole
    // block (every row-group in this block shares the one tile), `col` is
    // the contiguous fast axis so `w_tile[kk * n + col]` is a plain
    // coalesced read across the threads that use it.
    extern __shared__ __half w_tile[];

    const int row_local = threadIdx.x / THREADS_PER_ROW;
    const int lane = threadIdx.x % THREADS_PER_ROW;
    const int row = blockIdx.x * ROWS_PER_BLOCK + row_local;
    const bool row_valid = row < m;
    // Lane `lane` owns columns `lane, lane+THREADS_PER_ROW, ...` up to
    // `COLS_PER_THREAD` of them -- a small, fully compile-time-unrolled
    // fixed-size array (`COLS_PER_THREAD=2`), not one sized or indexed by a
    // runtime bound, so this does not reintroduce the dynamic-index
    // register-spill trap the very first version's `acc[48]` was built to
    // avoid.
    int col[COLS_PER_THREAD];
    bool col_valid[COLS_PER_THREAD];
    #pragma unroll
    for (int j = 0; j < COLS_PER_THREAD; ++j) {
        col[j] = lane + j * THREADS_PER_ROW;
        col_valid[j] = col[j] < n;
    }

    const __half* wbase = (const __half*)w;
    float acc[COLS_PER_THREAD];
    #pragma unroll
    for (int j = 0; j < COLS_PER_THREAD; ++j) acc[j] = 0.0f;

    for (int k0 = 0; k0 < k; k0 += NARROW_GEMM_KTILE) {
        const int ktile = min(NARROW_GEMM_KTILE, k - k0);

        // `w` is row-major `[n, k]` (ggml's own convention -- a row is
        // contiguous in `k`), so the coalesced way to read it is "`kk` is
        // the fast index, `c` the slow one", the opposite of `w_tile`'s own
        // layout (`c` fast, so the *compute* loop's `w_tile[kk*n+col]` reads
        // are contiguous across threads instead). A first version of this
        // loop used `kk = idx / n; c = idx % n` -- matching `w_tile`'s
        // layout directly, simplest to write -- and it was a real,
        // measured regression: that decomposition makes `c` the fast index,
        // so consecutive threads read `wbase[c*k+...]` for consecutive `c`,
        // which is a `k`-element (10KB) stride between them -- the load
        // phase paid almost exactly the same scattered-read penalty the
        // "thread = row" false start paid for `x`, just for `w` instead.
        // The shared-memory *write* this version does instead
        // (`w_tile[kk*n+c]`, stride `n` between consecutive `kk`) has some
        // bank conflict, but shared memory bandwidth is high enough that
        // this trade is a net real win -- confirmed by rerunning the
        // isolated benchmark, not assumed.
        for (int idx = threadIdx.x; idx < NARROW_GEMM_KTILE * n; idx += blockDim.x) {
            const int c = idx / NARROW_GEMM_KTILE;
            const int kk = idx % NARROW_GEMM_KTILE;
            w_tile[(size_t)kk * n + c] = (kk < ktile) ? wbase[(size_t)c * k + (k0 + kk)] : __float2half(0.0f);
        }
        __syncthreads();

        if (row_valid) {
            const float* xr = x + (size_t)row * k + k0;
            for (int kk = 0; kk < ktile; ++kk) {
                // Every thread with this `row_local` reads the identical
                // address here -- a broadcast, not a per-thread stream --
                // which is what makes "thread = column" coalesce where the
                // false start's "thread = row" did not.
                const float xv = xr[kk];
                const __half* wt = w_tile + (size_t)kk * n;
                #pragma unroll
                for (int j = 0; j < COLS_PER_THREAD; ++j) {
                    if (col_valid[j]) acc[j] += xv * __half2float(wt[col[j]]);
                }
            }
        }
        // Next iteration overwrites `w_tile` -- every thread must be done
        // reading this tile first.
        __syncthreads();
    }

    if (row_valid) {
        #pragma unroll
        for (int j = 0; j < COLS_PER_THREAD; ++j) {
            if (col_valid[j]) out[(size_t)row * n + col[j]] = acc[j];
        }
    }
}
