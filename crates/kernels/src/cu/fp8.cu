// FP8 E4M3 weights with a 128x128 block scale grid, read four rows at a time.
//
// The 27B ships FP8 and tuili was dequantizing it to f16 at load, which is
// correct and costs a factor of two in the only resource decode has: a step
// reads every weight once, so 51 GiB of f16 takes twice as long as 29 GiB of
// FP8 no matter how good the arithmetic is. Measured, that was 13.2 tok/s
// against vLLM's 34 on the same checkpoint.
//
// Layout, one buffer so a `Matrix` stays a single allocation: the quants,
// permuted by `fp8::repack_rows`, then the scale grid as f32 —
// `ceil(n/128) * ceil(k/128)` of them, row-major. The grid's row index is the
// *output* row over 128 and its column index is the position along k over 128; a
// scale is shared by 128 rows and 128 columns at once, which is why a row's
// bytes are not self-describing the way a ggml block's are.
//
// The permutation interleaves every four rows, so the sixteen bytes at
//
//     group * 4 * k + chunk * 16
//
// are rows `4g..4g+4` at positions `4c..4c+4`, and one `uint4` load feeds
// sixteen products. Why: above one token this mat-vec was limited by memory
// *requests* rather than by bytes or by arithmetic. Row-major, a thread read one
// 4-byte weight word and then one activation load a token — three requests a
// group at two tokens — and L1 serves a bounded number of requests a cycle
// whatever their width. Four rows at once makes it a quarter of that and takes
// the FMAs per request from 8 to 32.
//
// Two attempts to get the same reuse *without* repacking are recorded on
// `fp8::BATCH_KERNELS`, and both were slower than no reuse at all: handling R
// rows a block in row-major order turns the weight stream into R runs 5120 bytes
// apart, and the weight stream is the part that is genuinely DRAM-bound.
//
// Every kernel here reads whole groups. That is not a preference: a single row's
// bytes are now strided by 16, so a one-row kernel would read them worse than
// the group does.

// E4M3: sign in bit 7, four exponent bits biased by 7, three mantissa bits. No
// infinities — 0x7F and 0xFF are the only NaNs, and 0x7E is the largest finite
// value at 448.
//
// Done with bit arithmetic rather than a lookup table: four integer ops beat a
// shared-memory load per weight, and this kernel is reading a byte per multiply
// so there is no room for a table access in the inner loop.
__device__ __forceinline__ float e4m3_to_f32(unsigned int b) {
    const unsigned int sign = (b & 0x80u) << 24;
    const int exp = (int)((b >> 3) & 0x0Fu);
    const unsigned int man = b & 0x07u;
    if (exp == 0) {
        // Subnormal: (man / 8) * 2^-6, which is man * 2^-9. Zero when man is 0,
        // and the sign still has to be carried for -0.
        const float v = (float)man * (1.0f / 512.0f);
        return sign ? -v : v;
    }
    if (exp == 0x0F && man == 0x07) {
        return __int_as_float(0x7fc00000);  // the only NaN pattern
    }
    // Normal: rebias 7 to 127 and shift the mantissa into f32's field.
    return __int_as_float(sign | ((unsigned int)(exp + 120) << 23) | (man << 20));
}

// Rows interleaved into one group. Must match `fp8::ROW_GROUP`, and must divide
// 128, or a group would straddle a scale-grid row and one `srow` pointer would
// stop being enough.
#define FP8_ROW_GROUP 4

// Attribution switches for `examples/fp8_row_cost.rs`, compiled in by
// `fp8::strip_flags()` when `TUILI_FP8_STRIP` asks. A marginal row costs 2.25 ms
// where its DRAM bytes are zero, and three end-to-end guesses at why were all
// wrong — so the remaining move is to take pieces out and see which one the cost
// follows. These produce wrong answers by construction and are never on in a
// serving build.
#ifndef FP8_STRIP_FMA
#define FP8_STRIP_FMA 0
#endif
#ifndef FP8_STRIP_REDUCE
#define FP8_STRIP_REDUCE 0
#endif

// One packed word into four scaled floats.
__device__ __forceinline__ void fp8_unpack4(unsigned int packed, float s,
                                            float out[4]) {
    out[0] = s * e4m3_to_f32(packed & 0xFFu);
    out[1] = s * e4m3_to_f32((packed >> 8) & 0xFFu);
    out[2] = s * e4m3_to_f32((packed >> 16) & 0xFFu);
    out[3] = s * e4m3_to_f32((packed >> 24) & 0xFFu);
}

// out[t, row] = sum over i of w[row, i] * x[t, i], four rows to a block.
//
// The scale is folded into each product rather than applied to a per-slice sum,
// which is what lets the loads pipeline. Distributing it is exact:
//
//     sum_kb s_kb * sum_{i in kb} w_i x_i  ==  sum_kb sum_{i in kb} s_kb w_i x_i
//
// An earlier version did the inner sum with a shuffle reduction and then scaled,
// which put a five-step cross-lane reduction between one load and the next — so
// a warp had at most one outstanding memory request and none of the latency was
// hidden. This way every lane keeps its own running totals, all the loads are
// independent, and there is exactly one reduction at the end.
//
// `x` is `[n_tokens, k]` and `out` is `[n_tokens, n]`, both row-major, matching
// what the rest of the engine passes around.
template <int TOKENS>
__device__ __forceinline__ void mmv_f8_group_body(
        float* __restrict__ out, const unsigned char* __restrict__ w,
        const float* __restrict__ x, int k, int n, int scale_cols, int n_tokens,
        int accum) {
    const int group = blockIdx.x;
    const int row0 = group * FP8_ROW_GROUP;
    if (row0 >= n) return;
    const int rows = (n - row0 < FP8_ROW_GROUP) ? (n - row0) : FP8_ROW_GROUP;

    // The scale grid sits past every quant byte, padding included, because
    // `fp8_bytes` rounds the row count up to a whole group.
    const int padded = ((n + FP8_ROW_GROUP - 1) / FP8_ROW_GROUP) * FP8_ROW_GROUP;
    const float* scales = (const float*)(w + (size_t)padded * k);
    const float* srow = scales + (size_t)(row0 / 128) * scale_cols;
    const unsigned char* wg = w + (size_t)group * FP8_ROW_GROUP * k;

    float acc[FP8_ROW_GROUP][TOKENS];
#pragma unroll
    for (int r = 0; r < FP8_ROW_GROUP; ++r) {
#pragma unroll
        for (int t = 0; t < TOKENS; ++t) acc[r][t] = 0.0f;
    }

    // Whether the activation can be read 16 bytes at a time. `x` is a *view*
    // here — the caller may hand over a slice of a larger buffer — so the base
    // pointer is checked rather than assumed. Row `t` starts at `x + t * k` and
    // `k` is a multiple of four (the launcher checks), so 16-byte alignment of
    // the base carries to every row. Same guard as `f32_to_f16_vec` in `ops.cu`,
    // and for the same reason: a misaligned `float4` is a fault, not a slow load.
    const bool xvec = ((size_t)(const void*)x % 16 == 0);

    const int chunks = k / 4;
    for (int c = threadIdx.x; c < chunks; c += blockDim.x) {
        const int i0 = c * 4;
        // `i0 >> 7` is the scale block along k. A four-element group never
        // straddles two, because 128 is a multiple of four.
        const float s = srow[i0 >> 7];

        // `FP8_ROW_GROUP * 4` bytes: every row of the group at four positions,
        // in `uint4`s. One request per four rows.
        float wv[FP8_ROW_GROUP][4];
#pragma unroll
        for (int q = 0; q < FP8_ROW_GROUP / 4; ++q) {
            const uint4 wq = *(const uint4*)(const void*)(
                wg + (size_t)c * (FP8_ROW_GROUP * 4) + (size_t)q * 16);
            fp8_unpack4(wq.x, s, wv[q * 4 + 0]);
            fp8_unpack4(wq.y, s, wv[q * 4 + 1]);
            fp8_unpack4(wq.z, s, wv[q * 4 + 2]);
            fp8_unpack4(wq.w, s, wv[q * 4 + 3]);
        }

        // The activation group, once, for all four rows to reuse.
#pragma unroll
        for (int t = 0; t < TOKENS; ++t) {
            if (t >= n_tokens) break;
            const float* xt = x + (size_t)t * k;
            float xv[4];
            if (xvec) {
                const float4 v = *(const float4*)(const void*)(xt + i0);
                xv[0] = v.x;
                xv[1] = v.y;
                xv[2] = v.z;
                xv[3] = v.w;
            } else {
#pragma unroll
                for (int j = 0; j < 4; ++j) xv[j] = xt[i0 + j];
            }
#if FP8_STRIP_FMA
            // A quarter of the arithmetic — one add per (row, token) instead of
            // four FMAs — while every accumulator stays live. Touching only
            // `acc[0][0]` would let ptxas delete fifteen of the sixteen chains
            // and drop the register count with them, which is a different
            // experiment than the one this is for.
#pragma unroll
            for (int r = 0; r < FP8_ROW_GROUP; ++r) acc[r][t] += wv[r][0] + xv[0];
#else
#pragma unroll
            for (int r = 0; r < FP8_ROW_GROUP; ++r) {
                acc[r][t] += wv[r][0] * xv[0] + wv[r][1] * xv[1]
                           + wv[r][2] * xv[2] + wv[r][3] * xv[3];
            }
#endif
        }
    }

    // One reduction per row per token. Warp-level first, then across warps
    // through shared memory — `block_reduce_sum` cannot be called in a loop,
    // since its static shared result would be overwritten while slower threads
    // still read it.
#if FP8_STRIP_REDUCE
    // Every accumulator consumed by plain adds, so all `rows * n_tokens` chains
    // stay live and the register count is unchanged — then one write from one
    // thread, with no shuffle, no shared memory and no barrier. Reducing only
    // `acc[0][0]` instead would have let ptxas delete the other chains, which is
    // how the first version of this switch reported a flat row curve: it was not
    // measuring a cheaper reduction, it was measuring less arithmetic.
    {
        float v = 0.0f;
#pragma unroll
        for (int r = 0; r < FP8_ROW_GROUP; ++r) {
#pragma unroll
            for (int t = 0; t < TOKENS; ++t) v += acc[r][t];
        }
        if (threadIdx.x == 0) out[row0] = v;
    }
    return;
#endif
    __shared__ float partial[32][FP8_ROW_GROUP][TOKENS];
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
    const int warps = blockDim.x / WARP_SIZE;
#pragma unroll
    for (int r = 0; r < FP8_ROW_GROUP; ++r) {
        if (r >= rows) break;
#pragma unroll
        for (int t = 0; t < TOKENS; ++t) {
            if (t >= n_tokens) break;
            float v = acc[r][t];
            for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
                v += __shfl_down_sync(FULL_MASK, v, off);
            }
            if (lane == 0) partial[warp][r][t] = v;
        }
    }
    __syncthreads();
    // One thread per (row, token) pair finishes the cross-warp sum.
    for (int i = threadIdx.x; i < rows * n_tokens; i += blockDim.x) {
        const int r = i / n_tokens;
        const int t = i % n_tokens;
        float sum = 0.0f;
        for (int wi = 0; wi < warps; ++wi) sum += partial[wi][r][t];
        float* o = out + (size_t)t * n + row0 + r;
        *o = accum ? *o + sum : sum;
    }
}

// The one-token case, which is every plain decode step. A dedicated
// instantiation rather than a call with `n_tokens = 1`, so the accumulator array
// is four floats and the token loop unrolls to nothing.
extern "C" __global__ void mmv_f8_block_f32(float* __restrict__ out,
                                            const unsigned char* __restrict__ w,
                                            const float* __restrict__ x,
                                            int k, int n, int scale_cols,
                                            int accum) {
    mmv_f8_group_body<1>(out, w, x, k, n, scale_cols, 1, accum);
}

extern "C" __global__ void mmv_f8_block_batch2_f32(
        float* __restrict__ out, const unsigned char* __restrict__ w,
        const float* __restrict__ x, int k, int n, int scale_cols, int n_tokens,
        int accum) {
    mmv_f8_group_body<2>(out, w, x, k, n, scale_cols, n_tokens, accum);
}

extern "C" __global__ void mmv_f8_block_batch4_f32(
        float* __restrict__ out, const unsigned char* __restrict__ w,
        const float* __restrict__ x, int k, int n, int scale_cols, int n_tokens,
        int accum) {
    mmv_f8_group_body<4>(out, w, x, k, n, scale_cols, n_tokens, accum);
}

extern "C" __global__ void mmv_f8_block_batch3_f32(
        float* __restrict__ out, const unsigned char* __restrict__ w,
        const float* __restrict__ x, int k, int n, int scale_cols, int n_tokens,
        int accum) {
    mmv_f8_group_body<3>(out, w, x, k, n, scale_cols, n_tokens, accum);
}

extern "C" __global__ void mmv_f8_block_batch5_f32(
        float* __restrict__ out, const unsigned char* __restrict__ w,
        const float* __restrict__ x, int k, int n, int scale_cols, int n_tokens,
        int accum) {
    mmv_f8_group_body<5>(out, w, x, k, n, scale_cols, n_tokens, accum);
}

extern "C" __global__ void mmv_f8_block_batch6_f32(
        float* __restrict__ out, const unsigned char* __restrict__ w,
        const float* __restrict__ x, int k, int n, int scale_cols, int n_tokens,
        int accum) {
    mmv_f8_group_body<6>(out, w, x, k, n, scale_cols, n_tokens, accum);
}

extern "C" __global__ void mmv_f8_block_batch7_f32(
        float* __restrict__ out, const unsigned char* __restrict__ w,
        const float* __restrict__ x, int k, int n, int scale_cols, int n_tokens,
        int accum) {
    mmv_f8_group_body<7>(out, w, x, k, n, scale_cols, n_tokens, accum);
}

extern "C" __global__ void mmv_f8_block_batch8_f32(
        float* __restrict__ out, const unsigned char* __restrict__ w,
        const float* __restrict__ x, int k, int n, int scale_cols, int n_tokens,
        int accum) {
    mmv_f8_group_body<8>(out, w, x, k, n, scale_cols, n_tokens, accum);
}

extern "C" __global__ void mmv_f8_block_batch16_f32(
        float* __restrict__ out, const unsigned char* __restrict__ w,
        const float* __restrict__ x, int k, int n, int scale_cols, int n_tokens,
        int accum) {
    mmv_f8_group_body<16>(out, w, x, k, n, scale_cols, n_tokens, accum);
}

// The same weights, dequantized to f16 in row-major order for the batched path.
//
// Prefill still goes through the f16 GEMM, so the bytes have to be expandable on
// the device rather than at load. This is the operation that used to happen on
// the host and cost 22 GiB of resident memory.
//
// One block per group per 128-wide slice of k, so a scale is read once a block
// and the read side stays in whole `uint4`s. The *write* side is four strided
// rows, which is the price of the layout and is paid on a path that is not
// decode.
extern "C" __global__ void dequant_f8_block_f16(__half* __restrict__ out,
                                                const unsigned char* __restrict__ w,
                                                int k, int n, int scale_cols) {
    const int kb = blockIdx.x;
    const int group = blockIdx.y;
    const int row0 = group * FP8_ROW_GROUP;
    if (row0 >= n) return;
    const int rows = (n - row0 < FP8_ROW_GROUP) ? (n - row0) : FP8_ROW_GROUP;

    const int padded = ((n + FP8_ROW_GROUP - 1) / FP8_ROW_GROUP) * FP8_ROW_GROUP;
    const float* scales = (const float*)(w + (size_t)padded * k);
    const float s = scales[(size_t)(row0 / 128) * scale_cols + kb];
    const unsigned char* wg = w + (size_t)group * FP8_ROW_GROUP * k;

    // 128 values of k is 32 chunks of four.
    const int c0 = kb * 32;
    const int chunks = k / 4;
    for (int c = c0 + threadIdx.x; c < c0 + 32 && c < chunks; c += blockDim.x) {
        unsigned int words[FP8_ROW_GROUP];
#pragma unroll
        for (int q = 0; q < FP8_ROW_GROUP / 4; ++q) {
            const uint4 wq = *(const uint4*)(const void*)(
                wg + (size_t)c * (FP8_ROW_GROUP * 4) + (size_t)q * 16);
            words[q * 4 + 0] = wq.x;
            words[q * 4 + 1] = wq.y;
            words[q * 4 + 2] = wq.z;
            words[q * 4 + 3] = wq.w;
        }
#pragma unroll
        for (int r = 0; r < FP8_ROW_GROUP; ++r) {
            if (r >= rows) break;
            __half* orow = out + (size_t)(row0 + r) * k + (size_t)c * 4;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                orow[j] = __float2half(
                    e4m3_to_f32((words[r] >> (8 * j)) & 0xFFu) * s);
            }
        }
    }
}

// The row interleave, on the device.
//
// `fp8::repack_rows` does the same permutation on the host, and doing it there
// costs 28 seconds of a 63-second load on the 27B: 7.4e9 four-byte moves, one
// core, no help from the memory system because the writes stride by sixteen. The
// device does the same permutation at DRAM speed.
//
// `src` is `[n, k]` row-major quants, `dst` is the interleaved form and must be
// `padded * k` bytes, where `padded` rounds `n` up to a whole group. One thread
// per four-byte chunk of the source.
extern "C" __global__ void fp8_repack_rows(unsigned char* __restrict__ dst,
                                          const unsigned char* __restrict__ src,
                                          int k, int n) {
    const int chunks = k / 4;
    const long long total = (long long)n * chunks;
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int row = (int)(i / chunks);
    const int c = (int)(i % chunks);
    const int g = row / FP8_ROW_GROUP;
    const int r = row % FP8_ROW_GROUP;
    // One aligned four-byte read, one aligned four-byte write.
    const unsigned int w = *(const unsigned int*)(const void*)(src + (size_t)row * k + (size_t)c * 4);
    *(unsigned int*)(void*)(dst + (size_t)g * FP8_ROW_GROUP * k
                            + (size_t)c * (FP8_ROW_GROUP * 4) + (size_t)r * 4) = w;
}

// ---- tensor cores ------------------------------------------------------------
//
// The batched mat-vec's marginal row is 81% per-token multiply-accumulate
// (`examples/fp8_row_cost.rs`, and see `fp8::BATCH_KERNELS` for how two earlier
// readings of that probe were wrong). Its inner loop issues sixteen scalar FMAs
// per chunk per token and lands at a seventh of the f32 FMA bound. One
// `mma.m16n8k16` does the same 2048 MACs in one instruction.
//
// Three parameters, each of which came from a measurement rather than a guess.
//
// **`WARPS`** — how many warps split a tile's k, and so how much of the machine
// a narrow matrix can fill. A block owns 16 output rows, so a matrix of `n` rows
// offers `n/16` blocks and `WARPS*n/16` warps, and the rate follows that number
// until it saturates near 48 warps an SM:
//
// ```text
//        n   blocks   warps/SM     GB/s      (WARPS = 8, one row)
//     5120      320       13.6      934
//     6144      384       16.3     1054
//    10240      640       27.2     1271
//    17408     1088       46.3     1413
// ```
//
// Most of the 27B's projections are 5120 or 6144 wide, which is why the kernel
// averaged 1281 GB/s across a decode step's 433 launches while this probe read
// 1413 on the widest one. Raising `WARPS` raises the warp count without changing
// the launch count or needing a cross-block reduction.
//
// **`GROUPS`** — fragment columns off one staged tile, so token counts up to
// `8 * GROUPS` cost one pass over the weights. Prefill is why: a 66-token prompt
// used to fall to the expansion path at five bytes a weight, 148 GB against 29.6
// for the 27B's forward.
//
// **The tile's row stride**, `K_TILE + 8` halves. Unpadded, the fragment gather
// (lane L reads row L/4 at half `(L%4)*2`) puts eight rows on one bank and
// conflicts eight ways. The `+8` shifts each row by four banks, so lane L lands
// on `((L/4)*((K_TILE+8)/2) + L%4) % 32`; both 128 and 512 give a stride of 4
// banks a row, hence a permutation of 0..31.
//
// Three things the existing layout gave for free: a 16-row tile never straddles
// a 128-row scale block, so one scale multiplies an f32 accumulator after each
// warp's MMAs and the operands are never touched; the repacked weights put a
// group's k-window in one contiguous run, so staging is `uint4`s; activations are
// `[token][k]` with k contiguous, which is already the `B` fragment's layout.
//
// The e4m3 unpack has to be `cvt.rn.f16x2.e4m3x2`. The arithmetic version ran
// 89M times a launch and cost 0.033 ms of the 0.096 the kernel first measured,
// which the hardware instruction takes to 0.063.

// Two e4m3 bytes to two halves in one instruction.
__device__ __forceinline__ unsigned e4m3x2_to_half2(unsigned short two) {
#if __CUDA_ARCH__ >= 890
    unsigned h;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h) : "h"(two));
    return h;
#else
    const __half2 v = __floats2half2_rn(e4m3_to_f32(two & 0xFFu),
                                        e4m3_to_f32((two >> 8) & 0xFFu));
    return *(const unsigned*)(const void*)&v;
#endif
}

template <int WARPS, int GROUPS>
__device__ __forceinline__ void mma_f8_body(
        float* __restrict__ out, const unsigned char* __restrict__ w,
        const float* __restrict__ x, int k, int n, int scale_cols,
        int n_tokens, int accum) {
#if __CUDA_ARCH__ >= 800
    constexpr int K_TILE = WARPS * 16;         // k staged per iteration
    constexpr int STRIDE = K_TILE + 8;         // tile row stride, in halves
    constexpr int CHUNKS = K_TILE / 4;         // 4-byte chunks a group a tile
    constexpr int SCALES = K_TILE / 128;       // scale blocks a tile

    const int row0 = blockIdx.x * 16;
    if (row0 >= n) return;
    const int tok0 = blockIdx.y * (GROUPS * 8);
    if (tok0 >= n_tokens) return;
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;

    // One allocation, two lives: the weight tile while the k loop runs, then the
    // cross-warp partials once it is done and a barrier has passed. Separate
    // arrays would want 49664 bytes at WARPS=32 and the static limit is 49152.
    extern __shared__ char fp8_smem[];
    __half* tile = (__half*)fp8_smem;                // 2 * 16 * STRIDE halves
    float* red = (float*)fp8_smem;                   // WARPS * GROUPS * 128

    const int padded = ((n + FP8_ROW_GROUP - 1) / FP8_ROW_GROUP) * FP8_ROW_GROUP;
    const float* scales = (const float*)(w + (size_t)padded * k);
    const float* srow = scales + (size_t)(row0 / 128) * scale_cols;

    // Fragment coordinates. `ar`/`bc`/`cr` are all `lane / 4`; naming them
    // separately keeps each use readable against the PTX tables.
    const int ar = mma_a_row(lane);
    const int bc = mma_b_col(lane);
    const int k0 = mma_k0_f16(lane);
    const int cr = mma_c_row(lane);
    const int cc = mma_c_col(lane);

    mma_c_f32 acc[GROUPS];
#pragma unroll
    for (int g = 0; g < GROUPS; ++g) acc[g] = mma_c_f32{{0.0f, 0.0f, 0.0f, 0.0f}};

    // Which slice of a staged tile this lane moves. A `uint4` is four rows of
    // a group at four k; `half` splits it into two `uint2`s of two rows each,
    // so a pair of threads moves what one thread used to. `ncu` found the old
    // single-thread-a-pair design idle exactly half the block during this
    // section, then paying for it below: every warp spent 17.8 of the 55
    // cycles between issued instructions stalled at the barrier waiting for
    // the working half to finish. Splitting the pair keeps the bytes moved
    // and their layout in `tile` identical to before -- `repack_rows` is
    // unchanged, and so is what ends up at `tl[(4*gl+half*2+r)*STRIDE+4*sc]`
    // for a given source byte -- it only changes which thread carries which
    // half of a `uint4`.
    constexpr int PAIRS = 4 * CHUNKS;
    const int half = threadIdx.x / PAIRS;
    const int idx = threadIdx.x % PAIRS;
    const int gl = idx / CHUNKS;                // group within the 16-row tile
    const int sc = idx % CHUNKS;                // chunk within the k window
    const unsigned char* ssrc = w
        + (size_t)(row0 / FP8_ROW_GROUP + gl) * FP8_ROW_GROUP * k
        + (size_t)sc * 16 + (size_t)half * 8;

    const int tiles = k / K_TILE;
    // Prologue, so the loop is always load-next-then-compute-this.
    uint2 q = *(const uint2*)(const void*)ssrc;

    for (int it = 0; it < tiles; ++it) {
        const int cur = it & 1;
        __half* tl = tile + (size_t)cur * 16 * STRIDE;
        // Unpack what was loaded last iteration. The scale is not applied here:
        // e4m3 values are exact in f16, and folding a scale in could overflow.
        {
            const unsigned int word[2] = {q.x, q.y};
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                const uint2 h = make_uint2(
                    e4m3x2_to_half2((unsigned short)(word[r] & 0xFFFFu)),
                    e4m3x2_to_half2((unsigned short)(word[r] >> 16)));
                *(uint2*)(void*)&tl[(4 * gl + half * 2 + r) * STRIDE + 4 * sc] = h;
            }
        }
        __syncthreads();

        // Issue the next tile's loads *before* this tile's MMAs, so their
        // latency is spent under arithmetic rather than under a barrier. They
        // land in registers and are not written to shared memory until the top
        // of the next iteration, which is what makes one barrier enough.
        if (it + 1 < tiles) {
            q = *(const uint2*)(const void*)(ssrc + (size_t)(it + 1) * K_TILE * 4);
        }

        // Warp `w` takes k [it*K_TILE + w*16, +16), whose scale block is
        // `w / 8` of the `SCALES` this tile spans.
        const int ko = it * K_TILE + warp * 16;
        const float s = srow[it * SCALES + warp / 8];
        const int t0h = warp * 16 + k0;
        mma_a_f16 a;
        a.x[0] = *(const unsigned*)(const void*)&tl[ar * STRIDE + t0h];
        a.x[1] = *(const unsigned*)(const void*)&tl[(ar + 8) * STRIDE + t0h];
        a.x[2] = *(const unsigned*)(const void*)&tl[ar * STRIDE + t0h + 8];
        a.x[3] = *(const unsigned*)(const void*)&tl[(ar + 8) * STRIDE + t0h + 8];

        // Columns past `n_tokens` are fed zero rather than read out of bounds,
        // so the accumulator's unused columns stay unused.
#pragma unroll
        for (int g = 0; g < GROUPS; ++g) {
            const int tok = tok0 + g * 8 + bc;
            mma_b_f16 b = {{0u, 0u}};
            if (tok < n_tokens) {
                const float* xp = x + (size_t)tok * k + ko + k0;
                const __half2 lo = __floats2half2_rn(xp[0], xp[1]);
                const __half2 hi = __floats2half2_rn(xp[8], xp[9]);
                b.x[0] = *(const unsigned*)(const void*)&lo;
                b.x[1] = *(const unsigned*)(const void*)&hi;
            }
            mma_c_f32 c_local = {{0.0f, 0.0f, 0.0f, 0.0f}};
            mma_f16(c_local, a, b);
#pragma unroll
            for (int i = 0; i < 4; ++i) acc[g].x[i] += s * c_local.x[i];
        }
    }

    // The tile is dead; the same bytes become the partials. Both barriers are
    // needed: one so no warp is still reading the tile, one so no warp reads a
    // partial before its owner wrote it.
    __syncthreads();
#pragma unroll
    for (int g = 0; g < GROUPS; ++g) {
        float* rg = red + (size_t)warp * GROUPS * 128 + g * 128;
        rg[cr * 8 + cc + 0] = acc[g].x[0];
        rg[cr * 8 + cc + 1] = acc[g].x[1];
        rg[(cr + 8) * 8 + cc + 0] = acc[g].x[2];
        rg[(cr + 8) * 8 + cc + 1] = acc[g].x[3];
    }
    __syncthreads();

    // `red` is row-major by output row (`rg[row * 8 + col]` above), and
    // reading it in that order is what the write already made cheap. `out`
    // is `[token][row]`, the opposite: consecutive rows of the same token are
    // the contiguous addresses, not consecutive tokens of the same row. The
    // original loop walked `red` in its own order and let the store fall
    // where it may -- one sector a thread, 16 of 32 bytes used, `ncu` put a
    // fifth of the kernel's stall cycles on exactly this. `r` is the
    // fast-varying half of the thread index below so 16 consecutive threads
    // land on 16 consecutive `out` addresses; `slot` still computes the same
    // row-major position into `red` that the write used, independent of the
    // order threads visit it in.
    for (int i = threadIdx.x; i < GROUPS * 128; i += blockDim.x) {
        const int g = i / 128;
        const int rem = i % 128;
        const int r = rem % 16;
        const int col = rem / 16;
        const int t = tok0 + g * 8 + col;
        if (t >= n_tokens || row0 + r >= n) continue;
        const int slot = g * 128 + r * 8 + col;
        float sum = 0.0f;
#pragma unroll
        for (int wi = 0; wi < WARPS; ++wi) sum += red[(size_t)wi * GROUPS * 128 + slot];
        float* o = out + (size_t)t * n + row0 + r;
        *o = accum ? *o + sum : sum;
    }
#else
    (void)out; (void)w; (void)x; (void)k; (void)n; (void)scale_cols;
    (void)n_tokens; (void)accum;
#endif
}

#define FP8_MMA_ENTRY(WARPS, GROUPS, NAME)                                     \
    extern "C" __global__ void NAME(                                           \
            float* __restrict__ out, const unsigned char* __restrict__ w,      \
            const float* __restrict__ x, int k, int n, int scale_cols,         \
            int n_tokens, int accum) {                                         \
        mma_f8_body<WARPS, GROUPS>(out, w, x, k, n, scale_cols, n_tokens,      \
                                   accum);                                     \
    }

FP8_MMA_ENTRY(8, 1, mma_f8_block_f32)
FP8_MMA_ENTRY(8, 2, mma_f8_block_g2_f32)
FP8_MMA_ENTRY(8, 4, mma_f8_block_g4_f32)
FP8_MMA_ENTRY(8, 8, mma_f8_block_g8_f32)
FP8_MMA_ENTRY(32, 1, mma_f8_block_w32_f32)
FP8_MMA_ENTRY(32, 2, mma_f8_block_w32_g2_f32)
