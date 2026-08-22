// FP8 E4M3 weights with a 128x128 block scale grid, read where they lie.
//
// The 27B ships FP8 and tuili was dequantizing it to f16 at load, which is
// correct and costs a factor of two in the only resource decode has: a step
// reads every weight once, so 51 GiB of f16 takes twice as long as 29 GiB of
// FP8 no matter how good the arithmetic is. Measured, that was 13.2 tok/s
// against vLLM's 34 on the same checkpoint.
//
// Layout, one buffer so a `Matrix` stays a single allocation: `n * k` quant
// bytes row-major, then the scale grid as f32, `ceil(n/128) * ceil(k/128)` of
// them, also row-major. The grid's row index is the *output* row over 128 and
// its column index is the position along k over 128 — a scale is shared by 128
// rows and 128 columns at once, which is why a row's bytes are not
// self-describing the way a ggml block's are.

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

// out[row] = sum over k of w[row, i] * x[i], with w's blocks scaled.
//
// One block an output row, threads striding over the row.
//
// The scale is folded into each product rather than applied to a per-slice sum,
// which is what lets the loads pipeline. Distributing it is exact:
//
//     sum_kb s_kb * sum_{i in kb} w_i x_i  ==  sum_kb sum_{i in kb} s_kb w_i x_i
//
// The first version did the inner sum with a shuffle reduction and then scaled,
// which put a five-step cross-lane reduction between one load and the next — so
// a warp had at most one outstanding memory request and none of the latency was
// hidden. This way every lane keeps its own running total, all the loads are
// independent, and there is exactly one reduction at the end. It costs one extra
// multiply per four weights instead of one per 128, which is nothing next to a
// byte of DRAM per weight.
//
// A lane reads four consecutive bytes as one aligned 32-bit load, so a warp's
// read is a single 128-byte transaction. `row * k` and `kb * 128` are multiples
// of four for every projection in this checkpoint, and the launcher checks it.
extern "C" __global__ void mmv_f8_block_f32(float* __restrict__ out,
                                            const unsigned char* __restrict__ w,
                                            const float* __restrict__ x,
                                            int k, int n, int scale_cols,
                                            int accum) {
    const int row = blockIdx.x;
    if (row >= n) return;

    const float* scales = (const float*)(w + (size_t)n * k);
    const float* srow = scales + (size_t)(row / 128) * scale_cols;
    const unsigned char* wrow = w + (size_t)row * k;

    const bool xvec = ((size_t)(const void*)x % 16 == 0);

    float acc = 0.0f;
    // Each thread walks the row in 4-element strides. `i0 / 128` is the slice it
    // is in, and a 4-element group never straddles two slices because 128 is a
    // multiple of 4 — which is what makes one scale per group correct.
    for (int i0 = threadIdx.x * 4; i0 < k; i0 += blockDim.x * 4) {
        const float s = srow[i0 >> 7];
        if (i0 + 3 < k && xvec) {
            const unsigned int packed = *(const unsigned int*)(wrow + i0);
            // One 16-byte activation load beside the one 4-byte weight load.
            // Same reasoning as the batched kernel below, where it matters more.
            const float4 xv = *(const float4*)(const void*)(x + i0);
            acc += s * (e4m3_to_f32(packed & 0xFFu) * xv.x
                      + e4m3_to_f32((packed >> 8) & 0xFFu) * xv.y
                      + e4m3_to_f32((packed >> 16) & 0xFFu) * xv.z
                      + e4m3_to_f32((packed >> 24) & 0xFFu) * xv.w);
        } else {
            for (int j = 0; j < 4 && i0 + j < k; ++j) {
                acc += s * e4m3_to_f32(wrow[i0 + j]) * x[i0 + j];
            }
        }
    }

    const float total = block_reduce_sum(acc);
    if (threadIdx.x == 0) out[row] = accum ? out[row] + total : total;
}

// The same, for a handful of tokens at once.
//
// This is the case the expansion path was getting badly wrong. Expanding a
// matrix to f16 and handing it to cuBLAS costs one byte read, two written and
// two read back — five bytes a weight against the two that resident f16 cost —
// and at a few tokens the weights still dominate, so it made batched decode
// 2.5x more memory traffic than before FP8 rather than less. The profiler put
// `dequant_f8_block` at 67% of a batch-32 step and batch scaling fell from
// 36.9x to 8.6x.
//
// Here each weight is read once and spent on every token, which is the whole
// point of batching. `TOKENS` is a compile-time bound so the accumulators live
// in registers; above it, a real GEMM is the right answer and the expansion
// path amortizes over enough tokens to be fine.
//
// `x` is `[n_tokens, k]` and `out` is `[n_tokens, n]`, both row-major, matching
// what the rest of the engine passes around.
// Output rows a block handles, per instantiation.
//
// `ROWS * TOKENS` is held constant at 32, because that product is the
// accumulator count and the accumulators are what the register file has to fit.
// The first attempt used ROWS 8 with TOKENS 8 — 64 accumulators plus 32
// activation registers — and it was 2.75x *slower* than no tiling at all: 95.7 ms
// against 34.8 for a two-row pass. Spilling in this inner loop costs far more
// than the L2 traffic the tiling saves.
//
// `#pragma unroll` over `TOKENS` with a runtime `break` does not shrink that:
// the compiler still allocates every slot. So the token count has to be a tight
// compile-time bound, which is why there are four instantiations rather than one
// wide one, and why the launcher dispatches on the actual count.
//
// Every ROWS must divide 128, or a block's rows straddle a scale-grid row and
// one `srow` pointer stops being enough.
// One. Two tilings were measured and both were worse; the reasoning and the
// numbers are on `BATCH_KERNELS` in `fp8.rs`. The machinery stays because it is
// the shape a shared-memory-staged version needs, and because `ROWS = 1` is the
// same arithmetic the untiled kernel did.
#define FP8_MMV_ROWS2 1
#define FP8_MMV_ROWS4 1
#define FP8_MMV_ROWS8 1
#define FP8_MMV_ROWS16 1

// The same, for a handful of tokens at once, ROWS output rows to a block.
//
// Two things happen here that did not before, and the second is the one that
// mattered.
//
// **Each weight is read once and spent on every token.** This is the case the
// expansion path was getting badly wrong: expanding a matrix to f16 and handing
// it to cuBLAS costs one byte read, two written and two read back — five bytes a
// weight against the two that resident f16 cost — so it made batched decode 2.5x
// more memory traffic than before FP8. The profiler put `dequant_f8_block` at
// 67% of a batch-32 step and batch scaling fell from 36.9x to 8.6x.
//
// **Each activation is read once and spent on every row.** With one output row
// to a block, every block reads the whole activation, so activation traffic is
// four bytes per *weight element* — 118 GB a token on the 27B against 29.6 GB of
// weights. It comes out of L2 rather than DRAM, which is why it was invisible in
// a DRAM-traffic argument, and at a few TB/s of L2 it is exactly the size of the
// measured cost: a second row cost 6.9 ms of a 27.9 ms step, and a third and a
// fourth cost 6.9 each, flat, for zero extra weight bytes.
//
// It is not the arithmetic and it is not the load count. At two tokens a thread
// does eight FMAs per four weight bytes — four FLOP a byte against this card's
// 64 — which is 0.5 ms a step; the extra scalar loads are 16 us. Both are two
// orders of magnitude too small. Only the bytes are the right size, and `ROWS`
// divides them.
//
// `ROWS` must divide 128 so that a block's rows never straddle a scale-grid
// boundary: the grid is 128 rows by 128 columns, so one `srow` pointer serves a
// whole block only if the block's rows share `row / 128`.
//
// `x` is `[n_tokens, k]` and `out` is `[n_tokens, n]`, both row-major, matching
// what the rest of the engine passes around.
template <int TOKENS, int ROWS>
__device__ __forceinline__ void mmv_f8_block_batch_body(
        float* __restrict__ out, const unsigned char* __restrict__ w,
        const float* __restrict__ x, int k, int n, int scale_cols, int n_tokens,
        int accum) {
    static_assert(128 % ROWS == 0, "a block's rows must share one scale row");
    const int row0 = blockIdx.x * ROWS;
    if (row0 >= n) return;
    const int rows = (n - row0 < ROWS) ? (n - row0) : ROWS;

    const float* scales = (const float*)(w + (size_t)n * k);
    const float* srow = scales + (size_t)(row0 / 128) * scale_cols;

    float acc[ROWS][TOKENS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
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

    for (int i0 = threadIdx.x * 4; i0 < k; i0 += blockDim.x * 4) {
        const float s = srow[i0 >> 7];
        const bool whole = (i0 + 3 < k);

        // The activation group, once, for every row below to reuse. This is the
        // reordering: activations in the outer position, weights in the inner.
        float xv[TOKENS][4];
#pragma unroll
        for (int t = 0; t < TOKENS; ++t) {
            if (t >= n_tokens) break;
            const float* xt = x + (size_t)t * k;
            if (whole && xvec) {
                const float4 v = *(const float4*)(const void*)(xt + i0);
                xv[t][0] = v.x;
                xv[t][1] = v.y;
                xv[t][2] = v.z;
                xv[t][3] = v.w;
            } else {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    xv[t][j] = (i0 + j < k) ? xt[i0 + j] : 0.0f;
                }
            }
        }

#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            if (r >= rows) break;
            const unsigned char* wrow = w + (size_t)(row0 + r) * k;
            float wv[4];
            if (whole) {
                const unsigned int packed = *(const unsigned int*)(wrow + i0);
                wv[0] = s * e4m3_to_f32(packed & 0xFFu);
                wv[1] = s * e4m3_to_f32((packed >> 8) & 0xFFu);
                wv[2] = s * e4m3_to_f32((packed >> 16) & 0xFFu);
                wv[3] = s * e4m3_to_f32((packed >> 24) & 0xFFu);
            } else {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    wv[j] = (i0 + j < k) ? s * e4m3_to_f32(wrow[i0 + j]) : 0.0f;
                }
            }
#pragma unroll
            for (int t = 0; t < TOKENS; ++t) {
                if (t >= n_tokens) break;
                acc[r][t] += wv[0] * xv[t][0] + wv[1] * xv[t][1]
                           + wv[2] * xv[t][2] + wv[3] * xv[t][3];
            }
        }
    }

    // One reduction per row per token. Warp-level first, then across warps
    // through shared memory — `block_reduce_sum` cannot be called in a loop,
    // since its static shared result would be overwritten while slower threads
    // still read it.
    __shared__ float partial[32][ROWS][TOKENS];
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
    const int warps = blockDim.x / WARP_SIZE;
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
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
    // One thread per (row, token) pair finishes the cross-warp sum. `rows *
    // n_tokens` is at most `ROWS * TOKENS`, which is under any block size here.
    for (int i = threadIdx.x; i < rows * n_tokens; i += blockDim.x) {
        const int r = i / n_tokens;
        const int t = i % n_tokens;
        float sum = 0.0f;
        for (int wi = 0; wi < warps; ++wi) sum += partial[wi][r][t];
        float* o = out + (size_t)t * n + row0 + r;
        *o = accum ? *o + sum : sum;
    }
}

extern "C" __global__ void mmv_f8_block_batch2_f32(
        float* __restrict__ out, const unsigned char* __restrict__ w,
        const float* __restrict__ x, int k, int n, int scale_cols, int n_tokens,
        int accum) {
    mmv_f8_block_batch_body<2, FP8_MMV_ROWS2>(out, w, x, k, n, scale_cols,
                                                  n_tokens, accum);
}

extern "C" __global__ void mmv_f8_block_batch4_f32(
        float* __restrict__ out, const unsigned char* __restrict__ w,
        const float* __restrict__ x, int k, int n, int scale_cols, int n_tokens,
        int accum) {
    mmv_f8_block_batch_body<4, FP8_MMV_ROWS4>(out, w, x, k, n, scale_cols,
                                                  n_tokens, accum);
}

extern "C" __global__ void mmv_f8_block_batch8_f32(
        float* __restrict__ out, const unsigned char* __restrict__ w,
        const float* __restrict__ x, int k, int n, int scale_cols, int n_tokens,
        int accum) {
    mmv_f8_block_batch_body<8, FP8_MMV_ROWS8>(out, w, x, k, n, scale_cols,
                                                  n_tokens, accum);
}

extern "C" __global__ void mmv_f8_block_batch16_f32(
        float* __restrict__ out, const unsigned char* __restrict__ w,
        const float* __restrict__ x, int k, int n, int scale_cols, int n_tokens,
        int accum) {
    mmv_f8_block_batch_body<16, FP8_MMV_ROWS16>(out, w, x, k, n, scale_cols,
                                                  n_tokens, accum);
}

// The same weights, dequantized to f16 for the batched path.
//
// Prefill still goes through the f16 GEMM, so the bytes have to be expandable
// on the device rather than at load. This is the operation that used to happen
// on the host and cost 22 GiB of resident memory.
//
// One block per 128-wide slice of one row, so a scale is read once per block.
extern "C" __global__ void dequant_f8_block_f16(__half* __restrict__ out,
                                                const unsigned char* __restrict__ w,
                                                int k, int n, int scale_cols) {
    const int kb = blockIdx.x;
    const int row = blockIdx.y;
    if (row >= n) return;

    const float* scales = (const float*)(w + (size_t)n * k);
    const float s = scales[(size_t)(row / 128) * scale_cols + kb];
    const unsigned char* wrow = w + (size_t)row * k;
    __half* orow = out + (size_t)row * k;

    const int base = kb * 128;
    for (int i = base + threadIdx.x; i < base + 128 && i < k; i += blockDim.x) {
        orow[i] = __float2half(e4m3_to_f32(wrow[i]) * s);
    }
}
