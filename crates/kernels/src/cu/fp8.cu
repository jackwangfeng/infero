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
#pragma unroll
            for (int r = 0; r < FP8_ROW_GROUP; ++r) {
                acc[r][t] += wv[r][0] * xv[0] + wv[r][1] * xv[1]
                           + wv[r][2] * xv[2] + wv[r][3] * xv[3];
            }
        }
    }

    // One reduction per row per token. Warp-level first, then across warps
    // through shared memory — `block_reduce_sum` cannot be called in a loop,
    // since its static shared result would be overwritten while slower threads
    // still read it.
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
