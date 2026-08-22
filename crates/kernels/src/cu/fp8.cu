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

    float acc = 0.0f;
    // Each thread walks the row in 4-element strides. `i0 / 128` is the slice it
    // is in, and a 4-element group never straddles two slices because 128 is a
    // multiple of 4 — which is what makes one scale per group correct.
    for (int i0 = threadIdx.x * 4; i0 < k; i0 += blockDim.x * 4) {
        const float s = srow[i0 >> 7];
        if (i0 + 3 < k) {
            const unsigned int packed = *(const unsigned int*)(wrow + i0);
            acc += s * (e4m3_to_f32(packed & 0xFFu) * x[i0]
                      + e4m3_to_f32((packed >> 8) & 0xFFu) * x[i0 + 1]
                      + e4m3_to_f32((packed >> 16) & 0xFFu) * x[i0 + 2]
                      + e4m3_to_f32((packed >> 24) & 0xFFu) * x[i0 + 3]);
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
template <int TOKENS>
__device__ __forceinline__ void mmv_f8_block_batch_body(
        float* __restrict__ out, const unsigned char* __restrict__ w,
        const float* __restrict__ x, int k, int n, int scale_cols, int n_tokens,
        int accum) {
    const int row = blockIdx.x;
    if (row >= n) return;

    const float* scales = (const float*)(w + (size_t)n * k);
    const float* srow = scales + (size_t)(row / 128) * scale_cols;
    const unsigned char* wrow = w + (size_t)row * k;

    float acc[TOKENS];
#pragma unroll
    for (int t = 0; t < TOKENS; ++t) acc[t] = 0.0f;

    for (int i0 = threadIdx.x * 4; i0 < k; i0 += blockDim.x * 4) {
        const float s = srow[i0 >> 7];
        float wv[4];
        if (i0 + 3 < k) {
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
        // The weight is in registers now; every token reuses it.
#pragma unroll
        for (int t = 0; t < TOKENS; ++t) {
            if (t >= n_tokens) break;
            const float* xt = x + (size_t)t * k;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                if (i0 + j < k) acc[t] += wv[j] * xt[i0 + j];
            }
        }
    }

    // One reduction a token. Warp-level first, then across warps through shared
    // memory — `block_reduce_sum` cannot be called in a loop, since its static
    // shared result would be overwritten while slower threads still read it.
    __shared__ float partial[32][TOKENS];
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
    const int warps = blockDim.x / WARP_SIZE;
#pragma unroll
    for (int t = 0; t < TOKENS; ++t) {
        if (t >= n_tokens) break;
        float v = acc[t];
        for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
            v += __shfl_down_sync(FULL_MASK, v, off);
        }
        if (lane == 0) partial[warp][t] = v;
    }
    __syncthreads();
    if (threadIdx.x < TOKENS && threadIdx.x < n_tokens) {
        const int t = threadIdx.x;
        float sum = 0.0f;
        for (int wi = 0; wi < warps; ++wi) sum += partial[wi][t];
        float* o = out + (size_t)t * n + row;
        *o = accum ? *o + sum : sum;
    }
}

extern "C" __global__ void mmv_f8_block_batch8_f32(
        float* __restrict__ out, const unsigned char* __restrict__ w,
        const float* __restrict__ x, int k, int n, int scale_cols, int n_tokens,
        int accum) {
    mmv_f8_block_batch_body<8>(out, w, x, k, n, scale_cols, n_tokens, accum);
}

extern "C" __global__ void mmv_f8_block_batch32_f32(
        float* __restrict__ out, const unsigned char* __restrict__ w,
        const float* __restrict__ x, int k, int n, int scale_cols, int n_tokens,
        int accum) {
    mmv_f8_block_batch_body<32>(out, w, x, k, n, scale_cols, n_tokens, accum);
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
