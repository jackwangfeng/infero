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
// One block an output row. Within it, one warp per 128-wide slice of k, because
// that is exactly the span a single scale covers: the warp reduces its slice
// with shuffles, multiplies by the scale once, and adds to a running total. A
// scale applied per weight instead would be 128 times the multiplies for the
// same answer; applied per row it would be the wrong answer.
//
// A lane reads four consecutive bytes as one aligned 32-bit load, so a warp's
// read of its slice is a single 128-byte transaction. Both `row * k` and
// `kb * 128` are multiples of four for every projection in this checkpoint, and
// the caller checks it.
extern "C" __global__ void mmv_f8_block_f32(float* __restrict__ out,
                                            const unsigned char* __restrict__ w,
                                            const float* __restrict__ x,
                                            int k, int n, int scale_cols,
                                            int accum) {
    const int row = blockIdx.x;
    if (row >= n) return;

    const int warps = blockDim.x / WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int n_kb = (k + 127) / 128;

    // The scale grid sits after the quants.
    const float* scales = (const float*)(w + (size_t)n * k);
    const float* srow = scales + (size_t)(row / 128) * scale_cols;
    const unsigned char* wrow = w + (size_t)row * k;

    float acc = 0.0f;
    for (int kb = warp; kb < n_kb; kb += warps) {
        const int base = kb * 128;
        float part = 0.0f;
        const int i0 = base + lane * 4;
        if (i0 + 3 < k) {
            const unsigned int packed = ((const unsigned int*)(wrow + base))[lane];
            part = e4m3_to_f32(packed & 0xFFu) * x[i0]
                 + e4m3_to_f32((packed >> 8) & 0xFFu) * x[i0 + 1]
                 + e4m3_to_f32((packed >> 16) & 0xFFu) * x[i0 + 2]
                 + e4m3_to_f32((packed >> 24) & 0xFFu) * x[i0 + 3];
        } else {
            // A trailing partial slice. Every projection in this checkpoint has
            // k a multiple of 128 so this never runs, but a kernel that reads
            // past the row when it does would be a silent corruption rather
            // than a crash.
            for (int j = 0; j < 4; ++j) {
                const int i = i0 + j;
                if (i < k) part += e4m3_to_f32(wrow[i]) * x[i];
            }
        }
        for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
            part += __shfl_down_sync(FULL_MASK, part, off);
        }
        if (lane == 0) acc += part * srow[kb];
    }

    // Only lane 0 of each warp carries a partial.
    const float total = block_reduce_sum(lane == 0 ? acc : 0.0f);
    // `accum` folds the residual add into the projection that feeds it, the way
    // the other mat-vecs do.
    if (threadIdx.x == 0) out[row] = accum ? out[row] + total : total;
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
