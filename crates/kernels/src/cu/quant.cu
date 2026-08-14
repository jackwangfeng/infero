// Quantized weight access: whole-matrix dequantization, row gathers for the
// embedding table, and the dequant-fused mat-vec that decoding lives on.
//
// A ggml weight matrix is row-major with `k` elements per row (`ne0`) and `n`
// rows (`ne1`), each row an integer number of quant blocks. Every kernel here
// takes the raw mapped bytes and decodes on the fly; weights are never
// materialized in full precision on the device.

// ---- single-element decoders -------------------------------------------
// Used by the row gather, where the access pattern is irregular anyway.

__device__ __forceinline__ float deq_f32(const void* w, size_t i) {
    return ((const float*)w)[i];
}

__device__ __forceinline__ float deq_f16(const void* w, size_t i) {
    return __half2float(((const __half*)w)[i]);
}

__device__ __forceinline__ float deq_q8_0(const void* w, size_t i) {
    const block_q8_0* b = (const block_q8_0*)w + i / QK8_0;
    return __half2float(b->d) * (float)b->qs[i % QK8_0];
}

// In every block-32 quant the low nibbles hold elements 0..15 and the high
// nibbles elements 16..31, so a block's two halves are interleaved on disk.
__device__ __forceinline__ float deq_q4_0(const void* w, size_t i) {
    const block_q4_0* b = (const block_q4_0*)w + i / QK4_0;
    const int j = i % QK4_0;
    const uint8_t q = b->qs[j % 16];
    const int v = ((j < 16) ? (q & 0xF) : (q >> 4)) - 8;
    return __half2float(b->d) * (float)v;
}

__device__ __forceinline__ float deq_q4_1(const void* w, size_t i) {
    const block_q4_1* b = (const block_q4_1*)w + i / QK4_0;
    const int j = i % QK4_0;
    const uint8_t q = b->qs[j % 16];
    const int v = (j < 16) ? (q & 0xF) : (q >> 4);
    return __half2float(b->d) * (float)v + __half2float(b->m);
}

__device__ __forceinline__ float deq_q4_g128(const void* w, size_t i) {
    const block_q4_g128* b = (const block_q4_g128*)w + i / QK_G128;
    const int j = i % QK_G128;
    const uint8_t q = b->qs[j % 64];
    const int v = (j < 64) ? (q & 0xF) : (q >> 4);
    return __low2float(b->ds) * (float)v - __high2float(b->ds);
}

__device__ __forceinline__ float deq_q5_0(const void* w, size_t i) {
    const block_q5_0* b = (const block_q5_0*)w + i / QK5_0;
    const int j = i % QK5_0;
    const uint32_t qh = load_qh(b->qh);
    const uint8_t q = b->qs[j % 16];
    const uint8_t hi = (j < 16) ? (((qh >> j) << 4) & 0x10)
                                : ((qh >> (j - 16 + 12)) & 0x10);
    const int v = (int)(((j < 16) ? (q & 0xF) : (q >> 4)) | hi) - 16;
    return __half2float(b->d) * (float)v;
}

__device__ __forceinline__ float deq_q5_1(const void* w, size_t i) {
    const block_q5_1* b = (const block_q5_1*)w + i / QK5_0;
    const int j = i % QK5_0;
    const uint32_t qh = load_qh(b->qh);
    const uint8_t q = b->qs[j % 16];
    const uint8_t hi = (j < 16) ? (((qh >> j) << 4) & 0x10)
                                : ((qh >> (j - 16 + 12)) & 0x10);
    const int v = (int)(((j < 16) ? (q & 0xF) : (q >> 4)) | hi);
    return __half2float(b->d) * (float)v + __half2float(b->m);
}

__device__ __forceinline__ float deq_q4_K(const void* w, size_t i) {
    const block_q4_K* b = (const block_q4_K*)w + i / QK_K;
    const int within = i % QK_K;
    const int group64 = within / 64;   // which pair of 32-element groups
    const int rem = within % 64;
    const int high = rem / 32;         // low nibbles first, then high
    const int l = rem % 32;

    uint8_t sc, m;
    q4k_scale_min(b->scales, group64 * 2 + high, &sc, &m);

    const uint8_t q = b->qs[group64 * 32 + l];
    const int nib = high ? (q >> 4) : (q & 0xF);
    return __half2float(b->d) * (float)sc * (float)nib
         - __half2float(b->dmin) * (float)m;
}

__device__ __forceinline__ float deq_q6_K(const void* w, size_t i) {
    const block_q6_K* b = (const block_q6_K*)w + i / QK_K;
    const int within = i % QK_K;
    const int n = within / 128;        // super-block half
    const int rem = within % 128;
    const int quarter = rem / 32;      // which of the four interleaved groups
    const int l = rem % 32;

    const uint8_t* ql = b->ql + n * 64;
    const uint8_t* qh = b->qh + n * 32;
    const int8_t* sc = b->scales + n * 8;

    const int lo_index = (quarter & 1) ? (l + 32) : l;
    const int shift = quarter * 2;
    const uint8_t low = (quarter < 2) ? (ql[lo_index] & 0xF) : (ql[lo_index] >> 4);
    const int q = (int)(low | (((qh[l] >> shift) & 3) << 4)) - 32;

    return __half2float(b->d) * (float)sc[quarter * 2 + l / 16] * (float)q;
}

// ---- row gather (embedding lookup) --------------------------------------

#define GATHER_KERNEL(NAME, DECODE)                                            \
    extern "C" __global__ void NAME(float* __restrict__ out,                   \
                                    const void* __restrict__ w,                \
                                    const int* __restrict__ rows, int k) {     \
        const int t = blockIdx.y;                                              \
        const int i = blockIdx.x * blockDim.x + threadIdx.x;                   \
        if (i >= k) return;                                                    \
        const size_t src = (size_t)rows[t] * k + i;                            \
        out[(size_t)t * k + i] = DECODE(w, src);                               \
    }

GATHER_KERNEL(gather_rows_f32, deq_f32)
GATHER_KERNEL(gather_rows_f16, deq_f16)
GATHER_KERNEL(gather_rows_q4_0, deq_q4_0)
GATHER_KERNEL(gather_rows_q4_1, deq_q4_1)
GATHER_KERNEL(gather_rows_q5_0, deq_q5_0)
GATHER_KERNEL(gather_rows_q5_1, deq_q5_1)
GATHER_KERNEL(gather_rows_q8_0, deq_q8_0)
GATHER_KERNEL(gather_rows_q4_K, deq_q4_K)
GATHER_KERNEL(gather_rows_q6_K, deq_q6_K)
GATHER_KERNEL(gather_rows_q4_g128, deq_q4_g128)

// ---- whole-matrix dequantization to f16 ---------------------------------
// Feeds the cuBLAS path used for prefill, where re-reading the weights once
// costs far less than doing a batched mat-vec.

#define DEQUANT_KERNEL(NAME, DECODE)                                           \
    extern "C" __global__ void NAME(__half* __restrict__ out,                  \
                                    const void* __restrict__ w, size_t n) {    \
        const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;        \
        if (i >= n) return;                                                    \
        out[i] = __float2half(DECODE(w, i));                                   \
    }

DEQUANT_KERNEL(dequant_f32_f16, deq_f32)
DEQUANT_KERNEL(dequant_f16_f16, deq_f16)
DEQUANT_KERNEL(dequant_q4_0_f16, deq_q4_0)
DEQUANT_KERNEL(dequant_q4_1_f16, deq_q4_1)
DEQUANT_KERNEL(dequant_q5_0_f16, deq_q5_0)
DEQUANT_KERNEL(dequant_q5_1_f16, deq_q5_1)
DEQUANT_KERNEL(dequant_q8_0_f16, deq_q8_0)
DEQUANT_KERNEL(dequant_q4_K_f16, deq_q4_K)
DEQUANT_KERNEL(dequant_q6_K_f16, deq_q6_K)
DEQUANT_KERNEL(dequant_q4_g128_f16, deq_q4_g128)

// The split Q8_0 layout: `total` int8 quants, then one scale per 32. The
// boundary is at `total` because the quants are one byte each.
extern "C" __global__ void dequant_q8_0s_f16(__half* __restrict__ out,
                                             const void* __restrict__ w,
                                             size_t total) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int8_t* q = (const int8_t*)w;
    const __half* sc = (const __half*)(const void*)(q + total);
    out[i] = __float2half(__half2float(sc[i / QK8_0]) * (float)q[i]);
}

// The transposed Q4_G128 layout, for the cuBLAS prefill path.
//
// Quants first — one 64-byte block per (row, 128 weights), the 4x4 matrix of
// 4-byte words inside it transposed — then one `__half2` per block. `total` is
// the element count, which is what locates the boundary: the quants are half a
// byte each, so the scales start at `total / 2`.
extern "C" __global__ void dequant_q4_g128t_f16(__half* __restrict__ out,
                                                const void* __restrict__ w,
                                                size_t total) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const uint8_t* q = (const uint8_t*)w;
    const __half2* sc = (const __half2*)(const void*)(q + total / 2);
    const size_t b = i / QK_G128;
    const int j = (int)(i % QK_G128);
    const int m = j % 64;
    // Undo the word transpose: old byte `m` is at new `c*16 + word*4 + e`.
    /* Bytes 1 and 2 of each word swap in the pack; see `awq::transpose_words`. */
    const int _p[4] = {0, 2, 1, 3};
    const int byte = ((m % 16) / 4) * 16 + (m / 16) * 4 + _p[m % 4];
    const uint8_t v = q[b * 64 + byte];
    const int code = (j < 64) ? (v & 0xF) : (v >> 4);
    const __half2 ds = sc[b];
    out[i] = __float2half((float)code * __low2float(ds) - __high2float(ds));
}


// ---- mat-vec ------------------------------------------------------------
// out[t, r] = dot(W[r, :], x[t, :]) for a handful of tokens.
//
// One block per (output row, token). Decoding happens block-at-a-time rather
// than element-at-a-time so the scale is read once per 32 or 256 values.

// Tokens a single block serves. A block decodes each weight element once and
// spends it on every token it holds, so the weight traffic of a `t`-token
// batch drops by this factor. That matters most for the vocab projection,
// which is the largest matrix in the model and is read once per row that
// wants logits.
#define GEMV_TOKENS 8

#define GEMV_PROLOGUE                                                          \
    const int row = blockIdx.x;                                                \
    if (row >= n) return;                                                      \
    const int token0 = blockIdx.y * GEMV_TOKENS;                               \
    const int ntok = min(GEMV_TOKENS, n_tokens - token0);                      \
    float acc[GEMV_TOKENS];                                                    \
    _Pragma("unroll") for (int t = 0; t < GEMV_TOKENS; ++t) acc[t] = 0.0f;

/// Spend one decoded weight element on every token this block holds.
///
/// The trip count is the compile-time `GEMV_TOKENS` with a predicate inside,
/// not the runtime `ntok`. A runtime bound leaves the compiler unable to prove
/// the indices, so `acc` lands in local memory — which for a mat-vec that is
/// otherwise pure streaming costs an order of magnitude.
#define GEMV_SPREAD(WV, I)                                                     \
    _Pragma("unroll")                                                          \
    for (int t = 0; t < GEMV_TOKENS; ++t) {                                    \
        if (t < ntok) acc[t] += (WV) * x[(size_t)(token0 + t) * k + (I)];      \
    }

#define GEMV_EPILOGUE                                                          \
    _Pragma("unroll")                                                          \
    for (int t = 0; t < GEMV_TOKENS; ++t) {                                    \
        if (t < ntok) {                                                        \
            const float total = block_reduce_sum(acc[t]);                      \
            if (threadIdx.x == 0) out[(size_t)(token0 + t) * n + row] = total; \
        }                                                                      \
    }

extern "C" __global__ void gemv_f32(float* __restrict__ out,
                                    const void* __restrict__ w,
                                    const float* __restrict__ x, int k, int n,
                                    int n_tokens) {
    GEMV_PROLOGUE
    const float* wr = (const float*)w + (size_t)row * k;
    for (int i = threadIdx.x; i < k; i += blockDim.x) GEMV_SPREAD(wr[i], i)
    GEMV_EPILOGUE
}

extern "C" __global__ void gemv_f16(float* __restrict__ out,
                                    const void* __restrict__ w,
                                    const float* __restrict__ x, int k, int n,
                                    int n_tokens) {
    GEMV_PROLOGUE
    const __half* wr = (const __half*)w + (size_t)row * k;
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        GEMV_SPREAD(__half2float(wr[i]), i)
    }
    GEMV_EPILOGUE
}

// The block-32 legacy quants decode one element at a time. Consecutive lanes
// land in the same block, so the scale and nibble loads coalesce and stay in
// L1 — close enough to a block-at-a-time version to not be worth the code.
#define GEMV_ELEMENTWISE(NAME, DECODE)                                         \
    extern "C" __global__ void NAME(float* __restrict__ out,                   \
                                    const void* __restrict__ w,                \
                                    const float* __restrict__ x, int k,        \
                                    int n, int n_tokens) {                                   \
        GEMV_PROLOGUE                                                          \
        const size_t base = (size_t)row * k;                                   \
        for (int i = threadIdx.x; i < k; i += blockDim.x) {                    \
            GEMV_SPREAD(DECODE(w, base + i), i)                                \
        }                                                                      \
        GEMV_EPILOGUE                                                          \
    }

GEMV_ELEMENTWISE(gemv_q4_0, deq_q4_0)
GEMV_ELEMENTWISE(gemv_q4_1, deq_q4_1)
GEMV_ELEMENTWISE(gemv_q5_0, deq_q5_0)
GEMV_ELEMENTWISE(gemv_q5_1, deq_q5_1)

// Quarter-blocks, not whole blocks: a 896-element row is only 28 Q8_0 blocks,
// so one thread per block would leave a 128-thread launch three quarters idle.
// Eight elements per thread keeps the scale load amortized and the launch full.
#define Q8_0_PER_THREAD 8

extern "C" __global__ void gemv_q8_0(float* __restrict__ out,
                                     const void* __restrict__ w,
                                     const float* __restrict__ x, int k, int n,
                                    int n_tokens) {
    GEMV_PROLOGUE
    const int nb = k / QK8_0;
    const int per_block = QK8_0 / Q8_0_PER_THREAD;
    const int chunks = nb * per_block;
    const block_q8_0* wr = (const block_q8_0*)w + (size_t)row * nb;

    for (int c = threadIdx.x; c < chunks; c += blockDim.x) {
        const block_q8_0* blk = wr + c / per_block;
        const int sub = (c % per_block) * Q8_0_PER_THREAD;
        const int base = (c / per_block) * QK8_0 + sub;
        const float d = __half2float(blk->d);
#pragma unroll
        for (int i = 0; i < Q8_0_PER_THREAD; ++i) {
            GEMV_SPREAD(d * (float)blk->qs[sub + i], base + i)
        }
    }
    GEMV_EPILOGUE
}

// The same mat-vec over the split layout: `k` contiguous int8 a row, then one
// scale per 32. A thread's run of weights is contiguous here where the packed
// form put a 2-byte scale in the middle of every 32, so the load can be as wide
// as the unroll — which is the whole point of the layout. See
// `mmq_load_w_q8_0s`.
extern "C" __global__ void gemv_q8_0s(float* __restrict__ out,
                                      const void* __restrict__ w,
                                      const float* __restrict__ x, int k, int n,
                                      int n_tokens) {
    GEMV_PROLOGUE
    const int nb = k / QK8_0;
    const int per_block = QK8_0 / Q8_0_PER_THREAD;
    const int chunks = nb * per_block;
    const int8_t* q = (const int8_t*)w + (size_t)row * k;
    const __half* sc = (const __half*)(const void*)((const int8_t*)w + (size_t)n * k)
                       + (size_t)row * nb;

    for (int c = threadIdx.x; c < chunks; c += blockDim.x) {
        const int b = c / per_block;
        const int sub = (c % per_block) * Q8_0_PER_THREAD;
        const int base = b * QK8_0 + sub;
        const float d = __half2float(sc[b]);
#pragma unroll
        for (int i = 0; i < Q8_0_PER_THREAD; ++i) {
            GEMV_SPREAD(d * (float)q[base + i], base + i)
        }
    }
    GEMV_EPILOGUE
}

extern "C" __global__ void gemv_q4_g128(float* __restrict__ out,
                                        const void* __restrict__ w,
                                        const float* __restrict__ x, int k,
                                        int n, int n_tokens) {
    GEMV_PROLOGUE
    const int nb = k / QK_G128;
    // One 32-weight quarter per thread — the same slice a Q8_1 activation
    // block covers, so the integer path and this one walk the row alike.
    const int chunks = nb * 4;
    const block_q4_g128* wr = (const block_q4_g128*)w + (size_t)row * nb;

    for (int c = threadIdx.x; c < chunks; c += blockDim.x) {
        const block_q4_g128* blk = wr + c / 4;
        const int q = c % 4;
        const float d = __low2float(blk->ds);
        const float dz = __high2float(blk->ds);
        const int byte0 = 32 * (q % 2);
        const int shift = 4 * (q / 2);
        const int base = (c / 4) * QK_G128 + 32 * q;
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            const int v = (blk->qs[byte0 + i] >> shift) & 0xF;
            GEMV_SPREAD(d * (float)v - dz, base + i)
        }
    }
    GEMV_EPILOGUE
}

extern "C" __global__ void gemv_q4_K(float* __restrict__ out,
                                     const void* __restrict__ w,
                                     const float* __restrict__ x, int k, int n,
                                    int n_tokens) {
    GEMV_PROLOGUE
    const int nb = k / QK_K;
    const block_q4_K* wr = (const block_q4_K*)w + (size_t)row * nb;

    // One thread per 32-element group rather than per 256-element super-block,
    // for the same reason as Q8_0: a row is only a handful of super-blocks.
    for (int c = threadIdx.x; c < nb * 8; c += blockDim.x) {
        const block_q4_K* blk = wr + c / 8;
        const int g = c % 8;
        const int base = (c / 8) * QK_K + g * 32;

        uint8_t sc, m;
        q4k_scale_min(blk->scales, g, &sc, &m);
        const int high = g & 1;
        const float d = __half2float(blk->d) * (float)sc;
        const float mn = __half2float(blk->dmin) * (float)m;

        // Four nibble-bytes at a time. Reading `qs` a byte at a time makes a
        // warp issue 32 scattered one-byte requests, and the memory system
        // fetches whole sectors for each — the delivered bandwidth ends up a
        // fraction of what the weight volume alone would suggest. `qs` sits 16
        // bytes into the block, so the word loads stay aligned.
        const uint32_t* q32 =
            (const uint32_t*)(const void*)(blk->qs + (g / 2) * 32);
#pragma unroll
        for (int w = 0; w < 8; ++w) {
            const uint32_t packed = q32[w];
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                const int byte = (packed >> (b * 8)) & 0xFF;
                const int nib = high ? (byte >> 4) : (byte & 0xF);
                GEMV_SPREAD(d * (float)nib - mn, base + w * 4 + b)
            }
        }
    }
    GEMV_EPILOGUE
}

extern "C" __global__ void gemv_q6_K(float* __restrict__ out,
                                     const void* __restrict__ w,
                                     const float* __restrict__ x, int k, int n,
                                    int n_tokens) {
    GEMV_PROLOGUE
    const int nb = k / QK_K;
    const block_q6_K* wr = (const block_q6_K*)w + (size_t)row * nb;

    // One thread per `l`, four output elements each, rather than one thread per
    // 256-element super-block. A row of 4096 is only sixteen super-blocks, so
    // the old split left a 32-thread block half idle with each active thread
    // walking 256 elements serially — and consecutive threads were 256 elements
    // apart, which coalesces into nothing. Here they read adjacent bytes.
    const int chunks = nb * 64;  // two halves x 32 positions
    for (int c = threadIdx.x; c < chunks; c += blockDim.x) {
        const int b = c / 64;
        const int rem = c % 64;
        const int n2 = rem / 32;
        const int l = rem % 32;

        const block_q6_K* blk = wr + b;
        const uint8_t* ql = blk->ql + n2 * 64;
        const uint8_t* qh = blk->qh + n2 * 32;
        const int8_t* sc = blk->scales + n2 * 8;
        const int base = b * QK_K + n2 * 128;
        const float d = __half2float(blk->d);

        const uint8_t h = qh[l];
        const int is = l / 16;
        const int q0 = (int)((ql[l] & 0xF) | (((h >> 0) & 3) << 4)) - 32;
        const int q1 = (int)((ql[l + 32] & 0xF) | (((h >> 2) & 3) << 4)) - 32;
        const int q2 = (int)((ql[l] >> 4) | (((h >> 4) & 3) << 4)) - 32;
        const int q3 = (int)((ql[l + 32] >> 4) | (((h >> 6) & 3) << 4)) - 32;
        GEMV_SPREAD(d * (float)sc[is + 0] * (float)q0, base + l)
        GEMV_SPREAD(d * (float)sc[is + 2] * (float)q1, base + l + 32)
        GEMV_SPREAD(d * (float)sc[is + 4] * (float)q2, base + l + 64)
        GEMV_SPREAD(d * (float)sc[is + 6] * (float)q3, base + l + 96)
    }
    GEMV_EPILOGUE
}
