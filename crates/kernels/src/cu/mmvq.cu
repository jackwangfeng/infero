// Integer mat-vec: quantize the activation to Q8_1, then dot it against the
// weights with `__dp4a` without ever turning a weight into a float.
//
// The per-type dot products below are ported from llama.cpp's
// `ggml/src/ggml-cuda/vecdotq.cuh` (MIT, Copyright (c) 2023-2026 The ggml
// authors) — see `vendor/LICENSE.ggml`. The bit manipulation is theirs; the
// launcher and the Q8_1 quantizer are ours.
//
// Why it is worth borrowing rather than deriving: the float path in `quant.cu`
// decodes every weight to f32, multiplies, and accumulates — several
// instructions and a register each, per weight. This path packs four weights
// and four activations into one 32-bit word each and retires them in a single
// `dp4a`. Same memory traffic, a fraction of the instructions, and the
// activation quantization is amortized over every row of the matrix.
//
// Only the batch-one case lives here. Above that the answer is a tensor-core
// GEMM, not a wider mat-vec.

#define QK8_1 32
#define QI8_1 8   // 32 int8 packed into 8 int32
#define QR4_K 2
#define QR6_K 2
#define QI6_K 32

// Activation block: `d` scales the quants, `s` is the sum of the original
// floats, which lets a weight format with a per-group offset (Q4_K's `dmin`,
// Q4_1's `m`) fold that offset in without a second pass.
typedef struct {
    __half2 ds;
    int8_t qs[QK8_1];
} block_q8_1;

__device__ __forceinline__ int tq_get_int_b2(const void* x, int i32) {
    // `qs` in a K-quant block is only 2-byte aligned.
    const uint16_t* x16 = (const uint16_t*)x;
    return ((int)x16[2 * i32 + 0] << 0) | ((int)x16[2 * i32 + 1] << 16);
}

__device__ __forceinline__ int tq_get_int_b4(const void* x, int i32) {
    return ((const int*)x)[i32];
}

// ---- activation quantization -------------------------------------------

// One warp per 32-element group: the lane holds one element, and the group's
// scale needs a reduction over exactly those 32 values.
extern "C" __global__ void quantize_q8_1_f32(block_q8_1* __restrict__ y,
                                             const float* __restrict__ x,
                                             int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane = threadIdx.x % WARP_SIZE;
    const int block = i / QK8_1;

    const float xi = i < n ? x[i] : 0.0f;
    const float amax = warp_reduce_max(fabsf(xi));
    const float sum = warp_reduce_sum(xi);

    const float d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : (int8_t)roundf(xi / d);

    if (i < n) y[block].qs[lane] = q;
    if (lane == 0 && i < n) {
        y[block].ds = __floats2half2_rn(d, sum);
    }
}

// ---- per-type dot products (ported from llama.cpp) ---------------------

/// One Q8_0 block against one Q8_1 block: 8 elements per call.
__device__ __forceinline__ float tq_dot_q8_0(const void* __restrict__ vbq,
                                             const block_q8_1* __restrict__ bq8_1,
                                             int kbx, int iqs) {
    const block_q8_0* bq8_0 = (const block_q8_0*)vbq + kbx;

    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        sumi = __dp4a(tq_get_int_b2(bq8_0->qs, iqs + i),
                      tq_get_int_b4(bq8_1->qs, iqs + i), sumi);
    }
    const float2 ds = __half22float2(bq8_1->ds);
    return __half2float(bq8_0->d) * ds.x * sumi;
}

/// One 16-element slice of a Q4_K super-block.
__device__ __forceinline__ float tq_dot_q4_K(const void* __restrict__ vbq,
                                             const block_q8_1* __restrict__ bq8_1,
                                             int kbx, int iqs) {
    const block_q4_K* bq4_K = (const block_q4_K*)vbq + kbx;

    // iqs is even, 0..30. bq8_offset selects which pair of activation blocks.
    const int bq8_offset = QR4_K * ((iqs / 2) / (QI8_1 / 2));

    const int* q4 = (const int*)(bq4_K->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
    int v[2];
    v[0] = q4[0];
    v[1] = q4[4];

    // Unpack the 6-bit scale/min pair for this slice out of the 12 packed bytes.
    const uint16_t* scales = (const uint16_t*)bq4_K->scales;
    uint16_t aux[2];
    const int j = bq8_offset / 2;
    if (j < 2) {
        aux[0] = scales[j + 0] & 0x3f3f;
        aux[1] = scales[j + 2] & 0x3f3f;
    } else {
        aux[0] = ((scales[j + 2] >> 0) & 0x0f0f) | ((scales[j - 2] & 0xc0c0) >> 2);
        aux[1] = ((scales[j + 2] >> 4) & 0x0f0f) | ((scales[j - 0] & 0xc0c0) >> 2);
    }
    const uint8_t* sc = (const uint8_t*)aux;
    const uint8_t* m = sc + 2;

    int u[2 * QR4_K];
    float d8[QR4_K];
#pragma unroll
    for (int i = 0; i < QR4_K; ++i) {
        const block_q8_1* bq8i = bq8_1 + bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);
        const int* q8 = (const int*)bq8i->qs + ((iqs / 2) % 4);
        u[2 * i + 0] = q8[0];
        u[2 * i + 1] = q8[4];
    }

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < QR4_K; ++i) {
        const int v0i = (v[0] >> (4 * i)) & 0x0F0F0F0F;
        const int v1i = (v[1] >> (4 * i)) & 0x0F0F0F0F;

        const int dot1 = __dp4a(v1i, u[2 * i + 1], __dp4a(v0i, u[2 * i + 0], 0));
        // Dotting against all-ones sums the activation quants, which is what
        // the per-group minimum has to be multiplied by.
        const int dot2 =
            __dp4a(0x01010101, u[2 * i + 1], __dp4a(0x01010101, u[2 * i + 0], 0));

        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);
    }

    // ggml keeps these as one `half2`; ours are two `__half` at the same
    // offsets, so read them individually.
    return __half2float(bq4_K->d) * sumf_d - __half2float(bq4_K->dmin) * sumf_m;
}

// One 32-weight quarter of a Q4_G128 block: exactly what a Q8_1 activation
// block covers, so `iqs` selects both at once.
//
//   iqs 0 -> low nibbles of bytes  0..31   (weights   0.. 31)
//   iqs 1 -> low nibbles of bytes 32..63   (weights  32.. 63)
//   iqs 2 -> high nibbles of bytes  0..31  (weights  64.. 95)
//   iqs 3 -> high nibbles of bytes 32..63  (weights  96..127)
//
// The zero point arrives already multiplied by the scale, so it applies to the
// activation block's own sum rather than to each weight — one `dp4a` against
// all-ones instead of 32 subtractions.
__device__ __forceinline__ float tq_dot_q4_g128(const void* __restrict__ vbq,
                                                const block_q8_1* __restrict__ bq8_1,
                                                int kbx, int iqs) {
    const block_q4_g128* bq = (const block_q4_g128*)vbq + kbx;
    const int* q4 = (const int*)(bq->qs + 32 * (iqs % 2));
    const int shift = 4 * (iqs / 2);

    const block_q8_1* bq8 = bq8_1 + iqs;
    const int* q8 = (const int*)bq8->qs;

    int dot = 0;
    int sum = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int v = (q4[i] >> shift) & 0x0F0F0F0F;
        dot = __dp4a(v, q8[i], dot);
        // Dotting the activations against all-ones sums them, which is what
        // the folded zero point multiplies.
        sum = __dp4a(0x01010101, q8[i], sum);
    }

    const float d8 = __low2float(bq8->ds);
    return d8 * (__low2float(bq->ds) * dot - __high2float(bq->ds) * sum);
}

/// One 8-element slice of a Q6_K super-block.
__device__ __forceinline__ float tq_dot_q6_K(const void* __restrict__ vbq,
                                             const block_q8_1* __restrict__ bq8_1,
                                             int kbx, int iqs) {
    const block_q6_K* bq6_K = (const block_q6_K*)vbq + kbx;

    const int bq8_offset =
        2 * QR6_K * (iqs / (QI6_K / 2)) + (iqs % (QI6_K / 2)) / (QI6_K / 4);
    const int scale_offset =
        (QI6_K / 4) * (iqs / (QI6_K / 2)) + (iqs % (QI6_K / 2)) / (QI6_K / 8);
    const int vh_shift = 2 * ((iqs % (QI6_K / 2)) / (QI6_K / 4));

    const int vl = tq_get_int_b2(bq6_K->ql, iqs);
    const int vh =
        tq_get_int_b2(bq6_K->qh,
                      (QI6_K / 4) * (iqs / (QI6_K / 2)) + iqs % (QI6_K / 4)) >>
        vh_shift;

    const int8_t* scales = bq6_K->scales + scale_offset;

    int u[QR6_K];
    float d8[QR6_K];
#pragma unroll
    for (int i = 0; i < QR6_K; ++i) {
        u[i] = tq_get_int_b4(bq8_1[bq8_offset + 2 * i].qs, iqs % QI8_1);
        d8[i] = __low2float(bq8_1[bq8_offset + 2 * i].ds);
    }

    float sumf = 0.0f;
#pragma unroll
    for (int i = 0; i < QR6_K; ++i) {
        const int sc = scales[4 * i];
        const int vil = (vl >> (4 * i)) & 0x0F0F0F0F;
        const int vih = ((vh >> (4 * i)) << 4) & 0x30303030;
        // Six-bit quants are stored biased; __vsubss4 unbiases four at once.
        const int vi = __vsubss4((vil | vih), 0x20202020);
        sumf += d8[i] * (__dp4a(vi, u[i], 0) * sc);
    }

    return __half2float(bq6_K->d) * sumf;
}

// ---- launcher -----------------------------------------------------------
//
// One block per output row, threads striding over the row's slices — the same
// shape as the float mat-vec in `quant.cu`, so only the inner product changes.
// `slices_per_block` is how many `iqs` values cover one weight block, and
// `elems_per_slice` how many weights each call retires.

#define MMVQ_KERNEL(NAME, DOT, SLICES, IQS_STEP, BLOCK_ELEMS)                  \
    extern "C" __global__ void NAME(float* __restrict__ out,                   \
                                    const void* __restrict__ w,                \
                                    const block_q8_1* __restrict__ y, int k,   \
                                    int n, int accum) {                        \
        const int row = blockIdx.x;                                            \
        if (row >= n) return;                                                  \
        const int nb = k / (BLOCK_ELEMS);                                      \
        const int wn = n;                                                      \
        const char* wbase = (const char*)w;                                    \
        const char* wr = (const char*)w + (size_t)row * nb * WEIGHT_STRIDE;    \
                                                                               \
        float acc = 0.0f;                                                      \
        for (int c = threadIdx.x; c < nb * (SLICES); c += blockDim.x) {        \
            const int kbx = c / (SLICES);                                      \
            const int iqs = (c % (SLICES)) * (IQS_STEP);                       \
            acc += DOT(wr, y + kbx * ((BLOCK_ELEMS) / QK8_1), kbx, iqs);       \
        }                                                                      \
                                                                               \
        const float total = block_reduce_sum(acc);                             \
        /* `accum` folds the residual add into the projection that feeds it:  */\
        /* the output and down projections are the two whose result goes      */\
        /* straight back into the residual stream, and doing it here saves a  */\
        /* kernel and three passes over the vector per layer.                 */\
        if (threadIdx.x == 0) out[row] = accum ? out[row] + total : total;     \
    }

// Q8_0: 32 weights per block, 4 slices of 8, iqs = 0,2,4,6.
#define WEIGHT_STRIDE (int)sizeof(block_q8_0)
MMVQ_KERNEL(mmvq_q8_0, tq_dot_q8_0, 4, 2, 32)
#undef WEIGHT_STRIDE

// Q4_K: 256 weights per super-block, 16 slices of 16, iqs = 0,2,..,30.
#define WEIGHT_STRIDE (int)sizeof(block_q4_K)
MMVQ_KERNEL(mmvq_q4_K, tq_dot_q4_K, 16, 2, 256)
#undef WEIGHT_STRIDE

// Q6_K: 256 weights per super-block, 32 slices of 8, iqs = 0..31.
#define WEIGHT_STRIDE (int)sizeof(block_q6_K)
MMVQ_KERNEL(mmvq_q6_K, tq_dot_q6_K, 32, 1, 256)
#undef WEIGHT_STRIDE

// The transposed Q4_G128 layout, same dot product.
//
// The quants for the whole matrix come first — `n * nb` blocks of 64 bytes —
// then the scales, and inside each 64-byte run the 4x4 matrix of 4-byte words
// is transposed — which is what
// makes a tensor-core lane's fragment one aligned 16-byte read. The mat-vec
// gains nothing from that and loses nothing either: it reads the same eight
// words per quarter, at computed offsets rather than consecutive ones.
//
// Quarter `iqs` is run `iqs % 2` and nibble half `iqs / 2` as before. Its word
// `t` held elements `4t .. 4t+3`, which the transpose moved to
// `(t % 4) * 16 + (run * 2 + t / 4) * 4`.
__device__ __forceinline__ float tq_dot_q4_g128t(const void* __restrict__ base,
                                                 int row, int nb, int nw,
                                                 const block_q8_1* __restrict__ bq8_1,
                                                 int kbx, int iqs) {
    const uint8_t* q = (const uint8_t*)base;
    const uint8_t* qs = q + ((size_t)row * nb + kbx) * 64;
    const __half2* sc = (const __half2*)(const void*)(q + (size_t)nw * nb * 64);
    const __half2 ds = sc[(size_t)row * nb + kbx];
    const int run = iqs % 2;
    const int shift = 4 * (iqs / 2);

    const block_q8_1* bq8 = bq8_1 + iqs;
    const int* q8 = (const int*)bq8->qs;

    int dot = 0;
    int sum = 0;
#pragma unroll
    for (int t = 0; t < 8; ++t) {
        const int off = (t % 4) * 16 + (run * 2 + t / 4) * 4;
        /* Bytes 1 and 2 swap in the pack, so undo that before the `dp4a`:
           the activation quarter is in element order and this is not. */
        const int w = __byte_perm(*(const int*)(const void*)(qs + off), 0,
                                  0x3120);
        const int v = (w >> shift) & 0x0F0F0F0Fu;
        dot = __dp4a(v, q8[t], dot);
        sum = __dp4a(0x01010101, q8[t], sum);
    }
    const float d8 = __low2float(bq8->ds);
    return d8 * (__low2float(ds) * dot - __high2float(ds) * sum);
}

// The macros hand a DOT four arguments and the transposed layout needs the
// block count too, so bind it here: the row base and `nb` are the same two
// numbers every caller already has.
#define TQ_DOT_G128T(WR, Y, KBX, IQS)                                          \
    tq_dot_q4_g128t(wbase, row, nb, wn, Y, KBX, IQS)

// Q4_G128: 128 weights per block, 4 slices of 32, iqs = 0..3.
#define WEIGHT_STRIDE (int)sizeof(block_q4_g128)
MMVQ_KERNEL(mmvq_q4_g128, tq_dot_q4_g128, 4, 1, 128)
MMVQ_KERNEL(mmvq_q4_g128t, TQ_DOT_G128T, 4, 1, 128)
#undef WEIGHT_STRIDE

// One warp per output row, several rows per block. Measured level; off by
// default.
//
// The block-per-row shape above gives each of its 256 threads a single slice —
// nine bytes of Q4_K — and then spends a 256-thread reduction with three
// barriers to add them up. A warp per row instead gives each lane eight slices,
// reduces with five shuffles and no barrier at all, and reads eight contiguous
// rows per block rather than one: 18 KB of streaming instead of 2.3 KB. Since
// the mat-vecs are 95% of a decode step, that looked like the step.
//
// It is not. Sweeping 2, 4, 8 and 16 rows against the block-per-row shape, each
// setting measured twice in opposite order so the card's thermal drift cancels,
// every one lands between 16.20 and 16.35 ms for the same 4.62 GB — under one
// percent apart, inside the noise. The kernel is waiting on memory the whole
// time, so what the threads do between loads does not show up. Kept behind
// `TUILI_MMVQ_ROWS` because the negative result is worth being able to re-run.

#define MMVQ_WARP_KERNEL(NAME, DOT, SLICES, IQS_STEP, BLOCK_ELEMS)             \
    extern "C" __global__ void NAME(float* __restrict__ out,                   \
                                    const void* __restrict__ w,                \
                                    const block_q8_1* __restrict__ y, int k,   \
                                    int n, int accum) {                        \
        const int lane = threadIdx.x;                                          \
        const int row = blockIdx.x * blockDim.y + threadIdx.y;                 \
        if (row >= n) return;                                                  \
        const int nb = k / (BLOCK_ELEMS);                                      \
        const int wn = n;                                                      \
        const char* wbase = (const char*)w;                                    \
        const char* wr = (const char*)w + (size_t)row * nb * WEIGHT_STRIDE;    \
                                                                               \
        float acc = 0.0f;                                                      \
        for (int c = lane; c < nb * (SLICES); c += WARP_SIZE) {                \
            const int kbx = c / (SLICES);                                      \
            const int iqs = (c % (SLICES)) * (IQS_STEP);                       \
            acc += DOT(wr, y + kbx * ((BLOCK_ELEMS) / QK8_1), kbx, iqs);       \
        }                                                                      \
                                                                               \
        acc = warp_reduce_sum(acc);                                            \
        if (lane == 0) out[row] = accum ? out[row] + acc : acc;                \
    }

#define WEIGHT_STRIDE (int)sizeof(block_q8_0)
MMVQ_WARP_KERNEL(mmvqw_q8_0, tq_dot_q8_0, 4, 2, 32)
#undef WEIGHT_STRIDE

#define WEIGHT_STRIDE (int)sizeof(block_q4_K)
MMVQ_WARP_KERNEL(mmvqw_q4_K, tq_dot_q4_K, 16, 2, 256)
#undef WEIGHT_STRIDE

#define WEIGHT_STRIDE (int)sizeof(block_q6_K)
MMVQ_WARP_KERNEL(mmvqw_q6_K, tq_dot_q6_K, 32, 1, 256)
#undef WEIGHT_STRIDE

#define WEIGHT_STRIDE (int)sizeof(block_q4_g128)
MMVQ_WARP_KERNEL(mmvqw_q4_g128, tq_dot_q4_g128, 4, 1, 128)
MMVQ_WARP_KERNEL(mmvqw_q4_g128t, TQ_DOT_G128T, 4, 1, 128)
#undef WEIGHT_STRIDE

// ---- fused mat-vec over matrices sharing one activation -----------------
//
// A decode step issues two hundred and twenty-five of these, and back to back
// they reach 328 GB/s where one measured alone reaches 392. The difference is
// not launch cost — putting the whole sequence in a CUDA graph changes nothing
// — it is that each kernel drains before the next can start, so every one of
// them ends with the machine emptying out.
//
// Three of the seven matrices in a layer read the same activation (Q, K and V),
// and two more share another (the FFN's gate and up). Merging each group into a
// single grid removes ninety-six of those drains per step and gives the
// remaining kernels a wider grid to tail off with. It is the same fusion vLLM
// and llama.cpp do, for the same reason.
//
// `blockIdx.x` runs over the concatenated rows; the branch that picks the
// matrix is uniform within a block. Every matrix in a group must share `k` and
// the weight type — in a Q4_K_M file the first layer's V projection is Q6_K
// while its siblings are Q4_K, so the caller checks before fusing.

// The dot product itself, once, for both arities below.
#define MMVQ_FUSED_BODY(DOT, SLICES, IQS_STEP, BLOCK_ELEMS)                    \
    const int nb = k / (BLOCK_ELEMS);                                          \
    const char* wr = w + (size_t)row * nb * WEIGHT_STRIDE;                     \
    float acc = 0.0f;                                                          \
    for (int c = threadIdx.x; c < nb * (SLICES); c += blockDim.x) {            \
        const int kbx = c / (SLICES);                                          \
        const int iqs = (c % (SLICES)) * (IQS_STEP);                           \
        acc += DOT(wr, y + kbx * ((BLOCK_ELEMS) / QK8_1), kbx, iqs);           \
    }                                                                          \
    const float total = block_reduce_sum(acc);                                 \
    if (threadIdx.x == 0) out[row] = total;

#define MMVQ_FUSED2(NAME, DOT, SLICES, IQS_STEP, BLOCK_ELEMS)                  \
    extern "C" __global__ void NAME(                                           \
        float* __restrict__ out0, float* __restrict__ out1,                    \
        const void* __restrict__ w0, const void* __restrict__ w1,              \
        const block_q8_1* __restrict__ y, int k, int n0, int n1) {             \
        int row = blockIdx.x;                                                  \
        float* out = out0;                                                     \
        const char* w = (const char*)w0;                                       \
        int wn = n0;                                                           \
        if (row >= n0) {                                                       \
            if (row >= n0 + n1) return;                                        \
            row -= n0;                                                         \
            out = out1;                                                        \
            w = (const char*)w1;                                               \
            wn = n1;                                                           \
        }                                                                      \
        const char* wbase = w;                                                 \
        MMVQ_FUSED_BODY(DOT, SLICES, IQS_STEP, BLOCK_ELEMS)                    \
    }

#define MMVQ_FUSED3(NAME, DOT, SLICES, IQS_STEP, BLOCK_ELEMS)                  \
    extern "C" __global__ void NAME(                                           \
        float* __restrict__ out0, float* __restrict__ out1,                    \
        float* __restrict__ out2, const void* __restrict__ w0,                 \
        const void* __restrict__ w1, const void* __restrict__ w2,              \
        const block_q8_1* __restrict__ y, int k, int n0, int n1, int n2) {     \
        int row = blockIdx.x;                                                  \
        float* out = out0;                                                     \
        const char* w = (const char*)w0;                                       \
        int wn = n0;                                                           \
        if (row >= n0 + n1) {                                                  \
            if (row >= n0 + n1 + n2) return;                                   \
            row -= n0 + n1;                                                    \
            out = out2;                                                        \
            w = (const char*)w2;                                               \
            wn = n2;                                                           \
        } else if (row >= n0) {                                                \
            row -= n0;                                                         \
            out = out1;                                                        \
            w = (const char*)w1;                                               \
            wn = n1;                                                           \
        }                                                                      \
        const char* wbase = w;                                                 \
        MMVQ_FUSED_BODY(DOT, SLICES, IQS_STEP, BLOCK_ELEMS)                    \
    }

#define WEIGHT_STRIDE (int)sizeof(block_q8_0)
MMVQ_FUSED2(mmvqf2_q8_0, tq_dot_q8_0, 4, 2, 32)
MMVQ_FUSED3(mmvqf3_q8_0, tq_dot_q8_0, 4, 2, 32)
#undef WEIGHT_STRIDE

#define WEIGHT_STRIDE (int)sizeof(block_q4_K)
MMVQ_FUSED2(mmvqf2_q4_K, tq_dot_q4_K, 16, 2, 256)
MMVQ_FUSED3(mmvqf3_q4_K, tq_dot_q4_K, 16, 2, 256)
#undef WEIGHT_STRIDE

#define WEIGHT_STRIDE (int)sizeof(block_q6_K)
MMVQ_FUSED2(mmvqf2_q6_K, tq_dot_q6_K, 32, 1, 256)
MMVQ_FUSED3(mmvqf3_q6_K, tq_dot_q6_K, 32, 1, 256)
#undef WEIGHT_STRIDE

#define WEIGHT_STRIDE (int)sizeof(block_q4_g128)
MMVQ_FUSED2(mmvqf2_q4_g128, tq_dot_q4_g128, 4, 1, 128)
MMVQ_FUSED2(mmvqf2_q4_g128t, TQ_DOT_G128T, 4, 1, 128)
MMVQ_FUSED3(mmvqf3_q4_g128, tq_dot_q4_g128, 4, 1, 128)
MMVQ_FUSED3(mmvqf3_q4_g128t, TQ_DOT_G128T, 4, 1, 128)
#undef WEIGHT_STRIDE

// ---- multi-token mat-vec ------------------------------------------------
//
// The single-token kernel above reaches 375 GB/s on this card, 93% of what a
// pure streaming read achieves, and the tensor-core GEMM that replaces it at
// batch reaches 94. The difference is not the tensor cores: it is that this
// kernel streams weights global -> registers -> dp4a and never stages them
// through shared memory at all.
//
// So rather than make the GEMM stage better, amortize this kernel over T
// tokens. The weight loads inside the dot product depend only on (row, kbx,
// iqs), all invariant in t, so unrolling the token loop lets the compiler hoist
// the decode and spend it on T activations. The arithmetic cost is real but
// small: at 32 tokens `ffn_gate` needs 3.75 GOP against roughly 76 TOPS of dp4a
// throughput, about 50 microseconds, against 13 ms of weight reading.
//
// This is the same trade `GEMV_SPREAD` makes in the float path, where it was
// worth 8x.

#define MMVQ_T_KERNEL(NAME, DOT, SLICES, IQS_STEP, BLOCK_ELEMS, T)             \
    extern "C" __global__ void NAME(float* __restrict__ out,                   \
                                    const void* __restrict__ w,                \
                                    const block_q8_1* __restrict__ y, int k,   \
                                    int n, int n_tokens) {                     \
        const int row = blockIdx.x;                                            \
        if (row >= n) return;                                                  \
        const int nb = k / (BLOCK_ELEMS);                                      \
        const int ny = k / QK8_1; /* Q8_1 blocks per token */                  \
        const int wn = n;                                                      \
        const char* wbase = (const char*)w;                                    \
        const char* wr = (const char*)w + (size_t)row * nb * WEIGHT_STRIDE;    \
        const int tok0 = blockIdx.y * (T);                                     \
                                                                               \
        const int tmax = max(0, min((T) - 1, n_tokens - 1 - tok0));            \
        float acc[T];                                                          \
        _Pragma("unroll") for (int t = 0; t < (T); ++t) acc[t] = 0.0f;          \
                                                                               \
        for (int c = threadIdx.x; c < nb * (SLICES); c += blockDim.x) {        \
            const int kbx = c / (SLICES);                                      \
            const int iqs = (c % (SLICES)) * (IQS_STEP);                       \
            const block_q8_1* yb = y + (size_t)tok0 * ny                       \
                                 + kbx * ((BLOCK_ELEMS) / QK8_1);              \
            /* No bounds test inside the unrolled body: a branch per token      \
               stops the compiler hoisting the weight decode, which is the      \
               entire point of the loop. Out-of-range tokens read a clamped     \
               row and their results are discarded at the store. */             \
            _Pragma("unroll") for (int t = 0; t < (T); ++t) {                   \
                const size_t off = (size_t)min(t, tmax) * ny;                  \
                acc[t] += DOT(wr, yb + off, kbx, iqs);                          \
            }                                                                  \
        }                                                                      \
                                                                               \
        /* One barrier for all T reductions rather than T block-wide ones. */  \
        __shared__ float red[T][WARP_SIZE];                                    \
        const int lane = threadIdx.x % WARP_SIZE;                              \
        const int warp = threadIdx.x / WARP_SIZE;                              \
        const int warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;            \
        _Pragma("unroll") for (int t = 0; t < (T); ++t) {                       \
            const float v = warp_reduce_sum(acc[t]);                           \
            if (lane == 0) red[t][warp] = v;                                   \
        }                                                                      \
        __syncthreads();                                                       \
        if (threadIdx.x < (T) && tok0 + (int)threadIdx.x < n_tokens) {         \
            float s = 0.0f;                                                    \
            for (int wi = 0; wi < warps; ++wi) s += red[threadIdx.x][wi];      \
            out[(size_t)(tok0 + threadIdx.x) * n + row] = s;                   \
        }                                                                      \
    }

#define WEIGHT_STRIDE (int)sizeof(block_q8_0)
MMVQ_T_KERNEL(mmvqt1_q8_0, tq_dot_q8_0, 4, 2, 32, 1)
MMVQ_T_KERNEL(mmvqt2_q8_0, tq_dot_q8_0, 4, 2, 32, 2)
MMVQ_T_KERNEL(mmvqt4_q8_0, tq_dot_q8_0, 4, 2, 32, 4)
MMVQ_T_KERNEL(mmvqt8_q8_0, tq_dot_q8_0, 4, 2, 32, 8)
MMVQ_T_KERNEL(mmvqt16_q8_0, tq_dot_q8_0, 4, 2, 32, 16)
#undef WEIGHT_STRIDE

#define WEIGHT_STRIDE (int)sizeof(block_q4_K)
MMVQ_T_KERNEL(mmvqt1_q4_K, tq_dot_q4_K, 16, 2, 256, 1)
MMVQ_T_KERNEL(mmvqt2_q4_K, tq_dot_q4_K, 16, 2, 256, 2)
MMVQ_T_KERNEL(mmvqt4_q4_K, tq_dot_q4_K, 16, 2, 256, 4)
MMVQ_T_KERNEL(mmvqt8_q4_K, tq_dot_q4_K, 16, 2, 256, 8)
MMVQ_T_KERNEL(mmvqt16_q4_K, tq_dot_q4_K, 16, 2, 256, 16)
#undef WEIGHT_STRIDE

#define WEIGHT_STRIDE (int)sizeof(block_q6_K)
MMVQ_T_KERNEL(mmvqt1_q6_K, tq_dot_q6_K, 32, 1, 256, 1)
MMVQ_T_KERNEL(mmvqt2_q6_K, tq_dot_q6_K, 32, 1, 256, 2)
MMVQ_T_KERNEL(mmvqt4_q6_K, tq_dot_q6_K, 32, 1, 256, 4)
MMVQ_T_KERNEL(mmvqt8_q6_K, tq_dot_q6_K, 32, 1, 256, 8)
MMVQ_T_KERNEL(mmvqt16_q6_K, tq_dot_q6_K, 32, 1, 256, 16)
#undef WEIGHT_STRIDE

#define WEIGHT_STRIDE (int)sizeof(block_q4_g128)
MMVQ_T_KERNEL(mmvqt1_q4_g128, tq_dot_q4_g128, 4, 1, 128, 1)
MMVQ_T_KERNEL(mmvqt1_q4_g128t, TQ_DOT_G128T, 4, 1, 128, 1)
MMVQ_T_KERNEL(mmvqt2_q4_g128, tq_dot_q4_g128, 4, 1, 128, 2)
MMVQ_T_KERNEL(mmvqt2_q4_g128t, TQ_DOT_G128T, 4, 1, 128, 2)
MMVQ_T_KERNEL(mmvqt4_q4_g128, tq_dot_q4_g128, 4, 1, 128, 4)
MMVQ_T_KERNEL(mmvqt4_q4_g128t, TQ_DOT_G128T, 4, 1, 128, 4)
MMVQ_T_KERNEL(mmvqt8_q4_g128, tq_dot_q4_g128, 4, 1, 128, 8)
MMVQ_T_KERNEL(mmvqt8_q4_g128t, TQ_DOT_G128T, 4, 1, 128, 8)
MMVQ_T_KERNEL(mmvqt16_q4_g128, tq_dot_q4_g128, 4, 1, 128, 16)
MMVQ_T_KERNEL(mmvqt16_q4_g128t, TQ_DOT_G128T, 4, 1, 128, 16)
#undef WEIGHT_STRIDE

// RMS norm that also emits the Q8_1 form of its own output.
//
// Every projection group is `rms_norm` followed immediately by
// `quantize_q8_1` reading back what the norm just wrote. Two launches and two
// trips through the normalized vector where one will do. At a batch of one
// these are latency, not bandwidth: 64 pairs per step, a few microseconds each,
// against a 15.5 ms step whose weight reading alone is 13 ms.
//
// The block computes the norm exactly as `rms_norm_f32` does, writes the f32
// result — the float mat-vec and the vocab projection still want it — and then
// each warp quantizes whole 32-element groups out of the values it just wrote.
// Registers each thread holds; the host picks a block size so that
// `blockDim.x * RMS_REGS >= d`.
#define RMS_REGS 8

// The same fusion, writing f16 instead of Q8_1.
//
// The f16-operand GEMM is the Q4_G128 default now, and it takes activations
// unquantized — so the Q8_1 half of `rms_norm_q8_1_f32` became a buffer nobody
// reads, and every projection then paid a separate `to_f16` over the same row.
// The profile put those at 1.8% and 3.6% of a batch-32 step: 5.4% spent
// producing an activation in the wrong format and then converting it.
//
// One row, one pass, both outputs. The projections that share an input — q, k,
// v and gate, up — share this too, which is what `matmul_pre`'s
// `pre_quantized` already arranges for the Q8_1 path.
// The residual add and the norm that always follows it, in one pass.
//
// A decode step does this twice a layer: `x += sublayer_out`, then normalize
// `x` into the next projection's operand. Separately that is two launches and
// two round trips over a 512 KB residual — 64 launches and 32 MB a step at a
// batch of 32 — where the row is already in registers here between the two.
// vLLM fuses the same pair (`triton_red_fused_fused_add_rms_norm` in its
// trace).
//
// `x` is updated in place because the residual stream is what the *next*
// sublayer adds to; only the normalized copies are new.
extern "C" __global__ void add_rms_norm_f16_f32(float* __restrict__ out,
                                                __half* __restrict__ hout,
                                                float* __restrict__ x,
                                                const float* __restrict__ b,
                                                const float* __restrict__ weight,
                                                int d, float eps) {
    const int token = blockIdx.x;
    float* row = x + (size_t)token * d;
    const float* brow = b + (size_t)token * d;
    float* orow = out + (size_t)token * d;
    __half* hrow = hout + (size_t)token * d;
    const int tid = threadIdx.x;

    float v[RMS_REGS];
    float acc = 0.0f;
#pragma unroll
    for (int k = 0; k < RMS_REGS; ++k) {
        const int i = k * blockDim.x + tid;
        v[k] = 0.0f;
        if (i < d) {
            v[k] = row[i] + brow[i];
            row[i] = v[k];
        }
        acc += v[k] * v[k];
    }
    const float scale = rsqrtf(block_reduce_sum(acc) / (float)d + eps);

#pragma unroll
    for (int k = 0; k < RMS_REGS; ++k) {
        const int i = k * blockDim.x + tid;
        if (i < d) {
            v[k] *= scale * weight[i];
            orow[i] = v[k];
            hrow[i] = __float2half(v[k]);
        }
    }
}

extern "C" __global__ void rms_norm_f16_f32(float* __restrict__ out,
                                            __half* __restrict__ hout,
                                            const float* __restrict__ x,
                                            const float* __restrict__ weight,
                                            int d, float eps) {
    const int token = blockIdx.x;
    const float* row = x + (size_t)token * d;
    float* orow = out + (size_t)token * d;
    __half* hrow = hout + (size_t)token * d;
    const int tid = threadIdx.x;

    float v[RMS_REGS];
    float acc = 0.0f;
#pragma unroll
    for (int k = 0; k < RMS_REGS; ++k) {
        const int i = k * blockDim.x + tid;
        v[k] = (i < d) ? row[i] : 0.0f;
        acc += v[k] * v[k];
    }
    const float scale = rsqrtf(block_reduce_sum(acc) / (float)d + eps);

#pragma unroll
    for (int k = 0; k < RMS_REGS; ++k) {
        const int i = k * blockDim.x + tid;
        if (i < d) {
            v[k] *= scale * weight[i];
            orow[i] = v[k];
            hrow[i] = __float2half(v[k]);
        }
    }
}

extern "C" __global__ void rms_norm_q8_1_f32(float* __restrict__ out,
                                             block_q8_1* __restrict__ qout,
                                             const float* __restrict__ x,
                                             const float* __restrict__ weight,
                                             int d, float eps) {
    const int token = blockIdx.x;
    const float* row = x + (size_t)token * d;
    float* orow = out + (size_t)token * d;
    const int tid = threadIdx.x;

    // The row stays in registers across all three phases. Reading it back from
    // global for the scale, and again for the quantisation, is what made this
    // kernel cost more than the two it replaced: one block has to walk the
    // whole row on its own, so every extra pass is the full latency again.
    float v[RMS_REGS];
    float acc = 0.0f;
#pragma unroll
    for (int k = 0; k < RMS_REGS; ++k) {
        const int i = k * blockDim.x + tid;
        v[k] = (i < d) ? row[i] : 0.0f;
        acc += v[k] * v[k];
    }
    const float scale = rsqrtf(block_reduce_sum(acc) / (float)d + eps);

#pragma unroll
    for (int k = 0; k < RMS_REGS; ++k) {
        const int i = k * blockDim.x + tid;
        if (i < d) {
            v[k] *= scale * weight[i];
            orow[i] = v[k];
        }
    }

    // Q8_1 block `b` covers elements 32b..32b+31, and with a stride of
    // blockDim.x those land in one warp's lane 0..31 at register slot
    // 32b / blockDim.x. So the layout the strided load already produced is
    // exactly the one the per-block scale wants — no shared memory, no
    // barrier, and no re-read.
    const int lane = tid % WARP_SIZE;
    const int warp = tid / WARP_SIZE;
    const int warps = blockDim.x / WARP_SIZE;
    const int n_blocks = d / QK8_1;
    block_q8_1* qrow = qout + (size_t)token * n_blocks;
#pragma unroll
    for (int k = 0; k < RMS_REGS; ++k) {
        const int b = k * warps + warp;
        if (b >= n_blocks) continue;
        const float amax = warp_reduce_max(fabsf(v[k]));
        const float sum = warp_reduce_sum(v[k]);
        const float dq = amax / 127.0f;
        qrow[b].qs[lane] = (amax == 0.0f) ? 0 : (int8_t)roundf(v[k] / dq);
        if (lane == 0) {
            qrow[b].ds = __floats2half2_rn(dq, sum);
        }
    }
}
