// Quantized mat-vec on Metal: the ggml block layouts and the kernels that
// decode them in place. The twin of the relevant half of `cu/quant.cu`.
//
// Weights stay in their GGUF block encoding on the device and are decoded
// inside the kernel that consumes them, which is the whole reason a 27B fits in
// 17.6 GB. Field order and padding must match ggml-common.h exactly: these
// structs are reinterpreted straight from the mapped file.

#define QK8_0 32
#define QK_K 256
#define K_SCALE_SIZE 12

typedef struct {
    half d;                 // scale
    char qs[QK8_0];
} block_q8_0;

typedef struct {
    half d;                         // super-block scale for the 6-bit scales
    half dmin;                      // super-block scale for the 6-bit mins
    uchar scales[K_SCALE_SIZE];     // 8 pairs of 6-bit scale/min
    uchar qs[QK_K / 2];             // 4-bit quants
} block_q4_K;

typedef struct {
    uchar ql[QK_K / 2];         // lower 4 bits
    uchar qh[QK_K / 4];         // upper 2 bits
    char scales[QK_K / 16];     // 8-bit block scales
    half d;                     // super-block scale
} block_q6_K;

/// Unpack the 6-bit scale/min pair `j` (0..7) out of a Q4_K super-block.
inline void q4k_scale_min(device const uchar* q, int j,
                          thread uchar* d, thread uchar* m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

// Tokens a single threadgroup serves. A group decodes each weight element once
// and spends it on every token it holds, so a `t`-token batch cuts the weight
// traffic by this factor -- which matters most for the vocab projection, the
// largest matrix in the model.
// Tokens a single threadgroup serves. A group decodes each weight element once
// and spends it on every token it holds, so a `t`-token batch cuts the weight
// traffic by this factor -- which matters most for the vocab projection, the
// largest matrix in the model.
#define GEMV_TOKENS 8

// The macros take the token count as a parameter so each mat-vec can be
// instantiated twice: once batched and once specialised for a single token.
//
// Why that is worth two kernels rather than one. `GEMV_SPREAD`'s trip count has
// to be a compile-time constant -- a runtime bound leaves the compiler unable to
// prove the indices and `acc` lands in device memory, which for a kernel that is
// otherwise pure streaming costs an order of magnitude. So the batched form runs
// eight iterations with a predicate inside, and at one token seven of them are
// dead: the predicate is false, but the loop, the compare and the address
// arithmetic are still emitted. Decode is *always* one token, and decode is the
// case this engine exists to serve.
#define GEMV_PROLOGUE(T)                                                       \
    BLOCK_REDUCE_SCRATCH                                                       \
    const int row = int(tgid.x);                                               \
    if (row >= n) return;                                                      \
    const int token0 = int(tgid.y) * (T);                                      \
    const int ntok = min((T), n_tokens - token0);                              \
    float acc[(T)];                                                            \
    for (int t = 0; t < (T); ++t) acc[t] = 0.0f;

/// Spend one decoded weight element on every token this group holds.
///
/// The weight is bound to a local *before* the token loop, and that is the whole
/// point of the braces. These are macros, so a weight expression passed in is
/// substituted textually -- written the obvious way, `d * float(nib) - mn` lands
/// inside `for (int t ...)` and the dequantisation runs once per token instead of
/// once per element. It measured: two tokens cost 1.7x one token on Q4_K, whose
/// dequantisation is a shift, a mask, a convert and a fused multiply-add, while
/// on Q8_0 -- one multiply -- the same two tokens cost 1.1x. A mat-vec reads its
/// weights once at any token count, so 1.1x is the shape this should have had all
/// along and 1.7x was the tell.
///
/// The compiler will not always hoist it for us: the value is used only under
/// `if (t < ntok)`, so lifting it out of the loop means speculating a
/// computation the source guards, and it declines.
#define GEMV_SPREAD(T, WV, I)                                                  \
    {                                                                          \
        const float wv_ = (WV);                                                \
        for (int t = 0; t < (T); ++t) {                                        \
            if (t < ntok) acc[t] += wv_ * x[size_t(token0 + t) * k + (I)];     \
        }                                                                      \
    }

/// The same for four consecutive weight elements, taking the activations as one
/// 16-byte load a token instead of four 4-byte ones.
///
/// This is where the mat-vec's time actually goes, which took a measurement to
/// believe. Two tokens cost 1.7x one token rather than 1.0x -- a mat-vec reads
/// its weights once whatever the token count, so if the weight read were the
/// cost the second token would have been nearly free. It is not: the per-token
/// work is the activation load and the multiply-add, and at 32 scalar loads a
/// thread a token that is the majority of the kernel. It is also why decoding
/// one token sits at 27% of this machine's streaming ceiling on Q4_K while
/// reading only 4.5 bits a weight, and why loading the *weights* as `uint4`
/// bought 5%.
///
/// `packed_float4` rather than `float4`: the packed types carry their scalar's
/// alignment, 4 bytes, so this is well-defined for any float-aligned `x` at any
/// index that is a multiple of four. `float4` would require 16 and the
/// activation is a view into a scratch buffer whose element offset this kernel
/// cannot see. Every caller's `I` here is a multiple of four by construction --
/// a Q4_K group starts at a multiple of 32 -- so no host-side guard is needed.
#define GEMV_SPREAD4(T, W0, W1, W2, W3, I)                                     \
    {                                                                          \
        const float4 wv_ = float4((W0), (W1), (W2), (W3));                     \
        for (int t = 0; t < (T); ++t) {                                        \
            if (t < ntok) {                                                    \
                device const packed_float4* xp = (device const packed_float4*)  \
                    (x + size_t(token0 + t) * k + (I));                        \
                const packed_float4 xv = *xp;                                  \
                acc[t] += wv_[0] * xv[0] + wv_[1] * xv[1]                      \
                        + wv_[2] * xv[2] + wv_[3] * xv[3];                     \
            }                                                                  \
        }                                                                      \
    }

#define GEMV_EPILOGUE(T)                                                       \
    for (int t = 0; t < (T); ++t) {                                            \
        if (t < ntok) {                                                        \
            const float total = BLOCK_SUM(acc[t], tid.x, tgdim.x);            \
            if (tid.x == 0) out[size_t(token0 + t) * n + row] = total;         \
        }                                                                      \
    }

#define GEMV_ARGS                                                              \
    device float* out          [[buffer(0)]],                                  \
    device const void* w       [[buffer(1)]],                                  \
    device const float* x      [[buffer(2)]],                                  \
    constant int& k            [[buffer(3)]],                                  \
    constant int& n            [[buffer(4)]],                                  \
    constant int& n_tokens     [[buffer(5)]],                                  \
    uint3 tgid  [[threadgroup_position_in_grid]],                              \
    uint3 tid   [[thread_position_in_threadgroup]],                            \
    uint3 tgdim [[threads_per_threadgroup]]

// ---- the bodies, one per encoding -----------------------------------------

#define GEMV_BODY_F32(T)                                                       \
    GEMV_PROLOGUE(T)                                                           \
    device const float* wr = (device const float*)w + size_t(row) * k;         \
    for (int i = int(tid.x); i < k; i += int(tgdim.x)) GEMV_SPREAD(T, wr[i], i)\
    GEMV_EPILOGUE(T)

#define GEMV_BODY_F16(T)                                                       \
    GEMV_PROLOGUE(T)                                                           \
    device const half* wr = (device const half*)w + size_t(row) * k;           \
    for (int i = int(tid.x); i < k; i += int(tgdim.x)) {                       \
        GEMV_SPREAD(T, float(wr[i]), i)                                        \
    }                                                                          \
    GEMV_EPILOGUE(T)

/// Q8_0: one f16 scale per 32 int8 quants.
///
/// Eight elements a thread, which is four threads to a 32-quant block rather
/// than one. The count is the same contract `add_assign` has, read the other
/// way: the host sizes the threadgroup from `WeightType::gemv_work_items`, which
/// says `k / 8` for this encoding. A body that took a whole block a thread was
/// self-bounding and therefore correct, and left 96 of 256 threads with nothing
/// to do at this model's `k = 5120` -- 37% of the group idle on the 288 Q8_0
/// tensors of a Qwen3.8 checkpoint.
#define Q8_0_PER_THREAD 8

#define GEMV_BODY_Q8_0(T)                                                      \
    GEMV_PROLOGUE(T)                                                           \
    const int nb = k / QK8_0;                                                  \
    const int per_block = QK8_0 / Q8_0_PER_THREAD;                             \
    const int chunks = nb * per_block;                                         \
    device const block_q8_0* wr =                                              \
        (device const block_q8_0*)w + size_t(row) * nb;                        \
    for (int c = int(tid.x); c < chunks; c += int(tgdim.x)) {                   \
        device const block_q8_0* blk = wr + c / per_block;                     \
        const int sub = (c % per_block) * Q8_0_PER_THREAD;                     \
        const int base = (c / per_block) * QK8_0 + sub;                        \
        const float d = float(blk->d);                                         \
        for (int i = 0; i < Q8_0_PER_THREAD; i += 4) {                         \
            GEMV_SPREAD4(T,                                                    \
                         d * float(blk->qs[sub + i + 0]),                      \
                         d * float(blk->qs[sub + i + 1]),                      \
                         d * float(blk->qs[sub + i + 2]),                      \
                         d * float(blk->qs[sub + i + 3]),                      \
                         base + i)                                             \
        }                                                                      \
    }                                                                          \
    GEMV_EPILOGUE(T)

/// Q4_K: a 256-element super-block with eight 6-bit scale/min pairs. One thread
/// a 32-element group, for the same reason as Q8_0.
#define GEMV_BODY_Q4_K(T)                                                      \
    GEMV_PROLOGUE(T)                                                           \
    const int nb = k / QK_K;                                                   \
    device const block_q4_K* wr =                                              \
        (device const block_q4_K*)w + size_t(row) * nb;                        \
    for (int c = int(tid.x); c < nb * 8; c += int(tgdim.x)) {                   \
        device const block_q4_K* blk = wr + c / 8;                             \
        const int g = c % 8;                                                   \
        const int base = (c / 8) * QK_K + g * 32;                              \
        uchar sc, m;                                                           \
        q4k_scale_min(blk->scales, g, &sc, &m);                                \
        const int high = g & 1;                                                \
        const float d = float(blk->d) * float(sc);                             \
        const float mn = float(blk->dmin) * float(m);                          \
        device const uint4* q128 =                                             \
            (device const uint4*)(device const void*)(blk->qs + (g / 2) * 32);  \
        for (int v = 0; v < 2; ++v) {                                          \
            const uint4 quad = q128[v];                                        \
            for (int wi = 0; wi < 4; ++wi) {                                   \
                const uint packed = quad[wi];                                  \
                const int s0 = high ? 4 : 0;                                   \
                GEMV_SPREAD4(                                                  \
                    T,                                                         \
                    d * float((packed >> s0) & 0xF) - mn,                      \
                    d * float((packed >> (s0 + 8)) & 0xF) - mn,                \
                    d * float((packed >> (s0 + 16)) & 0xF) - mn,               \
                    d * float((packed >> (s0 + 24)) & 0xF) - mn,               \
                    base + v * 16 + wi * 4)                                    \
            }                                                                  \
        }                                                                      \
    }                                                                          \
    GEMV_EPILOGUE(T)

/// Q6_K: 4 low bits in `ql`, 2 high bits in `qh`, an int8 scale per 16. One
/// thread a `l`, four output elements each, so consecutive threads read
/// adjacent bytes.
#define GEMV_BODY_Q6_K(T)                                                      \
    GEMV_PROLOGUE(T)                                                           \
    const int nb = k / QK_K;                                                   \
    device const block_q6_K* wr =                                              \
        (device const block_q6_K*)w + size_t(row) * nb;                        \
    const int chunks = nb * 64;                                                \
    for (int c = int(tid.x); c < chunks; c += int(tgdim.x)) {                   \
        const int b = c / 64;                                                  \
        const int rem = c % 64;                                                \
        const int n2 = rem / 32;                                               \
        const int l = rem % 32;                                                \
        device const block_q6_K* blk = wr + b;                                 \
        device const uchar* ql = blk->ql + n2 * 64;                            \
        device const uchar* qh = blk->qh + n2 * 32;                            \
        device const char* sc = blk->scales + n2 * 8;                          \
        const int base = b * QK_K + n2 * 128;                                  \
        const float d = float(blk->d);                                         \
        const uchar h = qh[l];                                                 \
        const int is = l / 16;                                                 \
        const int q0 = int((ql[l] & 0xF) | (((h >> 0) & 3) << 4)) - 32;        \
        const int q1 = int((ql[l + 32] & 0xF) | (((h >> 2) & 3) << 4)) - 32;   \
        const int q2 = int((ql[l] >> 4) | (((h >> 4) & 3) << 4)) - 32;         \
        const int q3 = int((ql[l + 32] >> 4) | (((h >> 6) & 3) << 4)) - 32;    \
        /* Not `GEMV_SPREAD4`: this thread's four elements are `l`, `l + 32`,  \
           `l + 64` and `l + 96`, which is how Q6_K packs its high bits -- one  \
           `qh` byte carries two bits for each of four elements 32 apart. They  \
           are 128 bytes apart in the activation, so there is no 16-byte load   \
           that covers them and vectorising would mean changing which element   \
           each thread owns. This is the vocabulary projection only, already at  \
           37% of ceiling, and it is one launch a step against 448. */          \
        GEMV_SPREAD(T, d * float(sc[is + 0]) * float(q0), base + l)            \
        GEMV_SPREAD(T, d * float(sc[is + 2]) * float(q1), base + l + 32)       \
        GEMV_SPREAD(T, d * float(sc[is + 4]) * float(q2), base + l + 64)       \
        GEMV_SPREAD(T, d * float(sc[is + 6]) * float(q3), base + l + 96)       \
    }                                                                          \
    GEMV_EPILOGUE(T)

// ---- and the two instantiations of each -----------------------------------
//
// `gemv1_*` is the decode kernel. The host picks it whenever `n_tokens == 1`,
// which is every decode step, and the batched one for prefill.

kernel void gemv_f32(GEMV_ARGS)  { GEMV_BODY_F32(GEMV_TOKENS) }
kernel void gemv1_f32(GEMV_ARGS) { GEMV_BODY_F32(1) }
kernel void gemv2_f32(GEMV_ARGS) { GEMV_BODY_F32(2) }
kernel void gemv4_f32(GEMV_ARGS) { GEMV_BODY_F32(4) }
kernel void gemv_f16(GEMV_ARGS)  { GEMV_BODY_F16(GEMV_TOKENS) }
kernel void gemv1_f16(GEMV_ARGS) { GEMV_BODY_F16(1) }
kernel void gemv2_f16(GEMV_ARGS) { GEMV_BODY_F16(2) }
kernel void gemv4_f16(GEMV_ARGS) { GEMV_BODY_F16(4) }
kernel void gemv_q8_0(GEMV_ARGS)  { GEMV_BODY_Q8_0(GEMV_TOKENS) }
kernel void gemv1_q8_0(GEMV_ARGS) { GEMV_BODY_Q8_0(1) }
kernel void gemv2_q8_0(GEMV_ARGS) { GEMV_BODY_Q8_0(2) }
kernel void gemv4_q8_0(GEMV_ARGS) { GEMV_BODY_Q8_0(4) }
kernel void gemv_q4_K(GEMV_ARGS)  { GEMV_BODY_Q4_K(GEMV_TOKENS) }
kernel void gemv1_q4_K(GEMV_ARGS) { GEMV_BODY_Q4_K(1) }
kernel void gemv2_q4_K(GEMV_ARGS) { GEMV_BODY_Q4_K(2) }
kernel void gemv4_q4_K(GEMV_ARGS) { GEMV_BODY_Q4_K(4) }
kernel void gemv_q6_K(GEMV_ARGS)  { GEMV_BODY_Q6_K(GEMV_TOKENS) }
kernel void gemv1_q6_K(GEMV_ARGS) { GEMV_BODY_Q6_K(1) }
kernel void gemv2_q6_K(GEMV_ARGS) { GEMV_BODY_Q6_K(2) }
kernel void gemv4_q6_K(GEMV_ARGS) { GEMV_BODY_Q6_K(4) }

/// Dequantize one Q4_K row into f32.
///
/// The embedding table of a Q4_K_M build is Q4_K, and a batch of one token
/// wants exactly one of its 248320 rows -- so this reads that row rather than
/// expanding the table, which would be 1.3 GB of f32 for 5120 useful floats.
kernel void embed_row_q4_K(device float* out            [[buffer(0)]],
                           device const void* table     [[buffer(1)]],
                           device const int* rows       [[buffer(2)]],
                           constant int& d              [[buffer(3)]],
                           uint3 tgid  [[threadgroup_position_in_grid]],
                           uint3 tid   [[thread_position_in_threadgroup]],
                           uint3 tgdim [[threads_per_threadgroup]]) {
    const int nb = d / QK_K;
    const int r = int(tgid.y);
    device const block_q4_K* wr =
        (device const block_q4_K*)table + size_t(rows[r]) * nb;

    // One thread a 32-element group, the same split `gemv_q4_K` uses.
    const int groups = nb * 8;
    for (int c = int(tgid.x * tgdim.x + tid.x); c < groups;
         c += int(tgdim.x)) {
        device const block_q4_K* blk = wr + c / 8;
        const int g = c % 8;
        const int base = (c / 8) * QK_K + g * 32;
        uchar sc, m;
        q4k_scale_min(blk->scales, g, &sc, &m);
        const int high = g & 1;
        const float dd = float(blk->d) * float(sc);
        const float mn = float(blk->dmin) * float(m);
        device const uchar* qs = blk->qs + (g / 2) * 32;
        for (int j = 0; j < 32; ++j) {
            const int byte = int(qs[j]);
            const int nib = high ? (byte >> 4) : (byte & 0xF);
            out[size_t(r) * d + base + j] = dd * float(nib) - mn;
        }
    }
}

// ---- single-element decoders --------------------------------------------
//
// The row gather and the whole-matrix dequantisation are generated over these,
// exactly as `quant.cu` generates them: one function that decodes element `i`
// of a plane, and two macros that wrap it.

inline float deq_f32(device const void* w, size_t i) {
    return ((device const float*)w)[i];
}

inline float deq_f16(device const void* w, size_t i) {
    return float(((device const half*)w)[i]);
}

inline float deq_q8_0(device const void* w, size_t i) {
    device const block_q8_0* b = (device const block_q8_0*)w + i / QK8_0;
    return float(b->d) * float(b->qs[i % QK8_0]);
}

inline float deq_q4_K(device const void* w, size_t i) {
    device const block_q4_K* b = (device const block_q4_K*)w + i / QK_K;
    const int within = int(i % QK_K);
    const int group64 = within / 64;    // which pair of 32-element groups
    const int rem = within % 64;
    const int high = rem / 32;          // low nibbles first, then high
    const int l = rem % 32;

    uchar sc, m;
    q4k_scale_min(b->scales, group64 * 2 + high, &sc, &m);

    const uchar q = b->qs[group64 * 32 + l];
    const int nib = high ? (q >> 4) : (q & 0xF);
    return float(b->d) * float(sc) * float(nib) - float(b->dmin) * float(m);
}

inline float deq_q6_K(device const void* w, size_t i) {
    device const block_q6_K* b = (device const block_q6_K*)w + i / QK_K;
    const int within = int(i % QK_K);
    const int n = within / 128;         // super-block half
    const int rem = within % 128;
    const int quarter = rem / 32;       // which of the four interleaved groups
    const int l = rem % 32;

    device const uchar* ql = b->ql + n * 64;
    device const uchar* qh = b->qh + n * 32;
    device const char* sc = b->scales + n * 8;

    const int lo_index = (quarter & 1) ? (l + 32) : l;
    const int shift = quarter * 2;
    const uchar low = (quarter < 2) ? (ql[lo_index] & 0xF) : (ql[lo_index] >> 4);
    const int q = int(low | (((qh[l] >> shift) & 3) << 4)) - 32;

    return float(b->d) * float(sc[quarter * 2 + l / 16]) * float(q);
}

#define GATHER_KERNEL(NAME, DECODE)                                            \
    kernel void NAME(device float* out             [[buffer(0)]],              \
                     device const void* w          [[buffer(1)]],              \
                     device const int* rows        [[buffer(2)]],              \
                     constant int& k               [[buffer(3)]],              \
                     uint3 tgid  [[threadgroup_position_in_grid]],             \
                     uint3 tid   [[thread_position_in_threadgroup]],           \
                     uint3 tgdim [[threads_per_threadgroup]]) {                \
        const int t = int(tgid.y);                                             \
        const int i = int(tgid.x * tgdim.x + tid.x);                           \
        if (i >= k) return;                                                    \
        const size_t src = size_t(rows[t]) * k + i;                           \
        out[size_t(t) * k + i] = DECODE(w, src);                              \
    }

GATHER_KERNEL(gather_rows_f32, deq_f32)
GATHER_KERNEL(gather_rows_f16, deq_f16)
GATHER_KERNEL(gather_rows_q8_0, deq_q8_0)
GATHER_KERNEL(gather_rows_q4_K, deq_q4_K)
GATHER_KERNEL(gather_rows_q6_K, deq_q6_K)

/// Whole-matrix dequantisation to f16, which feeds the prefill GEMM.
#define DEQUANT_KERNEL(NAME, DECODE)                                           \
    kernel void NAME(device half* out              [[buffer(0)]],              \
                     device const void* w          [[buffer(1)]],              \
                     constant uint& n              [[buffer(2)]],              \
                     uint3 tgid  [[threadgroup_position_in_grid]],             \
                     uint3 tid   [[thread_position_in_threadgroup]],           \
                     uint3 tgdim [[threads_per_threadgroup]]) {                \
        const uint i = tgid.x * tgdim.x + tid.x;                               \
        if (i >= n) return;                                                    \
        out[i] = half(DECODE(w, size_t(i)));                                   \
    }

DEQUANT_KERNEL(dequant_f32_f16, deq_f32)
DEQUANT_KERNEL(dequant_f16_f16, deq_f16)
DEQUANT_KERNEL(dequant_q8_0_f16, deq_q8_0)
DEQUANT_KERNEL(dequant_q4_K_f16, deq_q4_K)
DEQUANT_KERNEL(dequant_q6_K_f16, deq_q6_K)

// ---- four output rows a threadgroup ----------------------------------------
//
// The experiment behind the rollout below it. `out[n] = W[n][k] . x[k]` reads
// n*k*4.5/8 bytes of Q4_K weight and, one row a threadgroup, re-reads the whole
// activation for every row: n*k*4 bytes, **7.1x the weight traffic**. The
// activation is 20 KiB and stays in cache, so this is not DRAM -- it is load
// issue, and it is why decoding one token sat at 36-45% of this machine's
// streaming ceiling while reading four and a half bits a weight, why loading
// the weights as `uint4` bought 5%, and why a second token is not free.
//
// Four rows in one group share one activation load, so that traffic divides by
// four. The weight loads do not change: each weight byte is still read exactly
// once, now by four streams instead of one.
#define GEMV_ROWS 4

#define GEMV_BODY_ROWS_Q4_K(T)                                                 \
    BLOCK_REDUCE_SCRATCH                                                       \
    const int row0 = int(tgid.x) * GEMV_ROWS;                                  \
    const int token0 = int(tgid.y) * (T);                                      \
    const int ntok = min((T), n_tokens - token0);                              \
    const int nb = k / QK_K;                                                   \
    float acc[GEMV_ROWS][(T)];                                                 \
    for (int r = 0; r < GEMV_ROWS; ++r)                                        \
        for (int t = 0; t < (T); ++t) acc[r][t] = 0.0f;                        \
    for (int c = int(tid.x); c < nb * 8; c += int(tgdim.x)) {                   \
        const int g = c % 8;                                                   \
        const int base = (c / 8) * QK_K + g * 32;                              \
        const int s0 = (g & 1) ? 4 : 0;                                        \
        float d[GEMV_ROWS], mn[GEMV_ROWS];                                     \
        device const uint4* q[GEMV_ROWS];                                      \
        for (int r = 0; r < GEMV_ROWS; ++r) {                                   \
            device const block_q4_K* blk = (device const block_q4_K*)w         \
                + size_t(min(row0 + r, n - 1)) * nb + c / 8;                   \
            uchar sc, m;                                                       \
            q4k_scale_min(blk->scales, g, &sc, &m);                            \
            d[r] = float(blk->d) * float(sc);                                  \
            mn[r] = float(blk->dmin) * float(m);                               \
            q[r] = (device const uint4*)(device const void*)                    \
                (blk->qs + (g / 2) * 32);                                      \
        }                                                                      \
        for (int v = 0; v < 2; ++v) {                                          \
            uint4 quad[GEMV_ROWS];                                             \
            for (int r = 0; r < GEMV_ROWS; ++r) quad[r] = q[r][v];             \
            for (int wi = 0; wi < 4; ++wi) {                                   \
                const int idx = base + v * 16 + wi * 4;                        \
                float4 wv[GEMV_ROWS];                                          \
                for (int r = 0; r < GEMV_ROWS; ++r) {                          \
                    const uint pk = quad[r][wi];                               \
                    wv[r] = float4(                                            \
                        d[r] * float((pk >> s0) & 0xF) - mn[r],                \
                        d[r] * float((pk >> (s0 + 8)) & 0xF) - mn[r],          \
                        d[r] * float((pk >> (s0 + 16)) & 0xF) - mn[r],         \
                        d[r] * float((pk >> (s0 + 24)) & 0xF) - mn[r]);        \
                }                                                              \
                /* One activation load a token, spent on all GEMV_ROWS rows. */\
                for (int t = 0; t < (T); ++t) {                                \
                    if (t >= ntok) continue;                                   \
                    const packed_float4 xv = *(device const packed_float4*)     \
                        (x + size_t(token0 + t) * k + idx);                    \
                    for (int r = 0; r < GEMV_ROWS; ++r) {                       \
                        acc[r][t] += wv[r][0] * xv[0] + wv[r][1] * xv[1]        \
                                   + wv[r][2] * xv[2] + wv[r][3] * xv[3];      \
                    }                                                          \
                }                                                              \
            }                                                                  \
        }                                                                      \
    }                                                                          \
    for (int r = 0; r < GEMV_ROWS; ++r) {                                       \
        for (int t = 0; t < (T); ++t) {                                        \
            const float total = BLOCK_SUM(acc[r][t], tid.x, tgdim.x);          \
            if (tid.x == 0 && t < ntok && row0 + r < n)                        \
                out[size_t(token0 + t) * n + row0 + r] = total;                \
        }                                                                      \
    }

kernel void gemv1x4_q4_K(GEMV_ARGS) { GEMV_BODY_ROWS_Q4_K(1) }
kernel void gemv2x4_q4_K(GEMV_ARGS) { GEMV_BODY_ROWS_Q4_K(2) }
kernel void gemv4x4_q4_K(GEMV_ARGS) { GEMV_BODY_ROWS_Q4_K(4) }

// ---- the narrow-batch matmul, on the matrix units ---------------------------
//
// Why this exists, and it is a number rather than a principle. A speculative
// verification pass is `k + 1` rows, and on CUDA that pass costs 1.06x a
// one-row step where here it costs 1.71x -- so speculation pays 27% there and
// exactly nothing here. The difference is not the drafter, whose acceptance is
// better here (1.85 against 1.72). It is one line in `matmul_pre`: "tensor cores
// whenever they will take the shape, which is every token count up to eight".
//
// An MMA instruction multiplies an 8x8 tile. One row fills an eighth of it and
// two rows fill a quarter, so on CUDA going from one row to two is *free* -- the
// instruction was already issued. The mat-vec above has no tile to fill: it
// spends one FMA per (element, token), so two tokens is two FMAs, and the
// per-token work it cannot amortise is measured at 1.24x-1.35x a shape and
// 1.79x in situ.
//
// Apple's `simdgroup_matrix` is the same instruction class. One simdgroup owns
// an 8-row by 8-token output tile and walks `k`; each weight is dequantised
// exactly once and that one dequantisation feeds all eight token columns. At two
// tokens six columns are padding and cost nothing, which is the entire point --
// the padding is what makes the second row free.
//
// It is worse than the scalar kernel at one token, where seven eighths of every
// MMA is waste and the scalar version spends exactly one FMA. So this is for two
// rows and up, and `gemv1_*` keeps the decode path.

#define MMA_ROWS 8
#define MMA_TOKS 8

/// Q4_K, eight rows by eight tokens a simdgroup.
///
/// A whole 256-element super-block is dequantised into threadgroup memory at
/// once and then fed to thirty-two MMAs. Staging per 8-wide sub-tile instead --
/// the obvious loop -- measured 2.5x the scalar kernel's one-token cost, because
/// it pays three threadgroup barriers to set up a single MMA: 1920 barriers a
/// tile against 40. The arithmetic is identical; only the ratio of setup to work
/// changed.
///
/// The dequantisation is arranged so a thread reads one contiguous span. Thread
/// `i` takes row `i / 4` and quarter `i % 4`, which is 32 bytes of `qs` -- and
/// those same 32 bytes carry group `2c` in their low nibbles and group `2c + 1`
/// in their high ones, so one read produces 64 values and needs two scales.
kernel void gemv_mma_q4_K(
    device float* out          [[buffer(0)]],
    device const void* w       [[buffer(1)]],
    device const float* x      [[buffer(2)]],
    constant int& k            [[buffer(3)]],
    constant int& n            [[buffer(4)]],
    constant int& n_tokens     [[buffer(5)]],
    uint3 tgid  [[threadgroup_position_in_grid]],
    uint3 tid   [[thread_position_in_threadgroup]]) {
    // One super-block of dequantised weight, transposed: `stage[kk][r]`, because
    // the MMA wants `C[t][r] = sum_kk A[t][kk] * B[kk][r]` and the weight arrives
    // as `[r][k]`. Transposing on the write costs nothing.
    // One 32-element group: 1 KiB, which is the only useful point on this curve.
    // A whole super-block -- 8 KiB -- measured 4.7x the scalar kernel, because at
    // 8 KiB a core holds four of these threadgroups and 128 threads hide nothing.
    // A single 8-wide sub-tile fits easily and measured 2.5x, because it pays
    // three barriers to set up one MMA. A group is four MMAs to a barrier pair
    // at 1 KiB, with neither problem.
    threadgroup float stage[32 * MMA_ROWS];
    threadgroup float sout[MMA_ROWS * MMA_TOKS];

    const int row0 = int(tgid.x) * MMA_ROWS;
    const int token0 = int(tgid.y) * MMA_TOKS;
    const int lane = int(tid.x);
    const int nb = k / QK_K;

    const int r = lane / 4;              // this thread's row within the tile
    const int c = lane % 4;             // and which eighth of the group it reads
    const int row = min(row0 + r, n - 1);

    simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (int b = 0; b < nb; ++b) {
        device const block_q4_K* blk =
            (device const block_q4_K*)w + size_t(row) * nb + b;
        for (int g = 0; g < 8; ++g) {
            uchar sc, m;
            q4k_scale_min(blk->scales, g, &sc, &m);
            const float d = float(blk->d) * float(sc);
            const float mn = float(blk->dmin) * float(m);
            const int s0 = (g & 1) ? 4 : 0;
            // 32 values a row, eight a thread, from one 8-byte span.
            device const uint2* q =
                (device const uint2*)(device const void*)(blk->qs + (g / 2) * 32 + c * 8);
            const uint2 pair = *q;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int u = 0; u < 2; ++u) {
                const uint pk = pair[u];
                for (int bi = 0; bi < 4; ++bi) {
                    const int byte = int((pk >> (bi * 8)) & 0xFF);
                    const int off = c * 8 + u * 4 + bi;   // 0..31
                    stage[off * MMA_ROWS + r] = d * float((byte >> s0) & 0xF) - mn;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            const int kbase = b * QK_K + g * 32;
            for (int sub = 0; sub < 4; ++sub) {
                simdgroup_float8x8 wt, xt;
                simdgroup_load(wt, stage + sub * MMA_ROWS * MMA_TOKS, MMA_ROWS);
                simdgroup_load(xt, x + size_t(token0) * k + kbase + sub * MMA_TOKS, k);
                simdgroup_multiply_accumulate(acc, xt, wt, acc);
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_store(acc, sout, MMA_ROWS);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int flat = lane; flat < MMA_ROWS * MMA_TOKS; flat += 32) {
        const int t = flat / MMA_ROWS;
        const int rr = flat % MMA_ROWS;
        if (token0 + t < n_tokens && row0 + rr < n) {
            out[size_t(token0 + t) * n + row0 + rr] = sout[flat];
        }
    }
}

#define MMA_WIDE_TILES 8

/// Q4_K, eight rows by up to 64 tokens (eight tiles of eight) a threadgroup.
///
/// `gemv_mma_q4_K` dequantises and MMAs one 8x8 tile a threadgroup, so a batch
/// past eight tokens pays for the same weight row again for every further
/// group of eight -- `grid.y` threadgroups sharing nothing, each re-fetching
/// and re-decoding the identical bytes. The decode into `stage` depends only
/// on which row this threadgroup owns, never on which tokens are being
/// multiplied; only the multiply-accumulate step needs the token index. This
/// keeps the identical decode and MMAs it against up to eight token-tiles
/// before moving to the next weight chunk, hoping a batch of 64 would pay the
/// weight stream once instead of eight times.
///
/// Measured on an M4 Max against the deployed kernel (`examples/
/// gemv_mma_wide_check.rs`), same answer to the bit at every token count:
/// 0.64x at eight tokens, falling to 0.51x by sixty-four -- *slower*, and
/// increasingly so. The "redundant" re-fetch this was written to remove was
/// evidently not costing much: eight `grid.y` threadgroups sharing one row
/// read the same few KiB of weight close together in time, which is exactly
/// what an L2 sized in MiB is for. What this version actually changed was
/// concurrency -- an 8x narrower `grid.y` is 8x fewer threadgroups in flight,
/// and eight live `simdgroup_float8x8` accumulators a thread is real register
/// pressure that likely caps how many of this fatter threadgroup fit on a
/// core at once. Trading real occupancy for a memory saving the cache had
/// already made this a net loss. Kept as the record of why: the next
/// idea for this range needs a different mechanism, not a wider tile.
kernel void gemv_mma_wide_q4_K(
        device float* out          [[buffer(0)]],
        device const void* w       [[buffer(1)]],
        device const float* x      [[buffer(2)]],
        constant int& k            [[buffer(3)]],
        constant int& n            [[buffer(4)]],
        constant int& n_tokens     [[buffer(5)]],
        uint3 tgid  [[threadgroup_position_in_grid]],
        uint3 tid   [[thread_position_in_threadgroup]]) {
    threadgroup float stage[32 * MMA_ROWS];

    const int row0 = int(tgid.x) * MMA_ROWS;
    const int token0 = int(tgid.y) * MMA_WIDE_TILES * MMA_TOKS;
    const int lane = int(tid.x);
    const int nb = k / QK_K;

    const int r = lane / 4;
    const int c = lane % 4;
    const int row = min(row0 + r, n - 1);

    // How many of this group's eight tiles hold a real token. Fewer only in
    // the last group of a batch that is not a multiple of 64; the decode
    // below runs the same either way, since it does not know about tokens.
    int n_tiles = 0;
    for (int i = 0; i < MMA_WIDE_TILES; ++i) {
        if (token0 + i * MMA_TOKS < n_tokens) n_tiles = i + 1;
    }

    simdgroup_float8x8 acc[MMA_WIDE_TILES];
    for (int i = 0; i < MMA_WIDE_TILES; ++i) {
        acc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }

    for (int b = 0; b < nb; ++b) {
        device const block_q4_K* blk =
            (device const block_q4_K*)w + size_t(row) * nb + b;
        for (int g = 0; g < 8; ++g) {
            uchar sc, m;
            q4k_scale_min(blk->scales, g, &sc, &m);
            const float d = float(blk->d) * float(sc);
            const float mn = float(blk->dmin) * float(m);
            const int s0 = (g & 1) ? 4 : 0;
            device const uint2* q =
                (device const uint2*)(device const void*)(blk->qs + (g / 2) * 32 + c * 8);
            const uint2 pair = *q;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int u = 0; u < 2; ++u) {
                const uint pk = pair[u];
                for (int bi = 0; bi < 4; ++bi) {
                    const int byte = int((pk >> (bi * 8)) & 0xFF);
                    const int off = c * 8 + u * 4 + bi;
                    stage[off * MMA_ROWS + r] = d * float((byte >> s0) & 0xF) - mn;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            const int kbase = b * QK_K + g * 32;
            for (int sub = 0; sub < 4; ++sub) {
                simdgroup_float8x8 wt;
                simdgroup_load(wt, stage + sub * MMA_ROWS * MMA_TOKS, MMA_ROWS);
                for (int tile = 0; tile < n_tiles; ++tile) {
                    simdgroup_float8x8 xt;
                    simdgroup_load(
                        xt,
                        x + size_t(token0 + tile * MMA_TOKS) * k + kbase + sub * MMA_TOKS,
                        k);
                    simdgroup_multiply_accumulate(acc[tile], xt, wt, acc[tile]);
                }
            }
        }
    }

    threadgroup float sout[MMA_ROWS * MMA_TOKS];
    for (int tile = 0; tile < n_tiles; ++tile) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_store(acc[tile], sout, MMA_ROWS);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int flat = lane; flat < MMA_ROWS * MMA_TOKS; flat += 32) {
            const int t = flat / MMA_ROWS;
            const int rr = flat % MMA_ROWS;
            const int tcol = token0 + tile * MMA_TOKS + t;
            if (tcol < n_tokens && row0 + rr < n) {
                out[size_t(tcol) * n + row0 + rr] = sout[flat];
            }
        }
    }
}
