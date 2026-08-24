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
