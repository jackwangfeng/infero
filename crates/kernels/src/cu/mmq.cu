// Quantized GEMM on the integer tensor cores.
//
// Structure follows llama.cpp's `mmq.cu` (MIT, see vendor/LICENSE.ggml), which
// vLLM also carries as its GGUF path: stage a tile of quantized weights and
// Q8_1 activations into shared memory, then run one `mma.m16n8k32.s8` per
// 32-element quantization group and fold the scales in float afterwards.
//
// Why the scales cannot stay in the integer accumulator: a ggml block carries
// its own scale every 32 elements, so an s32 accumulator may only span one
// group. K=32 per MMA makes that exact instead of approximate.
//
// This replaces `dequant_to_f16` + cuBLAS for batched decode and prefill. The
// old path materialized the whole weight matrix in f16 first, which costs one
// read of the quantized weights, one f16 write and one f16 read — roughly five
// times the traffic of reading the quantized weights alone.
//
// ---- where this kernel stands, and against what ----------------------------
//
// At 16 tokens on an A4000 this reaches 458 tok/s. A batch of one costs 12.6 ms
// and a batch of sixteen 34.9, so sixteen tokens add 22.3 ms of arithmetic:
// 223 GFLOP at 10 TOPS, against roughly 153 TOPS of int8 tensor-core throughput.
// The tensor cores are busy 6.5% of the time.
//
// Reading Marlin (see vendor/LICENSE.marlin) says why, and it is not what the
// obvious guesses were. Four differences, in the order they matter:
//
//   1. Its register tile is 64x64 per warp against this kernel's 8x16, so one
//      weight fragment feeds four to sixteen MMAs instead of one. Everything
//      around the MMA — the fragment load, the scale application, the address
//      arithmetic — is amortized that many times better. This is the 6.5%.
//   2. Global to shared is `cp.async` across four pipeline stages, so the copy
//      genuinely overlaps the MMAs. That also explains why double-buffering
//      this kernel measured as no change at all: without `cp.async` the copy
//      is synchronous whatever the buffering.
//   3. Its shared tile keeps the weights *packed* at four bits and unpacks in
//      registers. The staging below expands every nibble to its own int8, so
//      shared traffic is eight times larger than it needs to be.
//   4. Activations stay f16 and the weights dequantize to f16 with `lop3` —
//      two nibbles per instruction, via the trick of placing a 4-bit field in
//      the mantissa of f16 1024.0 and subtracting 1032. So there is no
//      activation quantization, no per-group activation scale, no stored sum
//      and no zero-point-times-sum term: the epilogue is one multiply per
//      128 weights.
//
// Measured, in the order they were tried, all against the AWQ 8B at 256 tokens
// of history:
//
//   `mmqd_*`  weight fragments straight from global, no shared tile   0%
//   `mmqp_*`  four groups under one scale, one s32 accumulation       0%
//   `mmqx_*`  a 32x32 or 128x32 register tile per warp                0%
//   gridDim.z slicing k, three ways                                   0%
//   gridDim.z slicing k, twelve ways                                +22%
//
// The last two are the same change. A 4096-row projection at 64 rows per block
// makes 64 blocks — 1.3 per SM — and the first attempt targeted `sm_count * 4`
// blocks, which asks for three slices and changes nothing. Targeting
// `sm_count * 16` asks for twelve and is worth 22% at eight tokens and 23% at
// sixteen. The device does not want merely enough blocks to be busy; it wants
// enough concurrent weight loads in flight to cover their latency, which is the
// same reason the mat-vec — one block per output row, 4096 of them — reads the
// same bytes three times faster than this kernel did.
//
// So the constraint was occupancy, and the three restructurings that measured
// nothing were all trading instructions inside a kernel that was waiting on
// memory. They stay as the negative results, and because the register tile in
// particular is what a Marlin-shaped rewrite would need anyway.

#define MMQ_M 16    // tokens per tile
// Weight rows per block is `warps * 8` — the MMA's N is 8 and each warp owns
// one. Wide blocks amortize the activation staging; narrow ones produce more
// blocks, which is what matters when the matrix is small relative to the GPU.
// An RTX PRO 6000 holds ~1128 of these at once, and a 4096-row projection at 32
// rows per block only makes 128 of them.
#define MMQ_MAX_WARPS 8
#define MMQ_MAX_ROWS (MMQ_MAX_WARPS * 8)
#define MMQ_K 256                            // k per tile step: one Q4_K super-block
#define MMQ_GROUPS (MMQ_K / 32)              // 8 quantization groups per tile

// Padded row stride for the int8 tiles. The MMA gather has lane L reading four
// bytes at row L/4, byte offset (L%4)*4, so an unpadded 256-byte stride puts
// all eight lanes sharing an (L%4) on one bank. 272 bytes = 68 words shifts
// each successive row by four banks and spreads the warp across all 32.
#define MMQ_STRIDE 272

typedef struct {
    __half2 ds;  // (scale, scale * sum(qs)) — the sum pays for Q4_K's mins
    int8_t qs[32];
} block_q8_1;

// ---- tile staging -------------------------------------------------------

// Zero the token rows this block will never fill. Rows past `n_tokens` stay
// zero for the whole k-loop, so they are written once here instead of being
// re-zeroed every step — at one token that is fifteen sixteenths of the
// activation staging removed, and decode is where it matters most.
__device__ __forceinline__ void mmq_zero_x(int8_t* xs, float* xd, float* xsum,
                                           int valid, int rows, int tid,
                                           int nthreads) {
    for (int i = tid + valid * (MMQ_K / 4); i < rows * (MMQ_K / 4);
         i += nthreads) {
        const int tl = i / (MMQ_K / 4);
        const int e = (i % (MMQ_K / 4)) * 4;
        *(uint32_t*)(void*)(xs + tl * MMQ_STRIDE + e) = 0;
    }
    for (int i = tid + valid * MMQ_GROUPS; i < rows * MMQ_GROUPS;
         i += nthreads) {
        xd[i] = 0.0f;
        xsum[i] = 0.0f;
    }
}

__device__ __forceinline__ void mmq_load_x(int8_t* xs, float* xd, float* xsum,
                                           const block_q8_1* __restrict__ x,
                                           int kb_total, int kb0, int tok0,
                                           int valid, int tid, int nthreads) {
    for (int i = tid; i < valid * MMQ_GROUPS; i += nthreads) {
        const int t = tok0 + i / MMQ_GROUPS;
        const int g = i % MMQ_GROUPS;
        const int kb = kb0 + g;
        float d = 0.0f, s = 0.0f;
        if (kb < kb_total) {
            const __half2 ds = x[(size_t)t * kb_total + kb].ds;
            d = __low2float(ds);
            s = __high2float(ds);
        }
        xd[i] = d;
        xsum[i] = s;
    }
    // Four elements per step. `block_q8_1::qs` sits four bytes into a 36-byte
    // block, so it is always word-aligned, and the destination stride is a
    // multiple of four — a byte-at-a-time version costs four times the
    // instructions and was outrunning the weight staging it exists to feed.
    for (int i = tid; i < valid * MMQ_K / 4; i += nthreads) {
        const int tl = i / (MMQ_K / 4);
        const int t = tok0 + tl;
        const int e = (i % (MMQ_K / 4)) * 4;
        const int kb = kb0 + e / 32;
        uint32_t v = 0;
        if (kb < kb_total) {
            v = *(const uint32_t*)(const void*)(x[(size_t)t * kb_total + kb].qs
                                                + (e % 32));
        }
        *(uint32_t*)(void*)(xs + tl * MMQ_STRIDE + e) = v;
    }
}

// Q8_0: the quants are already int8 and there is no min term.
__device__ __forceinline__ void mmq_load_w_q8_0(int8_t* ws, float* wd, float* wm,
                                                const block_q8_0* __restrict__ w,
                                                int kb_total, int kb0, int row0,
                                                int n, int rows, int tid,
                                                int nthreads) {
    for (int i = tid; i < rows * MMQ_GROUPS; i += nthreads) {
        const int r = i / MMQ_GROUPS;
        const int g = i % MMQ_GROUPS;
        const int gr = row0 + r;
        const int kb = kb0 + g;
        wd[i] = (gr < n && kb < kb_total)
                    ? __half2float(w[(size_t)gr * kb_total + kb].d)
                    : 0.0f;
        wm[i] = 0.0f;
    }
    // Two at a time, not four: a Q8_0 block is 34 bytes, so `qs` is only ever
    // halfword-aligned and a word load would fault on odd blocks.
    for (int i = tid; i < rows * MMQ_K / 2; i += nthreads) {
        const int r = i / (MMQ_K / 2);
        const int e = (i % (MMQ_K / 2)) * 2;
        const int gr = row0 + r;
        const int kb = kb0 + e / 32;
        uint16_t v = 0;
        if (gr < n && kb < kb_total) {
            v = *(const uint16_t*)(const void*)(w[(size_t)gr * kb_total + kb].qs
                                                + (e % 32));
        }
        *(uint16_t*)(void*)(ws + r * MMQ_STRIDE + e) = v;
    }
}

// Q8_0 with the quants and the scales in separate regions.
//
// `mmq_load_w_q8_0` above reads two bytes at a time and says why: a `block_q8_0`
// is 34 bytes, so `qs` is only ever halfword-aligned and a word load would fault
// on odd blocks. That is a layout problem, not a kernel one, and the vocab
// projection is the one matrix tuili quantizes itself — so the layout is ours to
// choose. Here a row's quants are one contiguous run of `k` bytes and the scales
// follow in a trailing region, which makes the tile load sixteen bytes a thread.
//
// It matters where L2 is small. On an A4000 the packed form reads 558 MB at 90
// GB/s against the card's 448, and the vocab projection is 18% of a batch-32
// step; on a Blackwell, whose 128 MB of L2 covers the whole matrix, the two
// layouts measured 789 GB/s against 837 and this would be worth nothing.
//
// Same bytes, same accumulation order, so the logits are unchanged and the
// batch-invariance the comment above `MMQ_SET` claims still holds.
__device__ __forceinline__ void mmq_load_w_q8_0s(int8_t* ws, float* wd, float* wm,
                                                 const int8_t* __restrict__ q,
                                                 const __half* __restrict__ sc,
                                                 int k_total, int kb_total, int kb0,
                                                 int row0, int n, int rows, int tid,
                                                 int nthreads) {
    for (int i = tid; i < rows * MMQ_GROUPS; i += nthreads) {
        const int r = i / MMQ_GROUPS;
        const int g = i % MMQ_GROUPS;
        const int gr = row0 + r;
        const int kb = kb0 + g;
        wd[i] = (gr < n && kb < kb_total)
                    ? __half2float(sc[(size_t)gr * kb_total + kb])
                    : 0.0f;
        wm[i] = 0.0f;
    }
    // Sixteen at a time. A row starts at `gr * k_total` and `k` is a multiple of
    // 32, so every address below is 16-byte aligned by construction.
    for (int i = tid; i < rows * (MMQ_K / 16); i += nthreads) {
        const int r = i / (MMQ_K / 16);
        const int e = (i % (MMQ_K / 16)) * 16;
        const int gr = row0 + r;
        const int ka = kb0 * 32 + e;
        uint4 v = make_uint4(0, 0, 0, 0);
        if (gr < n && ka + 16 <= k_total) {
            v = *(const uint4*)(const void*)(q + (size_t)gr * k_total + ka);
        }
        *(uint4*)(void*)(ws + r * MMQ_STRIDE + e) = v;
    }
}

// Q4_K: unsigned nibbles go into the tile as 0..15 and the 6-bit scale/min pair
// per group goes into wd/wm. The min becomes `-dmin*m`, applied against the
// activation's stored sum rather than through the MMA.
__device__ __forceinline__ void mmq_load_w_q4_K(int8_t* ws, float* wd, float* wm,
                                                const block_q4_K* __restrict__ w,
                                                int nsb, int sb, int row0, int n,
                                                int rows, int tid, int nthreads) {
    for (int i = tid; i < rows * MMQ_GROUPS; i += nthreads) {
        const int r = i / MMQ_GROUPS;
        const int g = i % MMQ_GROUPS;
        const int gr = row0 + r;
        float dd = 0.0f, mm = 0.0f;
        if (gr < n && sb < nsb) {
            const block_q4_K* b = w + (size_t)gr * nsb + sb;
            uint8_t sc, m;
            q4k_scale_min(b->scales, g, &sc, &m);
            dd = __half2float(b->d) * (float)sc;
            mm = -__half2float(b->dmin) * (float)m;
        }
        wd[i] = dd;
        wm[i] = mm;
    }

    // One unit is a (row, 32-byte nibble run); each expands to the two groups
    // that share it. Grid-strided so the block can be any width.
    for (int u = tid; u < rows * 4; u += nthreads) {
        const int r = u / 4;
        const int gp = u % 4;
        const int gr = row0 + r;
        const bool ok = (gr < n) && (sb < nsb);
        const block_q4_K* b = w + (size_t)gr * nsb + sb;
        const int g_lo = gp * 2;
        const int g_hi = g_lo + 1;
#pragma unroll
        for (int j = 0; j < 32; j += 4) {
            uint32_t packed = 0;
            if (ok) {
                packed = *(const uint32_t*)(const void*)(b->qs + gp * 32 + j);
            }
            uint32_t lo = 0, hi = 0;
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                const uint32_t byte = (packed >> (t * 8)) & 0xFFu;
                lo |= (byte & 0xFu) << (t * 8);
                hi |= (byte >> 4) << (t * 8);
            }
            *(uint32_t*)(void*)(ws + r * MMQ_STRIDE + g_lo * 32 + j) = lo;
            *(uint32_t*)(void*)(ws + r * MMQ_STRIDE + g_hi * 32 + j) = hi;
        }
    }
}

// Q4_G128: repacked AWQ. A 256-wide tile is exactly two blocks, and one scale
// covers 128 weights — four of the tile's eight groups — so the same scale and
// offset go into `wd`/`wm` four times rather than being read per group.
//
// The offset is `-scale * zero`, applied against the activation's stored sum
// like Q4_K's min, so the MMA never sees the zero point.
//
// `qs` starts four bytes into a 68-byte block, so both the block base and the
// nibble runs are word-aligned and the wide loads Q4_K uses are legal here too.
__device__ __forceinline__ void mmq_load_w_q4_g128(int8_t* ws, float* wd,
                                                   float* wm,
                                                   const block_q4_g128* __restrict__ w,
                                                   int nb, int tile, int row0,
                                                   int n, int rows, int tid,
                                                   int nthreads) {
    for (int i = tid; i < rows * MMQ_GROUPS; i += nthreads) {
        const int r = i / MMQ_GROUPS;
        const int g = i % MMQ_GROUPS;
        const int gr = row0 + r;
        // Two blocks per tile; groups 0..3 are the first, 4..7 the second.
        const int kb = tile * 2 + g / 4;
        float dd = 0.0f, mm = 0.0f;
        if (gr < n && kb < nb) {
            const __half2 ds = w[(size_t)gr * nb + kb].ds;
            dd = __low2float(ds);
            mm = -__high2float(ds);
        }
        wd[i] = dd;
        wm[i] = mm;
    }

    // One unit is a (row, 32-byte nibble run). Each run's low nibbles are one
    // group and its high nibbles the group two along, which is how the pack
    // arranges a block: byte `b` holds weight `b` and weight `b + 64`.
    for (int u = tid; u < rows * 4; u += nthreads) {
        const int r = u / 4;
        const int q = u % 4;
        const int gr = row0 + r;
        const int kb = tile * 2 + q / 2;
        const int run = q % 2;
        const bool ok = (gr < n) && (kb < nb);
        const block_q4_g128* b = w + (size_t)gr * nb + kb;
        const int g_lo = (q / 2) * 4 + run;
        const int g_hi = g_lo + 2;
#pragma unroll
        for (int j = 0; j < 32; j += 4) {
            uint32_t packed = 0;
            if (ok) {
                packed = *(const uint32_t*)(const void*)(b->qs + run * 32 + j);
            }
            uint32_t lo = 0, hi = 0;
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                const uint32_t byte = (packed >> (t * 8)) & 0xFFu;
                lo |= (byte & 0xFu) << (t * 8);
                hi |= (byte >> 4) << (t * 8);
            }
            *(uint32_t*)(void*)(ws + r * MMQ_STRIDE + g_lo * 32 + j) = lo;
            *(uint32_t*)(void*)(ws + r * MMQ_STRIDE + g_hi * 32 + j) = hi;
        }
    }
}

// Q6_K: six bits split across a low nibble and a two-bit high field, and a
// scale every *sixteen* elements rather than every 32. The scale granularity is
// why this type needs SPLIT=2 below — one MMA cannot span two scales.
//
// `block_q6_K` is 210 bytes, so a block base is only ever halfword-aligned and
// two-byte loads are the widest legal ones.
__device__ __forceinline__ void mmq_load_w_q6_K(int8_t* ws, float* wd, float* wm,
                                                const block_q6_K* __restrict__ w,
                                                int nsb, int sb, int row0, int n,
                                                int rows, int tid, int nthreads) {
    for (int i = tid; i < rows * MMQ_GROUPS * 2; i += nthreads) {
        const int r = i / (MMQ_GROUPS * 2);
        const int m = i % (MMQ_GROUPS * 2);  // half-group, 16 elements each
        const int gr = row0 + r;
        float dd = 0.0f;
        if (gr < n && sb < nsb) {
            const block_q6_K* b = w + (size_t)gr * nsb + sb;
            // ggml interleaves Q6_K's scales: element e belongs to scale
            // `(e/128)*8 + (e%32)/16 + ((e%128)/32)*2`.
            const int e = m * 16;
            const int hi = e / 128;
            const int rr = e % 128;
            dd = __half2float(b->d)
               * (float)b->scales[hi * 8 + (rr % 32) / 16 + (rr / 32) * 2];
        }
        wd[i] = dd;
        wm[i] = 0.0f;
    }

    // Each unit is one (row, 128-element half, pair of `l`) and expands to the
    // four quarters that share those two high-bit bytes.
    for (int u = tid; u < rows * 2 * 16; u += nthreads) {
        const int r = u / 32;
        const int hi = (u % 32) / 16;
        const int l0 = (u % 16) * 2;
        const int gr = row0 + r;
        if (gr >= n || sb >= nsb) {
            for (int j = 0; j < 2; ++j) {
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    ws[r * MMQ_STRIDE + hi * 128 + c * 32 + l0 + j] = 0;
                }
            }
            continue;
        }
        const block_q6_K* b = w + (size_t)gr * nsb + sb;
        const uint16_t qa = *(const uint16_t*)(const void*)(b->ql + hi * 64 + l0);
        const uint16_t qb =
            *(const uint16_t*)(const void*)(b->ql + hi * 64 + 32 + l0);
        const uint16_t qhv = *(const uint16_t*)(const void*)(b->qh + hi * 32 + l0);
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int a = (qa >> (j * 8)) & 0xFF;
            const int bb = (qb >> (j * 8)) & 0xFF;
            const int h = (qhv >> (j * 8)) & 0xFF;
            int8_t* dst = ws + r * MMQ_STRIDE + hi * 128 + l0 + j;
            dst[0] = (int8_t)(((a & 0xF) | (((h >> 0) & 3) << 4)) - 32);
            dst[32] = (int8_t)(((bb & 0xF) | (((h >> 2) & 3) << 4)) - 32);
            dst[64] = (int8_t)(((a >> 4) | (((h >> 4) & 3) << 4)) - 32);
            dst[96] = (int8_t)(((bb >> 4) | (((h >> 6) & 3) << 4)) - 32);
        }
    }
}

// ---- the tile loop ------------------------------------------------------
//
// Scale handling, since the loop below is dense: the row scales `wd`/`wm` do
// not depend on the token tile, so they are read once per group into registers
// rather than once per tile, and the per-token `xd`/`xsum` are read once for
// the two rows a lane accumulates. The scale path is 22% of this kernel at 32
// tokens — measured by compiling a variant with the whole thing replaced by a
// constant, not guessed. The Q4_K min term applies per 32-element group, hence
// the `h == 0` guard.
//
// Comments stay outside the macro body: a block comment spanning backslash
// continuations does not survive NVRTC's preprocessor.
//
// Warp `w` owns weight rows [w*8, w*8+8) and all 16 tokens. `out` is
// [n_tokens][n] row-major, matching the gemv path it replaces.

#define MMQ_BODY(WARPS, TILES, SPLIT, STAGE_W)                                 \
    /* Single-buffered on purpose. Staging is 25 us of this kernel's 39 us on  \
       an RTX PRO 6000 and looks like it should overlap with the MMAs, so a    \
       double-buffered version was written and measured: 38.4 us against 38.4, \
       and worse at 32 tokens (92.6 against 61.5) and on a 48-SM card (196     \
       against 174, where the doubled shared memory halves blocks per SM). The \
       overlap the barriers appear to prevent is evidently already happening. */\
    __shared__ int8_t ws[(WARPS) * 8 * MMQ_STRIDE];                            \
    __shared__ int8_t xs[(TILES) * MMQ_M * MMQ_STRIDE];                        \
    __shared__ float wd[(WARPS) * 8 * MMQ_GROUPS * (SPLIT)];                   \
    __shared__ float wm[(WARPS) * 8 * MMQ_GROUPS * (SPLIT)];                   \
    __shared__ float xd[(TILES) * MMQ_M * MMQ_GROUPS];                         \
    __shared__ float xsum[(TILES) * MMQ_M * MMQ_GROUPS];                       \
                                                                               \
    const int tid = threadIdx.x;                                               \
    const int lane = tid % WARP_SIZE;                                          \
    const int warp = tid / WARP_SIZE;                                          \
    const int nthreads = (WARPS) * WARP_SIZE;                                  \
    const int mrows = (WARPS) * 8;                                             \
    const int row0 = blockIdx.x * mrows;                                       \
    const int tok0 = blockIdx.y * (TILES) * MMQ_M;                             \
    const int kb_total = k / 32;                                               \
    const int n_tiles = (kb_total + MMQ_GROUPS - 1) / MMQ_GROUPS;              \
    const int x_rows = (TILES) * MMQ_M;                                        \
    const int x_valid = min(x_rows, max(0, n_tokens - tok0));                  \
                                                                               \
    const int ar = mma_a_row(lane);                                            \
    const int bc = mma_b_col(lane);                                            \
    const int kq = mma_k0(lane);                                               \
    const int cr = mma_c_row(lane);                                            \
    const int cc = mma_c_col(lane);                                            \
    const int wbase = warp * 8;                                                \
    const int wrow = wbase + bc;                                               \
                                                                               \
    float acc[TILES][4];                                                       \
    _Pragma("unroll") for (int u = 0; u < (TILES); ++u) {                       \
        _Pragma("unroll") for (int c = 0; c < 4; ++c) acc[u][c] = 0.0f;         \
    }                                                                          \
                                                                               \
    /* Masked token rows never change, so they are zeroed once in both halves. */ \
    mmq_zero_x(xs, xd, xsum, x_valid, x_rows, tid, nthreads);                  \
                                                                               \
    for (int kt = 0; kt < n_tiles; ++kt) {                                     \
        __syncthreads();                                                       \
        mmq_load_x(xs, xd, xsum, xq, kb_total, kt * MMQ_GROUPS, tok0, x_valid, \
                   tid, nthreads);                                             \
        STAGE_W(ws, wd, wm, kt);                                               \
        __syncthreads();                                                       \
                                                                               \
        _Pragma("unroll") for (int g = 0; g < MMQ_GROUPS; ++g) {                \
            const int8_t* bp = ws + wrow * MMQ_STRIDE + g * 32 + kq;      \
            const int bw0 = *(const int*)(const void*)bp;                      \
            const int bw1 = *(const int*)(const void*)(bp + 16);               \
                                                                               \
            /* SPLIT halves the K extent a scale has to cover. Registers 0/1 of \
               a fragment always hold k in [0,16) and 2/3 always hold [16,32),  \
               so zeroing one half of B isolates one 16-element scale group     \
               without touching the A side. SPLIT=1 folds this away. */         \
            _Pragma("unroll") for (int h = 0; h < (SPLIT); ++h) {               \
                mma_b_s8 b;                                                    \
                b.x[0] = ((SPLIT) == 1 || h == 0) ? bw0 : 0;                    \
                b.x[1] = ((SPLIT) == 1 || h == 1) ? bw1 : 0;                    \
                const int sg = g * (SPLIT) + h;                                 \
                const int r0 = (wbase + cc) * MMQ_GROUPS * (SPLIT) + sg;        \
                const int r1 = (wbase + cc + 1) * MMQ_GROUPS * (SPLIT) + sg;    \
                const float wd0 = wd[r0];                                     \
                const float wd1 = wd[r1];                                     \
                const float wm0 = ((h == 0) ? wm[r0] : 0.0f);                 \
                const float wm1 = ((h == 0) ? wm[r1] : 0.0f);                 \
                                                                               \
                _Pragma("unroll") for (int u = 0; u < (TILES); ++u) {           \
                    mma_a_s8 a;                                                \
                    mma_c_s32 d = {{0, 0, 0, 0}};                              \
                    ldmatrix_a_s8(a, xs + u * MMQ_M * MMQ_STRIDE + g * 32,     \
                                  MMQ_STRIDE);                                 \
                    mma_s8(d, a, b);                                           \
                                                                               \
                    const int t0 = (u * MMQ_M + cr) * MMQ_GROUPS + g;          \
                    const int t1 = (u * MMQ_M + cr + 8) * MMQ_GROUPS + g;      \
                    const float xdl = xd[t0], xsl = xsum[t0];                 \
                    const float xdh = xd[t1], xsh = xsum[t1];                 \
                    acc[u][0] += wd0 * xdl * (float)d.x[0] + wm0 * xsl;       \
                    acc[u][1] += wd1 * xdl * (float)d.x[1] + wm1 * xsl;       \
                    acc[u][2] += wd0 * xdh * (float)d.x[2] + wm0 * xsh;       \
                    acc[u][3] += wd1 * xdh * (float)d.x[3] + wm1 * xsh;       \
                }                                                              \
            }                                                                  \
        }                                                                      \
    }                                                                          \
                                                                               \
    const int orow = row0 + wbase + cc;                                        \
    _Pragma("unroll") for (int u = 0; u < (TILES); ++u) {                       \
        const int ot0 = tok0 + u * MMQ_M + cr;                                 \
        const int ot1 = ot0 + 8;                                               \
        if (ot0 < n_tokens) {                                                  \
            if (orow < n) out[(size_t)ot0 * n + orow] = acc[u][0];             \
            if (orow + 1 < n) out[(size_t)ot0 * n + orow + 1] = acc[u][1];     \
        }                                                                      \
        if (ot1 < n_tokens) {                                                  \
            if (orow < n) out[(size_t)ot1 * n + orow] = acc[u][2];             \
            if (orow + 1 < n) out[(size_t)ot1 * n + orow + 1] = acc[u][3];     \
        }                                                                      \
    }

// The token-tile count is a compile-time choice because the accumulators must
// live in registers. Results do not depend on it, nor on the double buffering:
// token `t` always accumulates over `k` in the same order with the same scales.
// That is what lets the vocab projection use this kernel at every row count
// without the logits depending on batch size.

#define MMQ_SET(SUFFIX, WARPS, TILES)                                          \
    extern "C" __global__ __launch_bounds__((WARPS) * WARP_SIZE) void          \
    mmq##SUFFIX##_q8_0(float* __restrict__ out, const void* __restrict__ wv,   \
                       const void* __restrict__ xv, int k, int n,              \
                       int n_tokens) {                                         \
        const block_q8_0* wq = (const block_q8_0*)wv;                          \
        const block_q8_1* xq = (const block_q8_1*)xv;                          \
        MMQ_BODY(WARPS, TILES, 1, MMQ_W_Q8_0)                                  \
    }                                                                          \
    extern "C" __global__ __launch_bounds__((WARPS) * WARP_SIZE) void          \
    mmq##SUFFIX##_q8_0s(float* __restrict__ out, const void* __restrict__ wv,  \
                        const void* __restrict__ xv, int k, int n,             \
                        int n_tokens) {                                        \
        const int8_t* wq = (const int8_t*)wv;                                  \
        const __half* wsc = (const __half*)(const void*)(wq + (size_t)n * k);  \
        const block_q8_1* xq = (const block_q8_1*)xv;                          \
        MMQ_BODY(WARPS, TILES, 1, MMQ_W_Q8_0S)                                 \
    }                                                                          \
    extern "C" __global__ __launch_bounds__((WARPS) * WARP_SIZE) void          \
    mmq##SUFFIX##_q4_K(float* __restrict__ out, const void* __restrict__ wv,   \
                       const void* __restrict__ xv, int k, int n,              \
                       int n_tokens) {                                         \
        const block_q4_K* wq = (const block_q4_K*)wv;                          \
        const block_q8_1* xq = (const block_q8_1*)xv;                          \
        MMQ_BODY(WARPS, TILES, 1, MMQ_W_Q4_K)                                  \
    }                                                                          \
    extern "C" __global__ __launch_bounds__((WARPS) * WARP_SIZE) void          \
    mmq##SUFFIX##_q6_K(float* __restrict__ out, const void* __restrict__ wv,   \
                       const void* __restrict__ xv, int k, int n,              \
                       int n_tokens) {                                         \
        const block_q6_K* wq = (const block_q6_K*)wv;                          \
        const block_q8_1* xq = (const block_q8_1*)xv;                          \
        MMQ_BODY(WARPS, TILES, 2, MMQ_W_Q6_K)                                  \
    }                                                                          \
    extern "C" __global__ __launch_bounds__((WARPS) * WARP_SIZE) void          \
    mmq##SUFFIX##_q4_g128(float* __restrict__ out,                             \
                          const void* __restrict__ wv,                         \
                          const void* __restrict__ xv, int k, int n,           \
                          int n_tokens) {                                      \
        const block_q4_g128* wq = (const block_q4_g128*)wv;                    \
        const block_q8_1* xq = (const block_q8_1*)xv;                          \
        MMQ_BODY(WARPS, TILES, 1, MMQ_W_Q4_G128)                               \
    }

#define MMQ_W_Q8_0(WS, WD, WM, KT)                                             \
    mmq_load_w_q8_0(WS, WD, WM, wq, kb_total, (KT) * MMQ_GROUPS, row0, n,      \
                    mrows, tid, nthreads)
#define MMQ_W_Q8_0S(WS, WD, WM, KT)                                            \
    mmq_load_w_q8_0s(WS, WD, WM, wq, wsc, k, k / 32, (KT) * MMQ_GROUPS, row0, n, \
                     mrows, tid, nthreads)
#define MMQ_W_Q4_K(WS, WD, WM, KT)                                             \
    mmq_load_w_q4_K(WS, WD, WM, wq, k / QK_K, KT, row0, n, mrows, tid, nthreads)
#define MMQ_W_Q6_K(WS, WD, WM, KT)                                             \
    mmq_load_w_q6_K(WS, WD, WM, wq, k / QK_K, KT, row0, n, mrows, tid, nthreads)
#define MMQ_W_Q4_G128(WS, WD, WM, KT)                                          \
    mmq_load_w_q4_g128(WS, WD, WM, wq, k / QK_G128, KT, row0, n, mrows, tid,   \
                       nthreads)

// Named `mmq[w<warps>][<tiles>]_<type>`; the plain name is the 4-warp,
// one-tile shape that a 48-SM card wants.
MMQ_SET(, 4, 1)
MMQ_SET(2, 4, 2)
MMQ_SET(w1, 1, 1)
MMQ_SET(w1_2, 1, 2)
MMQ_SET(w2, 2, 1)
MMQ_SET(w2_2, 2, 2)
// Eight warps, so 64 weight rows per block. llama.cpp's tuned Ampere table
// asks for 128 rows and 256 threads for Q4_K; 128 rows does not fit the 48 KB
// static shared-memory limit with this tile layout, so this is the widest step
// toward it that does.
MMQ_SET(w8, 8, 1)
MMQ_SET(w8_2, 8, 2)

// ---- direct-B pipeline, for Q4_G128 -------------------------------------
//
// Measured level with the staged path; kept as the negative result and as the
// starting point for the register-tile rewrite, which needs this operand path.
//
// The staged pipeline above spends 68% of its time filling shared memory and
// 17% in the tensor cores, and the two do not overlap. Double-buffering it
// changed nothing (38.4 us against 38.4), which says the staging is not waiting
// on memory — it is doing work. Per 256-weight tile row it reads 128 packed
// bytes, expands every nibble into its own `int8`, and writes 256 bytes back to
// shared, so shared traffic is twice the packed size and every weight costs a
// shift, a mask and a store before an MMA ever sees it.
//
// Removing all of that is worth nothing: 263 tok/s against 263 at eight tokens,
// 457 against 457 at sixteen, 504 against 502 at thirty-two. An earlier reading
// of +10% was an artifact of selecting the two arms through two different
// switches. What the 68% figure measures is where the *instructions* go, not
// what the kernel is waiting for.
//
// None of that buys anything for the weights. Each warp owns eight rows that no
// other warp reads, so the staging's only service is coalescing — and the B
// fragment is already coalesced without it. Lane L addresses row `L/4` and k
// offset `(L%4)*4`, and registers 2/3 hold k+16, so the four lanes sharing a
// row cover bytes 0..31 of that row's group: one full 32-byte sector, no waste.
//
// So read B straight from global into registers and unpack it there. One `&`
// per fragment replaces the whole staging pass, the weight half of the barrier
// disappears, and the unpack lands inside the MMA loop where the tensor cores
// have something to overlap with. Activations keep their shared tile: those
// *are* shared between warps.
//
// Only Q4_G128 gets this. A ggml K-quant interleaves its groups inside a byte
// run and packs its scales six bits at a time, so its fragment is not a
// contiguous sector and the unpack is not one instruction.

// Where group `g` of a tile lives inside a repacked AWQ row.
//
//   g 0..3 -> first block, g 4..7 -> second
//   run = g % 2       which 32-byte half of the block
//   high = (g % 4) / 2  low nibbles hold k, high nibbles hold k + 64
__device__ __forceinline__ int mmq_g_block(int g) { return g / 4; }
__device__ __forceinline__ int mmq_g_run(int g) { return g % 2; }
__device__ __forceinline__ int mmq_g_high(int g) { return (g % 4) / 2; }

/* Marlin's per-tile lock does not port to this partition. Written, measured,
   reverted.
 
   The output has to start at zero because a run that straddles a row group
   accumulates, and those memsets are 132 a step and about 0.12 ms of a 6.4 ms
   step — invisible in a kernel profile, which is why they survived this long.
   Marlin removes the equivalent with a counter per output tile: the block
   holding a tile's first k-chunk stores, the rest wait for their predecessor
   and add behind it. Ported here — `locks[nt]` counting the k-tiles already in
   `out`, `ld.global.acquire.gpu` to spin and `fence.acq_rel.gpu` to release,
   gated on `occupancy_max_active_blocks_per_multiprocessor` so a waiter cannot
   spin on an unscheduled block — it is *correct* and three times slower: 311
   tok/s against 1050, `layers_ms` 92 against 26.
 
   The reason is the shape of the partition rather than the mechanism. A block
   here owns a long run — about five row groups on an A4000, one k-tile short of
   two on a Blackwell — and its *first* row group is the one straddled with the
   *previous* block's *last*. So block b waits for block b-1 to finish
   everything, b-1 waits for b-2, and the grid serializes into a chain. Marlin's
   slices are one row group each, so its chain is one add long.
 
   Both of the things that would have to change together were then written and
   measured too. The straddled head group processed *last*, so the waits form a
   short cascade after every block's independent work — the head never waits, so
   this does remove the chain — and the lock path refused unless
   `iters >= k_tiles`, which bounds a row group to two contributors and leaves
   `gate_up` as the only eligible matrix at this grid.

   It is correct at every shape and token count, it does not hang, and it is
   **3.3% slower**: 4848 tok/s against 5012, `layers_ms` 5.880 against 5.671.

   The cost is not the locks. The same binary with `TUILI_MMQ_LOCKS=0` — the
   reordering kept, the memset back — measures 4862, so **splitting the run into
   two passes is itself worth -3%**, against the 2.2% the memsets cost. The
   `MMQ_Y_LOADW` hand-over carries a k-tile of weights across row-group
   iterations, and a second pass restarts that pipeline; the nested loop also
   costs registers where this kernel has none to spare.

   So the memset stays, and the reason is worth stating plainly: this partition
   balances 448 row groups over 376 blocks by splitting some of them, splitting
   requires accumulation, accumulation requires either a zeroed target or an
   order, and every way of imposing an order costs more than the zeroing does.
 
   The first version also hung the server on its first long prompt while all 170
   assertions passed: `blockIdx.y` is the token-tile dimension, so a prefill
   launch has sixteen slices sharing one counter per row group, and the
   residency the waiting depends on is the whole grid rather than `gridDim.x`.
   A correctness test for this belongs at more than one token tile. */

/* One k-slice is a plain store; several have to be summed, and `atomicAdd`
   costs less than a partial buffer and a second pass over it. The caller
   zeroes `out` when it splits. Float addition is not associative, so a split
   launch is not bit-reproducible — which is why `tests/q4g128.rs` checks
   against the dequantized weights rather than against another kernel. */
#define MMQ_PUT(IDX, V)                                                        \
    do {                                                                       \
        if (splits > 1) {                                                      \
            atomicAdd(out + (IDX), (V));                                       \
        } else {                                                               \
            out[IDX] = (V);                                                    \
        }                                                                      \
    } while (0)

#define MMQ_DIRECT_BODY(WARPS, TILES)                                          \
    __shared__ int8_t xs[(TILES) * MMQ_M * MMQ_STRIDE];                        \
    __shared__ float xd[(TILES) * MMQ_M * MMQ_GROUPS];                         \
    __shared__ float xsum[(TILES) * MMQ_M * MMQ_GROUPS];                       \
    /* Two scales per row per tile — one per 128-weight block — rather than    \
       one per 32-weight group: four of the eight groups share each. */        \
    __shared__ float wd[(WARPS) * 8 * 2];                                      \
    __shared__ float wm[(WARPS) * 8 * 2];                                      \
                                                                               \
    const int tid = threadIdx.x;                                               \
    const int lane = tid % WARP_SIZE;                                          \
    const int warp = tid / WARP_SIZE;                                          \
    const int nthreads = (WARPS) * WARP_SIZE;                                  \
    const int mrows = (WARPS) * 8;                                             \
    const int row0 = blockIdx.x * mrows;                                       \
    const int tok0 = blockIdx.y * (TILES) * MMQ_M;                             \
    const int kb_total = k / 32;                                               \
    const int nb = k / QK_G128;                                                \
    const int all_tiles = (kb_total + MMQ_GROUPS - 1) / MMQ_GROUPS;            \
    /* Slice k across `gridDim.z`. One block per 64 weight rows makes 64       \
       blocks for a 4096-row projection: 1.3 per SM, 22% occupancy, and the    \
       reason this kernel reads the same bytes five times slower than the      \
       mat-vec does. Splitting k is the only axis that adds blocks without     \
       taking threads from somewhere else — splitting rows conserves them. */  \
    const int splits = gridDim.z;                                              \
    const int per_split = (all_tiles + splits - 1) / splits;                   \
    const int tile_lo = (int)blockIdx.z * per_split;                           \
    const int tile_hi = min(all_tiles, tile_lo + per_split);                   \
    const int x_rows = (TILES) * MMQ_M;                                        \
    const int x_valid = min(x_rows, max(0, n_tokens - tok0));                  \
                                                                               \
    const int bc = mma_b_col(lane);                                            \
    const int kq = mma_k0(lane);                                               \
    const int cr = mma_c_row(lane);                                            \
    const int cc = mma_c_col(lane);                                            \
    const int wbase = warp * 8;                                                \
    const int brow = row0 + wbase + bc;                                        \
    const bool brow_ok = brow < n;                                             \
    /* This lane's byte cursor: its row, its four k, and the +16 partner. */   \
    const block_q4_g128* bsrc = wq + (brow_ok ? (size_t)brow * nb : 0);        \
                                                                               \
    float acc[TILES][4];                                                       \
    _Pragma("unroll") for (int u = 0; u < (TILES); ++u) {                       \
        _Pragma("unroll") for (int c = 0; c < 4; ++c) acc[u][c] = 0.0f;         \
    }                                                                          \
                                                                               \
    mmq_zero_x(xs, xd, xsum, x_valid, x_rows, tid, nthreads);                  \
                                                                               \
    for (int kt = tile_lo; kt < tile_hi; ++kt) {                               \
        __syncthreads();                                                       \
        mmq_load_x(xs, xd, xsum, xq, kb_total, kt * MMQ_GROUPS, tok0, x_valid, \
                   tid, nthreads);                                             \
        /* Scales only: two words per row, against the staged path's           \
           row-by-nibble expansion. */                                         \
        for (int i = tid; i < mrows * 2; i += nthreads) {                       \
            const int r = i / 2;                                               \
            const int h = i % 2;                                                \
            const int gr = row0 + r;                                            \
            const int kb = kt * 2 + h;                                          \
            float dd = 0.0f, mm = 0.0f;                                         \
            if (gr < n && kb < nb) {                                            \
                const __half2 ds = wq[(size_t)gr * nb + kb].ds;                 \
                dd = __low2float(ds);                                           \
                mm = -__high2float(ds);                                         \
            }                                                                   \
            wd[i] = dd;                                                         \
            wm[i] = mm;                                                         \
        }                                                                       \
        __syncthreads();                                                        \
                                                                               \
        /* Groups g and g+2 are the two nibble halves of the same bytes, so     \
           walk the four byte runs and feed two groups from each. Reading per   \
           group loaded every weight byte twice. */                             \
        _Pragma("unroll") for (int q = 0; q < 4; ++q) {                         \
            const int kb = kt * 2 + q / 2;                                      \
            const uint8_t* pq = bsrc[kb].qs + (q % 2) * 32 + kq;                \
            uint32_t v0 = 0, v1 = 0;                                            \
            if (brow_ok && kb < nb) {                                           \
                v0 = *(const uint32_t*)(const void*)pq;                         \
                v1 = *(const uint32_t*)(const void*)(pq + 16);                  \
            }                                                                   \
            _Pragma("unroll") for (int h = 0; h < 2; ++h) {                     \
            const int g = (q / 2) * 4 + (q % 2) + h * 2;                        \
            mma_b_s8 b;                                                         \
            b.x[0] = (int)((v0 >> (h * 4)) & 0x0F0F0F0Fu);                      \
            b.x[1] = (int)((v1 >> (h * 4)) & 0x0F0F0F0Fu);                      \
                                                                                \
            const int s0 = (wbase + cc) * 2 + (q / 2);                           \
            const int s1 = (wbase + cc + 1) * 2 + (q / 2);                       \
            const float wd0 = wd[s0];                                            \
            const float wd1 = wd[s1];                                            \
            const float wm0 = wm[s0];                                            \
            const float wm1 = wm[s1];                                            \
                                                                                \
            _Pragma("unroll") for (int u = 0; u < (TILES); ++u) {                \
                mma_a_s8 a;                                                      \
                mma_c_s32 d = {{0, 0, 0, 0}};                                    \
                ldmatrix_a_s8(a, xs + u * MMQ_M * MMQ_STRIDE + g * 32,           \
                              MMQ_STRIDE);                                       \
                mma_s8(d, a, b);                                                 \
                                                                                \
                const int t0 = (u * MMQ_M + cr) * MMQ_GROUPS + g;                \
                const int t1 = (u * MMQ_M + cr + 8) * MMQ_GROUPS + g;            \
                const float xdl = xd[t0], xsl = xsum[t0];                        \
                const float xdh = xd[t1], xsh = xsum[t1];                        \
                acc[u][0] += wd0 * xdl * (float)d.x[0] + wm0 * xsl;              \
                acc[u][1] += wd1 * xdl * (float)d.x[1] + wm1 * xsl;              \
                acc[u][2] += wd0 * xdh * (float)d.x[2] + wm0 * xsh;              \
                acc[u][3] += wd1 * xdh * (float)d.x[3] + wm1 * xsh;              \
            }                                                                    \
            }                                                                    \
        }                                                                        \
    }                                                                            \
                                                                                 \
    _Pragma("unroll") for (int u = 0; u < (TILES); ++u) {                        \
        const int orow = row0 + wbase + cc;                                      \
        const int ot0 = tok0 + u * MMQ_M + cr;                                   \
        const int ot1 = ot0 + 8;                                                 \
        if (ot0 < n_tokens) {                                                    \
            if (orow < n) MMQ_PUT((size_t)ot0 * n + orow, acc[u][0]);            \
            if (orow + 1 < n)                                                    \
                MMQ_PUT((size_t)ot0 * n + orow + 1, acc[u][1]);                  \
        }                                                                        \
        if (ot1 < n_tokens) {                                                    \
            if (orow < n) MMQ_PUT((size_t)ot1 * n + orow, acc[u][2]);            \
            if (orow + 1 < n)                                                    \
                MMQ_PUT((size_t)ot1 * n + orow + 1, acc[u][3]);                  \
        }                                                                        \
    }

// ---- striped scheduling, ported from Marlin -----------------------------
//
// See vendor/LICENSE.marlin. The index arithmetic here follows
// `marlin_template.h`; the inner loop is the one above.
//
// The kernels above size their grid from the matrix: one block per group of
// weight rows, each walking all of k. For a 4096-row projection that is 64
// blocks — 1.3 per SM — and no amount of restructuring inside the block fixes
// a device that is 78% idle. Adding `gridDim.z` slices of k fixed the block
// count and broke the reduction instead: every slice of every row group had to
// be summed, so the cost was O(slices * outputs) and at 32 tokens it ate the
// gain whole.
//
// Marlin sizes the grid from the *machine* and then partitions the work. The
// (row group, k chunk) pairs are flattened k-major, and block `b` takes the
// contiguous run `[iters*b, iters*(b+1))` of that list — so a block may finish
// one row group and continue into the next. The point is not that k gets split;
// it is that k gets split *as little as the balance requires*, and only where a
// run happens to straddle a boundary. Most row groups still come out of a
// single block and store directly. Marlin orders the stragglers with a lock and
// a per-slice index; this takes the simpler route of an atomic add, which is
// sound because the traffic is now O(boundaries) rather than O(slices).
//
// Measured level with the `gridDim.z` split it was meant to replace: 328.8
// against 327.3 tok/s at eight tokens and 557.8 against 565.3 at sixteen,
// across two, four, eight and sixteen blocks per SM. The reduction traffic it
// saves was never the constraint — the block count was, and the cruder split
// had already supplied that. Kept because a partition sized from the device is
// the shape a `cp.async` pipeline would need, and because the negative result
// is worth being able to re-run.

#define MMQ_STRIPED_BODY(WARPS, TILES)                                         \
    __shared__ int8_t xs[(TILES) * MMQ_M * MMQ_STRIDE];                        \
    __shared__ float xd[(TILES) * MMQ_M * MMQ_GROUPS];                         \
    __shared__ float xsum[(TILES) * MMQ_M * MMQ_GROUPS];                       \
    __shared__ float wd[(WARPS) * 8 * 2];                                      \
    __shared__ float wm[(WARPS) * 8 * 2];                                      \
                                                                               \
    const int tid = threadIdx.x;                                               \
    const int lane = tid % WARP_SIZE;                                          \
    const int warp = tid / WARP_SIZE;                                          \
    const int nthreads = (WARPS) * WARP_SIZE;                                  \
    const int mrows = (WARPS) * 8;                                             \
    const int tok0 = blockIdx.y * (TILES) * MMQ_M;                             \
    const int kb_total = k / 32;                                               \
    const int nblocks = k / QK_G128;                                           \
    const int k_tiles = (kb_total + MMQ_GROUPS - 1) / MMQ_GROUPS;              \
    const int n_tiles = (n + mrows - 1) / mrows;                               \
    const int x_rows = (TILES) * MMQ_M;                                        \
    const int x_valid = min(x_rows, max(0, n_tokens - tok0));                  \
                                                                               \
    const int bc = mma_b_col(lane);                                            \
    const int kq = mma_k0(lane);                                               \
    const int cr = mma_c_row(lane);                                            \
    const int cc = mma_c_col(lane);                                            \
    const int wbase = warp * 8;                                                \
                                                                               \
    /* This block's contiguous run of the flattened (row group, k chunk) list. */\
    const int total = n_tiles * k_tiles;                                       \
    const int iters = (total + (int)gridDim.x - 1) / (int)gridDim.x;           \
    int flat = iters * (int)blockIdx.x;                                        \
    const int flat_end = min(total, flat + iters);                             \
                                                                               \
    mmq_zero_x(xs, xd, xsum, x_valid, x_rows, tid, nthreads);                  \
                                                                               \
    while (flat < flat_end) {                                                  \
        const int nt = flat / k_tiles;                                         \
        const int kt_lo = flat % k_tiles;                                      \
        /* Stop at this row group's end; the next iteration picks up the next. */\
        const int kt_hi = min(k_tiles, kt_lo + (flat_end - flat));             \
        const int row0 = nt * mrows;                                           \
                                                                               \
        const block_q4_g128* bsrc;                                             \
        const int brow = row0 + wbase + bc;                                    \
        const bool brow_ok = brow < n;                                         \
        bsrc = wq + (brow_ok ? (size_t)brow * nblocks : 0);                    \
                                                                               \
        float acc[TILES][4];                                                   \
        _Pragma("unroll") for (int u = 0; u < (TILES); ++u) {                   \
            _Pragma("unroll") for (int c = 0; c < 4; ++c) acc[u][c] = 0.0f;     \
        }                                                                      \
                                                                               \
        for (int kt = kt_lo; kt < kt_hi; ++kt) {                               \
            __syncthreads();                                                   \
            mmq_load_x(xs, xd, xsum, xq, kb_total, kt * MMQ_GROUPS, tok0,      \
                       x_valid, tid, nthreads);                                \
            for (int i = tid; i < mrows * 2; i += nthreads) {                   \
                const int r = i / 2;                                            \
                const int h = i % 2;                                            \
                const int gr = row0 + r;                                        \
                const int kb = kt * 2 + h;                                     \
                float dd = 0.0f, mm = 0.0f;                                     \
                if (gr < n && kb < nblocks) {                                   \
                    const __half2 ds = wq[(size_t)gr * nblocks + kb].ds;        \
                    dd = __low2float(ds);                                       \
                    mm = -__high2float(ds);                                     \
                }                                                               \
                wd[i] = dd;                                                     \
                wm[i] = mm;                                                     \
            }                                                                   \
            __syncthreads();                                                    \
                                                                                \
            _Pragma("unroll") for (int q = 0; q < 4; ++q) {                      \
                const int kb = kt * 2 + q / 2;                                  \
                const uint8_t* pq = bsrc[kb].qs + (q % 2) * 32 + kq;            \
                uint32_t v0 = 0, v1 = 0;                                        \
                if (brow_ok && kb < nblocks) {                                  \
                    v0 = *(const uint32_t*)(const void*)pq;                     \
                    v1 = *(const uint32_t*)(const void*)(pq + 16);              \
                }                                                               \
                _Pragma("unroll") for (int h = 0; h < 2; ++h) {                  \
                    const int g = (q / 2) * 4 + (q % 2) + h * 2;                 \
                    mma_b_s8 b;                                                  \
                    b.x[0] = (int)((v0 >> (h * 4)) & 0x0F0F0F0Fu);               \
                    b.x[1] = (int)((v1 >> (h * 4)) & 0x0F0F0F0Fu);               \
                    const int s0 = (wbase + cc) * 2 + (q / 2);                   \
                    const int s1 = (wbase + cc + 1) * 2 + (q / 2);               \
                    const float wd0 = wd[s0], wd1 = wd[s1];                      \
                    const float wm0 = wm[s0], wm1 = wm[s1];                      \
                    _Pragma("unroll") for (int u = 0; u < (TILES); ++u) {        \
                        mma_a_s8 a;                                              \
                        mma_c_s32 d = {{0, 0, 0, 0}};                            \
                        ldmatrix_a_s8(a, xs + u * MMQ_M * MMQ_STRIDE + g * 32,   \
                                      MMQ_STRIDE);                               \
                        mma_s8(d, a, b);                                         \
                        const int t0 = (u * MMQ_M + cr) * MMQ_GROUPS + g;        \
                        const int t1 = (u * MMQ_M + cr + 8) * MMQ_GROUPS + g;    \
                        const float xdl = xd[t0], xsl = xsum[t0];                \
                        const float xdh = xd[t1], xsh = xsum[t1];                \
                        acc[u][0] += wd0 * xdl * (float)d.x[0] + wm0 * xsl;      \
                        acc[u][1] += wd1 * xdl * (float)d.x[1] + wm1 * xsl;      \
                        acc[u][2] += wd0 * xdh * (float)d.x[2] + wm0 * xsh;      \
                        acc[u][3] += wd1 * xdh * (float)d.x[3] + wm1 * xsh;      \
                    }                                                            \
                }                                                                \
            }                                                                    \
        }                                                                        \
                                                                                 \
        /* A run that covered this row group's whole k stores; a partial one     \
           adds. Most runs are whole, which is the point of the partition. */    \
        const bool whole = (kt_lo == 0) && (kt_hi == k_tiles);                   \
        const int orow = row0 + wbase + cc;                                      \
        _Pragma("unroll") for (int u = 0; u < (TILES); ++u) {                     \
            const int ot0 = tok0 + u * MMQ_M + cr;                               \
            const int ot1 = ot0 + 8;                                             \
            if (ot0 < n_tokens) {                                                \
                if (orow < n) MMQ_PUT2(whole, (size_t)ot0 * n + orow, acc[u][0]);\
                if (orow + 1 < n)                                                \
                    MMQ_PUT2(whole, (size_t)ot0 * n + orow + 1, acc[u][1]);      \
            }                                                                    \
            if (ot1 < n_tokens) {                                                \
                if (orow < n) MMQ_PUT2(whole, (size_t)ot1 * n + orow, acc[u][2]);\
                if (orow + 1 < n)                                                \
                    MMQ_PUT2(whole, (size_t)ot1 * n + orow + 1, acc[u][3]);      \
            }                                                                    \
        }                                                                        \
        flat += kt_hi - kt_lo;                                                   \
        __syncthreads();                                                         \
    }

#define MMQ_PUT2(WHOLE, IDX, V)                                                \
    do {                                                                       \
        if (WHOLE) {                                                           \
            out[IDX] = (V);                                                    \
        } else {                                                               \
            atomicAdd(out + (IDX), (V));                                       \
        }                                                                      \
    } while (0)

#define MMQ_STRIPED_SET(SUFFIX, WARPS, TILES)                                  \
    extern "C" __global__ __launch_bounds__((WARPS) * WARP_SIZE) void          \
    mmqs##SUFFIX##_q4_g128(float* __restrict__ out,                            \
                           const void* __restrict__ wv,                        \
                           const void* __restrict__ xv, int k, int n,          \
                           int n_tokens) {                                     \
        const block_q4_g128* wq = (const block_q4_g128*)wv;                    \
        const block_q8_1* xq = (const block_q8_1*)xv;                          \
        MMQ_STRIPED_BODY(WARPS, TILES)                                         \
    }

MMQ_STRIPED_SET(, 4, 1)
MMQ_STRIPED_SET(2, 4, 2)
MMQ_STRIPED_SET(w8, 8, 1)
MMQ_STRIPED_SET(w8_2, 8, 2)

// What the epilogue costs, priced before it is rewritten.
//
// Identical to `MMQ_DIRECT_BODY` except that the four 32-groups covered by one
// Q4_G128 scale accumulate in the `s32` accumulator and convert to float once,
// instead of converting and scaling per group. That is only legal if the
// activation scale is also constant across the four, which today it is not —
// a Q8_1 block is 32 elements — so this computes the wrong answer. It exists
// to say whether making the activation blocks 128 wide is worth the change.
//
// If the tensor cores were the limit this would measure the same as the real
// kernel. Whatever it saves is what the per-group epilogue costs.
#define MMQ_EPILOGUE_PROBE_BODY(WARPS, TILES)                                  \
    __shared__ int8_t xs[(TILES) * MMQ_M * MMQ_STRIDE];                        \
    __shared__ float xd[(TILES) * MMQ_M * MMQ_GROUPS];                         \
    __shared__ float xsum[(TILES) * MMQ_M * MMQ_GROUPS];                       \
    __shared__ float wd[(WARPS) * 8 * 2];                                      \
    __shared__ float wm[(WARPS) * 8 * 2];                                      \
                                                                               \
    const int tid = threadIdx.x;                                               \
    const int lane = tid % WARP_SIZE;                                          \
    const int warp = tid / WARP_SIZE;                                          \
    const int nthreads = (WARPS) * WARP_SIZE;                                  \
    const int mrows = (WARPS) * 8;                                             \
    const int row0 = blockIdx.x * mrows;                                       \
    const int tok0 = blockIdx.y * (TILES) * MMQ_M;                             \
    const int kb_total = k / 32;                                               \
    const int nb = k / QK_G128;                                                \
    const int n_tiles = (kb_total + MMQ_GROUPS - 1) / MMQ_GROUPS;              \
    const int x_rows = (TILES) * MMQ_M;                                        \
    const int x_valid = min(x_rows, max(0, n_tokens - tok0));                  \
                                                                               \
    const int bc = mma_b_col(lane);                                            \
    const int kq = mma_k0(lane);                                               \
    const int cr = mma_c_row(lane);                                            \
    const int cc = mma_c_col(lane);                                            \
    const int wbase = warp * 8;                                                \
    const int brow = row0 + wbase + bc;                                        \
    const bool brow_ok = brow < n;                                             \
    const block_q4_g128* bsrc = wq + (brow_ok ? (size_t)brow * nb : 0);        \
                                                                               \
    float acc[TILES][4];                                                       \
    _Pragma("unroll") for (int u = 0; u < (TILES); ++u) {                       \
        _Pragma("unroll") for (int c = 0; c < 4; ++c) acc[u][c] = 0.0f;         \
    }                                                                          \
                                                                               \
    mmq_zero_x(xs, xd, xsum, x_valid, x_rows, tid, nthreads);                  \
                                                                               \
    for (int kt = 0; kt < n_tiles; ++kt) {                                     \
        __syncthreads();                                                       \
        mmq_load_x(xs, xd, xsum, xq, kb_total, kt * MMQ_GROUPS, tok0, x_valid, \
                   tid, nthreads);                                             \
        for (int i = tid; i < mrows * 2; i += nthreads) {                       \
            const int r = i / 2;                                                \
            const int h = i % 2;                                                \
            const int gr = row0 + r;                                            \
            const int kb = kt * 2 + h;                                          \
            float dd = 0.0f, mm = 0.0f;                                         \
            if (gr < n && kb < nb) {                                            \
                const __half2 ds = wq[(size_t)gr * nb + kb].ds;                 \
                dd = __low2float(ds);                                           \
                mm = -__high2float(ds);                                         \
            }                                                                   \
            wd[i] = dd;                                                         \
            wm[i] = mm;                                                         \
        }                                                                       \
        __syncthreads();                                                        \
                                                                                \
        /* One 128-weight block at a time: four groups, one scale, one float    \
           conversion at the end. */                                            \
        _Pragma("unroll") for (int blk = 0; blk < 2; ++blk) {                    \
            int isum[TILES][4];                                                  \
            _Pragma("unroll") for (int u = 0; u < (TILES); ++u) {                \
                _Pragma("unroll") for (int c = 0; c < 4; ++c) isum[u][c] = 0;    \
            }                                                                    \
            _Pragma("unroll") for (int r = 0; r < 2; ++r) {                      \
                const int kb = kt * 2 + blk;                                     \
                const uint8_t* pq = bsrc[kb].qs + r * 32 + kq;                   \
                uint32_t v0 = 0, v1 = 0;                                         \
                if (brow_ok && kb < nb) {                                        \
                    v0 = *(const uint32_t*)(const void*)pq;                      \
                    v1 = *(const uint32_t*)(const void*)(pq + 16);               \
                }                                                                \
                _Pragma("unroll") for (int h = 0; h < 2; ++h) {                  \
                    const int g = blk * 4 + r + h * 2;                           \
                    mma_b_s8 b;                                                  \
                    b.x[0] = (int)((v0 >> (h * 4)) & 0x0F0F0F0Fu);               \
                    b.x[1] = (int)((v1 >> (h * 4)) & 0x0F0F0F0Fu);               \
                    _Pragma("unroll") for (int u = 0; u < (TILES); ++u) {        \
                        mma_a_s8 a;                                              \
                        mma_c_s32 d = {{0, 0, 0, 0}};                            \
                        ldmatrix_a_s8(a, xs + u * MMQ_M * MMQ_STRIDE + g * 32,   \
                                      MMQ_STRIDE);                               \
                        mma_s8(d, a, b);                                         \
                        _Pragma("unroll") for (int c = 0; c < 4; ++c) {          \
                            isum[u][c] += d.x[c];                                \
                        }                                                        \
                    }                                                            \
                }                                                                \
            }                                                                    \
            const int s0 = (wbase + cc) * 2 + blk;                               \
            const int s1 = (wbase + cc + 1) * 2 + blk;                           \
            const float wd0 = wd[s0], wd1 = wd[s1];                              \
            const float wm0 = wm[s0], wm1 = wm[s1];                              \
            _Pragma("unroll") for (int u = 0; u < (TILES); ++u) {                \
                const int t0 = (u * MMQ_M + cr) * MMQ_GROUPS + blk * 4;          \
                const int t1 = (u * MMQ_M + cr + 8) * MMQ_GROUPS + blk * 4;      \
                const float xdl = xd[t0], xsl = xsum[t0];                        \
                const float xdh = xd[t1], xsh = xsum[t1];                        \
                acc[u][0] += wd0 * xdl * (float)isum[u][0] + wm0 * xsl;          \
                acc[u][1] += wd1 * xdl * (float)isum[u][1] + wm1 * xsl;          \
                acc[u][2] += wd0 * xdh * (float)isum[u][2] + wm0 * xsh;          \
                acc[u][3] += wd1 * xdh * (float)isum[u][3] + wm1 * xsh;          \
            }                                                                    \
        }                                                                        \
    }                                                                            \
                                                                                 \
    _Pragma("unroll") for (int u = 0; u < (TILES); ++u) {                        \
        const int orow = row0 + wbase + cc;                                      \
        const int ot0 = tok0 + u * MMQ_M + cr;                                   \
        const int ot1 = ot0 + 8;                                                 \
        if (ot0 < n_tokens) {                                                    \
            if (orow < n) out[(size_t)ot0 * n + orow] = acc[u][0];               \
            if (orow + 1 < n) out[(size_t)ot0 * n + orow + 1] = acc[u][1];       \
        }                                                                        \
        if (ot1 < n_tokens) {                                                    \
            if (orow < n) out[(size_t)ot1 * n + orow] = acc[u][2];               \
            if (orow + 1 < n) out[(size_t)ot1 * n + orow + 1] = acc[u][3];       \
        }                                                                        \
    }

#define MMQ_PROBE_SET(SUFFIX, WARPS, TILES)                                    \
    extern "C" __global__ __launch_bounds__((WARPS) * WARP_SIZE) void          \
    mmqp##SUFFIX##_q4_g128(float* __restrict__ out,                            \
                           const void* __restrict__ wv,                        \
                           const void* __restrict__ xv, int k, int n,          \
                           int n_tokens) {                                     \
        const block_q4_g128* wq = (const block_q4_g128*)wv;                    \
        const block_q8_1* xq = (const block_q8_1*)xv;                          \
        MMQ_EPILOGUE_PROBE_BODY(WARPS, TILES)                                  \
    }

MMQ_PROBE_SET(, 4, 1)
MMQ_PROBE_SET(2, 4, 2)
MMQ_PROBE_SET(w8, 8, 1)
MMQ_PROBE_SET(w8_2, 8, 2)

#define MMQ_DIRECT_SET(SUFFIX, WARPS, TILES)                                   \
    extern "C" __global__ __launch_bounds__((WARPS) * WARP_SIZE) void          \
    mmqd##SUFFIX##_q4_g128(float* __restrict__ out,                            \
                           const void* __restrict__ wv,                        \
                           const void* __restrict__ xv, int k, int n,          \
                           int n_tokens) {                                     \
        const block_q4_g128* wq = (const block_q4_g128*)wv;                    \
        const block_q8_1* xq = (const block_q8_1*)xv;                          \
        MMQ_DIRECT_BODY(WARPS, TILES)                                          \
    }

MMQ_DIRECT_SET(, 4, 1)
MMQ_DIRECT_SET(2, 4, 2)
MMQ_DIRECT_SET(w2, 2, 1)
MMQ_DIRECT_SET(w2_2, 2, 2)
MMQ_DIRECT_SET(w8, 8, 1)
MMQ_DIRECT_SET(w8_2, 8, 2)

// ---- wide register tile, for Q4_G128 ------------------------------------
//
// The measurements above say the constraint is not the shared traffic and not
// the epilogue's instruction count: it is that one MMA carries about fifteen
// other instructions, so the tensor cores are busy 6.5% of the time. The fix is
// structural — give each warp more output to hold, so every operand load and
// every scale read is spread over more MMAs.
//
// A warp here owns `NBLK` blocks of eight weight rows instead of one, and
// `TILES` token tiles as before. Within a group of 32 weights:
//
//   the A fragment  loads once per token tile   and feeds NBLK MMAs
//   xd / xsum       read once per token tile    and feed  NBLK MMAs
//   wd / wm         read once per weight block  and feed  TILES MMAs
//   the B fragments cost one word pair per two groups per weight block
//
// At NBLK=4, TILES=2 that is eight MMAs against roughly fifty-four other
// instructions rather than one against fifteen. Marlin goes further still — a
// 64x64 tile per warp, `cp.async` staging and an f16 operand path — but the
// tile is what has to come first, because everything else is a way of hiding
// latency that only pays once there is arithmetic to hide it behind.
//
// The accumulators are `NBLK * TILES * 4` floats, so NBLK=4 with TILES=2 is 32
// of them plus the eight word pairs and the A fragment. That is the register
// budget this shape trades occupancy for.

#define MMQ_WIDE_BODY(WARPS, NBLK, TILES)                                      \
    __shared__ int8_t xs[(TILES) * MMQ_M * MMQ_STRIDE];                        \
    __shared__ float xd[(TILES) * MMQ_M * MMQ_GROUPS];                         \
    __shared__ float xsum[(TILES) * MMQ_M * MMQ_GROUPS];                       \
    /* One scale and offset per row per 128-weight block. */                   \
    __shared__ float wd[(WARPS) * (NBLK) * 8 * 2];                             \
    __shared__ float wm[(WARPS) * (NBLK) * 8 * 2];                             \
                                                                               \
    const int tid = threadIdx.x;                                               \
    const int lane = tid % WARP_SIZE;                                          \
    const int warp = tid / WARP_SIZE;                                          \
    const int nthreads = (WARPS) * WARP_SIZE;                                  \
    const int mrows = (WARPS) * (NBLK) * 8;                                     \
    const int row0 = blockIdx.x * mrows;                                        \
    const int tok0 = blockIdx.y * (TILES) * MMQ_M;                              \
    const int kb_total = k / 32;                                                \
    const int nb_total = k / QK_G128;                                           \
    const int n_tiles = (kb_total + MMQ_GROUPS - 1) / MMQ_GROUPS;               \
    const int x_rows = (TILES) * MMQ_M;                                         \
    const int x_valid = min(x_rows, max(0, n_tokens - tok0));                   \
                                                                                \
    const int bc = mma_b_col(lane);                                             \
    const int kq = mma_k0(lane);                                                \
    const int cr = mma_c_row(lane);                                             \
    const int cc = mma_c_col(lane);                                             \
    /* This warp's rows are contiguous: NBLK blocks of eight. */                \
    const int wbase = warp * (NBLK) * 8;                                        \
                                                                                \
    const block_q4_g128* bsrc[NBLK];                                            \
    bool brow_ok[NBLK];                                                         \
    _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {                        \
        const int r = row0 + wbase + j * 8 + bc;                                 \
        brow_ok[j] = r < n;                                                      \
        bsrc[j] = wq + (brow_ok[j] ? (size_t)r * nb_total : 0);                  \
    }                                                                           \
                                                                                \
    float acc[NBLK][TILES][4];                                                  \
    _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {                        \
        _Pragma("unroll") for (int u = 0; u < (TILES); ++u) {                    \
            _Pragma("unroll") for (int c = 0; c < 4; ++c) acc[j][u][c] = 0.0f;   \
        }                                                                       \
    }                                                                           \
                                                                                \
    mmq_zero_x(xs, xd, xsum, x_valid, x_rows, tid, nthreads);                   \
                                                                                \
    for (int kt = 0; kt < n_tiles; ++kt) {                                      \
        __syncthreads();                                                        \
        mmq_load_x(xs, xd, xsum, xq, kb_total, kt * MMQ_GROUPS, tok0, x_valid,  \
                   tid, nthreads);                                              \
        for (int i = tid; i < mrows * 2; i += nthreads) {                        \
            const int r = i / 2;                                                 \
            const int h = i % 2;                                                 \
            const int gr = row0 + r;                                             \
            const int kb = kt * 2 + h;                                          \
            float dd = 0.0f, mm = 0.0f;                                          \
            if (gr < n && kb < nb_total) {                                       \
                const __half2 ds = wq[(size_t)gr * nb_total + kb].ds;            \
                dd = __low2float(ds);                                            \
                mm = -__high2float(ds);                                          \
            }                                                                    \
            wd[i] = dd;                                                          \
            wm[i] = mm;                                                          \
        }                                                                        \
        __syncthreads();                                                         \
                                                                                 \
        /* Two 128-weight blocks per 256-wide tile, two byte runs each, and two  \
           nibble halves per run: the eight groups, four word pairs. */          \
        _Pragma("unroll") for (int blk = 0; blk < 2; ++blk) {                     \
            const int kb = kt * 2 + blk;                                          \
            _Pragma("unroll") for (int run = 0; run < 2; ++run) {                 \
                uint32_t v0[NBLK], v1[NBLK];                                      \
                _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {              \
                    v0[j] = 0;                                                    \
                    v1[j] = 0;                                                    \
                    if (brow_ok[j] && kb < nb_total) {                            \
                        const uint8_t* pq = bsrc[j][kb].qs + run * 32 + kq;        \
                        v0[j] = *(const uint32_t*)(const void*)pq;                 \
                        v1[j] = *(const uint32_t*)(const void*)(pq + 16);          \
                    }                                                             \
                }                                                                 \
                _Pragma("unroll") for (int h = 0; h < 2; ++h) {                    \
                    const int g = blk * 4 + run + h * 2;                           \
                    mma_a_s8 a[TILES];                                             \
                    _Pragma("unroll") for (int u = 0; u < (TILES); ++u) {          \
                        ldmatrix_a_s8(a[u], xs + u * MMQ_M * MMQ_STRIDE + g * 32,  \
                                      MMQ_STRIDE);                                 \
                    }                                                             \
                    _Pragma("unroll") for (int u = 0; u < (TILES); ++u) {          \
                        const int t0 = (u * MMQ_M + cr) * MMQ_GROUPS + g;          \
                        const int t1 = (u * MMQ_M + cr + 8) * MMQ_GROUPS + g;      \
                        const float xdl = xd[t0], xsl = xsum[t0];                  \
                        const float xdh = xd[t1], xsh = xsum[t1];                  \
                        _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {       \
                            mma_b_s8 b;                                            \
                            b.x[0] = (int)((v0[j] >> (h * 4)) & 0x0F0F0F0Fu);       \
                            b.x[1] = (int)((v1[j] >> (h * 4)) & 0x0F0F0F0Fu);       \
                            mma_c_s32 d = {{0, 0, 0, 0}};                          \
                            mma_s8(d, a[u], b);                                    \
                            const int s0 = (wbase + j * 8 + cc) * 2 + blk;          \
                            const int s1 = (wbase + j * 8 + cc + 1) * 2 + blk;      \
                            const float wd0 = wd[s0], wd1 = wd[s1];                 \
                            const float wm0 = wm[s0], wm1 = wm[s1];                 \
                            acc[j][u][0] += wd0 * xdl * (float)d.x[0] + wm0 * xsl;  \
                            acc[j][u][1] += wd1 * xdl * (float)d.x[1] + wm1 * xsl;  \
                            acc[j][u][2] += wd0 * xdh * (float)d.x[2] + wm0 * xsh;  \
                            acc[j][u][3] += wd1 * xdh * (float)d.x[3] + wm1 * xsh;  \
                        }                                                           \
                    }                                                              \
                }                                                                  \
            }                                                                      \
        }                                                                          \
    }                                                                              \
                                                                                   \
    _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {                            \
        const int orow = row0 + wbase + j * 8 + cc;                                 \
        _Pragma("unroll") for (int u = 0; u < (TILES); ++u) {                        \
            const int ot0 = tok0 + u * MMQ_M + cr;                                  \
            const int ot1 = ot0 + 8;                                                \
            if (ot0 < n_tokens) {                                                   \
                if (orow < n) out[(size_t)ot0 * n + orow] = acc[j][u][0];           \
                if (orow + 1 < n) out[(size_t)ot0 * n + orow + 1] = acc[j][u][1];   \
            }                                                                       \
            if (ot1 < n_tokens) {                                                   \
                if (orow < n) out[(size_t)ot1 * n + orow] = acc[j][u][2];           \
                if (orow + 1 < n) out[(size_t)ot1 * n + orow + 1] = acc[j][u][3];   \
            }                                                                       \
        }                                                                           \
    }

#define MMQ_WIDE_SET(SUFFIX, WARPS, NBLK, TILES)                               \
    extern "C" __global__ __launch_bounds__((WARPS) * WARP_SIZE) void          \
    mmqx##SUFFIX##_q4_g128(float* __restrict__ out,                            \
                           const void* __restrict__ wv,                        \
                           const void* __restrict__ xv, int k, int n,          \
                           int n_tokens) {                                     \
        const block_q4_g128* wq = (const block_q4_g128*)wv;                    \
        const block_q8_1* xq = (const block_q8_1*)xv;                          \
        MMQ_WIDE_BODY(WARPS, NBLK, TILES)                                      \
    }

/* Named mmqx<nblk>w<warps>[_<tiles>]; the rows a block covers is
   warps * nblk * 8. */
MMQ_WIDE_SET(2w4, 4, 2, 1)
MMQ_WIDE_SET(2w4_2, 4, 2, 2)
MMQ_WIDE_SET(4w2, 2, 4, 1)
MMQ_WIDE_SET(4w2_2, 2, 4, 2)
MMQ_WIDE_SET(4w4, 4, 4, 1)
MMQ_WIDE_SET(4w4_2, 4, 4, 2)
MMQ_WIDE_SET(8w2, 2, 8, 1)
MMQ_WIDE_SET(8w2_2, 2, 8, 2)

// ---- nibbles to f16 without an integer accumulator ----------------------
//
// `dequant.h`'s `lop3` path. A 4-bit field placed in the mantissa of f16
// 1024.0 (`EX = 0x64006400`) reads back as `1024 + n`, because 1024 is
// `2^10 * 1.0` and the mantissa has ten bits — so adding `n` to the mantissa
// adds exactly `n` to the value. One `lop3` does the mask and the or for two
// nibbles at once, and `hsub2` against 1024 recovers `n` exactly, both sides
// being integers a half represents without error.
//
// The subtraction has to happen before the scale, not folded into it. Folding
// gives `(1024 + n) * s - (1024 + z) * s`, two quantities near 2.0 whose
// difference is near 0.03, and an f16 near 2.0 has an ulp of 0.002 — the
// answer would be noise. Marlin splits it the same way and for the same
// reason.
//
// After the subtraction there is no such hazard: `n` is an exact integer in
// [0, 15] and `n * s - s * z` is an ordinary `hfma2` on quantities of the same
// size. The result has four bits of information in it, so representing it as a
// half loses nothing that was there.

template <int lut>
__device__ __forceinline__ uint32_t mmq_lop3(uint32_t a, uint32_t b,
                                             uint32_t c) {
    uint32_t r;
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                 : "=r"(r)
                 : "r"(a), "r"(b), "r"(c), "n"(lut));
    return r;
}

// Four packed weight bytes to four scaled halves, as the two registers of a
// `m16n8k16` B fragment.
//
// `v` is the word this lane reads out of a `block_q4_g128` run; `h` picks the
// nibble half, which is what selects between the two groups sharing those
// bytes. The pairing is (byte 0, byte 1) into the first register and
// (byte 2, byte 3) into the second, which is why `prmt` runs first: `lop3`
// masks bits 0-3 and 16-19, so the two bytes of a pair have to sit in bytes 0
// and 2 of the word.
//
// That pairing is a choice, and it is the one that keeps the *activation* side
// contiguous — see `MMQ_F16_K` below.
__device__ __forceinline__ void mmq_deq4_f16(uint32_t v, int h, __half2 s2,
                                             __half2 m2, unsigned* f) {
    uint32_t p0 = __byte_perm(v, 0, 0x4140);  // [b0, 0, b1, 0]
    uint32_t p1 = __byte_perm(v, 0, 0x4342);  // [b2, 0, b3, 0]
    if (h) {
        p0 >>= 4;
        p1 >>= 4;
    }
    const uint32_t MASK = 0x000f000fu;
    const uint32_t EX = 0x64006400u;
    uint32_t q0 = mmq_lop3<(0xf0 & 0xcc) | 0xaa>(p0, MASK, EX);
    uint32_t q1 = mmq_lop3<(0xf0 & 0xcc) | 0xaa>(p1, MASK, EX);
    const __half2 bias = __float2half2_rn(1024.0f);
    const __half2 w0 = __hsub2(*(const __half2*)(const void*)&q0, bias);
    const __half2 w1 = __hsub2(*(const __half2*)(const void*)&q1, bias);
    const __half2 r0 = __hfma2(w0, s2, m2);
    const __half2 r1 = __hfma2(w1, s2, m2);
    f[0] = *(const unsigned*)(const void*)&r0;
    f[1] = *(const unsigned*)(const void*)&r1;
}

// Where a fragment's k lands in the block's own numbering.
//
// The pack was laid out for the s8 path, where a lane reads four *consecutive*
// weights. An `m16n8k16` B fragment instead wants k `(lane%4)*2, +1` in its
// first register and `+8, +9` in its second. Those are different orders, and
// the cheap way out is not to repack: A and B only have to agree with *each
// other* on what k means, and the activation side is staged by this kernel.
//
// So the weights keep their order and the numbering bends. Fragment position
// k' maps to the block's weight j as below, and the activation gather reads
// the halves at `j` — which, for this pairing, is eight contiguous bytes at
// offset `8 * (lane % 4)`. One 8-byte load per row against the s8 path's two
// 4-byte ones.
__device__ __forceinline__ int mmq_f16_k(int kp) {
    const int hi = kp / 8;          // which B register
    const int c = (kp % 8) / 2;     // lane % 4
    const int r = kp % 2;
    return c * 4 + hi * 2 + r;
}

// One warp, one 128-weight block of one row, dequantized to the logical k it
// claims. Both the `lop3` path and the numbering above are under test here: a
// wrong permutation writes a real weight to the wrong k, which the reference
// catches, rather than producing a plausible average.
extern "C" __global__ void mmq_deq4_f16_probe(const void* __restrict__ wv,
                                              float* __restrict__ out) {
    const block_q4_g128* b = (const block_q4_g128*)wv;
    const int lane = threadIdx.x;
    if (lane >= 4) return;
    const int c = lane;
    const __half2 ds = b->ds;
    const __half2 s2 = __float2half2_rn(__low2float(ds));
    const __half2 m2 = __float2half2_rn(-__high2float(ds));

#pragma unroll
    for (int run = 0; run < 2; ++run) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int gl = run + 2 * h;  // group within the block
#pragma unroll
            for (int mma = 0; mma < 2; ++mma) {
                const uint32_t v = *(const uint32_t*)(const void*)(
                    b->qs + run * 32 + mma * 16 + c * 4);
                unsigned f[2];
                mmq_deq4_f16(v, h, s2, m2, f);
                const __half* hv = (const __half*)(const void*)f;
#pragma unroll
                for (int t = 0; t < 4; ++t) {
                    const int j = mma * 16 + c * 4 + t;
                    out[gl * 32 + j] = __half2float(hv[t]);
                }
            }
        }
    }
}

// ---- cp.async ring buffer over the wide tile, for Q4_G128 ---------------
//
// Step one of the port in vendor/marlin/README.md. The wide tile above gives
// the MMAs something to be overlapped with; this gives the overlap. The pair
// has to land together, which is why this is a copy of `MMQ_WIDE_BODY` with
// its activation staging replaced rather than a new tile shape.
//
// What changes, and why each piece has to change with the others:
//
//   * The activation tile is `STAGES` deep and filled by `cp.async`, so the
//     copy for tile kt+STAGES-1 is in flight while tile kt is multiplying.
//     The loop is `marlin_template.h` 1516-1580 with one k-tile per stage.
//   * `cp.async` is a *copy*, not a transform, so the shared tile has to be
//     the `block_q8_1` bytes verbatim — 36 bytes per group, scale and quants
//     interleaved — instead of the expanded `xs`/`xd`/`xsum` the staged path
//     builds. That deletes the staging pass entirely: 18 `cp.async.cg` per
//     token row per tile against 64 word loads and 64 word stores.
//   * A 36-byte group stride is not 16-byte aligned, so `ldmatrix` cannot
//     address it and the A fragment goes back to the four-scalar gather. That
//     is +3 instructions per fragment, paid once per NBLK MMAs, against the
//     whole staging pass removed.
//   * The weight scales move from shared to registers, read straight from
//     global. Not an optimization: with them in shared the loop needs a second
//     `__syncthreads` per tile to publish them, and a barrier between the
//     `cp.async` issue and the MMAs is exactly the overlap this exists to buy.
//     They are two f16 in one word and every lane in the warp hits the same
//     lines, so this is an L1 read, not a memory one.
//
// The B operand stays on `mmqd_*`'s direct global-to-register path: step two
// of the porting order says Marlin's packed shared weights are an optimization
// on top, not a prerequisite, and mixing them in here would make a wrong
// answer unattributable.
//
// Measured on an A4000 against the AWQ 8B at 256 tokens of history, in tok/s,
// each arm cold and selected through `TUILI_MMQ_VARIANT`:
//
//                       batch 8   16     32
//   mmqd (the default)     329   565    527
//   mmqx4w4                200   362    505     the wide tile alone
//   mmqa4w4s4              204   380    579
//   mmqa2w4s4              212   395    623
//
// So the pair is what the README predicted and each half alone is not: the wide
// tile by itself is 4% *down* at 32 tokens, and pipelining its activations puts
// it 18% up. That is the first structural change on this kernel to measure
// anything — five before it measured zero.
//
// This is only the first of Marlin's two pipeline levels; `mmqr_*` below adds
// the second and is worth another 14%, so read these numbers as a step rather
// than as what the shape is worth.
//
// It is also still a 30% loss at 16 tokens and worse below, which is why the
// default stays `mmqd`. The reason is the one this file keeps running into: a
// block here covers 64 or 128 weight rows, so a 4096-row projection makes 32 to
// 64 blocks against 48 SMs, and four stages of ring buffer cut the resident
// blocks further. At 32 tokens there is finally enough arithmetic per block to
// pay for that; at 8 there is not. Marlin answers this with a partition sized
// from the device rather than the matrix, which `mmqs_*` above already has.
//
// The shape sweep bottoms out rather than running to the narrowest block: 64
// rows per block (`2w4`) beats both 128 (`4w4`, 579) and 32 (`2w2`, 530) at 32
// tokens, while 32 rows is the best of the three at 16. Two stages measured
// level with four at 128 rows and slightly under at 64 (612 against 623), so
// the depth is worth less than the tile width — but it is not worth nothing,
// which is what a synchronous double buffer measured.

// Bytes one token row of a 256-wide tile occupies in `block_q8_1` form, and
// the padded stride the ring buffer uses.
//
// The pad is what keeps the A gather conflict-free. Lane L reads row L/4 at
// byte offset (L%4)*4, so with a stride of S words the warp lands on banks
// (L/4)*S + L%4 (mod 32), and eight rows cover all 32 banks exactly when
// S % 32 is 4, 12, 20 or 28. 288 bytes gives S = 72 and a two-way conflict;
// 304 gives S = 76, which is 12 mod 32. 304 is also a multiple of 16, which
// `cp.async.cg` requires of its destination.
#define MMQ_XA_ROW (MMQ_GROUPS * 36)
#define MMQ_XA_STRIDE (MMQ_XA_ROW + 16)

// `cp_async4` and friends from `marlin.cuh`, raw PTX so nothing here needs
// `cuda_pipeline.h` and the module stays reachable from NVRTC.
//
// The predicate is spelled as a source size rather than Marlin's `@p`: a false
// predicate has to leave *zeros* behind, not leave the destination alone. A
// stage buffer is reused every STAGES tiles, so a k-tile that runs past the end
// of the row would otherwise multiply against the quants of tile kt-STAGES.
// Callers pass a valid source address either way — a zero-byte copy still forms
// the address.
__device__ __forceinline__ void mmq_cp_async16(void* dst, const void* src,
                                               bool pred) {
#if __CUDA_ARCH__ >= 800
    const unsigned s = (unsigned)__cvta_generic_to_shared(dst);
    const int sz = pred ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s),
                 "l"(src), "r"(sz));
#else
    (void)dst; (void)src; (void)pred;
#endif
}

#if __CUDA_ARCH__ >= 800
#define MMQ_CP_ASYNC_FENCE() asm volatile("cp.async.commit_group;\n" ::)
#define MMQ_CP_ASYNC_WAIT(N) asm volatile("cp.async.wait_group %0;\n" ::"n"(N))
#else
#define MMQ_CP_ASYNC_FENCE() do {} while (0)
#define MMQ_CP_ASYNC_WAIT(N) do {} while (0)
#endif

// Issue one stage: every token row's 288 bytes for tile TILE into buffer BUF.
//
// Rows past `x_valid` are skipped — they were zeroed once before the loop and
// nothing writes them again, which is the same trick `mmq_zero_x` plays and for
// the same reason: at one token that is fifteen sixteenths of the traffic.
//
// A chunk is 16 bytes and a tile row is 288, so chunks never straddle the end
// of the quantized row as long as k is a multiple of 128 — which Q4_G128's
// group size already forces, and the launch asserts.
/* LIMIT is one past the last tile worth fetching. It is `n_tiles` for the
   kernels that walk all of k, and the end of this block's run for the striped
   schedule, which stops at a row-group boundary. */
#define MMQ_XA_FETCH(BUF, TILE, LIMIT)                                         \
    do {                                                                       \
        const int _tl = (TILE);                                                \
        const bool _live = _tl < (LIMIT);                                      \
        for (int _i = tid; _i < x_valid * (MMQ_XA_ROW / 16); _i += nthreads) { \
            const int _r = _i / (MMQ_XA_ROW / 16);                             \
            const int _c = _i % (MMQ_XA_ROW / 16);                             \
            const bool _hit =                                                  \
                _live && (_tl * MMQ_XA_ROW + _c * 16 + 16 <= kb_total * 36);   \
            const size_t _off =                                                \
                ((size_t)(tok0 + _r) * kb_total + (size_t)_tl * MMQ_GROUPS)    \
                    * 36 + _c * 16;                                            \
            mmq_cp_async16(                                                    \
                xa + ((BUF) * x_rows + _r) * MMQ_XA_STRIDE + _c * 16,          \
                xbytes + (_hit ? _off : 0), _hit);                             \
        }                                                                      \
        MMQ_CP_ASYNC_FENCE();                                                  \
    } while (0)

/* `cp.async.cg` requires a 16-byte-aligned destination and an `int8_t` array
   is only guaranteed one, so say so rather than rely on what nvcc happens to
   emit — a misaligned base faults at run time, on a shape that may not be the
   one the tests cover. `MMQ_XA_STRIDE` keeps every row on the same boundary. */
#define MMQ_ASYNC_BODY(WARPS, NBLK, TILES, STAGES)                             \
    __shared__ __align__(16)                                                   \
        int8_t xa[(STAGES) * (TILES) * MMQ_M * MMQ_XA_STRIDE];                 \
                                                                               \
    const int tid = threadIdx.x;                                               \
    const int lane = tid % WARP_SIZE;                                          \
    const int warp = tid / WARP_SIZE;                                          \
    const int nthreads = (WARPS) * WARP_SIZE;                                  \
    const int mrows = (WARPS) * (NBLK) * 8;                                    \
    const int row0 = blockIdx.x * mrows;                                       \
    const int tok0 = blockIdx.y * (TILES) * MMQ_M;                             \
    const int kb_total = k / 32;                                               \
    const int nb_total = k / QK_G128;                                          \
    const int n_tiles = (kb_total + MMQ_GROUPS - 1) / MMQ_GROUPS;              \
    const int x_rows = (TILES) * MMQ_M;                                        \
    const int x_valid = min(x_rows, max(0, n_tokens - tok0));                  \
    const uint8_t* xbytes = (const uint8_t*)xq;                                \
                                                                               \
    const int ar = mma_a_row(lane);                                            \
    const int bc = mma_b_col(lane);                                            \
    const int kq = mma_k0(lane);                                               \
    const int cr = mma_c_row(lane);                                            \
    const int cc = mma_c_col(lane);                                            \
    const int wbase = warp * (NBLK) * 8;                                       \
                                                                               \
    const block_q4_g128* bsrc[NBLK];                                           \
    bool brow_ok[NBLK];                                                        \
    _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {                       \
        const int r = row0 + wbase + j * 8 + bc;                               \
        brow_ok[j] = r < n;                                                    \
        bsrc[j] = wq + (brow_ok[j] ? (size_t)r * nb_total : 0);                \
    }                                                                          \
                                                                               \
    float acc[NBLK][TILES][4];                                                 \
    _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {                       \
        _Pragma("unroll") for (int u = 0; u < (TILES); ++u) {                  \
            _Pragma("unroll") for (int c = 0; c < 4; ++c) acc[j][u][c] = 0.0f; \
        }                                                                      \
    }                                                                          \
                                                                               \
    /* Masked token rows, in every stage, once. */                             \
    if (x_valid < x_rows) {                                                    \
        const int per = (x_rows - x_valid) * (MMQ_XA_ROW / 4);                 \
        for (int i = tid; i < (STAGES) * per; i += nthreads) {                 \
            const int s = i / per;                                            \
            const int j = i % per;                                            \
            const int r = x_valid + j / (MMQ_XA_ROW / 4);                     \
            const int e = (j % (MMQ_XA_ROW / 4)) * 4;                         \
            *(uint32_t*)(void*)(xa + (s * x_rows + r) * MMQ_XA_STRIDE + e) = 0;\
        }                                                                      \
    }                                                                          \
                                                                               \
    /* Prime the pipe: STAGES-1 tiles in flight before the first multiply. */  \
    _Pragma("unroll") for (int s = 0; s < (STAGES) - 1; ++s) {                 \
        MMQ_XA_FETCH(s, s, n_tiles);                                           \
    }                                                                          \
                                                                               \
    for (int kt = 0; kt < n_tiles; ++kt) {                                     \
        /* At most STAGES-2 groups outstanding means tile kt has landed. The   \
           barrier is what publishes it to the other threads — `wait_group`    \
           only orders this thread's own copies — and it doubles as the        \
           release of the buffer about to be overwritten, which held tile      \
           kt-1 and was read to the end of the previous iteration. */          \
        MMQ_CP_ASYNC_WAIT((STAGES) - 2);                                       \
        __syncthreads();                                                       \
        MMQ_XA_FETCH((kt + (STAGES) - 1) % (STAGES), kt + (STAGES) - 1,        \
                     n_tiles);                                                 \
        const int8_t* xbuf = xa + (kt % (STAGES)) * x_rows * MMQ_XA_STRIDE;    \
                                                                               \
        _Pragma("unroll") for (int blk = 0; blk < 2; ++blk) {                  \
            const int kb = kt * 2 + blk;                                       \
            const bool kb_ok = kb < nb_total;                                  \
            /* One word per row holds both the scale and the zero offset. */   \
            float swd0[NBLK], swd1[NBLK], swm0[NBLK], swm1[NBLK];              \
            _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {               \
                const int sr = row0 + wbase + j * 8 + cc;                      \
                __half2 s0 = __floats2half2_rn(0.0f, 0.0f);                    \
                __half2 s1 = s0;                                               \
                if (kb_ok && sr < n) s0 = wq[(size_t)sr * nb_total + kb].ds;   \
                if (kb_ok && sr + 1 < n)                                       \
                    s1 = wq[(size_t)(sr + 1) * nb_total + kb].ds;              \
                swd0[j] = __low2float(s0);                                     \
                swm0[j] = -__high2float(s0);                                   \
                swd1[j] = __low2float(s1);                                     \
                swm1[j] = -__high2float(s1);                                   \
            }                                                                  \
            _Pragma("unroll") for (int run = 0; run < 2; ++run) {              \
                uint32_t v0[NBLK], v1[NBLK];                                   \
                _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {           \
                    v0[j] = 0;                                                 \
                    v1[j] = 0;                                                 \
                    if (brow_ok[j] && kb_ok) {                                 \
                        const uint8_t* pq = bsrc[j][kb].qs + run * 32 + kq;    \
                        v0[j] = *(const uint32_t*)(const void*)pq;             \
                        v1[j] = *(const uint32_t*)(const void*)(pq + 16);      \
                    }                                                          \
                }                                                              \
                _Pragma("unroll") for (int h = 0; h < 2; ++h) {                \
                    const int g = blk * 4 + run + h * 2;                       \
                    /* Group g starts 4 bytes into its 36-byte block: the      \
                       scale word comes first, the 32 quants after it. */      \
                    mma_a_s8 a[TILES];                                         \
                    _Pragma("unroll") for (int u = 0; u < (TILES); ++u) {      \
                        const int8_t* ap = xbuf                                \
                                         + (u * MMQ_M + ar) * MMQ_XA_STRIDE    \
                                         + g * 36 + 4 + kq;                    \
                        const int8_t* aq = ap + 8 * MMQ_XA_STRIDE;             \
                        a[u].x[0] = *(const int*)(const void*)ap;              \
                        a[u].x[1] = *(const int*)(const void*)aq;              \
                        a[u].x[2] = *(const int*)(const void*)(ap + 16);       \
                        a[u].x[3] = *(const int*)(const void*)(aq + 16);       \
                    }                                                          \
                    _Pragma("unroll") for (int u = 0; u < (TILES); ++u) {      \
                        const __half2 dl = *(const __half2*)(const void*)(     \
                            xbuf + (u * MMQ_M + cr) * MMQ_XA_STRIDE + g * 36); \
                        const __half2 dh = *(const __half2*)(const void*)(     \
                            xbuf + (u * MMQ_M + cr + 8) * MMQ_XA_STRIDE        \
                            + g * 36);                                         \
                        const float xdl = __low2float(dl);                     \
                        const float xsl = __high2float(dl);                    \
                        const float xdh = __low2float(dh);                     \
                        const float xsh = __high2float(dh);                    \
                        _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {   \
                            mma_b_s8 b;                                        \
                            b.x[0] = (int)((v0[j] >> (h * 4)) & 0x0F0F0F0Fu);  \
                            b.x[1] = (int)((v1[j] >> (h * 4)) & 0x0F0F0F0Fu);  \
                            mma_c_s32 d = {{0, 0, 0, 0}};                      \
                            mma_s8(d, a[u], b);                                \
                            acc[j][u][0] +=                                    \
                                swd0[j] * xdl * (float)d.x[0] + swm0[j] * xsl; \
                            acc[j][u][1] +=                                    \
                                swd1[j] * xdl * (float)d.x[1] + swm1[j] * xsl; \
                            acc[j][u][2] +=                                    \
                                swd0[j] * xdh * (float)d.x[2] + swm0[j] * xsh; \
                            acc[j][u][3] +=                                    \
                                swd1[j] * xdh * (float)d.x[3] + swm1[j] * xsh; \
                        }                                                      \
                    }                                                          \
                }                                                              \
            }                                                                  \
        }                                                                      \
    }                                                                          \
                                                                               \
    _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {                       \
        const int orow = row0 + wbase + j * 8 + cc;                            \
        _Pragma("unroll") for (int u = 0; u < (TILES); ++u) {                  \
            const int ot0 = tok0 + u * MMQ_M + cr;                             \
            const int ot1 = ot0 + 8;                                           \
            if (ot0 < n_tokens) {                                              \
                if (orow < n) out[(size_t)ot0 * n + orow] = acc[j][u][0];      \
                if (orow + 1 < n)                                              \
                    out[(size_t)ot0 * n + orow + 1] = acc[j][u][1];            \
            }                                                                  \
            if (ot1 < n_tokens) {                                              \
                if (orow < n) out[(size_t)ot1 * n + orow] = acc[j][u][2];      \
                if (orow + 1 < n)                                              \
                    out[(size_t)ot1 * n + orow + 1] = acc[j][u][3];            \
            }                                                                  \
        }                                                                      \
    }

#define MMQ_ASYNC_SET(SUFFIX, WARPS, NBLK, TILES, STAGES)                      \
    extern "C" __global__ __launch_bounds__((WARPS) * WARP_SIZE) void          \
    mmqa##SUFFIX##_q4_g128(float* __restrict__ out,                            \
                           const void* __restrict__ wv,                        \
                           const void* __restrict__ xv, int k, int n,          \
                           int n_tokens) {                                     \
        const block_q4_g128* wq = (const block_q4_g128*)wv;                    \
        const block_q8_1* xq = (const block_q8_1*)xv;                          \
        MMQ_ASYNC_BODY(WARPS, NBLK, TILES, STAGES)                             \
    }

/* Named mmqa<nblk>w<warps>s<stages>[_<tiles>]. The stage count is in the name
   because the ring buffer is 4.8 KB per stage per token tile, which is where
   this shape spends its occupancy: four stages and one token tile is 19 KB
   against the wide tile's 7.4, so five resident blocks per SM against
   thirteen. That trade pays at 32 tokens and not below — see the table above,
   which is why two and four stages are both still here. */
MMQ_ASYNC_SET(4w4s4, 4, 4, 1, 4)
MMQ_ASYNC_SET(4w4s4_2, 4, 4, 2, 4)
MMQ_ASYNC_SET(4w4s2, 4, 4, 1, 2)
MMQ_ASYNC_SET(4w4s2_2, 4, 4, 2, 2)
MMQ_ASYNC_SET(4w2s4, 2, 4, 1, 4)
MMQ_ASYNC_SET(4w2s4_2, 2, 4, 2, 4)
MMQ_ASYNC_SET(2w4s4, 4, 2, 1, 4)
MMQ_ASYNC_SET(2w4s4_2, 4, 2, 2, 4)
MMQ_ASYNC_SET(2w4s2, 4, 2, 1, 2)
MMQ_ASYNC_SET(2w4s2_2, 4, 2, 2, 2)
MMQ_ASYNC_SET(2w2s4, 2, 2, 1, 4)
MMQ_ASYNC_SET(2w2s4_2, 2, 2, 2, 4)

// ---- the second pipeline level, for Q4_G128 -----------------------------
//
// `mmqa_*` above has Marlin's global-to-shared pipeline and not its
// shared-to-register one. Marlin's is two levels: `fetch_to_registers(k + 1,
// pipe)` (`marlin_template.h:853`) fills `frag_a[(k+1) % 2]` while `matmul(k)`
// multiplies out of `frag_a[k % 2]`, so every operand load — the `ldsm` from
// shared and, here, the weight words from global — is a full k-substep ahead
// of the MMA that consumes it. This adds that level.
//
// One k-substep here is a (128-weight block, byte run) pair, four to a tile,
// because that is the unit one pair of weight words serves: groups g and g+2
// are its two nibble halves. So `b_sh_wr_iters` is 4 and the register buffer
// holds one step's worth — the NBLK weight word pairs and their scales, and
// both halves' A fragments and activation scales.
//
// Where the `cp.async` issue sits is not a free choice, and this is the part
// that is easy to get subtly wrong. Marlin issues it at `k == b_sh_wr_iters -
// 2` (`marlin_template.h:1566`), overwriting the buffer *one stage behind* the
// current one, and then waits. That position is what makes the overwrite safe:
// the last read of that buffer happened at the same position one stage earlier,
// with `wait_for_stage`'s `__syncthreads` immediately after it. Issue it at the
// top of the tile instead — which is what `mmqa_*` does, and can afford to,
// having no reads outstanding — and a thread still one step behind is reading
// the bytes being overwritten, with no barrier between them.
//
// The other half of that choice: the last step of a tile prefetches the *first*
// step of the next one, out of a buffer the barrier at step 2 has just
// published. That is what lets the register pipeline run unbroken across the
// tile boundary rather than draining and re-priming four times per k-tile.
//
// Measured against `mmqa_*` of the same shape, which differs from these by this
// level and nothing else (tok/s, AWQ 8B, 256 tokens of history):
//
//                    batch 8   16    32
//   mmqa2w4s4          211    395   613
//   mmqr2w4s4          242    447   702      +14%  +13%  +14%
//   mmqa4w4s4          204    380   579
//   mmqr4w4s4          206    388   604      +1%   +2%   +4%
//
// Worth more than the level below it and, unlike that one, worth the same at
// every batch width — which makes sense: the first level hides the weight
// stream, and how much of that there is to hide per block scales with the
// tokens. This level hides an operand load behind the MMA that precedes it,
// and that ratio does not move with the batch.
//
// The gain is 14% at 64 weight rows per block and inside the noise at 128. Not
// register pressure: the driver reports every one of these kernels limited by
// its `__launch_bounds__` rather than below it, so registers are not the
// constraint on any of them. It is the block count this file keeps arriving at
// — 128 rows per block is 32 blocks for a 4096-row projection against 48 SMs,
// and hiding latency inside a block does nothing for an SM that has no block.
//
// One caution about reading the small differences: three cold runs of the same
// `mmqa2w4s4` binary measured 617, 623 and 613 at 32 tokens. That is 0.8%, not
// the 0.2% the README claims for `batch_bench`, so the 4% above is about three
// times the spread and the 14% is far outside it.

// One step's registers. `SLOT` is `step % 2` and has to fold to a constant, so
// every loop that reaches this is fully unrolled.
#define MMQ_RP_FETCH(NBLK, TILES, SLOT, ST, KTN, BUF)                          \
    do {                                                                       \
        const int _blk = (ST) / 2;                                             \
        const int _run = (ST) % 2;                                             \
        const int _kb = (KTN) * 2 + _blk;                                      \
        const bool _kb_ok = _kb < nb_total;                                    \
        _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {                   \
            v0[SLOT][j] = 0;                                                   \
            v1[SLOT][j] = 0;                                                   \
            if (brow_ok[j] && _kb_ok) {                                        \
                const uint8_t* pq = bsrc[j][_kb].qs + _run * 32 + kq;          \
                v0[SLOT][j] = *(const uint32_t*)(const void*)pq;               \
                v1[SLOT][j] = *(const uint32_t*)(const void*)(pq + 16);        \
            }                                                                  \
            const int _sr = row0 + wbase + j * 8 + cc;                         \
            __half2 _s0 = __floats2half2_rn(0.0f, 0.0f);                       \
            __half2 _s1 = _s0;                                                 \
            if (_kb_ok && _sr < n)                                             \
                _s0 = wq[(size_t)_sr * nb_total + _kb].ds;                     \
            if (_kb_ok && _sr + 1 < n)                                         \
                _s1 = wq[(size_t)(_sr + 1) * nb_total + _kb].ds;               \
            swd0[SLOT][j] = __low2float(_s0);                                  \
            swm0[SLOT][j] = -__high2float(_s0);                                \
            swd1[SLOT][j] = __low2float(_s1);                                  \
            swm1[SLOT][j] = -__high2float(_s1);                                \
        }                                                                      \
        _Pragma("unroll") for (int h = 0; h < 2; ++h) {                        \
            const int _g = _blk * 4 + _run + h * 2;                            \
            _Pragma("unroll") for (int u = 0; u < (TILES); ++u) {              \
                const int8_t* _ap = (BUF)                                      \
                                  + (u * MMQ_M + ar) * MMQ_XA_STRIDE           \
                                  + _g * 36 + 4 + kq;                          \
                const int8_t* _aq = _ap + 8 * MMQ_XA_STRIDE;                   \
                fa[SLOT][h][u].x[0] = *(const int*)(const void*)_ap;           \
                fa[SLOT][h][u].x[1] = *(const int*)(const void*)_aq;           \
                fa[SLOT][h][u].x[2] = *(const int*)(const void*)(_ap + 16);    \
                fa[SLOT][h][u].x[3] = *(const int*)(const void*)(_aq + 16);    \
                const __half2 _dl = *(const __half2*)(const void*)(            \
                    (BUF) + (u * MMQ_M + cr) * MMQ_XA_STRIDE + _g * 36);       \
                const __half2 _dh = *(const __half2*)(const void*)(            \
                    (BUF) + (u * MMQ_M + cr + 8) * MMQ_XA_STRIDE + _g * 36);   \
                fxdl[SLOT][h][u] = __low2float(_dl);                           \
                fxsl[SLOT][h][u] = __high2float(_dl);                          \
                fxdh[SLOT][h][u] = __low2float(_dh);                           \
                fxsh[SLOT][h][u] = __high2float(_dh);                          \
            }                                                                  \
        }                                                                      \
    } while (0)

/* Registers only — every operand was loaded a step ago. This is the whole
   point of the level: `marlin_template.h`'s `matmul` touches no memory. */
#define MMQ_RP_MUL(NBLK, TILES, SLOT)                                          \
    _Pragma("unroll") for (int h = 0; h < 2; ++h) {                            \
        _Pragma("unroll") for (int u = 0; u < (TILES); ++u) {                  \
            _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {               \
                mma_b_s8 b;                                                    \
                b.x[0] = (int)((v0[SLOT][j] >> (h * 4)) & 0x0F0F0F0Fu);        \
                b.x[1] = (int)((v1[SLOT][j] >> (h * 4)) & 0x0F0F0F0Fu);        \
                mma_c_s32 d = {{0, 0, 0, 0}};                                  \
                mma_s8(d, fa[SLOT][h][u], b);                                  \
                acc[j][u][0] += swd0[SLOT][j] * fxdl[SLOT][h][u]               \
                                    * (float)d.x[0]                            \
                              + swm0[SLOT][j] * fxsl[SLOT][h][u];              \
                acc[j][u][1] += swd1[SLOT][j] * fxdl[SLOT][h][u]               \
                                    * (float)d.x[1]                            \
                              + swm1[SLOT][j] * fxsl[SLOT][h][u];              \
                acc[j][u][2] += swd0[SLOT][j] * fxdh[SLOT][h][u]               \
                                    * (float)d.x[2]                            \
                              + swm0[SLOT][j] * fxsh[SLOT][h][u];              \
                acc[j][u][3] += swd1[SLOT][j] * fxdh[SLOT][h][u]               \
                                    * (float)d.x[3]                            \
                              + swm1[SLOT][j] * fxsh[SLOT][h][u];              \
            }                                                                  \
        }                                                                      \
    }

#define MMQ_REGPIPE_BODY(WARPS, NBLK, TILES, STAGES)                           \
    __shared__ __align__(16)                                                   \
        int8_t xa[(STAGES) * (TILES) * MMQ_M * MMQ_XA_STRIDE];                 \
                                                                               \
    const int tid = threadIdx.x;                                               \
    const int lane = tid % WARP_SIZE;                                          \
    const int warp = tid / WARP_SIZE;                                          \
    const int nthreads = (WARPS) * WARP_SIZE;                                  \
    const int mrows = (WARPS) * (NBLK) * 8;                                    \
    const int row0 = blockIdx.x * mrows;                                       \
    const int tok0 = blockIdx.y * (TILES) * MMQ_M;                             \
    const int kb_total = k / 32;                                               \
    const int nb_total = k / QK_G128;                                          \
    const int n_tiles = (kb_total + MMQ_GROUPS - 1) / MMQ_GROUPS;              \
    const int x_rows = (TILES) * MMQ_M;                                        \
    const int x_valid = min(x_rows, max(0, n_tokens - tok0));                  \
    const uint8_t* xbytes = (const uint8_t*)xq;                                \
                                                                               \
    const int ar = mma_a_row(lane);                                            \
    const int bc = mma_b_col(lane);                                            \
    const int kq = mma_k0(lane);                                               \
    const int cr = mma_c_row(lane);                                            \
    const int cc = mma_c_col(lane);                                            \
    const int wbase = warp * (NBLK) * 8;                                       \
                                                                               \
    const block_q4_g128* bsrc[NBLK];                                           \
    bool brow_ok[NBLK];                                                        \
    _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {                       \
        const int r = row0 + wbase + j * 8 + bc;                               \
        brow_ok[j] = r < n;                                                    \
        bsrc[j] = wq + (brow_ok[j] ? (size_t)r * nb_total : 0);                \
    }                                                                          \
                                                                               \
    uint32_t v0[2][NBLK], v1[2][NBLK];                                         \
    float swd0[2][NBLK], swd1[2][NBLK], swm0[2][NBLK], swm1[2][NBLK];          \
    mma_a_s8 fa[2][2][TILES];                                                  \
    float fxdl[2][2][TILES], fxsl[2][2][TILES];                                \
    float fxdh[2][2][TILES], fxsh[2][2][TILES];                                \
                                                                               \
    float acc[NBLK][TILES][4];                                                 \
    _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {                       \
        _Pragma("unroll") for (int u = 0; u < (TILES); ++u) {                  \
            _Pragma("unroll") for (int c = 0; c < 4; ++c) acc[j][u][c] = 0.0f; \
        }                                                                      \
    }                                                                          \
                                                                               \
    if (x_valid < x_rows) {                                                    \
        const int per = (x_rows - x_valid) * (MMQ_XA_ROW / 4);                 \
        for (int i = tid; i < (STAGES) * per; i += nthreads) {                 \
            const int s = i / per;                                             \
            const int j = i % per;                                             \
            const int r = x_valid + j / (MMQ_XA_ROW / 4);                      \
            const int e = (j % (MMQ_XA_ROW / 4)) * 4;                          \
            *(uint32_t*)(void*)(xa + (s * x_rows + r) * MMQ_XA_STRIDE + e) = 0;\
        }                                                                      \
    }                                                                          \
                                                                               \
    /* `start_pipes`: STAGES-1 tiles in flight, then the first register step   \
       out of the one that has landed. */                                      \
    _Pragma("unroll") for (int s = 0; s < (STAGES) - 1; ++s) {                 \
        MMQ_XA_FETCH(s, s, n_tiles);                                           \
    }                                                                          \
    MMQ_CP_ASYNC_WAIT((STAGES) - 2);                                           \
    __syncthreads();                                                           \
    MMQ_RP_FETCH(NBLK, TILES, 0, 0, 0, xa);                                    \
                                                                               \
    for (int kt = 0; kt < n_tiles; ++kt) {                                     \
        const int8_t* xbuf = xa + (kt % (STAGES)) * x_rows * MMQ_XA_STRIDE;    \
        const int8_t* xnxt =                                                   \
            xa + ((kt + 1) % (STAGES)) * x_rows * MMQ_XA_STRIDE;               \
        _Pragma("unroll") for (int s = 0; s < 4; ++s) {                        \
            /* Step s+1, wrapping into the next tile at the last step. Past    \
               the end that reads a buffer whose contents are never            \
               multiplied, which costs the loads and nothing else. */          \
            if (s < 3) {                                                       \
                MMQ_RP_FETCH(NBLK, TILES, (s + 1) % 2, s + 1, kt, xbuf);       \
            } else {                                                           \
                MMQ_RP_FETCH(NBLK, TILES, (s + 1) % 2, 0, kt + 1, xnxt);       \
            }                                                                  \
            if (s == 2) {                                                      \
                MMQ_XA_FETCH((kt + (STAGES) - 1) % (STAGES),                   \
                             kt + (STAGES) - 1, n_tiles);                      \
                MMQ_CP_ASYNC_WAIT((STAGES) - 2);                               \
                __syncthreads();                                               \
            }                                                                  \
            MMQ_RP_MUL(NBLK, TILES, s % 2);                                    \
        }                                                                      \
    }                                                                          \
                                                                               \
    _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {                       \
        const int orow = row0 + wbase + j * 8 + cc;                            \
        _Pragma("unroll") for (int u = 0; u < (TILES); ++u) {                  \
            const int ot0 = tok0 + u * MMQ_M + cr;                             \
            const int ot1 = ot0 + 8;                                           \
            if (ot0 < n_tokens) {                                              \
                if (orow < n) out[(size_t)ot0 * n + orow] = acc[j][u][0];      \
                if (orow + 1 < n)                                              \
                    out[(size_t)ot0 * n + orow + 1] = acc[j][u][1];            \
            }                                                                  \
            if (ot1 < n_tokens) {                                              \
                if (orow < n) out[(size_t)ot1 * n + orow] = acc[j][u][2];      \
                if (orow + 1 < n)                                              \
                    out[(size_t)ot1 * n + orow + 1] = acc[j][u][3];            \
            }                                                                  \
        }                                                                      \
    }

#define MMQ_REGPIPE_SET(SUFFIX, WARPS, NBLK, TILES, STAGES)                    \
    extern "C" __global__ __launch_bounds__((WARPS) * WARP_SIZE) void          \
    mmqr##SUFFIX##_q4_g128(float* __restrict__ out,                            \
                           const void* __restrict__ wv,                        \
                           const void* __restrict__ xv, int k, int n,          \
                           int n_tokens) {                                     \
        const block_q4_g128* wq = (const block_q4_g128*)wv;                    \
        const block_q8_1* xq = (const block_q8_1*)xv;                          \
        MMQ_REGPIPE_BODY(WARPS, NBLK, TILES, STAGES)                           \
    }

/* Same names and shapes as `mmqa_*`, so the two are directly A/B-able and the
   second pipeline level is the only difference between them. The accumulation
   order is identical, so they should agree to rounding — which is what the
   test asserts, and is a sharper check than either against the float answer. */
MMQ_REGPIPE_SET(2w4s4, 4, 2, 1, 4)
MMQ_REGPIPE_SET(2w4s4_2, 4, 2, 2, 4)
MMQ_REGPIPE_SET(4w4s4, 4, 4, 1, 4)
MMQ_REGPIPE_SET(4w4s4_2, 4, 4, 2, 4)
MMQ_REGPIPE_SET(2w2s4, 2, 2, 1, 4)
MMQ_REGPIPE_SET(2w2s4_2, 2, 2, 2, 4)

// ---- the striped partition, re-run against the register pipeline --------
//
// `mmqs_*` above measured level with the `gridDim.z` split it was meant to
// replace, and the note there says why that null result was not surprising: the
// reduction traffic it saves was never the constraint, the block count was, and
// the cruder split already supplied the block count.
//
// That reasoning does not carry to `mmqr_*`. Its blocks are wider (64 or 128
// weight rows against 32) and hold four stages of ring buffer, so it makes
// fewer of them and fits fewer per SM — and the level-two measurement points at
// the block count directly, being worth 14% at 64 rows per block and 4% at 128.
// A partition sized from the device rather than the matrix is the one lever
// that adds blocks without taking threads or shared memory from anywhere, which
// is exactly the shape of that deficit. So the null result is worth re-running
// rather than inherited.
//
// The partition is `MMQ_STRIPED_BODY`'s, unchanged: (row group, k chunk) pairs
// flattened k-major, block `b` taking the contiguous run
// `[iters*b, iters*(b+1))`, whole runs storing and straddling ones adding. The
// inner loop is `MMQ_REGPIPE_BODY`'s.
//
// Two things the pipeline forces on top of that partition:
//
//   * The ring is indexed from the *run's* first tile, not from zero, and the
//     fetch limit is the run's end rather than `n_tiles`. A run stops at a row
//     group boundary, and fetching past it would stream weights for a row group
//     this block is not going to multiply.
//   * Each run drains with `cp_async_wait<0>` and a barrier before the next one
//     re-primes. Without it a copy still in flight from the tail of one run
//     lands in a buffer the next run has already filled. Runs are a whole row
//     group's k in the common case, so this costs one drain per row group.
//
// The null result did not hold. Against `mmqr2w4s4`, the same inner loop
// walking all of k (tok/s, AWQ 8B, 256 tokens of history):
//
//                 batch 4    8    16    32
//   mmqr2w4s4       123   244   447   713
//   mmqsr, bps 2    149   288   518   716
//   mmqsr, bps 4    130   251   459   733
//   mmqsr, bps 8    143   276   492   741
//   mmqsr, bps 16   163   312   548   734
//   mmqsr, bps 24   163   313   552   711
//   mmqsr, bps 32   168   321   565   708
//   mmqsr, bps 48   168   325   581   699
//
// Worth 30% at four tokens and 16-30% through the middle, which is the range
// this whole line of work had been losing in. It buys nothing at 32, where
// there were already enough blocks.
//
// Two things in that table are worth carrying forward:
//
//   * The two ends want opposite grids. Sixteen tokens rises to bps 48 and
//     beyond the sweep; 32 tokens peaks at 8 and falls off. The difference
//     between them is the token-tile count — one tile at 16, two at 32 — so a
//     block at 32 holds twice the arithmetic per k-tile and fewer of them
//     saturate the device, while more only add reduction traffic. That reading
//     is consistent with the table and is *not* separately verified; it is why
//     the default in `lib.rs` keys off `tiles` rather than off a measured
//     constant.
//   * bps 4, which is `mmqs`'s default and was this variant's until the sweep,
//     is the worst point in it — below bps 2 at every width. The curve is not
//     monotonic at the low end and no mechanism offered here explains that, so
//     it stays a measurement. Anyone benchmarking this variant without setting
//     `TUILI_MMQ_BPS` used to land exactly on the dip.
//
// Where that leaves the port, against the `mmqd` default measured cold from the
// same binary: 168 against 177 at four tokens, 323 against 329 at eight, 580
// against 565 at sixteen, 741 against 527 at thirty-two. At or above the
// default from eight tokens up, and 41% above it at thirty-two.

#define MMQ_STRIPED_REGPIPE_BODY(WARPS, NBLK, TILES, STAGES)                   \
    __shared__ __align__(16)                                                   \
        int8_t xa[(STAGES) * (TILES) * MMQ_M * MMQ_XA_STRIDE];                 \
                                                                               \
    const int tid = threadIdx.x;                                               \
    const int lane = tid % WARP_SIZE;                                          \
    const int warp = tid / WARP_SIZE;                                          \
    const int nthreads = (WARPS) * WARP_SIZE;                                  \
    const int mrows = (WARPS) * (NBLK) * 8;                                    \
    const int tok0 = blockIdx.y * (TILES) * MMQ_M;                             \
    const int kb_total = k / 32;                                               \
    const int nb_total = k / QK_G128;                                          \
    const int k_tiles = (kb_total + MMQ_GROUPS - 1) / MMQ_GROUPS;              \
    const int row_tiles = (n + mrows - 1) / mrows;                             \
    const int x_rows = (TILES) * MMQ_M;                                        \
    const int x_valid = min(x_rows, max(0, n_tokens - tok0));                  \
    const uint8_t* xbytes = (const uint8_t*)xq;                                \
                                                                               \
    const int ar = mma_a_row(lane);                                            \
    const int bc = mma_b_col(lane);                                            \
    const int kq = mma_k0(lane);                                               \
    const int cr = mma_c_row(lane);                                            \
    const int cc = mma_c_col(lane);                                            \
    const int wbase = warp * (NBLK) * 8;                                       \
                                                                               \
    /* This block's contiguous run of the flattened list. Uniform across the    \
       block, so every barrier below is reached by every thread. */            \
    const int total = row_tiles * k_tiles;                                     \
    const int iters = (total + (int)gridDim.x - 1) / (int)gridDim.x;           \
    int flat = iters * (int)blockIdx.x;                                        \
    const int flat_end = min(total, flat + iters);                             \
                                                                               \
    uint32_t v0[2][NBLK], v1[2][NBLK];                                         \
    float swd0[2][NBLK], swd1[2][NBLK], swm0[2][NBLK], swm1[2][NBLK];          \
    mma_a_s8 fa[2][2][TILES];                                                  \
    float fxdl[2][2][TILES], fxsl[2][2][TILES];                                \
    float fxdh[2][2][TILES], fxsh[2][2][TILES];                                \
    const block_q4_g128* bsrc[NBLK];                                           \
    bool brow_ok[NBLK];                                                        \
    float acc[NBLK][TILES][4];                                                 \
                                                                               \
    /* Masked token rows never change across runs. */                          \
    if (x_valid < x_rows) {                                                    \
        const int per = (x_rows - x_valid) * (MMQ_XA_ROW / 4);                 \
        for (int i = tid; i < (STAGES) * per; i += nthreads) {                 \
            const int s = i / per;                                            \
            const int j = i % per;                                            \
            const int r = x_valid + j / (MMQ_XA_ROW / 4);                     \
            const int e = (j % (MMQ_XA_ROW / 4)) * 4;                         \
            *(uint32_t*)(void*)(xa + (s * x_rows + r) * MMQ_XA_STRIDE + e) = 0;\
        }                                                                      \
    }                                                                          \
                                                                               \
    while (flat < flat_end) {                                                  \
        const int nt = flat / k_tiles;                                         \
        const int kt_lo = flat % k_tiles;                                      \
        const int kt_hi = min(k_tiles, kt_lo + (flat_end - flat));             \
        const int row0 = nt * mrows;                                           \
                                                                               \
        _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {                   \
            const int r = row0 + wbase + j * 8 + bc;                           \
            brow_ok[j] = r < n;                                                \
            bsrc[j] = wq + (brow_ok[j] ? (size_t)r * nb_total : 0);            \
            _Pragma("unroll") for (int u = 0; u < (TILES); ++u) {              \
                _Pragma("unroll") for (int c = 0; c < 4; ++c)                  \
                    acc[j][u][c] = 0.0f;                                       \
            }                                                                  \
        }                                                                      \
                                                                               \
        /* Prime this run's pipe. The ring is indexed from `kt_lo`. */         \
        _Pragma("unroll") for (int s = 0; s < (STAGES) - 1; ++s) {             \
            MMQ_XA_FETCH(s, kt_lo + s, kt_hi);                                 \
        }                                                                      \
        MMQ_CP_ASYNC_WAIT((STAGES) - 2);                                       \
        __syncthreads();                                                       \
        MMQ_RP_FETCH(NBLK, TILES, 0, 0, kt_lo, xa);                            \
                                                                               \
        for (int kt = kt_lo; kt < kt_hi; ++kt) {                               \
            const int pos = (kt - kt_lo) % (STAGES);                           \
            const int8_t* xbuf = xa + pos * x_rows * MMQ_XA_STRIDE;            \
            const int8_t* xnxt =                                               \
                xa + ((pos + 1) % (STAGES)) * x_rows * MMQ_XA_STRIDE;          \
            _Pragma("unroll") for (int s = 0; s < 4; ++s) {                    \
                if (s < 3) {                                                   \
                    MMQ_RP_FETCH(NBLK, TILES, (s + 1) % 2, s + 1, kt, xbuf);   \
                } else {                                                       \
                    MMQ_RP_FETCH(NBLK, TILES, (s + 1) % 2, 0, kt + 1, xnxt);   \
                }                                                              \
                if (s == 2) {                                                  \
                    MMQ_XA_FETCH((pos + (STAGES) - 1) % (STAGES),              \
                                 kt + (STAGES) - 1, kt_hi);                    \
                    MMQ_CP_ASYNC_WAIT((STAGES) - 2);                           \
                    __syncthreads();                                           \
                }                                                              \
                MMQ_RP_MUL(NBLK, TILES, s % 2);                                \
            }                                                                  \
        }                                                                      \
                                                                               \
        /* Drain before the next run re-primes over the same buffers. */       \
        MMQ_CP_ASYNC_WAIT(0);                                                  \
        __syncthreads();                                                       \
                                                                               \
        const bool whole = (kt_lo == 0) && (kt_hi == k_tiles);                 \
        _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {                   \
            const int orow = row0 + wbase + j * 8 + cc;                        \
            _Pragma("unroll") for (int u = 0; u < (TILES); ++u) {              \
                const int ot0 = tok0 + u * MMQ_M + cr;                         \
                const int ot1 = ot0 + 8;                                       \
                if (ot0 < n_tokens) {                                          \
                    if (orow < n)                                              \
                        MMQ_PUT2(whole, (size_t)ot0 * n + orow, acc[j][u][0]); \
                    if (orow + 1 < n)                                          \
                        MMQ_PUT2(whole, (size_t)ot0 * n + orow + 1,            \
                                 acc[j][u][1]);                                \
                }                                                              \
                if (ot1 < n_tokens) {                                          \
                    if (orow < n)                                              \
                        MMQ_PUT2(whole, (size_t)ot1 * n + orow, acc[j][u][2]); \
                    if (orow + 1 < n)                                          \
                        MMQ_PUT2(whole, (size_t)ot1 * n + orow + 1,            \
                                 acc[j][u][3]);                                \
                }                                                              \
            }                                                                  \
        }                                                                      \
        flat += kt_hi - kt_lo;                                                 \
    }

#define MMQ_STRIPED_REGPIPE_SET(SUFFIX, WARPS, NBLK, TILES, STAGES)            \
    extern "C" __global__ __launch_bounds__((WARPS) * WARP_SIZE) void          \
    mmqsr##SUFFIX##_q4_g128(float* __restrict__ out,                           \
                            const void* __restrict__ wv,                       \
                            const void* __restrict__ xv, int k, int n,         \
                            int n_tokens) {                                    \
        const block_q4_g128* wq = (const block_q4_g128*)wv;                    \
        const block_q8_1* xq = (const block_q8_1*)xv;                          \
        MMQ_STRIPED_REGPIPE_BODY(WARPS, NBLK, TILES, STAGES)                   \
    }

/* Same shapes as `mmqr_*`, so the partition is the only difference and
   `TUILI_MMQ_BPS` sets the blocks per SM the grid is sized to. */
MMQ_STRIPED_REGPIPE_SET(2w4s4, 4, 2, 1, 4)
MMQ_STRIPED_REGPIPE_SET(2w4s4_2, 4, 2, 2, 4)
MMQ_STRIPED_REGPIPE_SET(2w2s4, 2, 2, 1, 4)
MMQ_STRIPED_REGPIPE_SET(2w2s4_2, 2, 2, 2, 4)

// ---- deeper weight prefetch ---------------------------------------------
//
// Nine structural changes have been measured on this kernel and they sort by
// exactly one property: the six that removed instructions were worth nothing,
// and the four that put more weight loads in flight at once were worth 14-30%
// each. The porting order's answer to that is step two, staging the weights in
// shared through `cp.async`.
//
// This does the same thing without the shared memory, and there is a specific
// reason to try it that way first. A warp here owns its own weight rows and no
// other warp reads them, so a shared tile buys no reuse — the note above
// `MMQ_DIRECT_BODY` established that years of this file ago, and measured the
// staging's removal at 0%. All a shared tile would buy is the *asynchrony*,
// and it would cost `mrows * 128` bytes a stage: 32 KB on top of the
// activation ring's 19 at four stages and 64 weight rows, which is one block
// per SM. `mmqf_*` just measured what happens when this kernel spends shared
// memory to save work — the trade went the wrong way by 7%.
//
// The weights do not need shared memory to be prefetched. They come from
// global and nothing synchronizes them, so the register pipeline can run them
// arbitrarily far ahead; it is only the *activation* side that is pinned one
// step behind the `cp.async` ring's barrier. So the two operands get different
// depths: activations stay at two slots, weights get `DEPTH`.
//
// At DEPTH 2 this is `mmqsr_*` exactly, which is the A/B.

// Every slot index below has to fold to a compile-time constant or the ring
// is not a ring — an array indexed by a runtime value goes to local memory,
// and a spilled operand ring is slower than no ring at all. Measured: writing
// the slot as `gs % DEPTH`, with `gs` carrying the k-tile counter, cost 2x at
// every batch width even at depth 2, where the arithmetic is identical to
// `mmqsr_*` bit for bit.
//
// What makes it constant: a k-tile is exactly four steps, so for any DEPTH
// that divides 4, `(4 * tile + s) % DEPTH` is just `s % DEPTH`, and `s` comes
// from an unrolled loop. That is also why DEPTH is 2 or 4 and not 8 — at 8 the
// index depends on the tile's parity, which would need the tile loop unrolled
// by two as well.
#define MMQ_DB_RD(S, D) ((S) % (D))
#define MMQ_DB_WR(S, D) (((S) + (D) - 1) % (D))
#define MMQ_DB_WR_STEP(S, D) (((S) + (D) - 1) % 4)
#define MMQ_DB_WR_AHEAD(S, D) (((S) + (D) - 1) / 4)

// Weights and their scales for one step, into slot `SLOT` of the deep ring.
// `KTN` is the k-tile and `ST` the step within it; both callers pass `ST` as a
// constant.
#define MMQ_DB_FETCH_W(NBLK, SLOT, KTN, ST)                                    \
    do {                                                                       \
        const int _ktn = (KTN);                                                \
        const int _st = (ST);                                                  \
        const int _blk = _st / 2;                                              \
        const int _run = _st % 2;                                              \
        const int _kb = _ktn * 2 + _blk;                                       \
        const bool _ok = (_ktn < kt_hi) && (_kb < nb_total);                   \
        _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {                   \
            v0[SLOT][j] = 0;                                                   \
            v1[SLOT][j] = 0;                                                   \
            if (_ok && brow_ok[j]) {                                           \
                const uint8_t* _pq = bsrc[j][_kb].qs + _run * 32 + kq;         \
                v0[SLOT][j] = *(const uint32_t*)(const void*)_pq;              \
                v1[SLOT][j] = *(const uint32_t*)(const void*)(_pq + 16);       \
            }                                                                  \
            const int _sr = row0 + wbase + j * 8 + cc;                         \
            __half2 _s0 = __floats2half2_rn(0.0f, 0.0f);                       \
            __half2 _s1 = _s0;                                                 \
            if (_ok && _sr < n) _s0 = wq[(size_t)_sr * nb_total + _kb].ds;     \
            if (_ok && _sr + 1 < n)                                            \
                _s1 = wq[(size_t)(_sr + 1) * nb_total + _kb].ds;               \
            swd0[SLOT][j] = __low2float(_s0);                                  \
            swm0[SLOT][j] = -__high2float(_s0);                                \
            swd1[SLOT][j] = __low2float(_s1);                                  \
            swm1[SLOT][j] = -__high2float(_s1);                                \
        }                                                                      \
    } while (0)

// The activation half of a step: A fragments and the per-token scales, out of
// the shared ring. Two slots, because the ring's barrier is what bounds it.
#define MMQ_DB_FETCH_A(TILES, SLOT, ST, BUF)                                   \
    do {                                                                       \
        const int _blk = (ST) / 2;                                             \
        const int _run = (ST) % 2;                                             \
        _Pragma("unroll") for (int h = 0; h < 2; ++h) {                        \
            const int _g = _blk * 4 + _run + h * 2;                            \
            _Pragma("unroll") for (int u = 0; u < (TILES); ++u) {              \
                const int8_t* _ap = (BUF)                                      \
                                  + (u * MMQ_M + ar) * MMQ_XA_STRIDE           \
                                  + _g * 36 + 4 + kq;                          \
                const int8_t* _aq = _ap + 8 * MMQ_XA_STRIDE;                   \
                fa[SLOT][h][u].x[0] = *(const int*)(const void*)_ap;           \
                fa[SLOT][h][u].x[1] = *(const int*)(const void*)_aq;           \
                fa[SLOT][h][u].x[2] = *(const int*)(const void*)(_ap + 16);    \
                fa[SLOT][h][u].x[3] = *(const int*)(const void*)(_aq + 16);    \
                const __half2 _dl = *(const __half2*)(const void*)(            \
                    (BUF) + (u * MMQ_M + cr) * MMQ_XA_STRIDE + _g * 36);       \
                const __half2 _dh = *(const __half2*)(const void*)(            \
                    (BUF) + (u * MMQ_M + cr + 8) * MMQ_XA_STRIDE + _g * 36);   \
                fxdl[SLOT][h][u] = __low2float(_dl);                           \
                fxsl[SLOT][h][u] = __high2float(_dl);                          \
                fxdh[SLOT][h][u] = __low2float(_dh);                           \
                fxsh[SLOT][h][u] = __high2float(_dh);                          \
            }                                                                  \
        }                                                                      \
    } while (0)

// The same fetch with the per-token scales left behind. They are four floats a
// token tile and they double-buffer into ~32 registers, which at two token
// tiles is the difference between three resident blocks and four — and unlike
// the A fragments they are cheap to re-read, being shared-memory loads with no
// global latency behind them.
#define MMQ_DB_FETCH_AF(TILES, SLOT, ST, BUF)                                  \
    do {                                                                       \
        const int _blk = (ST) / 2;                                             \
        const int _run = (ST) % 2;                                             \
        _Pragma("unroll") for (int h = 0; h < 2; ++h) {                        \
            const int _g = _blk * 4 + _run + h * 2;                            \
            _Pragma("unroll") for (int u = 0; u < (TILES); ++u) {              \
                const int8_t* _ap = (BUF)                                      \
                                  + (u * MMQ_M + ar) * MMQ_XA_STRIDE           \
                                  + _g * 36 + 4 + kq;                          \
                const int8_t* _aq = _ap + 8 * MMQ_XA_STRIDE;                   \
                fa[SLOT][h][u].x[0] = *(const int*)(const void*)_ap;           \
                fa[SLOT][h][u].x[1] = *(const int*)(const void*)_aq;           \
                fa[SLOT][h][u].x[2] = *(const int*)(const void*)(_ap + 16);    \
                fa[SLOT][h][u].x[3] = *(const int*)(const void*)(_aq + 16);    \
            }                                                                  \
        }                                                                      \
    } while (0)

// `MMQ_DB_MUL` with the token scales read here rather than carried. `BUF` and
// `ST` name the step being multiplied, which is the one fetched a step ago.
#define MMQ_DB_MUL_L(NBLK, TILES, ASLOT, WSLOT, ST, BUF)                       \
    _Pragma("unroll") for (int h = 0; h < 2; ++h) {                            \
        const int _g = ((ST) / 2) * 4 + ((ST) % 2) + h * 2;                    \
        _Pragma("unroll") for (int u = 0; u < (TILES); ++u) {                  \
            const __half2 _dl = *(const __half2*)(const void*)(                \
                (BUF) + (u * MMQ_M + cr) * MMQ_XA_STRIDE + _g * 36);           \
            const __half2 _dh = *(const __half2*)(const void*)(                \
                (BUF) + (u * MMQ_M + cr + 8) * MMQ_XA_STRIDE + _g * 36);       \
            const float _xdl = __low2float(_dl), _xsl = __high2float(_dl);     \
            const float _xdh = __low2float(_dh), _xsh = __high2float(_dh);     \
            _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {               \
                mma_b_s8 b;                                                    \
                b.x[0] = (int)((v0[WSLOT][j] >> (h * 4)) & 0x0F0F0F0Fu);       \
                b.x[1] = (int)((v1[WSLOT][j] >> (h * 4)) & 0x0F0F0F0Fu);       \
                mma_c_s32 d = {{0, 0, 0, 0}};                                  \
                mma_s8(d, fa[ASLOT][h][u], b);                                 \
                acc[j][u][0] += swd0[WSLOT][j] * _xdl * (float)d.x[0]          \
                              + swm0[WSLOT][j] * _xsl;                         \
                acc[j][u][1] += swd1[WSLOT][j] * _xdl * (float)d.x[1]          \
                              + swm1[WSLOT][j] * _xsl;                         \
                acc[j][u][2] += swd0[WSLOT][j] * _xdh * (float)d.x[2]          \
                              + swm0[WSLOT][j] * _xsh;                         \
                acc[j][u][3] += swd1[WSLOT][j] * _xdh * (float)d.x[3]          \
                              + swm1[WSLOT][j] * _xsh;                         \
            }                                                                  \
        }                                                                      \
    }

#define MMQ_DB_MUL(NBLK, TILES, ASLOT, WSLOT)                                  \
    _Pragma("unroll") for (int h = 0; h < 2; ++h) {                            \
        _Pragma("unroll") for (int u = 0; u < (TILES); ++u) {                  \
            _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {               \
                mma_b_s8 b;                                                    \
                b.x[0] = (int)((v0[WSLOT][j] >> (h * 4)) & 0x0F0F0F0Fu);       \
                b.x[1] = (int)((v1[WSLOT][j] >> (h * 4)) & 0x0F0F0F0Fu);       \
                mma_c_s32 d = {{0, 0, 0, 0}};                                  \
                mma_s8(d, fa[ASLOT][h][u], b);                                 \
                acc[j][u][0] += swd0[WSLOT][j] * fxdl[ASLOT][h][u]             \
                                    * (float)d.x[0]                            \
                              + swm0[WSLOT][j] * fxsl[ASLOT][h][u];            \
                acc[j][u][1] += swd1[WSLOT][j] * fxdl[ASLOT][h][u]             \
                                    * (float)d.x[1]                            \
                              + swm1[WSLOT][j] * fxsl[ASLOT][h][u];            \
                acc[j][u][2] += swd0[WSLOT][j] * fxdh[ASLOT][h][u]             \
                                    * (float)d.x[2]                            \
                              + swm0[WSLOT][j] * fxsh[ASLOT][h][u];            \
                acc[j][u][3] += swd1[WSLOT][j] * fxdh[ASLOT][h][u]             \
                                    * (float)d.x[3]                            \
                              + swm1[WSLOT][j] * fxsh[ASLOT][h][u];            \
            }                                                                  \
        }                                                                      \
    }

#define MMQ_DEEPB_BODY(WARPS, NBLK, TILES, STAGES, DEPTH)                      \
    __shared__ __align__(16)                                                   \
        int8_t xa[(STAGES) * (TILES) * MMQ_M * MMQ_XA_STRIDE];                 \
                                                                               \
    const int tid = threadIdx.x;                                               \
    const int lane = tid % WARP_SIZE;                                          \
    const int warp = tid / WARP_SIZE;                                          \
    const int nthreads = (WARPS) * WARP_SIZE;                                  \
    const int mrows = (WARPS) * (NBLK) * 8;                                    \
    const int tok0 = blockIdx.y * (TILES) * MMQ_M;                             \
    const int kb_total = k / 32;                                               \
    const int nb_total = k / QK_G128;                                          \
    const int k_tiles = (kb_total + MMQ_GROUPS - 1) / MMQ_GROUPS;              \
    const int row_tiles = (n + mrows - 1) / mrows;                             \
    const int x_rows = (TILES) * MMQ_M;                                        \
    const int x_valid = min(x_rows, max(0, n_tokens - tok0));                  \
    const uint8_t* xbytes = (const uint8_t*)xq;                                \
                                                                               \
    const int ar = mma_a_row(lane);                                            \
    const int bc = mma_b_col(lane);                                            \
    const int kq = mma_k0(lane);                                               \
    const int cr = mma_c_row(lane);                                            \
    const int cc = mma_c_col(lane);                                            \
    const int wbase = warp * (NBLK) * 8;                                       \
                                                                               \
    const int total = row_tiles * k_tiles;                                     \
    const int iters = (total + (int)gridDim.x - 1) / (int)gridDim.x;           \
    int flat = iters * (int)blockIdx.x;                                        \
    const int flat_end = min(total, flat + iters);                             \
                                                                               \
    uint32_t v0[DEPTH][NBLK], v1[DEPTH][NBLK];                                 \
    float swd0[DEPTH][NBLK], swd1[DEPTH][NBLK];                                \
    float swm0[DEPTH][NBLK], swm1[DEPTH][NBLK];                                \
    mma_a_s8 fa[2][2][TILES];                                                  \
    float fxdl[2][2][TILES], fxsl[2][2][TILES];                                \
    float fxdh[2][2][TILES], fxsh[2][2][TILES];                                \
    const block_q4_g128* bsrc[NBLK];                                           \
    bool brow_ok[NBLK];                                                        \
    float acc[NBLK][TILES][4];                                                 \
                                                                               \
    if (x_valid < x_rows) {                                                    \
        const int per = (x_rows - x_valid) * (MMQ_XA_ROW / 4);                 \
        for (int i = tid; i < (STAGES) * per; i += nthreads) {                 \
            const int s = i / per;                                             \
            const int j = i % per;                                             \
            const int r = x_valid + j / (MMQ_XA_ROW / 4);                      \
            const int e = (j % (MMQ_XA_ROW / 4)) * 4;                          \
            *(uint32_t*)(void*)(xa + (s * x_rows + r) * MMQ_XA_STRIDE + e) = 0;\
        }                                                                      \
    }                                                                          \
                                                                               \
    while (flat < flat_end) {                                                  \
        const int nt = flat / k_tiles;                                         \
        const int kt_lo = flat % k_tiles;                                      \
        const int kt_hi = min(k_tiles, kt_lo + (flat_end - flat));             \
        const int row0 = nt * mrows;                                           \
                                                                               \
        _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {                   \
            const int r = row0 + wbase + j * 8 + bc;                           \
            brow_ok[j] = r < n;                                                \
            bsrc[j] = wq + (brow_ok[j] ? (size_t)r * nb_total : 0);            \
            _Pragma("unroll") for (int u = 0; u < (TILES); ++u) {              \
                _Pragma("unroll") for (int c = 0; c < 4; ++c)                  \
                    acc[j][u][c] = 0.0f;                                       \
            }                                                                  \
        }                                                                      \
                                                                               \
        _Pragma("unroll") for (int s = 0; s < (STAGES) - 1; ++s) {             \
            MMQ_XA_FETCH(s, kt_lo + s, kt_hi);                                 \
        }                                                                      \
        MMQ_CP_ASYNC_WAIT((STAGES) - 2);                                       \
        __syncthreads();                                                       \
        /* Prime both rings. The weight ring runs DEPTH-1 steps ahead and       \
           crosses k-tiles freely; nothing publishes it. */                    \
        _Pragma("unroll") for (int d = 0; d < (DEPTH) - 1; ++d) {              \
            MMQ_DB_FETCH_W(NBLK, d, kt_lo, d);                                 \
        }                                                                      \
        MMQ_DB_FETCH_A(TILES, 0, 0, xa);                                       \
                                                                               \
        for (int kt = kt_lo; kt < kt_hi; ++kt) {                               \
            const int pos = (kt - kt_lo) % (STAGES);                           \
            const int8_t* xbuf = xa + pos * x_rows * MMQ_XA_STRIDE;            \
            const int8_t* xnxt =                                               \
                xa + ((pos + 1) % (STAGES)) * x_rows * MMQ_XA_STRIDE;          \
            _Pragma("unroll") for (int s = 0; s < 4; ++s) {                    \
                MMQ_DB_FETCH_W(NBLK, MMQ_DB_WR(s, DEPTH),                      \
                               kt + MMQ_DB_WR_AHEAD(s, DEPTH),                 \
                               MMQ_DB_WR_STEP(s, DEPTH));                      \
                if (s < 3) {                                                   \
                    MMQ_DB_FETCH_A(TILES, (s + 1) % 2, s + 1, xbuf);           \
                } else {                                                       \
                    MMQ_DB_FETCH_A(TILES, (s + 1) % 2, 0, xnxt);               \
                }                                                              \
                if (s == 2) {                                                  \
                    MMQ_XA_FETCH((pos + (STAGES) - 1) % (STAGES),              \
                                 kt + (STAGES) - 1, kt_hi);                    \
                    MMQ_CP_ASYNC_WAIT((STAGES) - 2);                           \
                    __syncthreads();                                           \
                }                                                              \
                MMQ_DB_MUL(NBLK, TILES, s % 2, MMQ_DB_RD(s, DEPTH));           \
            }                                                                  \
        }                                                                      \
                                                                               \
        MMQ_CP_ASYNC_WAIT(0);                                                  \
        __syncthreads();                                                       \
                                                                               \
        const bool whole = (kt_lo == 0) && (kt_hi == k_tiles);                 \
        _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {                   \
            const int orow = row0 + wbase + j * 8 + cc;                        \
            _Pragma("unroll") for (int u = 0; u < (TILES); ++u) {              \
                const int ot0 = tok0 + u * MMQ_M + cr;                         \
                const int ot1 = ot0 + 8;                                       \
                if (ot0 < n_tokens) {                                          \
                    if (orow < n)                                              \
                        MMQ_PUT2(whole, (size_t)ot0 * n + orow, acc[j][u][0]); \
                    if (orow + 1 < n)                                          \
                        MMQ_PUT2(whole, (size_t)ot0 * n + orow + 1,            \
                                 acc[j][u][1]);                                \
                }                                                              \
                if (ot1 < n_tokens) {                                          \
                    if (orow < n)                                              \
                        MMQ_PUT2(whole, (size_t)ot1 * n + orow, acc[j][u][2]); \
                    if (orow + 1 < n)                                          \
                        MMQ_PUT2(whole, (size_t)ot1 * n + orow + 1,            \
                                 acc[j][u][3]);                                \
                }                                                              \
            }                                                                  \
        }                                                                      \
        flat += kt_hi - kt_lo;                                                 \
    }


// ---- what the two-token-tile shape is actually spending -----------------
//
// At 32 tokens `mmql1w4s2d2` runs 185 GB/s against the 294 its access pattern
// reaches at 8, and the shape is not short of memory bandwidth — forcing one
// token tile there hits 300 GB/s of real traffic. So it is short of something
// else, and per k-tile per warp there are two candidates that scale with the
// token tiles where the weight loads do not: 64 A-fragment shared loads and 64
// accumulator updates.
//
// These price them the way `mmq_noA_q4_K` and `mmq_noscale_q4_K` priced the
// staged kernel: identical in every other respect, wrong on purpose.

#define MMQ_NOA_FETCH_AF(TILES, SLOT, ST, BUF)                                 \
    do {                                                                       \
        const int _blk = (ST) / 2;                                             \
        const int _run = (ST) % 2;                                             \
        _Pragma("unroll") for (int h = 0; h < 2; ++h) {                        \
            const int _g = _blk * 4 + _run + h * 2;                            \
            _Pragma("unroll") for (int u = 0; u < (TILES); ++u) {              \
                const int8_t* _ap = (BUF)                                      \
                                  + (u * MMQ_M + ar) * MMQ_XA_STRIDE           \
                                  + _g * 36 + 4 + kq;                          \
                const int8_t* _aq = _ap + 8 * MMQ_XA_STRIDE;                   \
                fa[SLOT][h][u].x[0] = 0x01010101;                              \
                fa[SLOT][h][u].x[1] = 0x01010101;                              \
                fa[SLOT][h][u].x[2] = 0x01010101;                              \
                fa[SLOT][h][u].x[3] = 0x01010101;                              \
                (void)_ap; (void)_aq;                                          \
            }                                                                  \
        }                                                                      \
    } while (0)

// The MMAs and the operand loads stay; the scale application does not.
#define MMQ_NOE_MUL(NBLK, TILES, ASLOT, WSLOT, ST, BUF)                        \
    _Pragma("unroll") for (int h = 0; h < 2; ++h) {                            \
        _Pragma("unroll") for (int u = 0; u < (TILES); ++u) {                  \
            _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {               \
                mma_b_s8 b;                                                    \
                b.x[0] = (int)((v0[WSLOT][j] >> (h * 4)) & 0x0F0F0F0Fu);       \
                b.x[1] = (int)((v1[WSLOT][j] >> (h * 4)) & 0x0F0F0F0Fu);       \
                mma_c_s32 d = {{0, 0, 0, 0}};                                  \
                mma_s8(d, fa[ASLOT][h][u], b);                                 \
                acc[j][u][0] += 1.5f * (float)d.x[0];                          \
                acc[j][u][1] += 1.5f * (float)d.x[1];                          \
                acc[j][u][2] += 1.5f * (float)d.x[2];                          \
                acc[j][u][3] += 1.5f * (float)d.x[3];                          \
            }                                                                  \
        }                                                                      \
        (void)(ST); (void)(BUF);                                               \
    }

#define MMQ_LEANB_BODY(WARPS, NBLK, TILES, STAGES, DEPTH)                      \
    __shared__ __align__(16)                                                   \
        int8_t xa[(STAGES) * (TILES) * MMQ_M * MMQ_XA_STRIDE];                 \
                                                                               \
    const int tid = threadIdx.x;                                               \
    const int lane = tid % WARP_SIZE;                                          \
    const int warp = tid / WARP_SIZE;                                          \
    const int nthreads = (WARPS) * WARP_SIZE;                                  \
    const int mrows = (WARPS) * (NBLK) * 8;                                    \
    const int tok0 = blockIdx.y * (TILES) * MMQ_M;                             \
    const int kb_total = k / 32;                                               \
    const int nb_total = k / QK_G128;                                          \
    const int k_tiles = (kb_total + MMQ_GROUPS - 1) / MMQ_GROUPS;              \
    const int row_tiles = (n + mrows - 1) / mrows;                             \
    const int x_rows = (TILES) * MMQ_M;                                        \
    const int x_valid = min(x_rows, max(0, n_tokens - tok0));                  \
    const uint8_t* xbytes = (const uint8_t*)xq;                                \
                                                                               \
    const int ar = mma_a_row(lane);                                            \
    const int bc = mma_b_col(lane);                                            \
    const int kq = mma_k0(lane);                                               \
    const int cr = mma_c_row(lane);                                            \
    const int cc = mma_c_col(lane);                                            \
    const int wbase = warp * (NBLK) * 8;                                       \
                                                                               \
    const int total = row_tiles * k_tiles;                                     \
    const int iters = (total + (int)gridDim.x - 1) / (int)gridDim.x;           \
    int flat = iters * (int)blockIdx.x;                                        \
    const int flat_end = min(total, flat + iters);                             \
                                                                               \
    uint32_t v0[DEPTH][NBLK], v1[DEPTH][NBLK];                                 \
    float swd0[DEPTH][NBLK], swd1[DEPTH][NBLK];                                \
    float swm0[DEPTH][NBLK], swm1[DEPTH][NBLK];                                \
    mma_a_s8 fa[2][2][TILES];                                                  \
    const block_q4_g128* bsrc[NBLK];                                           \
    bool brow_ok[NBLK];                                                        \
    float acc[NBLK][TILES][4];                                                 \
                                                                               \
    if (x_valid < x_rows) {                                                    \
        const int per = (x_rows - x_valid) * (MMQ_XA_ROW / 4);                 \
        for (int i = tid; i < (STAGES) * per; i += nthreads) {                 \
            const int s = i / per;                                             \
            const int j = i % per;                                             \
            const int r = x_valid + j / (MMQ_XA_ROW / 4);                      \
            const int e = (j % (MMQ_XA_ROW / 4)) * 4;                          \
            *(uint32_t*)(void*)(xa + (s * x_rows + r) * MMQ_XA_STRIDE + e) = 0;\
        }                                                                      \
    }                                                                          \
                                                                               \
    while (flat < flat_end) {                                                  \
        const int nt = flat / k_tiles;                                         \
        const int kt_lo = flat % k_tiles;                                      \
        const int kt_hi = min(k_tiles, kt_lo + (flat_end - flat));             \
        const int row0 = nt * mrows;                                           \
                                                                               \
        _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {                   \
            const int r = row0 + wbase + j * 8 + bc;                           \
            brow_ok[j] = r < n;                                                \
            bsrc[j] = wq + (brow_ok[j] ? (size_t)r * nb_total : 0);            \
            _Pragma("unroll") for (int u = 0; u < (TILES); ++u) {              \
                _Pragma("unroll") for (int c = 0; c < 4; ++c)                  \
                    acc[j][u][c] = 0.0f;                                       \
            }                                                                  \
        }                                                                      \
                                                                               \
        _Pragma("unroll") for (int s = 0; s < (STAGES) - 1; ++s) {             \
            MMQ_XA_FETCH(s, kt_lo + s, kt_hi);                                 \
        }                                                                      \
        MMQ_CP_ASYNC_WAIT((STAGES) - 2);                                       \
        __syncthreads();                                                       \
        /* Prime both rings. The weight ring runs DEPTH-1 steps ahead and       \
           crosses k-tiles freely; nothing publishes it. */                    \
        _Pragma("unroll") for (int d = 0; d < (DEPTH) - 1; ++d) {              \
            MMQ_DB_FETCH_W(NBLK, d, kt_lo, d);                                 \
        }                                                                      \
        MMQ_DB_FETCH_AF(TILES, 0, 0, xa);                                       \
                                                                               \
        for (int kt = kt_lo; kt < kt_hi; ++kt) {                               \
            const int pos = (kt - kt_lo) % (STAGES);                           \
            const int8_t* xbuf = xa + pos * x_rows * MMQ_XA_STRIDE;            \
            const int8_t* xnxt =                                               \
                xa + ((pos + 1) % (STAGES)) * x_rows * MMQ_XA_STRIDE;          \
            _Pragma("unroll") for (int s = 0; s < 4; ++s) {                    \
                MMQ_DB_FETCH_W(NBLK, MMQ_DB_WR(s, DEPTH),                      \
                               kt + MMQ_DB_WR_AHEAD(s, DEPTH),                 \
                               MMQ_DB_WR_STEP(s, DEPTH));                      \
                if (s < 3) {                                                   \
                    MMQ_DB_FETCH_AF(TILES, (s + 1) % 2, s + 1, xbuf);           \
                } else {                                                       \
                    MMQ_DB_FETCH_AF(TILES, (s + 1) % 2, 0, xnxt);               \
                }                                                              \
                if (s == 2) {                                                  \
                    MMQ_XA_FETCH((pos + (STAGES) - 1) % (STAGES),              \
                                 kt + (STAGES) - 1, kt_hi);                    \
                    MMQ_CP_ASYNC_WAIT((STAGES) - 2);                           \
                    __syncthreads();                                           \
                }                                                              \
                MMQ_DB_MUL_L(NBLK, TILES, s % 2, MMQ_DB_RD(s, DEPTH), s,      \
                             xbuf);           \
            }                                                                  \
        }                                                                      \
                                                                               \
        MMQ_CP_ASYNC_WAIT(0);                                                  \
        __syncthreads();                                                       \
                                                                               \
        const bool whole = (kt_lo == 0) && (kt_hi == k_tiles);                 \
        _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {                   \
            const int orow = row0 + wbase + j * 8 + cc;                        \
            _Pragma("unroll") for (int u = 0; u < (TILES); ++u) {              \
                const int ot0 = tok0 + u * MMQ_M + cr;                         \
                const int ot1 = ot0 + 8;                                       \
                if (ot0 < n_tokens) {                                          \
                    if (orow < n)                                              \
                        MMQ_PUT2(whole, (size_t)ot0 * n + orow, acc[j][u][0]); \
                    if (orow + 1 < n)                                          \
                        MMQ_PUT2(whole, (size_t)ot0 * n + orow + 1,            \
                                 acc[j][u][1]);                                \
                }                                                              \
                if (ot1 < n_tokens) {                                          \
                    if (orow < n)                                              \
                        MMQ_PUT2(whole, (size_t)ot1 * n + orow, acc[j][u][2]); \
                    if (orow + 1 < n)                                          \
                        MMQ_PUT2(whole, (size_t)ot1 * n + orow + 1,            \
                                 acc[j][u][3]);                                \
                }                                                              \
            }                                                                  \
        }                                                                      \
        flat += kt_hi - kt_lo;                                                 \
    }

#define MMQ_NOA_BODY(WARPS, NBLK, TILES, STAGES, DEPTH)                      \
    __shared__ __align__(16)                                                   \
        int8_t xa[(STAGES) * (TILES) * MMQ_M * MMQ_XA_STRIDE];                 \
                                                                               \
    const int tid = threadIdx.x;                                               \
    const int lane = tid % WARP_SIZE;                                          \
    const int warp = tid / WARP_SIZE;                                          \
    const int nthreads = (WARPS) * WARP_SIZE;                                  \
    const int mrows = (WARPS) * (NBLK) * 8;                                    \
    const int tok0 = blockIdx.y * (TILES) * MMQ_M;                             \
    const int kb_total = k / 32;                                               \
    const int nb_total = k / QK_G128;                                          \
    const int k_tiles = (kb_total + MMQ_GROUPS - 1) / MMQ_GROUPS;              \
    const int row_tiles = (n + mrows - 1) / mrows;                             \
    const int x_rows = (TILES) * MMQ_M;                                        \
    const int x_valid = min(x_rows, max(0, n_tokens - tok0));                  \
    const uint8_t* xbytes = (const uint8_t*)xq;                                \
                                                                               \
    const int ar = mma_a_row(lane);                                            \
    const int bc = mma_b_col(lane);                                            \
    const int kq = mma_k0(lane);                                               \
    const int cr = mma_c_row(lane);                                            \
    const int cc = mma_c_col(lane);                                            \
    const int wbase = warp * (NBLK) * 8;                                       \
                                                                               \
    const int total = row_tiles * k_tiles;                                     \
    const int iters = (total + (int)gridDim.x - 1) / (int)gridDim.x;           \
    int flat = iters * (int)blockIdx.x;                                        \
    const int flat_end = min(total, flat + iters);                             \
                                                                               \
    uint32_t v0[DEPTH][NBLK], v1[DEPTH][NBLK];                                 \
    float swd0[DEPTH][NBLK], swd1[DEPTH][NBLK];                                \
    float swm0[DEPTH][NBLK], swm1[DEPTH][NBLK];                                \
    mma_a_s8 fa[2][2][TILES];                                                  \
    const block_q4_g128* bsrc[NBLK];                                           \
    bool brow_ok[NBLK];                                                        \
    float acc[NBLK][TILES][4];                                                 \
                                                                               \
    if (x_valid < x_rows) {                                                    \
        const int per = (x_rows - x_valid) * (MMQ_XA_ROW / 4);                 \
        for (int i = tid; i < (STAGES) * per; i += nthreads) {                 \
            const int s = i / per;                                             \
            const int j = i % per;                                             \
            const int r = x_valid + j / (MMQ_XA_ROW / 4);                      \
            const int e = (j % (MMQ_XA_ROW / 4)) * 4;                          \
            *(uint32_t*)(void*)(xa + (s * x_rows + r) * MMQ_XA_STRIDE + e) = 0;\
        }                                                                      \
    }                                                                          \
                                                                               \
    while (flat < flat_end) {                                                  \
        const int nt = flat / k_tiles;                                         \
        const int kt_lo = flat % k_tiles;                                      \
        const int kt_hi = min(k_tiles, kt_lo + (flat_end - flat));             \
        const int row0 = nt * mrows;                                           \
                                                                               \
        _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {                   \
            const int r = row0 + wbase + j * 8 + bc;                           \
            brow_ok[j] = r < n;                                                \
            bsrc[j] = wq + (brow_ok[j] ? (size_t)r * nb_total : 0);            \
            _Pragma("unroll") for (int u = 0; u < (TILES); ++u) {              \
                _Pragma("unroll") for (int c = 0; c < 4; ++c)                  \
                    acc[j][u][c] = 0.0f;                                       \
            }                                                                  \
        }                                                                      \
                                                                               \
        _Pragma("unroll") for (int s = 0; s < (STAGES) - 1; ++s) {             \
            MMQ_XA_FETCH(s, kt_lo + s, kt_hi);                                 \
        }                                                                      \
        MMQ_CP_ASYNC_WAIT((STAGES) - 2);                                       \
        __syncthreads();                                                       \
        /* Prime both rings. The weight ring runs DEPTH-1 steps ahead and       \
           crosses k-tiles freely; nothing publishes it. */                    \
        _Pragma("unroll") for (int d = 0; d < (DEPTH) - 1; ++d) {              \
            MMQ_DB_FETCH_W(NBLK, d, kt_lo, d);                                 \
        }                                                                      \
        MMQ_NOA_FETCH_AF(TILES, 0, 0, xa);                                       \
                                                                               \
        for (int kt = kt_lo; kt < kt_hi; ++kt) {                               \
            const int pos = (kt - kt_lo) % (STAGES);                           \
            const int8_t* xbuf = xa + pos * x_rows * MMQ_XA_STRIDE;            \
            const int8_t* xnxt =                                               \
                xa + ((pos + 1) % (STAGES)) * x_rows * MMQ_XA_STRIDE;          \
            _Pragma("unroll") for (int s = 0; s < 4; ++s) {                    \
                MMQ_DB_FETCH_W(NBLK, MMQ_DB_WR(s, DEPTH),                      \
                               kt + MMQ_DB_WR_AHEAD(s, DEPTH),                 \
                               MMQ_DB_WR_STEP(s, DEPTH));                      \
                if (s < 3) {                                                   \
                    MMQ_NOA_FETCH_AF(TILES, (s + 1) % 2, s + 1, xbuf);           \
                } else {                                                       \
                    MMQ_NOA_FETCH_AF(TILES, (s + 1) % 2, 0, xnxt);               \
                }                                                              \
                if (s == 2) {                                                  \
                    MMQ_XA_FETCH((pos + (STAGES) - 1) % (STAGES),              \
                                 kt + (STAGES) - 1, kt_hi);                    \
                    MMQ_CP_ASYNC_WAIT((STAGES) - 2);                           \
                    __syncthreads();                                           \
                }                                                              \
                MMQ_DB_MUL_L(NBLK, TILES, s % 2, MMQ_DB_RD(s, DEPTH), s,      \
                             xbuf);           \
            }                                                                  \
        }                                                                      \
                                                                               \
        MMQ_CP_ASYNC_WAIT(0);                                                  \
        __syncthreads();                                                       \
                                                                               \
        const bool whole = (kt_lo == 0) && (kt_hi == k_tiles);                 \
        _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {                   \
            const int orow = row0 + wbase + j * 8 + cc;                        \
            _Pragma("unroll") for (int u = 0; u < (TILES); ++u) {              \
                const int ot0 = tok0 + u * MMQ_M + cr;                         \
                const int ot1 = ot0 + 8;                                       \
                if (ot0 < n_tokens) {                                          \
                    if (orow < n)                                              \
                        MMQ_PUT2(whole, (size_t)ot0 * n + orow, acc[j][u][0]); \
                    if (orow + 1 < n)                                          \
                        MMQ_PUT2(whole, (size_t)ot0 * n + orow + 1,            \
                                 acc[j][u][1]);                                \
                }                                                              \
                if (ot1 < n_tokens) {                                          \
                    if (orow < n)                                              \
                        MMQ_PUT2(whole, (size_t)ot1 * n + orow, acc[j][u][2]); \
                    if (orow + 1 < n)                                          \
                        MMQ_PUT2(whole, (size_t)ot1 * n + orow + 1,            \
                                 acc[j][u][3]);                                \
                }                                                              \
            }                                                                  \
        }                                                                      \
        flat += kt_hi - kt_lo;                                                 \
    }

#define MMQ_NOE_BODY(WARPS, NBLK, TILES, STAGES, DEPTH)                      \
    __shared__ __align__(16)                                                   \
        int8_t xa[(STAGES) * (TILES) * MMQ_M * MMQ_XA_STRIDE];                 \
                                                                               \
    const int tid = threadIdx.x;                                               \
    const int lane = tid % WARP_SIZE;                                          \
    const int warp = tid / WARP_SIZE;                                          \
    const int nthreads = (WARPS) * WARP_SIZE;                                  \
    const int mrows = (WARPS) * (NBLK) * 8;                                    \
    const int tok0 = blockIdx.y * (TILES) * MMQ_M;                             \
    const int kb_total = k / 32;                                               \
    const int nb_total = k / QK_G128;                                          \
    const int k_tiles = (kb_total + MMQ_GROUPS - 1) / MMQ_GROUPS;              \
    const int row_tiles = (n + mrows - 1) / mrows;                             \
    const int x_rows = (TILES) * MMQ_M;                                        \
    const int x_valid = min(x_rows, max(0, n_tokens - tok0));                  \
    const uint8_t* xbytes = (const uint8_t*)xq;                                \
                                                                               \
    const int ar = mma_a_row(lane);                                            \
    const int bc = mma_b_col(lane);                                            \
    const int kq = mma_k0(lane);                                               \
    const int cr = mma_c_row(lane);                                            \
    const int cc = mma_c_col(lane);                                            \
    const int wbase = warp * (NBLK) * 8;                                       \
                                                                               \
    const int total = row_tiles * k_tiles;                                     \
    const int iters = (total + (int)gridDim.x - 1) / (int)gridDim.x;           \
    int flat = iters * (int)blockIdx.x;                                        \
    const int flat_end = min(total, flat + iters);                             \
                                                                               \
    uint32_t v0[DEPTH][NBLK], v1[DEPTH][NBLK];                                 \
    float swd0[DEPTH][NBLK], swd1[DEPTH][NBLK];                                \
    float swm0[DEPTH][NBLK], swm1[DEPTH][NBLK];                                \
    mma_a_s8 fa[2][2][TILES];                                                  \
    const block_q4_g128* bsrc[NBLK];                                           \
    bool brow_ok[NBLK];                                                        \
    float acc[NBLK][TILES][4];                                                 \
                                                                               \
    if (x_valid < x_rows) {                                                    \
        const int per = (x_rows - x_valid) * (MMQ_XA_ROW / 4);                 \
        for (int i = tid; i < (STAGES) * per; i += nthreads) {                 \
            const int s = i / per;                                             \
            const int j = i % per;                                             \
            const int r = x_valid + j / (MMQ_XA_ROW / 4);                      \
            const int e = (j % (MMQ_XA_ROW / 4)) * 4;                          \
            *(uint32_t*)(void*)(xa + (s * x_rows + r) * MMQ_XA_STRIDE + e) = 0;\
        }                                                                      \
    }                                                                          \
                                                                               \
    while (flat < flat_end) {                                                  \
        const int nt = flat / k_tiles;                                         \
        const int kt_lo = flat % k_tiles;                                      \
        const int kt_hi = min(k_tiles, kt_lo + (flat_end - flat));             \
        const int row0 = nt * mrows;                                           \
                                                                               \
        _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {                   \
            const int r = row0 + wbase + j * 8 + bc;                           \
            brow_ok[j] = r < n;                                                \
            bsrc[j] = wq + (brow_ok[j] ? (size_t)r * nb_total : 0);            \
            _Pragma("unroll") for (int u = 0; u < (TILES); ++u) {              \
                _Pragma("unroll") for (int c = 0; c < 4; ++c)                  \
                    acc[j][u][c] = 0.0f;                                       \
            }                                                                  \
        }                                                                      \
                                                                               \
        _Pragma("unroll") for (int s = 0; s < (STAGES) - 1; ++s) {             \
            MMQ_XA_FETCH(s, kt_lo + s, kt_hi);                                 \
        }                                                                      \
        MMQ_CP_ASYNC_WAIT((STAGES) - 2);                                       \
        __syncthreads();                                                       \
        /* Prime both rings. The weight ring runs DEPTH-1 steps ahead and       \
           crosses k-tiles freely; nothing publishes it. */                    \
        _Pragma("unroll") for (int d = 0; d < (DEPTH) - 1; ++d) {              \
            MMQ_DB_FETCH_W(NBLK, d, kt_lo, d);                                 \
        }                                                                      \
        MMQ_DB_FETCH_AF(TILES, 0, 0, xa);                                       \
                                                                               \
        for (int kt = kt_lo; kt < kt_hi; ++kt) {                               \
            const int pos = (kt - kt_lo) % (STAGES);                           \
            const int8_t* xbuf = xa + pos * x_rows * MMQ_XA_STRIDE;            \
            const int8_t* xnxt =                                               \
                xa + ((pos + 1) % (STAGES)) * x_rows * MMQ_XA_STRIDE;          \
            _Pragma("unroll") for (int s = 0; s < 4; ++s) {                    \
                MMQ_DB_FETCH_W(NBLK, MMQ_DB_WR(s, DEPTH),                      \
                               kt + MMQ_DB_WR_AHEAD(s, DEPTH),                 \
                               MMQ_DB_WR_STEP(s, DEPTH));                      \
                if (s < 3) {                                                   \
                    MMQ_DB_FETCH_AF(TILES, (s + 1) % 2, s + 1, xbuf);           \
                } else {                                                       \
                    MMQ_DB_FETCH_AF(TILES, (s + 1) % 2, 0, xnxt);               \
                }                                                              \
                if (s == 2) {                                                  \
                    MMQ_XA_FETCH((pos + (STAGES) - 1) % (STAGES),              \
                                 kt + (STAGES) - 1, kt_hi);                    \
                    MMQ_CP_ASYNC_WAIT((STAGES) - 2);                           \
                    __syncthreads();                                           \
                }                                                              \
                MMQ_NOE_MUL(NBLK, TILES, s % 2, MMQ_DB_RD(s, DEPTH), s,      \
                             xbuf);           \
            }                                                                  \
        }                                                                      \
                                                                               \
        MMQ_CP_ASYNC_WAIT(0);                                                  \
        __syncthreads();                                                       \
                                                                               \
        const bool whole = (kt_lo == 0) && (kt_hi == k_tiles);                 \
        _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {                   \
            const int orow = row0 + wbase + j * 8 + cc;                        \
            _Pragma("unroll") for (int u = 0; u < (TILES); ++u) {              \
                const int ot0 = tok0 + u * MMQ_M + cr;                         \
                const int ot1 = ot0 + 8;                                       \
                if (ot0 < n_tokens) {                                          \
                    if (orow < n)                                              \
                        MMQ_PUT2(whole, (size_t)ot0 * n + orow, acc[j][u][0]); \
                    if (orow + 1 < n)                                          \
                        MMQ_PUT2(whole, (size_t)ot0 * n + orow + 1,            \
                                 acc[j][u][1]);                                \
                }                                                              \
                if (ot1 < n_tokens) {                                          \
                    if (orow < n)                                              \
                        MMQ_PUT2(whole, (size_t)ot1 * n + orow, acc[j][u][2]); \
                    if (orow + 1 < n)                                          \
                        MMQ_PUT2(whole, (size_t)ot1 * n + orow + 1,            \
                                 acc[j][u][3]);                                \
                }                                                              \
            }                                                                  \
        }                                                                      \
        flat += kt_hi - kt_lo;                                                 \
    }

#define MMQ_LEANB_SET(SUFFIX, WARPS, NBLK, TILES, STAGES, DEPTH)               \
    extern "C" __global__ __launch_bounds__((WARPS) * WARP_SIZE) void          \
    mmql##SUFFIX##_q4_g128(float* __restrict__ out,                            \
                           const void* __restrict__ wv,                        \
                           const void* __restrict__ xv, int k, int n,          \
                           int n_tokens) {                                     \
        const block_q4_g128* wq = (const block_q4_g128*)wv;                    \
        const block_q8_1* xq = (const block_q8_1*)xv;                          \
        MMQ_LEANB_BODY(WARPS, NBLK, TILES, STAGES, DEPTH)                      \
    }

// The narrow shapes, and why they win — this is the measurement that turned the
// whole line of work around.
//
// `mmq_bw_probe_w4` walks a weight matrix with this kernel's exact access
// pattern and nothing else, so it is the ceiling any shape of this kernel can
// reach. On `ffn_gate` (4096 by 14336) it is 294 GB/s, against 390 GB/s for the
// same bytes read sixteen at a time. Against that ceiling, per GEMM rather than
// per decode step:
//
//                     8 tokens   32 tokens
//   the probe's ceiling   294        294
//   mmqd                  240         93
//   mmqsr2w4s4            238        150
//   mmqb2w4s4d2           246        156
//   mmqb1w4s2d2           296        182
//   mmql1w4s2d2           299        185
//
// Two things fall out of that table and neither was expected.
//
// **At eight tokens this kernel is done.** 299 against a 294 ceiling is the
// ceiling. Nothing about scheduling, pipelining or tiling can move it, and the
// only thing that can is a wider load, which needs the weights repacked.
//
// **NBLK=1 wins**, which contradicts the premise the whole port was built on.
// The note above `MMQ_WIDE_BODY` argues a warp should own more output so every
// operand load is amortized further, and Marlin's 64x64 register tile is the
// model. Measured here, one block of eight weight rows per warp beats two by
// 9-21%, because the wide tile costs registers and registers cost resident
// blocks — 158 against 124 at two token tiles, three blocks against four.
//
// Amortization is the right idea on a machine whose kernel is issue-bound. This
// one is not; it is waiting on memory, and what it wants is warps.
//
// What is left, and where: at 32 tokens the kernel runs at 185 GB/s against the
// same 294 ceiling it reaches at 8. Forcing one token tile at 32 tokens gives
// 150 GB/s of *counted* weights, but that shape reads the weights twice, so the
// real traffic is 300 GB/s — at the ceiling again. So two token tiles is not
// short of memory bandwidth; it is short of everything else. Per k-tile per
// warp it issues 8 weight loads, 64 A-fragment shared loads and 64 accumulator
// updates, and the last two scale with the token tiles while the first does not.

/* mmql<nblk>w<warps>s<stages>d<depth>: `mmqb_*` with the per-token activation
   scales read at use instead of carried in registers. Worth 158 registers down
   to 124 at two token tiles, which is three resident blocks up to four. */
MMQ_LEANB_SET(1w4s2d2, 4, 1, 1, 2, 2)
MMQ_LEANB_SET(1w4s2d2_2, 4, 1, 2, 2, 2)
MMQ_LEANB_SET(1w4s4d2, 4, 1, 1, 4, 2)
MMQ_LEANB_SET(1w4s4d2_2, 4, 1, 2, 4, 2)
MMQ_LEANB_SET(2w4s2d2, 4, 2, 1, 2, 2)
MMQ_LEANB_SET(2w4s2d2_2, 4, 2, 2, 2, 2)

#define MMQ_PROBE2_SET(NAME, BODY, WARPS, NBLK, TILES, STAGES, DEPTH)          \
    extern "C" __global__ __launch_bounds__((WARPS) * WARP_SIZE) void          \
    NAME(float* __restrict__ out, const void* __restrict__ wv,                 \
         const void* __restrict__ xv, int k, int n, int n_tokens) {            \
        const block_q4_g128* wq = (const block_q4_g128*)wv;                    \
        const block_q8_1* xq = (const block_q8_1*)xv;                          \
        BODY(WARPS, NBLK, TILES, STAGES, DEPTH)                                \
    }

MMQ_PROBE2_SET(mmqna1w4s2d2_q4_g128, MMQ_NOA_BODY, 4, 1, 1, 2, 2)
MMQ_PROBE2_SET(mmqna1w4s2d2_2_q4_g128, MMQ_NOA_BODY, 4, 1, 2, 2, 2)
MMQ_PROBE2_SET(mmqne1w4s2d2_q4_g128, MMQ_NOE_BODY, 4, 1, 1, 2, 2)
MMQ_PROBE2_SET(mmqne1w4s2d2_2_q4_g128, MMQ_NOE_BODY, 4, 1, 2, 2, 2)

#define MMQ_DEEPB_SET(SUFFIX, WARPS, NBLK, TILES, STAGES, DEPTH)               \
    extern "C" __global__ __launch_bounds__((WARPS) * WARP_SIZE) void          \
    mmqb##SUFFIX##_q4_g128(float* __restrict__ out,                            \
                           const void* __restrict__ wv,                        \
                           const void* __restrict__ xv, int k, int n,          \
                           int n_tokens) {                                     \
        const block_q4_g128* wq = (const block_q4_g128*)wv;                    \
        const block_q8_1* xq = (const block_q8_1*)xv;                          \
        MMQ_DEEPB_BODY(WARPS, NBLK, TILES, STAGES, DEPTH)                      \
    }

// Measured, and it does not work (tok/s, AWQ 8B, 256 tokens of history):
//
//                   batch 4    8    16    32
//   mmqsr2w4s4        167    325   583   743
//   mmqb2w4s4d2       172    336   598   754     the control
//   mmqb2w4s4d4       159    310   559   724
//   mmqb2w2s4d4       150    301   550   617
//
// Depth 4 is 5-8% down at every width, and `kernel_registers` says exactly why:
// 168 registers a thread against depth 2's 128, which is 3 resident blocks per
// SM against 4.
//
// That is the whole answer to the idea, and it is worth stating plainly because
// it rules out a family of changes rather than one: **memory-level parallelism
// cannot be bought with registers here.** Registers are what buy occupancy, and
// occupancy is where the parallelism was coming from — so holding more loads in
// flight per thread costs more warps than it gains issue slots. Every deeper
// software pipeline over this operand runs into the same wall.
//
// Which leaves the other way to put more bytes in flight: make each load move
// more of them. This kernel reads weights four bytes at a time
// (`*(const uint32_t*)pq`, twice) because a lane's eight bytes are not
// contiguous in the AWQ pack. Marlin reads sixteen at a time — `const int4* B`
// at `marlin_template.h:59`, `cp_async4` at :757, `I4 frag_b_quant` at :689 —
// and it can because `gptq_marlin_repack` put each lane's whole fragment in one
// contiguous 16-byte chunk first. That is the piece of Marlin this port skipped
// twice, deliberately, to avoid touching the loader. On this evidence it is the
// piece that mattered.
//
// The control is 1-2% above `mmqsr_*` at every width rather than level, which
// is past the 0.8% run-to-run spread but not by much. The restructure hoists
// the weight scales into the same fetch as the weight words; no other
// difference. Treat it as level.

/* mmqb<nblk>w<warps>s<stages>d<depth>. Depth 2 is `mmqsr_*` and is the
   control; depth 4 puts three times the weight bytes in flight. 8 is not here
   because its slot index would not fold — see `MMQ_DB_RD`. */
MMQ_DEEPB_SET(2w4s4d2, 4, 2, 1, 4, 2)
MMQ_DEEPB_SET(2w4s4d2_2, 4, 2, 2, 4, 2)
MMQ_DEEPB_SET(2w4s4d4, 4, 2, 1, 4, 4)
MMQ_DEEPB_SET(2w4s4d4_2, 4, 2, 2, 4, 4)
MMQ_DEEPB_SET(2w2s4d4, 2, 2, 1, 4, 4)
MMQ_DEEPB_SET(2w2s4d4_2, 2, 2, 2, 4, 4)
/* Shapes aimed at the two-token-tile cliff: at TILES=2 the four-stage ring is
   38.9 KB and the body wants 215 registers, and both cap the SM at two
   resident blocks. Fewer stages attack the first, a narrower register tile the
   second. */
MMQ_DEEPB_SET(2w4s2d2, 4, 2, 1, 2, 2)
MMQ_DEEPB_SET(2w4s2d2_2, 4, 2, 2, 2, 2)
MMQ_DEEPB_SET(2w4s3d2, 4, 2, 1, 3, 2)
MMQ_DEEPB_SET(2w4s3d2_2, 4, 2, 2, 3, 2)
MMQ_DEEPB_SET(1w4s4d2, 4, 1, 1, 4, 2)
MMQ_DEEPB_SET(1w4s4d2_2, 4, 1, 2, 4, 2)
MMQ_DEEPB_SET(1w8s4d2, 8, 1, 1, 4, 2)
MMQ_DEEPB_SET(1w8s4d2_2, 8, 1, 2, 4, 2)
MMQ_DEEPB_SET(2w8s2d2, 8, 2, 1, 2, 2)
MMQ_DEEPB_SET(2w8s2d2_2, 8, 2, 2, 2, 2)
MMQ_DEEPB_SET(1w2s4d2, 2, 1, 1, 4, 2)
MMQ_DEEPB_SET(1w2s4d2_2, 2, 1, 2, 4, 2)
MMQ_DEEPB_SET(1w4s2d2, 4, 1, 1, 2, 2)
MMQ_DEEPB_SET(1w4s2d2_2, 4, 1, 2, 2, 2)
/* Eight stages fits at one token tile and not at two — 77.8 KB against the
   48 KB static cap — so it exists only in the narrow form. */
MMQ_DEEPB_SET(1w4s8d2, 4, 1, 1, 8, 2)
MMQ_DEEPB_SET(1w4s8d2_2, 4, 1, 2, 4, 2)

// ---- f16 operands, for Q4_G128 ------------------------------------------
//
// Step three of the port in vendor/marlin/README.md, and the one it calls the
// largest. Everything above multiplies int8 against int8 into an s32
// accumulator, which can only span one quantization group — so every 32
// weights the accumulator has to be drained, converted to float, scaled by the
// weight scale and the activation scale, and corrected by a zero-point-times-
// activation-sum term. That epilogue is four fused multiply-adds and two
// conversions per accumulator per group, and it is paid `NBLK * TILES` times.
//
// With f16 operands it is paid zero times. `mma.m16n8k16.f16` accumulates in
// f32, so the accumulator spans the whole k; the scale and the zero point fold
// into the B fragment as one `hfma2` while the nibbles are being unpacked
// anyway (`mmq_deq4_f16`); and the activations, being f16 rather than Q8_1,
// carry no per-block scale and no stored sum for a zero-point term to multiply.
// The epilogue is the store.
//
// What it costs instead: k per MMA halves, so twice the instructions; and the
// activations are 2 bytes an element rather than Q8_1's 1.125, so the ring
// buffer is 8.5 KB per stage per token tile against 4.8. Four stages and two
// token tiles is 68 KB, which is why this one is on dynamic shared memory —
// the 48 KB cap is on static `__shared__` only.
//
// The nibble-to-half path and the k numbering it implies are `mmq_deq4_f16`
// and `mmq_f16_k` above, both pinned by tests. The one structural consequence
// is here: A and B have to agree on what k means, and rather than repack the
// weights, the activation gather bends to the pack. A lane reads eight
// contiguous bytes at `8 * (lane % 4)` — one 8-byte load per row where the s8
// path did two 4-byte ones.

#define MMQ_XF_ROW (MMQ_K * 2)             // 256 halves of activation
// 544 is 16-aligned for `cp.async` and 32 mod 128, which is what makes the
// 8-byte gather conflict-free: lane L reads row L/4 at byte 8*(L%4), so the
// sixteen lanes of a phase cover all 32 banks exactly when the row stride is
// 8 words mod 32.
#define MMQ_XF_STRIDE (MMQ_XF_ROW + 32)

#define MMQ_XF_FETCH(BUF, TILE, LIMIT)                                         \
    do {                                                                       \
        const int _tl = (TILE);                                                \
        const bool _live = _tl < (LIMIT);                                      \
        for (int _i = tid; _i < x_valid * (MMQ_XF_ROW / 16); _i += nthreads) { \
            const int _r = _i / (MMQ_XF_ROW / 16);                             \
            const int _c = _i % (MMQ_XF_ROW / 16);                             \
            /* Eight halves a chunk, and k is a multiple of 128, so a chunk    \
               is wholly inside the row or wholly past it. */                  \
            const bool _hit = _live && (_tl * MMQ_K + _c * 8 + 8 <= k);        \
            const size_t _off =                                                \
                ((size_t)(tok0 + _r) * k + (size_t)_tl * MMQ_K) * 2            \
                + (size_t)_c * 16;                                             \
            mmq_cp_async16(                                                    \
                xf + ((BUF) * x_rows + _r) * MMQ_XF_STRIDE + _c * 16,          \
                xbytes + (_hit ? _off : 0), _hit);                             \
        }                                                                      \
        MMQ_CP_ASYNC_FENCE();                                                  \
    } while (0)


// The dequantization a Marlin-ordered pack would allow: the two bytes of a
// pair already sit where `lop3`'s mask wants them, so the `prmt` pair that
// `mmq_deq4_f16` needs to shuffle AWQ's order goes away. Two instructions of
// ten, per two weights.
__device__ __forceinline__ void mmq_deq4_f16_repacked(uint32_t v, int h,
                                                      __half2 s2, __half2 m2,
                                                      unsigned* f) {
    uint32_t p0 = v;
    uint32_t p1 = v >> 8;
    if (h) {
        p0 >>= 4;
        p1 >>= 4;
    }
    const uint32_t MASK = 0x000f000fu;
    const uint32_t EX = 0x64006400u;
    uint32_t q0 = mmq_lop3<(0xf0 & 0xcc) | 0xaa>(p0, MASK, EX);
    uint32_t q1 = mmq_lop3<(0xf0 & 0xcc) | 0xaa>(p1, MASK, EX);
    const __half2 bias = __float2half2_rn(1024.0f);
    const __half2 w0 = __hsub2(*(const __half2*)(const void*)&q0, bias);
    const __half2 w1 = __hsub2(*(const __half2*)(const void*)&q1, bias);
    const __half2 r0 = __hfma2(w0, s2, m2);
    const __half2 r1 = __hfma2(w1, s2, m2);
    f[0] = *(const unsigned*)(const void*)&r0;
    f[1] = *(const unsigned*)(const void*)&r1;
}

// ---- what a repacked weight layout would be worth -----------------------
//
// Three things the AWQ pack costs this kernel, all of them fixable only by
// reordering the weights at load: four-byte loads where sixteen would do, a
// `prmt` pair in every dequantization, and a scalar A gather where `ldmatrix`
// would serve if the fragment order were the standard one.
//
// Each is small; the question is what they are together, and the answer decides
// whether to touch the loader, `unpack_row`, the mat-vec, the float path and
// every test that pins them. So: the same kernel with all three assumed, wrong
// answers and right traffic.
//
// **It does not happen.** GB/s of weights on `ffn_gate`, three rounds each and
// stable to a digit:
//
//                        8 tokens   32 tokens
//   mmqf1w8s2 (as built)     330        214
//   mmqfp1w8s2 (16-byte)     306        210
//   mmqfp1w8s2x2 (2 x 8)     303        180
//   mmqfp1w8s2x4 (4 x 4)     292        184
//
// Level to 7% down, across three different request shapes over the same
// repacked layout. The wider load is not worth anything here, and at 8 tokens
// there was nothing to win in the first place: 330 against a 341 ceiling.
//
// Two things had to be fixed before that reading was worth anything, and both
// were the probe rather than the idea:
//
//   * `ldmatrix` on the A side measured 10% slower, because `MMQ_XF_STRIDE` is
//     544 — chosen so an 8-byte gather at `8 * (lane % 4)` is conflict-free,
//     and 8 words mod 32, which is exactly the stride that two-ways
//     `ldmatrix`. Whether `ldmatrix` helps is a question about the activation
//     tile, not about the weight layout, so it came out of this probe.
//   * `bsrc[j] - wq` is a pointer difference on a 68-byte struct, so it
//     compiles to a division by a non-power-of-two in the inner loop. That
//     alone was 8% at 32 tokens — the entire size of the effect being measured.
//
// Which is the standing lesson of this file, one more time: three of the four
// numbers this probe first produced were the instrument.


// ---- the same, with `ldmatrix` on the A side ----------------------------
//
// `mmqf_*` gathers each A fragment with two 8-byte loads because the fragment
// order it uses is not the one `ldmatrix` produces — it bent k to suit the AWQ
// weight pack rather than repack the weights. `f32_to_f16_kperm` in `ops.cu`
// bends the *activations* instead, which costs nothing (they are a thousandth
// of the bytes and are rewritten every step), and that puts the standard
// fragment order back within reach.
//
// The stride has to change with it. 544 was chosen so an 8-byte gather at
// `8 * (lane % 4)` is conflict-free, which needs 32 mod 128; `ldmatrix` wants
// its eight row addresses on distinct bank groups, which needs 16 mod 128. The
// two are mutually exclusive, so this shape gets its own.
//
// **And it loses.** GB/s of weights, two rounds, stable to a digit:
//
//                     ffn_gate  ffn_down  attn_q  attn_k
//   mmqf1w8s2  @ 8t      331       341      261     177
//   mmqm1w8s2  @ 8t      243       251      239     183
//   mmqf1w8s2  @32t      213       226      171     115
//   mmqm1w8s2  @32t      167       170      158     107
//
// 20-25% down at 32 tokens on the two matrices that matter, ahead only on the
// narrowest at eight. Two 8-byte gathers are two instructions but they are
// wide and conflict-free; `ldmatrix` is one instruction that serialises four
// 8x8 tiles internally. Fewer instructions is not the same as less work, which
// is the sixth time this file has had to write that down.
//
// Kept because the activation permutation it is built on (`f32_to_f16_kperm`)
// is the cheap half of the idea and would be needed again by anything wanting
// the standard fragment order — and because the negative is only worth having
// once it is A/B-able.
#define MMQ_XL_ROW (MMQ_K * 2)
#define MMQ_XL_STRIDE (MMQ_XL_ROW + 16)

#define MMQ_XL_FETCH(BUF, TILE, LIMIT)                                         \
    do {                                                                       \
        const int _tl = (TILE);                                                \
        const bool _live = _tl < (LIMIT);                                      \
        for (int _i = tid; _i < x_valid * (MMQ_XL_ROW / 16); _i += nthreads) { \
            const int _r = _i / (MMQ_XL_ROW / 16);                             \
            const int _c = _i % (MMQ_XL_ROW / 16);                             \
            const bool _hit = _live && (_tl * MMQ_K + _c * 8 + 8 <= k);        \
            const size_t _off =                                                \
                ((size_t)(tok0 + _r) * k + (size_t)_tl * MMQ_K) * 2            \
                + (size_t)_c * 16;                                             \
            mmq_cp_async16(                                                    \
                xf + ((BUF) * x_rows + _r) * MMQ_XL_STRIDE + _c * 16,          \
                xbytes + (_hit ? _off : 0), _hit);                             \
        }                                                                      \
        MMQ_CP_ASYNC_FENCE();                                                  \
    } while (0)

#define MMQ_F16_BODY(WARPS, NBLK, TILES, STAGES)                               \
    extern __shared__ __align__(16) int8_t xf[];                               \
                                                                               \
    const int tid = threadIdx.x;                                               \
    const int lane = tid % WARP_SIZE;                                          \
    const int warp = tid / WARP_SIZE;                                          \
    const int nthreads = (WARPS) * WARP_SIZE;                                  \
    const int mrows = (WARPS) * (NBLK) * 8;                                    \
    const int tok0 = blockIdx.y * (TILES) * MMQ_M;                             \
    const int nb_total = k / QK_G128;                                          \
    const int k_tiles = (k + MMQ_K - 1) / MMQ_K;                               \
    const int row_tiles = (n + mrows - 1) / mrows;                             \
    const int x_rows = (TILES) * MMQ_M;                                        \
    const int x_valid = min(x_rows, max(0, n_tokens - tok0));                  \
    const uint8_t* xbytes = (const uint8_t*)xv;                                \
                                                                               \
    const int ar = mma_a_row(lane);                                            \
    const int bc = mma_b_col(lane);                                            \
    const int cq = lane % 4;                                                   \
    const int kq = cq * 4;                                                     \
    const int cr = mma_c_row(lane);                                            \
    const int cc = mma_c_col(lane);                                            \
    const int wbase = warp * (NBLK) * 8;                                       \
                                                                               \
    /* The striped partition, which `mmqsr_*` measured worth 30% at four        \
       tokens. Same arithmetic here. */                                        \
    const int total = row_tiles * k_tiles;                                     \
    const int iters = (total + (int)gridDim.x - 1) / (int)gridDim.x;           \
    int flat = iters * (int)blockIdx.x;                                        \
    const int flat_end = min(total, flat + iters);                             \
                                                                               \
    const block_q4_g128* bsrc[NBLK];                                           \
    bool brow_ok[NBLK];                                                        \
    mma_c_f32 acc[NBLK][TILES];                                                \
                                                                               \
    if (x_valid < x_rows) {                                                    \
        const int per = (x_rows - x_valid) * (MMQ_XF_ROW / 4);                 \
        for (int i = tid; i < (STAGES) * per; i += nthreads) {                 \
            const int s = i / per;                                             \
            const int j = i % per;                                             \
            const int r = x_valid + j / (MMQ_XF_ROW / 4);                      \
            const int e = (j % (MMQ_XF_ROW / 4)) * 4;                          \
            *(uint32_t*)(void*)(xf + (s * x_rows + r) * MMQ_XF_STRIDE + e) = 0;\
        }                                                                      \
    }                                                                          \
                                                                               \
    while (flat < flat_end) {                                                  \
        const int nt = flat / k_tiles;                                         \
        const int kt_lo = flat % k_tiles;                                      \
        const int kt_hi = min(k_tiles, kt_lo + (flat_end - flat));             \
        const int row0 = nt * mrows;                                           \
                                                                               \
        _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {                   \
            const int r = row0 + wbase + j * 8 + bc;                           \
            brow_ok[j] = r < n;                                                \
            bsrc[j] = wq + (brow_ok[j] ? (size_t)r * nb_total : 0);            \
            _Pragma("unroll") for (int u = 0; u < (TILES); ++u) {              \
                _Pragma("unroll") for (int c = 0; c < 4; ++c)                  \
                    acc[j][u].x[c] = 0.0f;                                     \
            }                                                                  \
        }                                                                      \
                                                                               \
        _Pragma("unroll") for (int s = 0; s < (STAGES) - 1; ++s) {             \
            MMQ_XF_FETCH(s, kt_lo + s, kt_hi);                                 \
        }                                                                      \
                                                                               \
        for (int kt = kt_lo; kt < kt_hi; ++kt) {                               \
            const int pos = (kt - kt_lo) % (STAGES);                           \
            MMQ_CP_ASYNC_WAIT((STAGES) - 2);                                   \
            __syncthreads();                                                   \
            MMQ_XF_FETCH((pos + (STAGES) - 1) % (STAGES),                      \
                         kt + (STAGES) - 1, kt_hi);                            \
            const int8_t* xbuf = xf + pos * x_rows * MMQ_XF_STRIDE;            \
                                                                               \
            _Pragma("unroll") for (int blk = 0; blk < 2; ++blk) {              \
                const int kb = kt * 2 + blk;                                   \
                const bool kb_ok = kb < nb_total;                              \
                /* One scale and one zero per row per 128 weights, broadcast   \
                   into the halves the dequantization folds them into. */      \
                __half2 s2[NBLK], m2[NBLK];                                    \
                _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {           \
                    const int sr = row0 + wbase + j * 8 + bc;                  \
                    __half2 ds = __floats2half2_rn(0.0f, 0.0f);                \
                    if (kb_ok && brow_ok[j]) ds = bsrc[j][kb].ds;              \
                    s2[j] = __float2half2_rn(__low2float(ds));                 \
                    m2[j] = __float2half2_rn(-__high2float(ds));               \
                    (void)sr;                                                  \
                }                                                              \
                _Pragma("unroll") for (int run = 0; run < 2; ++run) {          \
                    uint32_t v0[NBLK], v1[NBLK];                               \
                    _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {       \
                        v0[j] = 0;                                             \
                        v1[j] = 0;                                             \
                        if (brow_ok[j] && kb_ok) {                             \
                            const uint8_t* pq =                                \
                                bsrc[j][kb].qs + run * 32 + kq;                \
                            v0[j] = *(const uint32_t*)(const void*)pq;         \
                            v1[j] = *(const uint32_t*)(const void*)(pq + 16);  \
                        }                                                      \
                    }                                                          \
                    _Pragma("unroll") for (int h = 0; h < 2; ++h) {            \
                        const int g = blk * 4 + run + h * 2;                   \
                        /* Two MMAs a group: k is 16 here, not 32. */          \
                        _Pragma("unroll") for (int mm = 0; mm < 2; ++mm) {     \
                            mma_a_f16 a[TILES];                                \
                            _Pragma("unroll") for (int u = 0; u < (TILES);     \
                                                   ++u) {                      \
                                const int8_t* ap =                             \
                                    xbuf + (u * MMQ_M + ar) * MMQ_XF_STRIDE    \
                                    + g * 64 + mm * 32 + cq * 8;               \
                                const int8_t* aq = ap + 8 * MMQ_XF_STRIDE;     \
                                const uint2 lo =                               \
                                    *(const uint2*)(const void*)ap;            \
                                const uint2 hi =                               \
                                    *(const uint2*)(const void*)aq;            \
                                a[u].x[0] = lo.x;                              \
                                a[u].x[1] = hi.x;                              \
                                a[u].x[2] = lo.y;                              \
                                a[u].x[3] = hi.y;                              \
                            }                                                  \
                            _Pragma("unroll") for (int j = 0; j < (NBLK);      \
                                                   ++j) {                      \
                                mma_b_f16 b;                                   \
                                mmq_deq4_f16(mm ? v1[j] : v0[j], h, s2[j],     \
                                             m2[j], b.x);                      \
                                _Pragma("unroll") for (int u = 0;              \
                                                       u < (TILES); ++u) {     \
                                    mma_f16(acc[j][u], a[u], b);               \
                                }                                              \
                            }                                                  \
                        }                                                      \
                    }                                                          \
                }                                                              \
            }                                                                  \
        }                                                                      \
                                                                               \
        MMQ_CP_ASYNC_WAIT(0);                                                  \
        __syncthreads();                                                       \
                                                                               \
        /* No epilogue: the accumulator is already the answer. */              \
        const bool whole = (kt_lo == 0) && (kt_hi == k_tiles);                 \
        _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {                   \
            const int orow = row0 + wbase + j * 8 + cc;                        \
            _Pragma("unroll") for (int u = 0; u < (TILES); ++u) {              \
                const int ot0 = tok0 + u * MMQ_M + cr;                         \
                const int ot1 = ot0 + 8;                                       \
                if (ot0 < n_tokens) {                                          \
                    if (orow < n)                                              \
                        MMQ_PUT2(whole, (size_t)ot0 * n + orow,                \
                                 acc[j][u].x[0]);                              \
                    if (orow + 1 < n)                                          \
                        MMQ_PUT2(whole, (size_t)ot0 * n + orow + 1,            \
                                 acc[j][u].x[1]);                              \
                }                                                              \
                if (ot1 < n_tokens) {                                          \
                    if (orow < n)                                              \
                        MMQ_PUT2(whole, (size_t)ot1 * n + orow,                \
                                 acc[j][u].x[2]);                              \
                    if (orow + 1 < n)                                          \
                        MMQ_PUT2(whole, (size_t)ot1 * n + orow + 1,            \
                                 acc[j][u].x[3]);                              \
                }                                                              \
            }                                                                  \
        }                                                                      \
        flat += kt_hi - kt_lo;                                                 \
    }

#define MMQ_F16REPACK_BODY(WARPS, NBLK, TILES, STAGES, SPLIT)                               \
    extern __shared__ __align__(16) int8_t xf[];                               \
                                                                               \
    const int tid = threadIdx.x;                                               \
    const int lane = tid % WARP_SIZE;                                          \
    const int warp = tid / WARP_SIZE;                                          \
    const int nthreads = (WARPS) * WARP_SIZE;                                  \
    const int mrows = (WARPS) * (NBLK) * 8;                                    \
    const int tok0 = blockIdx.y * (TILES) * MMQ_M;                             \
    const int nb_total = k / QK_G128;                                          \
    const int k_tiles = (k + MMQ_K - 1) / MMQ_K;                               \
    const int row_tiles = (n + mrows - 1) / mrows;                             \
    const int x_rows = (TILES) * MMQ_M;                                        \
    const int x_valid = min(x_rows, max(0, n_tokens - tok0));                  \
    const uint8_t* xbytes = (const uint8_t*)xv;                                \
                                                                               \
    const int ar = mma_a_row(lane);                                            \
    const int bc = mma_b_col(lane);                                            \
    const int cq = lane % 4;                                                   \
    const int kq = cq * 4;                                                     \
    const int cr = mma_c_row(lane);                                            \
    const int cc = mma_c_col(lane);                                            \
    const int wbase = warp * (NBLK) * 8;                                       \
                                                                               \
    /* The striped partition, which `mmqsr_*` measured worth 30% at four        \
       tokens. Same arithmetic here. */                                        \
    const int total = row_tiles * k_tiles;                                     \
    const int iters = (total + (int)gridDim.x - 1) / (int)gridDim.x;           \
    int flat = iters * (int)blockIdx.x;                                        \
    const int flat_end = min(total, flat + iters);                             \
                                                                               \
    const block_q4_g128* bsrc[NBLK];                                           \
    size_t browi[NBLK];                                                        \
    bool brow_ok[NBLK];                                                        \
    mma_c_f32 acc[NBLK][TILES];                                                \
                                                                               \
    if (x_valid < x_rows) {                                                    \
        const int per = (x_rows - x_valid) * (MMQ_XF_ROW / 4);                 \
        for (int i = tid; i < (STAGES) * per; i += nthreads) {                 \
            const int s = i / per;                                             \
            const int j = i % per;                                             \
            const int r = x_valid + j / (MMQ_XF_ROW / 4);                      \
            const int e = (j % (MMQ_XF_ROW / 4)) * 4;                          \
            *(uint32_t*)(void*)(xf + (s * x_rows + r) * MMQ_XF_STRIDE + e) = 0;\
        }                                                                      \
    }                                                                          \
                                                                               \
    while (flat < flat_end) {                                                  \
        const int nt = flat / k_tiles;                                         \
        const int kt_lo = flat % k_tiles;                                      \
        const int kt_hi = min(k_tiles, kt_lo + (flat_end - flat));             \
        const int row0 = nt * mrows;                                           \
                                                                               \
        _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {                   \
            const int r = row0 + wbase + j * 8 + bc;                           \
            brow_ok[j] = r < n;                                                \
            /* The row's byte offset, not its index: multiplying by the row  \
               stride inside the k-loop puts a 64-bit multiply on every weight \
               address and cost 13% when it was written that way. */          \
            browi[j] = brow_ok[j] ? (size_t)r * nb_total : 0;                  \
            bsrc[j] = wq + browi[j];                                           \
            _Pragma("unroll") for (int u = 0; u < (TILES); ++u) {              \
                _Pragma("unroll") for (int c = 0; c < 4; ++c)                  \
                    acc[j][u].x[c] = 0.0f;                                     \
            }                                                                  \
        }                                                                      \
                                                                               \
        _Pragma("unroll") for (int s = 0; s < (STAGES) - 1; ++s) {             \
            MMQ_XF_FETCH(s, kt_lo + s, kt_hi);                                 \
        }                                                                      \
                                                                               \
        for (int kt = kt_lo; kt < kt_hi; ++kt) {                               \
            const int pos = (kt - kt_lo) % (STAGES);                           \
            MMQ_CP_ASYNC_WAIT((STAGES) - 2);                                   \
            __syncthreads();                                                   \
            MMQ_XF_FETCH((pos + (STAGES) - 1) % (STAGES),                      \
                         kt + (STAGES) - 1, kt_hi);                            \
            const int8_t* xbuf = xf + pos * x_rows * MMQ_XF_STRIDE;            \
                                                                               \
            _Pragma("unroll") for (int blk = 0; blk < 2; ++blk) {              \
                const int kb = kt * 2 + blk;                                   \
                const bool kb_ok = kb < nb_total;                              \
                /* One scale and one zero per row per 128 weights, broadcast   \
                   into the halves the dequantization folds them into. */      \
                __half2 s2[NBLK], m2[NBLK];                                    \
                _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {           \
                    const int sr = row0 + wbase + j * 8 + bc;                  \
                    __half2 ds = __floats2half2_rn(0.0f, 0.0f);                \
                    if (kb_ok && brow_ok[j]) ds = bsrc[j][kb].ds;              \
                    s2[j] = __float2half2_rn(__low2float(ds));                 \
                    m2[j] = __float2half2_rn(-__high2float(ds));               \
                    (void)sr;                                                  \
                }                                                              \
                uint4 vv[NBLK];                                                \
                _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {           \
                    vv[j] = make_uint4(0, 0, 0, 0);                            \
                    if (brow_ok[j] && kb_ok) {                                 \
                        /* 16-byte aligned by construction; the value is wrong \
                           and the traffic is right, which is the point. */    \
                        /* From the row index, not from a pointer difference: \
                           `block_q4_g128` is 68 bytes, so `bsrc[j] - wq` is a \
                           division by a non-power-of-two in the inner loop.   \
                           That artefact cost this probe 8%, which is the      \
                           whole effect it was built to measure. */            \
                        const size_t _blkid = browi[j] + kb;                   \
                        const uint8_t* pq = (const uint8_t*)(const void*)wq     \
                                          + _blkid * 64 + cq * 16;             \
                        /* SPLIT trades one 16-byte request for two 8-byte    \
                           ones over the same bytes, which is what separates   \
                           request width from request count. */                \
                        if ((SPLIT) == 2) {                                    \
                            const uint2 _a = *(const uint2*)(const void*)pq;   \
                            const uint2 _b =                                   \
                                *(const uint2*)(const void*)(pq + 8);          \
                            vv[j] = make_uint4(_a.x, _a.y, _b.x, _b.y);        \
                        } else if ((SPLIT) == 4) {                             \
                            const uint32_t* _p = (const uint32_t*)(const void*)pq;\
                            vv[j] = make_uint4(_p[0], _p[1], _p[2], _p[3]);    \
                        } else {                                               \
                            vv[j] = *(const uint4*)(const void*)pq;            \
                        }                                                      \
                    }                                                          \
                }                                                              \
                _Pragma("unroll") for (int run = 0; run < 2; ++run) {          \
                    uint32_t v0[NBLK], v1[NBLK];                               \
                    _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {       \
                        v0[j] = run ? vv[j].z : vv[j].x;                       \
                        v1[j] = run ? vv[j].w : vv[j].y;                       \
                    }                                                          \
                    _Pragma("unroll") for (int h = 0; h < 2; ++h) {            \
                        const int g = blk * 4 + run + h * 2;                   \
                        /* Two MMAs a group: k is 16 here, not 32. */          \
                        _Pragma("unroll") for (int mm = 0; mm < 2; ++mm) {     \
                            mma_a_f16 a[TILES];                                \
                            _Pragma("unroll") for (int u = 0; u < (TILES);     \
                                                   ++u) {                      \
                                const int8_t* ap =                             \
                                    xbuf + (u * MMQ_M + ar) * MMQ_XF_STRIDE    \
                                    + g * 64 + mm * 32 + cq * 8;               \
                                const int8_t* aq = ap + 8 * MMQ_XF_STRIDE;     \
                                /* The A side stays on the scalar gather. An  \
                                   earlier cut of this probe used `ldmatrix`   \
                                   here and measured 10% slower, which was the \
                                   probe and not the idea: `MMQ_XF_STRIDE` is  \
                                   544, chosen so an 8-byte gather at          \
                                   `8 * (lane % 4)` is conflict-free, and 544  \
                                   is 8 words mod 32 — exactly the stride that \
                                   two-ways `ldmatrix`. Measuring that needs a \
                                   different tile, so it is a separate         \
                                   question from the weight layout. */         \
                                const uint2 lo =                               \
                                    *(const uint2*)(const void*)ap;            \
                                const uint2 hi =                               \
                                    *(const uint2*)(const void*)aq;            \
                                a[u].x[0] = lo.x;                              \
                                a[u].x[1] = hi.x;                              \
                                a[u].x[2] = lo.y;                              \
                                a[u].x[3] = hi.y;                              \
                            }                                                  \
                            _Pragma("unroll") for (int j = 0; j < (NBLK);      \
                                                   ++j) {                      \
                                mma_b_f16 b;                                   \
                                mmq_deq4_f16_repacked(mm ? v1[j] : v0[j], h,   \
                                                      s2[j], m2[j], b.x);                      \
                                _Pragma("unroll") for (int u = 0;              \
                                                       u < (TILES); ++u) {     \
                                    mma_f16(acc[j][u], a[u], b);               \
                                }                                              \
                            }                                                  \
                        }                                                      \
                    }                                                          \
                }                                                              \
            }                                                                  \
        }                                                                      \
                                                                               \
        MMQ_CP_ASYNC_WAIT(0);                                                  \
        __syncthreads();                                                       \
                                                                               \
        /* No epilogue: the accumulator is already the answer. */              \
        const bool whole = (kt_lo == 0) && (kt_hi == k_tiles);                 \
        _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {                   \
            const int orow = row0 + wbase + j * 8 + cc;                        \
            _Pragma("unroll") for (int u = 0; u < (TILES); ++u) {              \
                const int ot0 = tok0 + u * MMQ_M + cr;                         \
                const int ot1 = ot0 + 8;                                       \
                if (ot0 < n_tokens) {                                          \
                    if (orow < n)                                              \
                        MMQ_PUT2(whole, (size_t)ot0 * n + orow,                \
                                 acc[j][u].x[0]);                              \
                    if (orow + 1 < n)                                          \
                        MMQ_PUT2(whole, (size_t)ot0 * n + orow + 1,            \
                                 acc[j][u].x[1]);                              \
                }                                                              \
                if (ot1 < n_tokens) {                                          \
                    if (orow < n)                                              \
                        MMQ_PUT2(whole, (size_t)ot1 * n + orow,                \
                                 acc[j][u].x[2]);                              \
                    if (orow + 1 < n)                                          \
                        MMQ_PUT2(whole, (size_t)ot1 * n + orow + 1,            \
                                 acc[j][u].x[3]);                              \
                }                                                              \
            }                                                                  \
        }                                                                      \
        flat += kt_hi - kt_lo;                                                 \
    }

#define MMQ_F16REPACK_SET(SUFFIX, WARPS, NBLK, TILES, STAGES, SPLIT)           \
    extern "C" __global__ __launch_bounds__((WARPS) * WARP_SIZE) void          \
    mmqfp##SUFFIX##_q4_g128(float* __restrict__ out,                           \
                            const void* __restrict__ wv,                       \
                            const void* __restrict__ xv, int k, int n,         \
                            int n_tokens) {                                    \
        const block_q4_g128* wq = (const block_q4_g128*)wv;                    \
        MMQ_F16REPACK_BODY(WARPS, NBLK, TILES, STAGES, SPLIT)                  \
    }

MMQ_F16REPACK_SET(1w8s2, 8, 1, 1, 2, 1)
MMQ_F16REPACK_SET(1w8s2_2, 8, 1, 2, 2, 1)
MMQ_F16REPACK_SET(1w8s2x2, 8, 1, 1, 2, 2)
MMQ_F16REPACK_SET(1w8s2x2_2, 8, 1, 2, 2, 2)
MMQ_F16REPACK_SET(1w8s2x4, 8, 1, 1, 2, 4)
MMQ_F16REPACK_SET(1w8s2x4_2, 8, 1, 2, 2, 4)

#define MMQ_F16LM_BODY(WARPS, NBLK, TILES, STAGES)                               \
    extern __shared__ __align__(16) int8_t xf[];                               \
                                                                               \
    const int tid = threadIdx.x;                                               \
    const int lane = tid % WARP_SIZE;                                          \
    const int warp = tid / WARP_SIZE;                                          \
    const int nthreads = (WARPS) * WARP_SIZE;                                  \
    const int mrows = (WARPS) * (NBLK) * 8;                                    \
    const int tok0 = blockIdx.y * (TILES) * MMQ_M;                             \
    const int nb_total = k / QK_G128;                                          \
    const int k_tiles = (k + MMQ_K - 1) / MMQ_K;                               \
    const int row_tiles = (n + mrows - 1) / mrows;                             \
    const int x_rows = (TILES) * MMQ_M;                                        \
    const int x_valid = min(x_rows, max(0, n_tokens - tok0));                  \
    const uint8_t* xbytes = (const uint8_t*)xv;                                \
                                                                               \
    const int ar = mma_a_row(lane);                                            \
    const int bc = mma_b_col(lane);                                            \
    const int cq = lane % 4;                                                   \
    const int kq = cq * 4;                                                     \
    const int cr = mma_c_row(lane);                                            \
    const int cc = mma_c_col(lane);                                            \
    const int wbase = warp * (NBLK) * 8;                                       \
                                                                               \
    /* The striped partition, which `mmqsr_*` measured worth 30% at four        \
       tokens. Same arithmetic here. */                                        \
    const int total = row_tiles * k_tiles;                                     \
    const int iters = (total + (int)gridDim.x - 1) / (int)gridDim.x;           \
    int flat = iters * (int)blockIdx.x;                                        \
    const int flat_end = min(total, flat + iters);                             \
                                                                               \
    const block_q4_g128* bsrc[NBLK];                                           \
    bool brow_ok[NBLK];                                                        \
    mma_c_f32 acc[NBLK][TILES];                                                \
                                                                               \
    if (x_valid < x_rows) {                                                    \
        const int per = (x_rows - x_valid) * (MMQ_XL_ROW / 4);                 \
        for (int i = tid; i < (STAGES) * per; i += nthreads) {                 \
            const int s = i / per;                                             \
            const int j = i % per;                                             \
            const int r = x_valid + j / (MMQ_XL_ROW / 4);                      \
            const int e = (j % (MMQ_XL_ROW / 4)) * 4;                          \
            *(uint32_t*)(void*)(xf + (s * x_rows + r) * MMQ_XL_STRIDE + e) = 0;\
        }                                                                      \
    }                                                                          \
                                                                               \
    while (flat < flat_end) {                                                  \
        const int nt = flat / k_tiles;                                         \
        const int kt_lo = flat % k_tiles;                                      \
        const int kt_hi = min(k_tiles, kt_lo + (flat_end - flat));             \
        const int row0 = nt * mrows;                                           \
                                                                               \
        _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {                   \
            const int r = row0 + wbase + j * 8 + bc;                           \
            brow_ok[j] = r < n;                                                \
            bsrc[j] = wq + (brow_ok[j] ? (size_t)r * nb_total : 0);            \
            _Pragma("unroll") for (int u = 0; u < (TILES); ++u) {              \
                _Pragma("unroll") for (int c = 0; c < 4; ++c)                  \
                    acc[j][u].x[c] = 0.0f;                                     \
            }                                                                  \
        }                                                                      \
                                                                               \
        _Pragma("unroll") for (int s = 0; s < (STAGES) - 1; ++s) {             \
            MMQ_XL_FETCH(s, kt_lo + s, kt_hi);                                 \
        }                                                                      \
                                                                               \
        for (int kt = kt_lo; kt < kt_hi; ++kt) {                               \
            const int pos = (kt - kt_lo) % (STAGES);                           \
            MMQ_CP_ASYNC_WAIT((STAGES) - 2);                                   \
            __syncthreads();                                                   \
            MMQ_XL_FETCH((pos + (STAGES) - 1) % (STAGES),                      \
                         kt + (STAGES) - 1, kt_hi);                            \
            const int8_t* xbuf = xf + pos * x_rows * MMQ_XL_STRIDE;            \
                                                                               \
            _Pragma("unroll") for (int blk = 0; blk < 2; ++blk) {              \
                const int kb = kt * 2 + blk;                                   \
                const bool kb_ok = kb < nb_total;                              \
                /* One scale and one zero per row per 128 weights, broadcast   \
                   into the halves the dequantization folds them into. */      \
                __half2 s2[NBLK], m2[NBLK];                                    \
                _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {           \
                    const int sr = row0 + wbase + j * 8 + bc;                  \
                    __half2 ds = __floats2half2_rn(0.0f, 0.0f);                \
                    if (kb_ok && brow_ok[j]) ds = bsrc[j][kb].ds;              \
                    s2[j] = __float2half2_rn(__low2float(ds));                 \
                    m2[j] = __float2half2_rn(-__high2float(ds));               \
                    (void)sr;                                                  \
                }                                                              \
                _Pragma("unroll") for (int run = 0; run < 2; ++run) {          \
                    uint32_t v0[NBLK], v1[NBLK];                               \
                    _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {       \
                        v0[j] = 0;                                             \
                        v1[j] = 0;                                             \
                        if (brow_ok[j] && kb_ok) {                             \
                            const uint8_t* pq =                                \
                                bsrc[j][kb].qs + run * 32 + kq;                \
                            v0[j] = *(const uint32_t*)(const void*)pq;         \
                            v1[j] = *(const uint32_t*)(const void*)(pq + 16);  \
                        }                                                      \
                    }                                                          \
                    _Pragma("unroll") for (int h = 0; h < 2; ++h) {            \
                        const int g = blk * 4 + run + h * 2;                   \
                        /* Two MMAs a group: k is 16 here, not 32. */          \
                        _Pragma("unroll") for (int mm = 0; mm < 2; ++mm) {     \
                            mma_a_f16 a[TILES];                                \
                            _Pragma("unroll") for (int u = 0; u < (TILES);     \
                                                   ++u) {                      \
                                /* One instruction where the gather takes    \
                                   two, and the standard fragment order that   \
                                   `f32_to_f16_kperm` has already arranged the \
                                   activations for. */                         \
                                mma_a_s8 t;                                    \
                                ldmatrix_a_s8(t, xbuf                          \
                                    + (u * MMQ_M) * MMQ_XL_STRIDE              \
                                    + g * 64 + mm * 32, MMQ_XL_STRIDE);        \
                                a[u].x[0] = (unsigned)t.x[0];                  \
                                a[u].x[1] = (unsigned)t.x[1];                  \
                                a[u].x[2] = (unsigned)t.x[2];                  \
                                a[u].x[3] = (unsigned)t.x[3];                  \
                            }                                                  \
                            _Pragma("unroll") for (int j = 0; j < (NBLK);      \
                                                   ++j) {                      \
                                mma_b_f16 b;                                   \
                                mmq_deq4_f16(mm ? v1[j] : v0[j], h, s2[j],     \
                                             m2[j], b.x);                      \
                                _Pragma("unroll") for (int u = 0;              \
                                                       u < (TILES); ++u) {     \
                                    mma_f16(acc[j][u], a[u], b);               \
                                }                                              \
                            }                                                  \
                        }                                                      \
                    }                                                          \
                }                                                              \
            }                                                                  \
        }                                                                      \
                                                                               \
        MMQ_CP_ASYNC_WAIT(0);                                                  \
        __syncthreads();                                                       \
                                                                               \
        /* No epilogue: the accumulator is already the answer. */              \
        const bool whole = (kt_lo == 0) && (kt_hi == k_tiles);                 \
        _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {                   \
            const int orow = row0 + wbase + j * 8 + cc;                        \
            _Pragma("unroll") for (int u = 0; u < (TILES); ++u) {              \
                const int ot0 = tok0 + u * MMQ_M + cr;                         \
                const int ot1 = ot0 + 8;                                       \
                if (ot0 < n_tokens) {                                          \
                    if (orow < n)                                              \
                        MMQ_PUT2(whole, (size_t)ot0 * n + orow,                \
                                 acc[j][u].x[0]);                              \
                    if (orow + 1 < n)                                          \
                        MMQ_PUT2(whole, (size_t)ot0 * n + orow + 1,            \
                                 acc[j][u].x[1]);                              \
                }                                                              \
                if (ot1 < n_tokens) {                                          \
                    if (orow < n)                                              \
                        MMQ_PUT2(whole, (size_t)ot1 * n + orow,                \
                                 acc[j][u].x[2]);                              \
                    if (orow + 1 < n)                                          \
                        MMQ_PUT2(whole, (size_t)ot1 * n + orow + 1,            \
                                 acc[j][u].x[3]);                              \
                }                                                              \
            }                                                                  \
        }                                                                      \
        flat += kt_hi - kt_lo;                                                 \
    }

#define MMQ_F16LM_SET(SUFFIX, WARPS, NBLK, TILES, STAGES)                      \
    extern "C" __global__ __launch_bounds__((WARPS) * WARP_SIZE) void          \
    mmqm##SUFFIX##_q4_g128(float* __restrict__ out,                            \
                           const void* __restrict__ wv,                        \
                           const void* __restrict__ xv, int k, int n,          \
                           int n_tokens) {                                     \
        const block_q4_g128* wq = (const block_q4_g128*)wv;                    \
        MMQ_F16LM_BODY(WARPS, NBLK, TILES, STAGES)                             \
    }

MMQ_F16LM_SET(1w8s2, 8, 1, 1, 2)
MMQ_F16LM_SET(1w8s2_2, 8, 1, 2, 2)
MMQ_F16LM_SET(1w4s2, 4, 1, 1, 2)
MMQ_F16LM_SET(1w4s2_2, 4, 1, 2, 2)

#define MMQ_F16NOMMA_BODY(WARPS, NBLK, TILES, STAGES)                               \
    extern __shared__ __align__(16) int8_t xf[];                               \
                                                                               \
    const int tid = threadIdx.x;                                               \
    const int lane = tid % WARP_SIZE;                                          \
    const int warp = tid / WARP_SIZE;                                          \
    const int nthreads = (WARPS) * WARP_SIZE;                                  \
    const int mrows = (WARPS) * (NBLK) * 8;                                    \
    const int tok0 = blockIdx.y * (TILES) * MMQ_M;                             \
    const int nb_total = k / QK_G128;                                          \
    const int k_tiles = (k + MMQ_K - 1) / MMQ_K;                               \
    const int row_tiles = (n + mrows - 1) / mrows;                             \
    const int x_rows = (TILES) * MMQ_M;                                        \
    const int x_valid = min(x_rows, max(0, n_tokens - tok0));                  \
    const uint8_t* xbytes = (const uint8_t*)xv;                                \
                                                                               \
    const int ar = mma_a_row(lane);                                            \
    const int bc = mma_b_col(lane);                                            \
    const int cq = lane % 4;                                                   \
    const int kq = cq * 4;                                                     \
    const int cr = mma_c_row(lane);                                            \
    const int cc = mma_c_col(lane);                                            \
    const int wbase = warp * (NBLK) * 8;                                       \
                                                                               \
    /* The striped partition, which `mmqsr_*` measured worth 30% at four        \
       tokens. Same arithmetic here. */                                        \
    const int total = row_tiles * k_tiles;                                     \
    const int iters = (total + (int)gridDim.x - 1) / (int)gridDim.x;           \
    int flat = iters * (int)blockIdx.x;                                        \
    const int flat_end = min(total, flat + iters);                             \
                                                                               \
    const block_q4_g128* bsrc[NBLK];                                           \
    bool brow_ok[NBLK];                                                        \
    mma_c_f32 acc[NBLK][TILES];                                                \
                                                                               \
    if (x_valid < x_rows) {                                                    \
        const int per = (x_rows - x_valid) * (MMQ_XF_ROW / 4);                 \
        for (int i = tid; i < (STAGES) * per; i += nthreads) {                 \
            const int s = i / per;                                             \
            const int j = i % per;                                             \
            const int r = x_valid + j / (MMQ_XF_ROW / 4);                      \
            const int e = (j % (MMQ_XF_ROW / 4)) * 4;                          \
            *(uint32_t*)(void*)(xf + (s * x_rows + r) * MMQ_XF_STRIDE + e) = 0;\
        }                                                                      \
    }                                                                          \
                                                                               \
    while (flat < flat_end) {                                                  \
        const int nt = flat / k_tiles;                                         \
        const int kt_lo = flat % k_tiles;                                      \
        const int kt_hi = min(k_tiles, kt_lo + (flat_end - flat));             \
        const int row0 = nt * mrows;                                           \
                                                                               \
        _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {                   \
            const int r = row0 + wbase + j * 8 + bc;                           \
            brow_ok[j] = r < n;                                                \
            bsrc[j] = wq + (brow_ok[j] ? (size_t)r * nb_total : 0);            \
            _Pragma("unroll") for (int u = 0; u < (TILES); ++u) {              \
                _Pragma("unroll") for (int c = 0; c < 4; ++c)                  \
                    acc[j][u].x[c] = 0.0f;                                     \
            }                                                                  \
        }                                                                      \
                                                                               \
        _Pragma("unroll") for (int s = 0; s < (STAGES) - 1; ++s) {             \
            MMQ_XF_FETCH(s, kt_lo + s, kt_hi);                                 \
        }                                                                      \
                                                                               \
        for (int kt = kt_lo; kt < kt_hi; ++kt) {                               \
            const int pos = (kt - kt_lo) % (STAGES);                           \
            MMQ_CP_ASYNC_WAIT((STAGES) - 2);                                   \
            __syncthreads();                                                   \
            MMQ_XF_FETCH((pos + (STAGES) - 1) % (STAGES),                      \
                         kt + (STAGES) - 1, kt_hi);                            \
            const int8_t* xbuf = xf + pos * x_rows * MMQ_XF_STRIDE;            \
                                                                               \
            _Pragma("unroll") for (int blk = 0; blk < 2; ++blk) {              \
                const int kb = kt * 2 + blk;                                   \
                const bool kb_ok = kb < nb_total;                              \
                /* One scale and one zero per row per 128 weights, broadcast   \
                   into the halves the dequantization folds them into. */      \
                __half2 s2[NBLK], m2[NBLK];                                    \
                _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {           \
                    const int sr = row0 + wbase + j * 8 + bc;                  \
                    __half2 ds = __floats2half2_rn(0.0f, 0.0f);                \
                    if (kb_ok && brow_ok[j]) ds = bsrc[j][kb].ds;              \
                    s2[j] = __float2half2_rn(__low2float(ds));                 \
                    m2[j] = __float2half2_rn(-__high2float(ds));               \
                    (void)sr;                                                  \
                }                                                              \
                _Pragma("unroll") for (int run = 0; run < 2; ++run) {          \
                    uint32_t v0[NBLK], v1[NBLK];                               \
                    _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {       \
                        v0[j] = 0;                                             \
                        v1[j] = 0;                                             \
                        if (brow_ok[j] && kb_ok) {                             \
                            const uint8_t* pq =                                \
                                bsrc[j][kb].qs + run * 32 + kq;                \
                            v0[j] = *(const uint32_t*)(const void*)pq;         \
                            v1[j] = *(const uint32_t*)(const void*)(pq + 16);  \
                        }                                                      \
                    }                                                          \
                    _Pragma("unroll") for (int h = 0; h < 2; ++h) {            \
                        const int g = blk * 4 + run + h * 2;                   \
                        /* Two MMAs a group: k is 16 here, not 32. */          \
                        _Pragma("unroll") for (int mm = 0; mm < 2; ++mm) {     \
                            mma_a_f16 a[TILES];                                \
                            _Pragma("unroll") for (int u = 0; u < (TILES);     \
                                                   ++u) {                      \
                                const int8_t* ap =                             \
                                    xbuf + (u * MMQ_M + ar) * MMQ_XF_STRIDE    \
                                    + g * 64 + mm * 32 + cq * 8;               \
                                const int8_t* aq = ap + 8 * MMQ_XF_STRIDE;     \
                                const uint2 lo =                               \
                                    *(const uint2*)(const void*)ap;            \
                                const uint2 hi =                               \
                                    *(const uint2*)(const void*)aq;            \
                                a[u].x[0] = lo.x;                              \
                                a[u].x[1] = hi.x;                              \
                                a[u].x[2] = lo.y;                              \
                                a[u].x[3] = hi.y;                              \
                            }                                                  \
                            _Pragma("unroll") for (int j = 0; j < (NBLK);      \
                                                   ++j) {                      \
                                mma_b_f16 b;                                   \
                                mmq_deq4_f16(mm ? v1[j] : v0[j], h, s2[j],     \
                                             m2[j], b.x);                      \
                                _Pragma("unroll") for (int u = 0;              \
                                                       u < (TILES); ++u) {     \
                                    mma_f16(acc[j][u], a[u], b);               \
                                }                                              \
                            }                                                  \
                        }                                                      \
                    }                                                          \
                }                                                              \
            }                                                                  \
        }                                                                      \
                                                                               \
        MMQ_CP_ASYNC_WAIT(0);                                                  \
        __syncthreads();                                                       \
                                                                               \
        /* No epilogue: the accumulator is already the answer. */              \
        const bool whole = (kt_lo == 0) && (kt_hi == k_tiles);                 \
        _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {                   \
            const int orow = row0 + wbase + j * 8 + cc;                        \
            _Pragma("unroll") for (int u = 0; u < (TILES); ++u) {              \
                const int ot0 = tok0 + u * MMQ_M + cr;                         \
                const int ot1 = ot0 + 8;                                       \
                if (ot0 < n_tokens) {                                          \
                    if (orow < n)                                              \
                        MMQ_PUT2(whole, (size_t)ot0 * n + orow,                \
                                 acc[j][u].x[0]);                              \
                    if (orow + 1 < n)                                          \
                        MMQ_PUT2(whole, (size_t)ot0 * n + orow + 1,            \
                                 acc[j][u].x[1]);                              \
                }                                                              \
                if (ot1 < n_tokens) {                                          \
                    if (orow < n)                                              \
                        MMQ_PUT2(whole, (size_t)ot1 * n + orow,                \
                                 acc[j][u].x[2]);                              \
                    if (orow + 1 < n)                                          \
                        MMQ_PUT2(whole, (size_t)ot1 * n + orow + 1,            \
                                 acc[j][u].x[3]);                              \
                }                                                              \
            }                                                                  \
        }                                                                      \
        flat += kt_hi - kt_lo;                                                 \
    }


#define MMQ_F16NOACT_BODY(WARPS, NBLK, TILES, STAGES)                               \
    extern __shared__ __align__(16) int8_t xf[];                               \
                                                                               \
    const int tid = threadIdx.x;                                               \
    const int lane = tid % WARP_SIZE;                                          \
    const int warp = tid / WARP_SIZE;                                          \
    const int nthreads = (WARPS) * WARP_SIZE;                                  \
    const int mrows = (WARPS) * (NBLK) * 8;                                    \
    const int tok0 = blockIdx.y * (TILES) * MMQ_M;                             \
    const int nb_total = k / QK_G128;                                          \
    const int k_tiles = (k + MMQ_K - 1) / MMQ_K;                               \
    const int row_tiles = (n + mrows - 1) / mrows;                             \
    const int x_rows = (TILES) * MMQ_M;                                        \
    const int x_valid = min(x_rows, max(0, n_tokens - tok0));                  \
    const uint8_t* xbytes = (const uint8_t*)xv;                                \
                                                                               \
    const int ar = mma_a_row(lane);                                            \
    const int bc = mma_b_col(lane);                                            \
    const int cq = lane % 4;                                                   \
    const int kq = cq * 4;                                                     \
    const int cr = mma_c_row(lane);                                            \
    const int cc = mma_c_col(lane);                                            \
    const int wbase = warp * (NBLK) * 8;                                       \
                                                                               \
    /* The striped partition, which `mmqsr_*` measured worth 30% at four        \
       tokens. Same arithmetic here. */                                        \
    const int total = row_tiles * k_tiles;                                     \
    const int iters = (total + (int)gridDim.x - 1) / (int)gridDim.x;           \
    int flat = iters * (int)blockIdx.x;                                        \
    const int flat_end = min(total, flat + iters);                             \
                                                                               \
    const block_q4_g128* bsrc[NBLK];                                           \
    bool brow_ok[NBLK];                                                        \
    mma_c_f32 acc[NBLK][TILES];                                                \
                                                                               \
    if (x_valid < x_rows) {                                                    \
        const int per = (x_rows - x_valid) * (MMQ_XF_ROW / 4);                 \
        for (int i = tid; i < (STAGES) * per; i += nthreads) {                 \
            const int s = i / per;                                             \
            const int j = i % per;                                             \
            const int r = x_valid + j / (MMQ_XF_ROW / 4);                      \
            const int e = (j % (MMQ_XF_ROW / 4)) * 4;                          \
            *(uint32_t*)(void*)(xf + (s * x_rows + r) * MMQ_XF_STRIDE + e) = 0;\
        }                                                                      \
    }                                                                          \
                                                                               \
    while (flat < flat_end) {                                                  \
        const int nt = flat / k_tiles;                                         \
        const int kt_lo = flat % k_tiles;                                      \
        const int kt_hi = min(k_tiles, kt_lo + (flat_end - flat));             \
        const int row0 = nt * mrows;                                           \
                                                                               \
        _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {                   \
            const int r = row0 + wbase + j * 8 + bc;                           \
            brow_ok[j] = r < n;                                                \
            bsrc[j] = wq + (brow_ok[j] ? (size_t)r * nb_total : 0);            \
            _Pragma("unroll") for (int u = 0; u < (TILES); ++u) {              \
                _Pragma("unroll") for (int c = 0; c < 4; ++c)                  \
                    acc[j][u].x[c] = 0.0f;                                     \
            }                                                                  \
        }                                                                      \
                                                                               \
        _Pragma("unroll") for (int s = 0; s < (STAGES) - 1; ++s) {             \
            MMQ_XF_FETCH(s, kt_lo + s, kt_hi);                                 \
        }                                                                      \
                                                                               \
        for (int kt = kt_lo; kt < kt_hi; ++kt) {                               \
            const int pos = (kt - kt_lo) % (STAGES);                           \
            MMQ_CP_ASYNC_WAIT((STAGES) - 2);                                   \
            __syncthreads();                                                   \
            /* No refill: the tile keeps whatever the prologue left. Wrong  \
               answers, and every barrier and every read still there. */       \
            MMQ_CP_ASYNC_FENCE();                                              \
            const int8_t* xbuf = xf + pos * x_rows * MMQ_XF_STRIDE;            \
                                                                               \
            _Pragma("unroll") for (int blk = 0; blk < 2; ++blk) {              \
                const int kb = kt * 2 + blk;                                   \
                const bool kb_ok = kb < nb_total;                              \
                /* One scale and one zero per row per 128 weights, broadcast   \
                   into the halves the dequantization folds them into. */      \
                __half2 s2[NBLK], m2[NBLK];                                    \
                _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {           \
                    const int sr = row0 + wbase + j * 8 + bc;                  \
                    __half2 ds = __floats2half2_rn(0.0f, 0.0f);                \
                    if (kb_ok && brow_ok[j]) ds = bsrc[j][kb].ds;              \
                    s2[j] = __float2half2_rn(__low2float(ds));                 \
                    m2[j] = __float2half2_rn(-__high2float(ds));               \
                    (void)sr;                                                  \
                }                                                              \
                _Pragma("unroll") for (int run = 0; run < 2; ++run) {          \
                    uint32_t v0[NBLK], v1[NBLK];                               \
                    _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {       \
                        v0[j] = 0;                                             \
                        v1[j] = 0;                                             \
                        if (brow_ok[j] && kb_ok) {                             \
                            const uint8_t* pq =                                \
                                bsrc[j][kb].qs + run * 32 + kq;                \
                            v0[j] = *(const uint32_t*)(const void*)pq;         \
                            v1[j] = *(const uint32_t*)(const void*)(pq + 16);  \
                        }                                                      \
                    }                                                          \
                    _Pragma("unroll") for (int h = 0; h < 2; ++h) {            \
                        const int g = blk * 4 + run + h * 2;                   \
                        /* Two MMAs a group: k is 16 here, not 32. */          \
                        _Pragma("unroll") for (int mm = 0; mm < 2; ++mm) {     \
                            mma_a_f16 a[TILES];                                \
                            _Pragma("unroll") for (int u = 0; u < (TILES);     \
                                                   ++u) {                      \
                                const int8_t* ap =                             \
                                    xbuf + (u * MMQ_M + ar) * MMQ_XF_STRIDE    \
                                    + g * 64 + mm * 32 + cq * 8;               \
                                const int8_t* aq = ap + 8 * MMQ_XF_STRIDE;     \
                                const uint2 lo =                               \
                                    *(const uint2*)(const void*)ap;            \
                                const uint2 hi =                               \
                                    *(const uint2*)(const void*)aq;            \
                                a[u].x[0] = lo.x;                              \
                                a[u].x[1] = hi.x;                              \
                                a[u].x[2] = lo.y;                              \
                                a[u].x[3] = hi.y;                              \
                            }                                                  \
                            _Pragma("unroll") for (int j = 0; j < (NBLK);      \
                                                   ++j) {                      \
                                mma_b_f16 b;                                   \
                                mmq_deq4_f16(mm ? v1[j] : v0[j], h, s2[j],     \
                                             m2[j], b.x);                      \
                                _Pragma("unroll") for (int u = 0;              \
                                                       u < (TILES); ++u) {     \
                                    mma_f16(acc[j][u], a[u], b);               \
                                }                                              \
                            }                                                  \
                        }                                                      \
                    }                                                          \
                }                                                              \
            }                                                                  \
        }                                                                      \
                                                                               \
        MMQ_CP_ASYNC_WAIT(0);                                                  \
        __syncthreads();                                                       \
                                                                               \
        /* No epilogue: the accumulator is already the answer. */              \
        const bool whole = (kt_lo == 0) && (kt_hi == k_tiles);                 \
        _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {                   \
            const int orow = row0 + wbase + j * 8 + cc;                        \
            _Pragma("unroll") for (int u = 0; u < (TILES); ++u) {              \
                const int ot0 = tok0 + u * MMQ_M + cr;                         \
                const int ot1 = ot0 + 8;                                       \
                if (ot0 < n_tokens) {                                          \
                    if (orow < n)                                              \
                        MMQ_PUT2(whole, (size_t)ot0 * n + orow,                \
                                 acc[j][u].x[0]);                              \
                    if (orow + 1 < n)                                          \
                        MMQ_PUT2(whole, (size_t)ot0 * n + orow + 1,            \
                                 acc[j][u].x[1]);                              \
                }                                                              \
                if (ot1 < n_tokens) {                                          \
                    if (orow < n)                                              \
                        MMQ_PUT2(whole, (size_t)ot1 * n + orow,                \
                                 acc[j][u].x[2]);                              \
                    if (orow + 1 < n)                                          \
                        MMQ_PUT2(whole, (size_t)ot1 * n + orow + 1,            \
                                 acc[j][u].x[3]);                              \
                }                                                              \
            }                                                                  \
        }                                                                      \
        flat += kt_hi - kt_lo;                                                 \
    }

// The activation stream, priced the same way — and this one does not work.
// `mmqnx_*` fills the ring once in the prologue and never refills it: every
// barrier, every A-fragment read, every weight load and every MMA survives,
// and the `cp.async` traffic that carries the tokens does not.
//
//                    ffn_gate  ffn_down
//   mmqf1w8s2  @ 8t     329       341
//   mmqnx1w8s2 @ 8t     249       260
//   mmqf1w8s2  @32t     215       227
//   mmqnx1w8s2 @32t     189       192
//
// Removing the traffic made it 12-24% *slower*, so the probe changed something
// other than what it meant to and its number is not usable. Most likely the
// `cp.async` stream is what keeps the memory pipeline busy across the barrier,
// and without it the weight loads stand exposed — but that is a story, not a
// measurement.
//
// Third subtractive probe on this kernel to come back faster-when-fatter: the
// A-fragment stub (`mmqna_*`) did it, the `ldmatrix` cut did it, this does it.
// Deleting work from this kernel changes its schedule, and the schedule is what
// is being measured. `mmqnm_*` above is the exception and the reason to trust
// it: it measured *identical*, which is the one outcome a codegen artefact
// cannot fake.
#define MMQ_F16NOACT_SET(SUFFIX, WARPS, NBLK, TILES, STAGES)                   \
    extern "C" __global__ __launch_bounds__((WARPS) * WARP_SIZE) void          \
    mmqnx##SUFFIX##_q4_g128(float* __restrict__ out,                           \
                            const void* __restrict__ wv,                       \
                            const void* __restrict__ xv, int k, int n,         \
                            int n_tokens) {                                    \
        const block_q4_g128* wq = (const block_q4_g128*)wv;                    \
        MMQ_F16NOACT_BODY(WARPS, NBLK, TILES, STAGES)                          \
    }

MMQ_F16NOACT_SET(1w8s2, 8, 1, 1, 2)
MMQ_F16NOACT_SET(1w8s2_2, 8, 1, 2, 2)

// Half the activation traffic, every instruction intact.
//
// The arithmetic says this shape moves 16 KB of activations against 8 KB of
// weights per block per k-tile at 32 tokens, and 4 KB against 8 KB at eight —
// and it reaches 97% of the weight-read ceiling at eight tokens and 63% at
// thirty-two. If that ratio is the reason, halving the copies should show it.
//
// `MMQ_XH_FETCH` stages half the token rows and nothing else changes: same
// barriers, same ring, same A-fragment reads, same MMAs, same weight loads.
// That matters because three subtractive probes on this kernel have already
// come back faster-when-fatter by perturbing the schedule — this one perturbs
// only the byte count.
//
//                    ffn_gate  ffn_down  attn_q
//   mmqf1w8s2  @ 8t     330       341      260
//   mmqnh1w8s2 @ 8t     340       348      271
//   mmqf1w8s2  @32t     214       226        -
//   mmqnh1w8s2 @32t     229       240        -
//
// 6-7% at 32 tokens, 3-4% at eight. So the ratio is real and it is *small* —
// nothing like the 1.6x that closing to the weight-read ceiling would be. Which
// rules out the change it was scouting: staging activations at eight bits and
// widening them to f16 in registers would halve exactly these bytes, and 7% is
// not worth a second activation format and a dequantization on the A side.
//
// So at 32 tokens this kernel is not bound by the tensor cores (`mmqnm_*`, 0%),
// not by weight load width (`mmqfp_*`, 0%), not by the A-fragment order
// (`mmqm_*`, negative), and only 7% by the activation stream. What is left is
// the shared-memory footprint itself: two token tiles at two stages is 34.8 KB,
// which is two resident blocks per SM against five at one tile. That is the
// difference the 63% has to live in, and neither fewer stages nor more warps
// nor a narrower stride moves it.
#define MMQ_XH_FETCH(BUF, TILE, LIMIT)                                         \
    do {                                                                       \
        const int _tl = (TILE);                                                \
        const bool _live = _tl < (LIMIT);                                      \
        const int _rows = (x_valid + 1) / 2;                                   \
        for (int _i = tid; _i < _rows * (MMQ_XF_ROW / 16); _i += nthreads) {   \
            const int _r = _i / (MMQ_XF_ROW / 16);                             \
            const int _c = _i % (MMQ_XF_ROW / 16);                             \
            const bool _hit = _live && (_tl * MMQ_K + _c * 8 + 8 <= k);        \
            const size_t _off =                                                \
                ((size_t)(tok0 + _r) * k + (size_t)_tl * MMQ_K) * 2            \
                + (size_t)_c * 16;                                             \
            mmq_cp_async16(                                                    \
                xf + ((BUF) * x_rows + _r) * MMQ_XF_STRIDE + _c * 16,          \
                xbytes + (_hit ? _off : 0), _hit);                             \
        }                                                                      \
        MMQ_CP_ASYNC_FENCE();                                                  \
    } while (0)

#define MMQ_F16HALFACT_BODY(WARPS, NBLK, TILES, STAGES)                               \
    extern __shared__ __align__(16) int8_t xf[];                               \
                                                                               \
    const int tid = threadIdx.x;                                               \
    const int lane = tid % WARP_SIZE;                                          \
    const int warp = tid / WARP_SIZE;                                          \
    const int nthreads = (WARPS) * WARP_SIZE;                                  \
    const int mrows = (WARPS) * (NBLK) * 8;                                    \
    const int tok0 = blockIdx.y * (TILES) * MMQ_M;                             \
    const int nb_total = k / QK_G128;                                          \
    const int k_tiles = (k + MMQ_K - 1) / MMQ_K;                               \
    const int row_tiles = (n + mrows - 1) / mrows;                             \
    const int x_rows = (TILES) * MMQ_M;                                        \
    const int x_valid = min(x_rows, max(0, n_tokens - tok0));                  \
    const uint8_t* xbytes = (const uint8_t*)xv;                                \
                                                                               \
    const int ar = mma_a_row(lane);                                            \
    const int bc = mma_b_col(lane);                                            \
    const int cq = lane % 4;                                                   \
    const int kq = cq * 4;                                                     \
    const int cr = mma_c_row(lane);                                            \
    const int cc = mma_c_col(lane);                                            \
    const int wbase = warp * (NBLK) * 8;                                       \
                                                                               \
    /* The striped partition, which `mmqsr_*` measured worth 30% at four        \
       tokens. Same arithmetic here. */                                        \
    const int total = row_tiles * k_tiles;                                     \
    const int iters = (total + (int)gridDim.x - 1) / (int)gridDim.x;           \
    int flat = iters * (int)blockIdx.x;                                        \
    const int flat_end = min(total, flat + iters);                             \
                                                                               \
    const block_q4_g128* bsrc[NBLK];                                           \
    bool brow_ok[NBLK];                                                        \
    mma_c_f32 acc[NBLK][TILES];                                                \
                                                                               \
    if (x_valid < x_rows) {                                                    \
        const int per = (x_rows - x_valid) * (MMQ_XF_ROW / 4);                 \
        for (int i = tid; i < (STAGES) * per; i += nthreads) {                 \
            const int s = i / per;                                             \
            const int j = i % per;                                             \
            const int r = x_valid + j / (MMQ_XF_ROW / 4);                      \
            const int e = (j % (MMQ_XF_ROW / 4)) * 4;                          \
            *(uint32_t*)(void*)(xf + (s * x_rows + r) * MMQ_XF_STRIDE + e) = 0;\
        }                                                                      \
    }                                                                          \
                                                                               \
    while (flat < flat_end) {                                                  \
        const int nt = flat / k_tiles;                                         \
        const int kt_lo = flat % k_tiles;                                      \
        const int kt_hi = min(k_tiles, kt_lo + (flat_end - flat));             \
        const int row0 = nt * mrows;                                           \
                                                                               \
        _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {                   \
            const int r = row0 + wbase + j * 8 + bc;                           \
            brow_ok[j] = r < n;                                                \
            bsrc[j] = wq + (brow_ok[j] ? (size_t)r * nb_total : 0);            \
            _Pragma("unroll") for (int u = 0; u < (TILES); ++u) {              \
                _Pragma("unroll") for (int c = 0; c < 4; ++c)                  \
                    acc[j][u].x[c] = 0.0f;                                     \
            }                                                                  \
        }                                                                      \
                                                                               \
        _Pragma("unroll") for (int s = 0; s < (STAGES) - 1; ++s) {             \
            MMQ_XH_FETCH(s, kt_lo + s, kt_hi);                                 \
        }                                                                      \
                                                                               \
        for (int kt = kt_lo; kt < kt_hi; ++kt) {                               \
            const int pos = (kt - kt_lo) % (STAGES);                           \
            MMQ_CP_ASYNC_WAIT((STAGES) - 2);                                   \
            __syncthreads();                                                   \
            MMQ_XH_FETCH((pos + (STAGES) - 1) % (STAGES),                      \
                         kt + (STAGES) - 1, kt_hi);                            \
            const int8_t* xbuf = xf + pos * x_rows * MMQ_XF_STRIDE;            \
                                                                               \
            _Pragma("unroll") for (int blk = 0; blk < 2; ++blk) {              \
                const int kb = kt * 2 + blk;                                   \
                const bool kb_ok = kb < nb_total;                              \
                /* One scale and one zero per row per 128 weights, broadcast   \
                   into the halves the dequantization folds them into. */      \
                __half2 s2[NBLK], m2[NBLK];                                    \
                _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {           \
                    const int sr = row0 + wbase + j * 8 + bc;                  \
                    __half2 ds = __floats2half2_rn(0.0f, 0.0f);                \
                    if (kb_ok && brow_ok[j]) ds = bsrc[j][kb].ds;              \
                    s2[j] = __float2half2_rn(__low2float(ds));                 \
                    m2[j] = __float2half2_rn(-__high2float(ds));               \
                    (void)sr;                                                  \
                }                                                              \
                _Pragma("unroll") for (int run = 0; run < 2; ++run) {          \
                    uint32_t v0[NBLK], v1[NBLK];                               \
                    _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {       \
                        v0[j] = 0;                                             \
                        v1[j] = 0;                                             \
                        if (brow_ok[j] && kb_ok) {                             \
                            const uint8_t* pq =                                \
                                bsrc[j][kb].qs + run * 32 + kq;                \
                            v0[j] = *(const uint32_t*)(const void*)pq;         \
                            v1[j] = *(const uint32_t*)(const void*)(pq + 16);  \
                        }                                                      \
                    }                                                          \
                    _Pragma("unroll") for (int h = 0; h < 2; ++h) {            \
                        const int g = blk * 4 + run + h * 2;                   \
                        /* Two MMAs a group: k is 16 here, not 32. */          \
                        _Pragma("unroll") for (int mm = 0; mm < 2; ++mm) {     \
                            mma_a_f16 a[TILES];                                \
                            _Pragma("unroll") for (int u = 0; u < (TILES);     \
                                                   ++u) {                      \
                                const int8_t* ap =                             \
                                    xbuf + (u * MMQ_M + ar) * MMQ_XF_STRIDE    \
                                    + g * 64 + mm * 32 + cq * 8;               \
                                const int8_t* aq = ap + 8 * MMQ_XF_STRIDE;     \
                                const uint2 lo =                               \
                                    *(const uint2*)(const void*)ap;            \
                                const uint2 hi =                               \
                                    *(const uint2*)(const void*)aq;            \
                                a[u].x[0] = lo.x;                              \
                                a[u].x[1] = hi.x;                              \
                                a[u].x[2] = lo.y;                              \
                                a[u].x[3] = hi.y;                              \
                            }                                                  \
                            _Pragma("unroll") for (int j = 0; j < (NBLK);      \
                                                   ++j) {                      \
                                mma_b_f16 b;                                   \
                                mmq_deq4_f16(mm ? v1[j] : v0[j], h, s2[j],     \
                                             m2[j], b.x);                      \
                                _Pragma("unroll") for (int u = 0;              \
                                                       u < (TILES); ++u) {     \
                                    mma_f16(acc[j][u], a[u], b);               \
                                }                                              \
                            }                                                  \
                        }                                                      \
                    }                                                          \
                }                                                              \
            }                                                                  \
        }                                                                      \
                                                                               \
        MMQ_CP_ASYNC_WAIT(0);                                                  \
        __syncthreads();                                                       \
                                                                               \
        /* No epilogue: the accumulator is already the answer. */              \
        const bool whole = (kt_lo == 0) && (kt_hi == k_tiles);                 \
        _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {                   \
            const int orow = row0 + wbase + j * 8 + cc;                        \
            _Pragma("unroll") for (int u = 0; u < (TILES); ++u) {              \
                const int ot0 = tok0 + u * MMQ_M + cr;                         \
                const int ot1 = ot0 + 8;                                       \
                if (ot0 < n_tokens) {                                          \
                    if (orow < n)                                              \
                        MMQ_PUT2(whole, (size_t)ot0 * n + orow,                \
                                 acc[j][u].x[0]);                              \
                    if (orow + 1 < n)                                          \
                        MMQ_PUT2(whole, (size_t)ot0 * n + orow + 1,            \
                                 acc[j][u].x[1]);                              \
                }                                                              \
                if (ot1 < n_tokens) {                                          \
                    if (orow < n)                                              \
                        MMQ_PUT2(whole, (size_t)ot1 * n + orow,                \
                                 acc[j][u].x[2]);                              \
                    if (orow + 1 < n)                                          \
                        MMQ_PUT2(whole, (size_t)ot1 * n + orow + 1,            \
                                 acc[j][u].x[3]);                              \
                }                                                              \
            }                                                                  \
        }                                                                      \
        flat += kt_hi - kt_lo;                                                 \
    }
#define MMQ_F16HALFACT_SET(SUFFIX, WARPS, NBLK, TILES, STAGES)                 \
    extern "C" __global__ __launch_bounds__((WARPS) * WARP_SIZE) void          \
    mmqnh##SUFFIX##_q4_g128(float* __restrict__ out,                           \
                            const void* __restrict__ wv,                       \
                            const void* __restrict__ xv, int k, int n,         \
                            int n_tokens) {                                    \
        const block_q4_g128* wq = (const block_q4_g128*)wv;                    \
        MMQ_F16HALFACT_BODY(WARPS, NBLK, TILES, STAGES)                        \
    }

MMQ_F16HALFACT_SET(1w8s2, 8, 1, 1, 2)
MMQ_F16HALFACT_SET(1w8s2_2, 8, 1, 2, 2)

// Half the *ring*, not just half the traffic.
//
// `mmqnh_*` above halved the activation copies and measured 7%, which read as
// a small effect — but it kept the ring the same size, and the ring is what
// caps this shape. At two token tiles and two stages it is 34.8 KB, which is
// two resident blocks per SM against five at one tile. `mmqnr_*` halves the
// allocation as well, so both token tiles read the same rows: wrong answers,
// and the footprint an eight-bit activation ring would actually have.
#define MMQ_F16HALFRING_BODY(WARPS, NBLK, TILES, STAGES)                               \
    extern __shared__ __align__(16) int8_t xf[];                               \
                                                                               \
    const int tid = threadIdx.x;                                               \
    const int lane = tid % WARP_SIZE;                                          \
    const int warp = tid / WARP_SIZE;                                          \
    const int nthreads = (WARPS) * WARP_SIZE;                                  \
    const int mrows = (WARPS) * (NBLK) * 8;                                    \
    const int tok0 = blockIdx.y * (TILES) * MMQ_M;                             \
    const int nb_total = k / QK_G128;                                          \
    const int k_tiles = (k + MMQ_K - 1) / MMQ_K;                               \
    const int row_tiles = (n + mrows - 1) / mrows;                             \
    const int x_rows = ((TILES) * MMQ_M) / 2;                                        \
    const int x_valid = min(x_rows, max(0, n_tokens - tok0));                  \
    const uint8_t* xbytes = (const uint8_t*)xv;                                \
                                                                               \
    const int ar = mma_a_row(lane);                                            \
    const int bc = mma_b_col(lane);                                            \
    const int cq = lane % 4;                                                   \
    const int kq = cq * 4;                                                     \
    const int cr = mma_c_row(lane);                                            \
    const int cc = mma_c_col(lane);                                            \
    const int wbase = warp * (NBLK) * 8;                                       \
                                                                               \
    /* The striped partition, which `mmqsr_*` measured worth 30% at four        \
       tokens. Same arithmetic here. */                                        \
    const int total = row_tiles * k_tiles;                                     \
    const int iters = (total + (int)gridDim.x - 1) / (int)gridDim.x;           \
    int flat = iters * (int)blockIdx.x;                                        \
    const int flat_end = min(total, flat + iters);                             \
                                                                               \
    const block_q4_g128* bsrc[NBLK];                                           \
    bool brow_ok[NBLK];                                                        \
    mma_c_f32 acc[NBLK][TILES];                                                \
                                                                               \
    if (x_valid < x_rows) {                                                    \
        const int per = (x_rows - x_valid) * (MMQ_XF_ROW / 4);                 \
        for (int i = tid; i < (STAGES) * per; i += nthreads) {                 \
            const int s = i / per;                                             \
            const int j = i % per;                                             \
            const int r = x_valid + j / (MMQ_XF_ROW / 4);                      \
            const int e = (j % (MMQ_XF_ROW / 4)) * 4;                          \
            *(uint32_t*)(void*)(xf + (s * x_rows + r) * MMQ_XF_STRIDE + e) = 0;\
        }                                                                      \
    }                                                                          \
                                                                               \
    while (flat < flat_end) {                                                  \
        const int nt = flat / k_tiles;                                         \
        const int kt_lo = flat % k_tiles;                                      \
        const int kt_hi = min(k_tiles, kt_lo + (flat_end - flat));             \
        const int row0 = nt * mrows;                                           \
                                                                               \
        _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {                   \
            const int r = row0 + wbase + j * 8 + bc;                           \
            brow_ok[j] = r < n;                                                \
            bsrc[j] = wq + (brow_ok[j] ? (size_t)r * nb_total : 0);            \
            _Pragma("unroll") for (int u = 0; u < (TILES); ++u) {              \
                _Pragma("unroll") for (int c = 0; c < 4; ++c)                  \
                    acc[j][u].x[c] = 0.0f;                                     \
            }                                                                  \
        }                                                                      \
                                                                               \
        _Pragma("unroll") for (int s = 0; s < (STAGES) - 1; ++s) {             \
            MMQ_XF_FETCH(s, kt_lo + s, kt_hi);                                 \
        }                                                                      \
                                                                               \
        for (int kt = kt_lo; kt < kt_hi; ++kt) {                               \
            const int pos = (kt - kt_lo) % (STAGES);                           \
            MMQ_CP_ASYNC_WAIT((STAGES) - 2);                                   \
            __syncthreads();                                                   \
            MMQ_XF_FETCH((pos + (STAGES) - 1) % (STAGES),                      \
                         kt + (STAGES) - 1, kt_hi);                            \
            const int8_t* xbuf = xf + pos * x_rows * MMQ_XF_STRIDE;            \
                                                                               \
            _Pragma("unroll") for (int blk = 0; blk < 2; ++blk) {              \
                const int kb = kt * 2 + blk;                                   \
                const bool kb_ok = kb < nb_total;                              \
                /* One scale and one zero per row per 128 weights, broadcast   \
                   into the halves the dequantization folds them into. */      \
                __half2 s2[NBLK], m2[NBLK];                                    \
                _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {           \
                    const int sr = row0 + wbase + j * 8 + bc;                  \
                    __half2 ds = __floats2half2_rn(0.0f, 0.0f);                \
                    if (kb_ok && brow_ok[j]) ds = bsrc[j][kb].ds;              \
                    s2[j] = __float2half2_rn(__low2float(ds));                 \
                    m2[j] = __float2half2_rn(-__high2float(ds));               \
                    (void)sr;                                                  \
                }                                                              \
                _Pragma("unroll") for (int run = 0; run < 2; ++run) {          \
                    uint32_t v0[NBLK], v1[NBLK];                               \
                    _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {       \
                        v0[j] = 0;                                             \
                        v1[j] = 0;                                             \
                        if (brow_ok[j] && kb_ok) {                             \
                            const uint8_t* pq =                                \
                                bsrc[j][kb].qs + run * 32 + kq;                \
                            v0[j] = *(const uint32_t*)(const void*)pq;         \
                            v1[j] = *(const uint32_t*)(const void*)(pq + 16);  \
                        }                                                      \
                    }                                                          \
                    _Pragma("unroll") for (int h = 0; h < 2; ++h) {            \
                        const int g = blk * 4 + run + h * 2;                   \
                        /* Two MMAs a group: k is 16 here, not 32. */          \
                        _Pragma("unroll") for (int mm = 0; mm < 2; ++mm) {     \
                            mma_a_f16 a[TILES];                                \
                            _Pragma("unroll") for (int u = 0; u < (TILES);     \
                                                   ++u) {                      \
                                const int8_t* ap =                             \
                                    xbuf + (((u * MMQ_M + ar) % x_rows)) * MMQ_XF_STRIDE    \
                                    + g * 64 + mm * 32 + cq * 8;               \
                                const int8_t* aq = ap; /* half ring */         \
                                const uint2 lo =                               \
                                    *(const uint2*)(const void*)ap;            \
                                const uint2 hi =                               \
                                    *(const uint2*)(const void*)aq;            \
                                a[u].x[0] = lo.x;                              \
                                a[u].x[1] = hi.x;                              \
                                a[u].x[2] = lo.y;                              \
                                a[u].x[3] = hi.y;                              \
                            }                                                  \
                            _Pragma("unroll") for (int j = 0; j < (NBLK);      \
                                                   ++j) {                      \
                                mma_b_f16 b;                                   \
                                mmq_deq4_f16(mm ? v1[j] : v0[j], h, s2[j],     \
                                             m2[j], b.x);                      \
                                _Pragma("unroll") for (int u = 0;              \
                                                       u < (TILES); ++u) {     \
                                    mma_f16(acc[j][u], a[u], b);               \
                                }                                              \
                            }                                                  \
                        }                                                      \
                    }                                                          \
                }                                                              \
            }                                                                  \
        }                                                                      \
                                                                               \
        MMQ_CP_ASYNC_WAIT(0);                                                  \
        __syncthreads();                                                       \
                                                                               \
        /* No epilogue: the accumulator is already the answer. */              \
        const bool whole = (kt_lo == 0) && (kt_hi == k_tiles);                 \
        _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {                   \
            const int orow = row0 + wbase + j * 8 + cc;                        \
            _Pragma("unroll") for (int u = 0; u < (TILES); ++u) {              \
                const int ot0 = tok0 + u * MMQ_M + cr;                         \
                const int ot1 = ot0 + 8;                                       \
                if (ot0 < n_tokens) {                                          \
                    if (orow < n)                                              \
                        MMQ_PUT2(whole, (size_t)ot0 * n + orow,                \
                                 acc[j][u].x[0]);                              \
                    if (orow + 1 < n)                                          \
                        MMQ_PUT2(whole, (size_t)ot0 * n + orow + 1,            \
                                 acc[j][u].x[1]);                              \
                }                                                              \
                if (ot1 < n_tokens) {                                          \
                    if (orow < n)                                              \
                        MMQ_PUT2(whole, (size_t)ot1 * n + orow,                \
                                 acc[j][u].x[2]);                              \
                    if (orow + 1 < n)                                          \
                        MMQ_PUT2(whole, (size_t)ot1 * n + orow + 1,            \
                                 acc[j][u].x[3]);                              \
                }                                                              \
            }                                                                  \
        }                                                                      \
        flat += kt_hi - kt_lo;                                                 \
    }
#define MMQ_F16HALFRING_SET(SUFFIX, WARPS, NBLK, TILES, STAGES)                \
    extern "C" __global__ __launch_bounds__((WARPS) * WARP_SIZE) void          \
    mmqnr##SUFFIX##_q4_g128(float* __restrict__ out,                           \
                            const void* __restrict__ wv,                       \
                            const void* __restrict__ xv, int k, int n,         \
                            int n_tokens) {                                    \
        const block_q4_g128* wq = (const block_q4_g128*)wv;                    \
        MMQ_F16HALFRING_BODY(WARPS, NBLK, TILES, STAGES)                       \
    }

MMQ_F16HALFRING_SET(1w8s2, 8, 1, 1, 2)
MMQ_F16HALFRING_SET(1w8s2_2, 8, 1, 2, 2)

// ---- eight-bit activations under the f16 tensor cores -------------------
//
// The f16 kernel reaches 97% of its weight-read ceiling at 8 tokens and 63% at
// 32, and the two configurations differ in exactly one thing that survived
// every probe: 32 resident warps per SM against 16. Two token tiles of f16
// activations at two stages is 34.8 KB, which caps the SM at two blocks of
// eight warps; one tile is 17.4 KB and fits four. `mmql_*` lands on the same 16
// warps by a different route — four warps and four blocks — and measures the
// same shortfall, which is why the number to move is warps and not blocks.
//
// A half is what makes the ring too big, and it does not have to be a half.
// Q8_1 activations are 1.125 bytes an element, so the same two tiles at two
// stages are 19.5 KB and the SM holds four blocks again — and, unlike the
// integer kernels, nothing forces the *accumulator* back to s32: the weights
// still dequantize to whole `(w - z) * s` values in f16, the activation carries
// its own scale, and f32 accumulation spans the whole k. So the epilogue stays
// gone.
//
// The cost is on the A side, where four int8 have to become four halves. That
// is `dequant.h`'s int8 path (lines 226-236): `prmt` drops each byte into the
// mantissa of f16 1024, `hsub2` takes the bias back out. Q8_1 stores signed
// two's complement where Marlin's variant stores offset-128, so an `xor` with
// 0x80808080 converts one to the other first and the bias becomes 1152.
//
// **And the cost wins.** GB/s of weights:
//
//                   ffn_gate       ffn_down
//                  8t     32t     8t     32t
//   mmqf1w8s2     331     214    342     227
//   mmqe1w8s2     224     151    190     128
//
// 30% down, and down by as much at eight tokens — where the ring was already
// small enough for four blocks and the only change is the widening — as at
// thirty-two, where the occupancy was supposed to pay for it. So the extra ten
// instructions an A fragment cost more than doubling the resident warps saves,
// and the ring size was never the thing.
//
// Which corrects something this file concluded two sections ago. `mmqnm_*`
// measured the tensor cores free and it was tempting to read that as "this
// kernel has ALU to spare". It does not: the *same* MMAs with ten more
// instructions on the operand path lose 30%. Free tensor cores mean the MMA
// pipe is idle, not that the issue slots are.
__device__ __forceinline__ void mmq_i8_to_f16(uint32_t v, __half2 scale,
                                              unsigned* f) {
    // Signed to offset-128, so the byte lands in the mantissa as a positive
    // integer and one subtraction recovers the value.
    const uint32_t q = v ^ 0x80808080u;
    const uint32_t EX = 0x64646464u;
    uint32_t lo = __byte_perm(q, EX, 0x5250);
    uint32_t hi = __byte_perm(q, EX, 0x5351);
    const __half2 bias = __float2half2_rn(1152.0f);
    const __half2 a0 = __hsub2(*(const __half2*)(const void*)&lo, bias);
    const __half2 a1 = __hsub2(*(const __half2*)(const void*)&hi, bias);
    const __half2 r0 = __hmul2(a0, scale);
    const __half2 r1 = __hmul2(a1, scale);
    f[0] = *(const unsigned*)(const void*)&r0;
    f[1] = *(const unsigned*)(const void*)&r1;
}

#define MMQ_E8_BODY(WARPS, NBLK, TILES, STAGES)                               \
    extern __shared__ __align__(16) int8_t xa[];                               \
                                                                               \
    const int tid = threadIdx.x;                                               \
    const int lane = tid % WARP_SIZE;                                          \
    const int warp = tid / WARP_SIZE;                                          \
    const int nthreads = (WARPS) * WARP_SIZE;                                  \
    const int mrows = (WARPS) * (NBLK) * 8;                                    \
    const int tok0 = blockIdx.y * (TILES) * MMQ_M;                             \
    const int kb_total = k / 32;                                               \
    const int nb_total = k / QK_G128;                                          \
    const int k_tiles = (k + MMQ_K - 1) / MMQ_K;                               \
    const int row_tiles = (n + mrows - 1) / mrows;                             \
    const int x_rows = (TILES) * MMQ_M;                                        \
    const int x_valid = min(x_rows, max(0, n_tokens - tok0));                  \
    const uint8_t* xbytes = (const uint8_t*)xq;                                \
                                                                               \
    const int ar = mma_a_row(lane);                                            \
    const int bc = mma_b_col(lane);                                            \
    const int cq = lane % 4;                                                   \
    const int kq = cq * 4;                                                     \
    const int cr = mma_c_row(lane);                                            \
    const int cc = mma_c_col(lane);                                            \
    const int wbase = warp * (NBLK) * 8;                                       \
                                                                               \
    /* The striped partition, which `mmqsr_*` measured worth 30% at four        \
       tokens. Same arithmetic here. */                                        \
    const int total = row_tiles * k_tiles;                                     \
    const int iters = (total + (int)gridDim.x - 1) / (int)gridDim.x;           \
    int flat = iters * (int)blockIdx.x;                                        \
    const int flat_end = min(total, flat + iters);                             \
                                                                               \
    const block_q4_g128* bsrc[NBLK];                                           \
    bool brow_ok[NBLK];                                                        \
    mma_c_f32 acc[NBLK][TILES];                                                \
                                                                               \
    if (x_valid < x_rows) {                                                    \
        const int per = (x_rows - x_valid) * (MMQ_XA_ROW / 4);                 \
        for (int i = tid; i < (STAGES) * per; i += nthreads) {                 \
            const int s = i / per;                                             \
            const int j = i % per;                                             \
            const int r = x_valid + j / (MMQ_XA_ROW / 4);                      \
            const int e = (j % (MMQ_XA_ROW / 4)) * 4;                          \
            *(uint32_t*)(void*)(xa + (s * x_rows + r) * MMQ_XA_STRIDE + e) = 0;\
        }                                                                      \
    }                                                                          \
                                                                               \
    while (flat < flat_end) {                                                  \
        const int nt = flat / k_tiles;                                         \
        const int kt_lo = flat % k_tiles;                                      \
        const int kt_hi = min(k_tiles, kt_lo + (flat_end - flat));             \
        const int row0 = nt * mrows;                                           \
                                                                               \
        _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {                   \
            const int r = row0 + wbase + j * 8 + bc;                           \
            brow_ok[j] = r < n;                                                \
            bsrc[j] = wq + (brow_ok[j] ? (size_t)r * nb_total : 0);            \
            _Pragma("unroll") for (int u = 0; u < (TILES); ++u) {              \
                _Pragma("unroll") for (int c = 0; c < 4; ++c)                  \
                    acc[j][u].x[c] = 0.0f;                                     \
            }                                                                  \
        }                                                                      \
                                                                               \
        _Pragma("unroll") for (int s = 0; s < (STAGES) - 1; ++s) {             \
            MMQ_XA_FETCH(s, kt_lo + s, kt_hi);                                 \
        }                                                                      \
                                                                               \
        for (int kt = kt_lo; kt < kt_hi; ++kt) {                               \
            const int pos = (kt - kt_lo) % (STAGES);                           \
            MMQ_CP_ASYNC_WAIT((STAGES) - 2);                                   \
            __syncthreads();                                                   \
            MMQ_XA_FETCH((pos + (STAGES) - 1) % (STAGES),                      \
                         kt + (STAGES) - 1, kt_hi);                            \
            const int8_t* xbuf = xa + pos * x_rows * MMQ_XA_STRIDE;            \
                                                                               \
            _Pragma("unroll") for (int blk = 0; blk < 2; ++blk) {              \
                const int kb = kt * 2 + blk;                                   \
                const bool kb_ok = kb < nb_total;                              \
                /* One scale and one zero per row per 128 weights, broadcast   \
                   into the halves the dequantization folds them into. */      \
                __half2 s2[NBLK], m2[NBLK];                                    \
                _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {           \
                    const int sr = row0 + wbase + j * 8 + bc;                  \
                    __half2 ds = __floats2half2_rn(0.0f, 0.0f);                \
                    if (kb_ok && brow_ok[j]) ds = bsrc[j][kb].ds;              \
                    s2[j] = __float2half2_rn(__low2float(ds));                 \
                    m2[j] = __float2half2_rn(-__high2float(ds));               \
                    (void)sr;                                                  \
                }                                                              \
                _Pragma("unroll") for (int run = 0; run < 2; ++run) {          \
                    uint32_t v0[NBLK], v1[NBLK];                               \
                    _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {       \
                        v0[j] = 0;                                             \
                        v1[j] = 0;                                             \
                        if (brow_ok[j] && kb_ok) {                             \
                            const uint8_t* pq =                                \
                                bsrc[j][kb].qs + run * 32 + kq;                \
                            v0[j] = *(const uint32_t*)(const void*)pq;         \
                            v1[j] = *(const uint32_t*)(const void*)(pq + 16);  \
                        }                                                      \
                    }                                                          \
                    _Pragma("unroll") for (int h = 0; h < 2; ++h) {            \
                        const int g = blk * 4 + run + h * 2;                   \
                        /* Two MMAs a group: k is 16 here, not 32. */          \
                        _Pragma("unroll") for (int mm = 0; mm < 2; ++mm) {     \
                            mma_a_f16 a[TILES];                                \
                            _Pragma("unroll") for (int u = 0; u < (TILES);     \
                                                   ++u) {                      \
                                /* Four int8 per row, widened here. The     \
                                   fragment sits inside one 32-group and one   \
                                   token row, so one scale covers it. */       \
                                const int8_t* _gb =                            \
                                    xbuf + (u * MMQ_M + ar) * MMQ_XA_STRIDE    \
                                    + g * 36;                                  \
                                const int8_t* ap = _gb + 4 + mm * 16 + cq * 4; \
                                const int8_t* aq = ap + 8 * MMQ_XA_STRIDE;     \
                                const __half2 _sa = __half2half2(__low2half(   \
                                    *(const __half2*)(const void*)_gb));       \
                                const __half2 _sb = __half2half2(__low2half(   \
                                    *(const __half2*)(const void*)(            \
                                        _gb + 8 * MMQ_XA_STRIDE)));            \
                                unsigned _fa[2], _fb[2];                       \
                                mmq_i8_to_f16(                                 \
                                    *(const uint32_t*)(const void*)ap, _sa,     \
                                    _fa);                                      \
                                mmq_i8_to_f16(                                 \
                                    *(const uint32_t*)(const void*)aq, _sb,     \
                                    _fb);                                      \
                                a[u].x[0] = _fa[0];                            \
                                a[u].x[1] = _fb[0];                            \
                                a[u].x[2] = _fa[1];                            \
                                a[u].x[3] = _fb[1];                            \
                            }                                                  \
                            _Pragma("unroll") for (int j = 0; j < (NBLK);      \
                                                   ++j) {                      \
                                mma_b_f16 b;                                   \
                                mmq_deq4_f16(mm ? v1[j] : v0[j], h, s2[j],     \
                                             m2[j], b.x);                      \
                                _Pragma("unroll") for (int u = 0;              \
                                                       u < (TILES); ++u) {     \
                                    mma_f16(acc[j][u], a[u], b);               \
                                }                                              \
                            }                                                  \
                        }                                                      \
                    }                                                          \
                }                                                              \
            }                                                                  \
        }                                                                      \
                                                                               \
        MMQ_CP_ASYNC_WAIT(0);                                                  \
        __syncthreads();                                                       \
                                                                               \
        /* No epilogue: the accumulator is already the answer. */              \
        const bool whole = (kt_lo == 0) && (kt_hi == k_tiles);                 \
        _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {                   \
            const int orow = row0 + wbase + j * 8 + cc;                        \
            _Pragma("unroll") for (int u = 0; u < (TILES); ++u) {              \
                const int ot0 = tok0 + u * MMQ_M + cr;                         \
                const int ot1 = ot0 + 8;                                       \
                if (ot0 < n_tokens) {                                          \
                    if (orow < n)                                              \
                        MMQ_PUT2(whole, (size_t)ot0 * n + orow,                \
                                 acc[j][u].x[0]);                              \
                    if (orow + 1 < n)                                          \
                        MMQ_PUT2(whole, (size_t)ot0 * n + orow + 1,            \
                                 acc[j][u].x[1]);                              \
                }                                                              \
                if (ot1 < n_tokens) {                                          \
                    if (orow < n)                                              \
                        MMQ_PUT2(whole, (size_t)ot1 * n + orow,                \
                                 acc[j][u].x[2]);                              \
                    if (orow + 1 < n)                                          \
                        MMQ_PUT2(whole, (size_t)ot1 * n + orow + 1,            \
                                 acc[j][u].x[3]);                              \
                }                                                              \
            }                                                                  \
        }                                                                      \
        flat += kt_hi - kt_lo;                                                 \
    }
#define MMQ_E8_SET(SUFFIX, WARPS, NBLK, TILES, STAGES)                         \
    extern "C" __global__ __launch_bounds__((WARPS) * WARP_SIZE) void          \
    mmqe##SUFFIX##_q4_g128(float* __restrict__ out,                            \
                           const void* __restrict__ wv,                        \
                           const void* __restrict__ xv, int k, int n,          \
                           int n_tokens) {                                     \
        const block_q4_g128* wq = (const block_q4_g128*)wv;                    \
        const block_q8_1* xq = (const block_q8_1*)xv;                          \
        MMQ_E8_BODY(WARPS, NBLK, TILES, STAGES)                                \
    }

MMQ_E8_SET(1w8s2, 8, 1, 1, 2)
MMQ_E8_SET(1w8s2_2, 8, 1, 2, 2)
MMQ_E8_SET(1w8s4, 8, 1, 1, 4)
MMQ_E8_SET(1w8s4_2, 8, 1, 2, 4)
MMQ_E8_SET(1w4s2, 4, 1, 1, 2)
MMQ_E8_SET(1w4s2_2, 4, 1, 2, 2)
/* Four token tiles, which is where the Q8_1 ring earns its keep: 64 tokens to
   one weight pass at 38.9 KB and two resident blocks, where the f16 ring at
   four tiles is 69.6 KB and one. Per pass this loses 30% to the widening; per
   *token* above 32 it should not, because the f16 shape is reading the weights
   twice there. */
MMQ_E8_SET(1w8s2_4, 8, 1, 4, 2)
MMQ_E8_SET(1w4s2_4, 4, 1, 4, 2)
MMQ_E8_SET(1w8s2_6, 8, 1, 6, 2)

// ---- one weight pass, several token tiles, one tile of ring ------------
//
// At 8 tokens this kernel runs at 97% of its weight-read ceiling and at 32 it
// runs at 63%, and the difference is resident warps: one token tile of f16
// activations at two stages is 17.4 KB and the SM holds four blocks of eight
// warps, two tiles is 34.8 KB and it holds two. Every attempt to shrink the
// two-tile ring has cost more than it saved — eight-bit activations lose 30% to
// the widening (`mmqe_*`), a narrower stride does not exist, fewer stages is
// not a thing at two.
//
// So do not shrink it: do not allocate it. A block covers `GROUPS * 16` tokens
// but stages one sixteen at a time, and the weights for a k-tile stay in
// registers across all of them. The pipeline unit becomes (k-tile, group)
// rather than k-tile, which the existing ring already expresses — it is the
// same `cp.async` ring with twice the units through it and none of the extra
// footprint.
//
// What it costs is that a group's staging can only be issued one unit ahead
// rather than a whole k-tile ahead, the weight registers are live across
// `GROUPS` matmuls instead of one, and — the part that turned out to matter —
// there is a `__syncthreads` per group rather than per k-tile.
//
// **It loses, and in losing it refutes the reason it was built.** GB/s of
// weights:
//
//                   ffn_gate       ffn_down
//                  8t     32t     8t     32t
//   mmqf1w8s2     331     214    342     226
//   mmqg1w8s2     300     201    320     206
//
// `kernel_registers` puts `mmqg1w8s2_2` at 66 registers and a 17.4 KB ring, so
// three resident blocks per SM, against `mmqf1w8s2_2`'s 80 registers and 34.8
// KB, which is two. **Half again the resident warps and 6% slower.**
//
// That kills the explanation this file had been carrying for the 32-token
// deficit — that 32 warps per SM buy 97% of the weight-read ceiling and 16 buy
// 63%. Two kernels now sit on both sides of that: `mmql_*` at 16 warps by four
// blocks of four warps measures the same shortfall as `mmqf_*` at 16 by two of
// eight, which looked like confirmation, and `mmqg_*` at 24 warps measures
// *worse*. Resident warps are not what separates 8 tokens from 32 here.
//
// What separates them, on this evidence, is the barrier count: `mmqf_*` at two
// token tiles crosses one `__syncthreads` per k-tile and `mmqg_*` crosses two,
// and that 2x is worth more than 1.5x the blocks. Which is a hypothesis with
// one measurement behind it, not a conclusion.
#define MMQ_G_BODY(WARPS, NBLK, GROUPS, STAGES)                                \
    extern __shared__ __align__(16) int8_t xf[];                               \
                                                                               \
    const int tid = threadIdx.x;                                               \
    const int lane = tid % WARP_SIZE;                                          \
    const int warp = tid / WARP_SIZE;                                          \
    const int nthreads = (WARPS) * WARP_SIZE;                                  \
    const int mrows = (WARPS) * (NBLK) * 8;                                    \
    const int tok0 = blockIdx.y * (GROUPS) * MMQ_M;                            \
    const int nb_total = k / QK_G128;                                          \
    const int k_tiles = (k + MMQ_K - 1) / MMQ_K;                               \
    const int row_tiles = (n + mrows - 1) / mrows;                             \
    const uint8_t* xbytes = (const uint8_t*)xv;                                \
                                                                               \
    const int ar = mma_a_row(lane);                                            \
    const int bc = mma_b_col(lane);                                            \
    const int cq = lane % 4;                                                   \
    const int kq = cq * 4;                                                     \
    const int cr = mma_c_row(lane);                                            \
    const int cc = mma_c_col(lane);                                            \
    const int wbase = warp * (NBLK) * 8;                                       \
                                                                               \
    const int total = row_tiles * k_tiles;                                     \
    const int iters = (total + (int)gridDim.x - 1) / (int)gridDim.x;           \
    int flat = iters * (int)blockIdx.x;                                        \
    const int flat_end = min(total, flat + iters);                             \
                                                                               \
    const block_q4_g128* bsrc[NBLK];                                           \
    bool brow_ok[NBLK];                                                        \
    mma_c_f32 acc[NBLK][GROUPS];                                               \
    /* One k-tile's weights, held across every token group: [blk][run][j] for  \
       the quants and [blk][j] for the scales. This is the whole point of the  \
       shape — the ring is one token tile, and the weights are what does not   \
       have to be read again for the next one. */                              \
    uint32_t wv0[2][2][NBLK], wv1[2][2][NBLK];                                 \
    __half2 ws2[2][NBLK], wm2[2][NBLK];                                        \
                                                                               \
    while (flat < flat_end) {                                                  \
        const int nt = flat / k_tiles;                                         \
        const int kt_lo = flat % k_tiles;                                      \
        const int kt_hi = min(k_tiles, kt_lo + (flat_end - flat));             \
        const int row0 = nt * mrows;                                           \
        const int units = (kt_hi - kt_lo) * (GROUPS);                          \
                                                                               \
        _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {                   \
            const int r = row0 + wbase + j * 8 + bc;                           \
            brow_ok[j] = r < n;                                                \
            bsrc[j] = wq + (brow_ok[j] ? (size_t)r * nb_total : 0);            \
            _Pragma("unroll") for (int g = 0; g < (GROUPS); ++g) {             \
                _Pragma("unroll") for (int c = 0; c < 4; ++c)                  \
                    acc[j][g].x[c] = 0.0f;                                     \
            }                                                                  \
        }                                                                      \
                                                                               \
        _Pragma("unroll") for (int s = 0; s < (STAGES) - 1; ++s) {             \
            MMQ_G_FETCH(GROUPS, s, s);                                         \
        }                                                                      \
                                                                               \
        for (int kt = kt_lo; kt < kt_hi; ++kt) {                               \
            /* Once per k-tile, not once per group. */                         \
            _Pragma("unroll") for (int blk = 0; blk < 2; ++blk) {              \
                const int kb = kt * 2 + blk;                                   \
                const bool kb_ok = kb < nb_total;                              \
                _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {           \
                    __half2 ds = __floats2half2_rn(0.0f, 0.0f);                \
                    if (kb_ok && brow_ok[j]) ds = bsrc[j][kb].ds;              \
                    ws2[blk][j] = __float2half2_rn(__low2float(ds));           \
                    wm2[blk][j] = __float2half2_rn(-__high2float(ds));         \
                }                                                              \
                _Pragma("unroll") for (int run = 0; run < 2; ++run) {          \
                    _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {       \
                        wv0[blk][run][j] = 0;                                  \
                        wv1[blk][run][j] = 0;                                  \
                        if (brow_ok[j] && kb_ok) {                             \
                            const uint8_t* pq =                                \
                                bsrc[j][kb].qs + run * 32 + kq;                \
                            wv0[blk][run][j] =                                 \
                                *(const uint32_t*)(const void*)pq;             \
                            wv1[blk][run][j] =                                 \
                                *(const uint32_t*)(const void*)(pq + 16);      \
                        }                                                      \
                    }                                                          \
                }                                                              \
            }                                                                  \
                                                                               \
            _Pragma("unroll") for (int gp = 0; gp < (GROUPS); ++gp) {          \
                const int un = (kt - kt_lo) * (GROUPS) + gp;                   \
                const int pos = un % (STAGES);                                 \
                MMQ_CP_ASYNC_WAIT((STAGES) - 2);                               \
                __syncthreads();                                               \
                MMQ_G_FETCH(GROUPS, (pos + (STAGES) - 1) % (STAGES),           \
                            un + (STAGES) - 1);                                \
                const int8_t* xbuf = xf + pos * MMQ_M * MMQ_XF_STRIDE;         \
                                                                               \
                _Pragma("unroll") for (int blk = 0; blk < 2; ++blk) {          \
                    _Pragma("unroll") for (int run = 0; run < 2; ++run) {      \
                        _Pragma("unroll") for (int h = 0; h < 2; ++h) {        \
                            const int g = blk * 4 + run + h * 2;               \
                            _Pragma("unroll") for (int mm = 0; mm < 2; ++mm) { \
                                const int8_t* ap = xbuf + ar * MMQ_XF_STRIDE   \
                                                 + g * 64 + mm * 32 + cq * 8;  \
                                const int8_t* aq = ap + 8 * MMQ_XF_STRIDE;     \
                                const uint2 lo =                               \
                                    *(const uint2*)(const void*)ap;            \
                                const uint2 hi =                               \
                                    *(const uint2*)(const void*)aq;            \
                                mma_a_f16 a;                                   \
                                a.x[0] = lo.x;                                 \
                                a.x[1] = hi.x;                                 \
                                a.x[2] = lo.y;                                 \
                                a.x[3] = hi.y;                                 \
                                _Pragma("unroll") for (int j = 0; j < (NBLK);  \
                                                       ++j) {                  \
                                    mma_b_f16 b;                               \
                                    mmq_deq4_f16(mm ? wv1[blk][run][j]         \
                                                    : wv0[blk][run][j],        \
                                                 h, ws2[blk][j], wm2[blk][j],  \
                                                 b.x);                         \
                                    mma_f16(acc[j][gp], a, b);                 \
                                }                                              \
                            }                                                  \
                        }                                                      \
                    }                                                          \
                }                                                              \
            }                                                                  \
        }                                                                      \
                                                                               \
        MMQ_CP_ASYNC_WAIT(0);                                                  \
        __syncthreads();                                                       \
                                                                               \
        const bool whole = (kt_lo == 0) && (kt_hi == k_tiles);                 \
        _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {                   \
            const int orow = row0 + wbase + j * 8 + cc;                        \
            _Pragma("unroll") for (int g = 0; g < (GROUPS); ++g) {             \
                const int ot0 = tok0 + g * MMQ_M + cr;                         \
                const int ot1 = ot0 + 8;                                       \
                if (ot0 < n_tokens) {                                          \
                    if (orow < n)                                              \
                        MMQ_PUT2(whole, (size_t)ot0 * n + orow,                \
                                 acc[j][g].x[0]);                              \
                    if (orow + 1 < n)                                          \
                        MMQ_PUT2(whole, (size_t)ot0 * n + orow + 1,            \
                                 acc[j][g].x[1]);                              \
                }                                                              \
                if (ot1 < n_tokens) {                                          \
                    if (orow < n)                                              \
                        MMQ_PUT2(whole, (size_t)ot1 * n + orow,                \
                                 acc[j][g].x[2]);                              \
                    if (orow + 1 < n)                                          \
                        MMQ_PUT2(whole, (size_t)ot1 * n + orow + 1,            \
                                 acc[j][g].x[3]);                              \
                }                                                              \
            }                                                                  \
        }                                                                      \
        flat += kt_hi - kt_lo;                                                 \
    }

/* Stage one sixteen-token tile for pipeline unit UN, which names a k-tile and
   a token group. Rows past the batch and k past the row are zero-filled by the
   copy itself, which matters here because a slot is reused by groups with
   different valid counts. */
#define MMQ_G_FETCH(GROUPS, BUF, UN)                                           \
    do {                                                                       \
        const int _un = (UN);                                                  \
        const bool _live = _un < units;                                        \
        const int _kt = kt_lo + (_live ? _un / (GROUPS) : 0);                  \
        const int _gp = _live ? _un % (GROUPS) : 0;                            \
        const int _t0 = tok0 + _gp * MMQ_M;                                    \
        const int _valid = min(MMQ_M, max(0, n_tokens - _t0));                 \
        for (int _i = tid; _i < MMQ_M * (MMQ_XF_ROW / 16); _i += nthreads) {   \
            const int _r = _i / (MMQ_XF_ROW / 16);                             \
            const int _c = _i % (MMQ_XF_ROW / 16);                             \
            const bool _hit = _live && _r < _valid                             \
                            && (_kt * MMQ_K + _c * 8 + 8 <= k);                \
            const size_t _off =                                                \
                ((size_t)(_t0 + _r) * k + (size_t)_kt * MMQ_K) * 2             \
                + (size_t)_c * 16;                                             \
            mmq_cp_async16(                                                    \
                xf + ((BUF) * MMQ_M + _r) * MMQ_XF_STRIDE + _c * 16,           \
                xbytes + (_hit ? _off : 0), _hit);                             \
        }                                                                      \
        MMQ_CP_ASYNC_FENCE();                                                  \
    } while (0)

#define MMQ_G_SET(SUFFIX, WARPS, NBLK, GROUPS, STAGES)                         \
    extern "C" __global__ __launch_bounds__((WARPS) * WARP_SIZE) void          \
    mmqg##SUFFIX##_q4_g128(float* __restrict__ out,                            \
                           const void* __restrict__ wv,                        \
                           const void* __restrict__ xv, int k, int n,          \
                           int n_tokens) {                                     \
        const block_q4_g128* wq = (const block_q4_g128*)wv;                    \
        MMQ_G_BODY(WARPS, NBLK, GROUPS, STAGES)                                \
    }

MMQ_G_SET(1w8s2, 8, 1, 1, 2)
MMQ_G_SET(1w8s2_2, 8, 1, 2, 2)
MMQ_G_SET(1w8s2_4, 8, 1, 4, 2)
MMQ_G_SET(1w8s3_2, 8, 1, 2, 3)
MMQ_G_SET(1w4s2_2, 4, 1, 2, 2)

// ---- the weight repack, for real this time ------------------------------
//
// `mmqfp_*` assumed a repacked layout and measured level, but it was a probe:
// wrong answers, an address forced into alignment, a dequantization pairing
// that did not match. This is the layout itself.
//
// A lane's B fragment is four words — (run 0 low, run 0 high, run 1 low, run 1
// high) — read four bytes at a time because in an AWQ pack they are 16 bytes
// apart. Transposing the 4x4 matrix of 4-byte words inside each 64-byte `qs`
// puts them side by side:
//
//   new[c*16 + q*4 + i] = old[q*16 + c*4 + i]      c, q in 0..4, i in 0..4
//
// after which lane `c` reads its whole fragment as one `uint4` at `c*16`. The
// dequantization does not change at all — the word order the transpose
// produces is exactly the order `mmq_deq4_f16` already consumes.
//
// The block layout has to change with it. `qs` sits four bytes into a 68-byte
// block, so `qs + c*16` is never 16-byte aligned; padding the block to 80 costs
// 17.6% of the weight bytes, which is the whole budget. So the scales move out:
// `n * k / 2` bytes of quants, then one `__half2` per row per 128 weights.
// Same total, and every block's quants land on a 64-byte boundary.

#define MMQ_Z_BODY(WARPS, NBLK, TILES, STAGES)                               \
    extern __shared__ __align__(16) int8_t xf[];                               \
                                                                               \
    const int tid = threadIdx.x;                                               \
    const int lane = tid % WARP_SIZE;                                          \
    const int warp = tid / WARP_SIZE;                                          \
    const int nthreads = (WARPS) * WARP_SIZE;                                  \
    const int mrows = (WARPS) * (NBLK) * 8;                                    \
    const int tok0 = blockIdx.y * (TILES) * MMQ_M;                             \
    const int nb_total = k / QK_G128;                                          \
    const int k_tiles = (k + MMQ_K - 1) / MMQ_K;                               \
    const int row_tiles = (n + mrows - 1) / mrows;                             \
    const int x_rows = (TILES) * MMQ_M;                                        \
    const int x_valid = min(x_rows, max(0, n_tokens - tok0));                  \
    const uint8_t* xbytes = (const uint8_t*)xv;                                \
                                                                               \
    const int ar = mma_a_row(lane);                                            \
    const int bc = mma_b_col(lane);                                            \
    const int cq = lane % 4;                                                   \
    const int kq = cq * 4;                                                     \
    const int cr = mma_c_row(lane);                                            \
    const int cc = mma_c_col(lane);                                            \
    const int wbase = warp * (NBLK) * 8;                                       \
                                                                               \
    /* The striped partition, which `mmqsr_*` measured worth 30% at four        \
       tokens. Same arithmetic here. */                                        \
    const int total = row_tiles * k_tiles;                                     \
    const int iters = (total + (int)gridDim.x - 1) / (int)gridDim.x;           \
    int flat = iters * (int)blockIdx.x;                                        \
    const int flat_end = min(total, flat + iters);                             \
                                                                               \
    /* `n * nb` blocks of 64 quant bytes, then `n * nb` scale pairs. Same      \
       total as the packed 68-byte blocks, and the quants are 16-byte aligned   \
       where the blocks are not — which is the point, since a lane's fragment   \
       is one `uint4`. */                                                      \
    const uint8_t* qbase = (const uint8_t*)wq;                                 \
    const __half2* sbase =                                                     \
        (const __half2*)(const void*)(qbase + (size_t)n * nb_total * 64);      \
    size_t browi[NBLK];                                                        \
    bool brow_ok[NBLK];                                                        \
    mma_c_f32 acc[NBLK][TILES];                                                \
                                                                               \
    if (x_valid < x_rows) {                                                    \
        const int per = (x_rows - x_valid) * (MMQ_XF_ROW / 4);                 \
        for (int i = tid; i < (STAGES) * per; i += nthreads) {                 \
            const int s = i / per;                                             \
            const int j = i % per;                                             \
            const int r = x_valid + j / (MMQ_XF_ROW / 4);                      \
            const int e = (j % (MMQ_XF_ROW / 4)) * 4;                          \
            *(uint32_t*)(void*)(xf + (s * x_rows + r) * MMQ_XF_STRIDE + e) = 0;\
        }                                                                      \
    }                                                                          \
                                                                               \
    while (flat < flat_end) {                                                  \
        const int nt = flat / k_tiles;                                         \
        const int kt_lo = flat % k_tiles;                                      \
        const int kt_hi = min(k_tiles, kt_lo + (flat_end - flat));             \
        const int row0 = nt * mrows;                                           \
                                                                               \
        _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {                   \
            const int r = row0 + wbase + j * 8 + bc;                           \
            brow_ok[j] = r < n;                                                \
            /* The row's byte offset, not its index: multiplying by the row  \
               stride inside the k-loop puts a 64-bit multiply on every weight \
               address and cost 13% when it was written that way. */          \
            browi[j] = brow_ok[j] ? (size_t)r * nb_total : 0;                  \
            _Pragma("unroll") for (int u = 0; u < (TILES); ++u) {              \
                _Pragma("unroll") for (int c = 0; c < 4; ++c)                  \
                    acc[j][u].x[c] = 0.0f;                                     \
            }                                                                  \
        }                                                                      \
                                                                               \
        _Pragma("unroll") for (int s = 0; s < (STAGES) - 1; ++s) {             \
            MMQ_XF_FETCH(s, kt_lo + s, kt_hi);                                 \
        }                                                                      \
                                                                               \
        for (int kt = kt_lo; kt < kt_hi; ++kt) {                               \
            const int pos = (kt - kt_lo) % (STAGES);                           \
            MMQ_CP_ASYNC_WAIT((STAGES) - 2);                                   \
            __syncthreads();                                                   \
            MMQ_XF_FETCH((pos + (STAGES) - 1) % (STAGES),                      \
                         kt + (STAGES) - 1, kt_hi);                            \
            const int8_t* xbuf = xf + pos * x_rows * MMQ_XF_STRIDE;            \
                                                                               \
            _Pragma("unroll") for (int blk = 0; blk < 2; ++blk) {              \
                const int kb = kt * 2 + blk;                                   \
                const bool kb_ok = kb < nb_total;                              \
                /* One scale and one zero per row per 128 weights, broadcast   \
                   into the halves the dequantization folds them into. */      \
                __half2 s2[NBLK], m2[NBLK];                                    \
                _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {           \
                    const int sr = row0 + wbase + j * 8 + bc;                  \
                    __half2 ds = __floats2half2_rn(0.0f, 0.0f);                \
                    if (kb_ok && brow_ok[j])                                   \
                        ds = sbase[browi[j] + kb];                             \
                    s2[j] = __float2half2_rn(__low2float(ds));                 \
                    m2[j] = __float2half2_rn(-__high2float(ds));               \
                    (void)sr;                                                  \
                }                                                              \
                /* One 16-byte request for the four words this lane used   \
                   to fetch separately: the transpose put (run 0 low, run 0   \
                   high, run 1 low, run 1 high) side by side. */              \
                uint4 wv4[NBLK];                                              \
                _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {          \
                    wv4[j] = make_uint4(0, 0, 0, 0);                          \
                    if (brow_ok[j] && kb_ok) {                                \
                        wv4[j] = *(const uint4*)(const void*)(                \
                            qbase + (browi[j] + kb) * 64 + cq * 16);          \
                    }                                                          \
                }                                                              \
                _Pragma("unroll") for (int run = 0; run < 2; ++run) {          \
                    uint32_t v0[NBLK], v1[NBLK];                               \
                    _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {       \
                        v0[j] = run ? wv4[j].z : wv4[j].x;                     \
                        v1[j] = run ? wv4[j].w : wv4[j].y;                     \
                    }                                                          \
                    _Pragma("unroll") for (int h = 0; h < 2; ++h) {            \
                        const int g = blk * 4 + run + h * 2;                   \
                        /* Two MMAs a group: k is 16 here, not 32. */          \
                        _Pragma("unroll") for (int mm = 0; mm < 2; ++mm) {     \
                            mma_a_f16 a[TILES];                                \
                            _Pragma("unroll") for (int u = 0; u < (TILES);     \
                                                   ++u) {                      \
                                const int8_t* ap =                             \
                                    xbuf + (u * MMQ_M + ar) * MMQ_XF_STRIDE    \
                                    + g * 64 + mm * 32 + cq * 8;               \
                                const int8_t* aq = ap + 8 * MMQ_XF_STRIDE;     \
                                const uint2 lo =                               \
                                    *(const uint2*)(const void*)ap;            \
                                const uint2 hi =                               \
                                    *(const uint2*)(const void*)aq;            \
                                a[u].x[0] = lo.x;                              \
                                a[u].x[1] = hi.x;                              \
                                a[u].x[2] = lo.y;                              \
                                a[u].x[3] = hi.y;                              \
                            }                                                  \
                            _Pragma("unroll") for (int j = 0; j < (NBLK);      \
                                                   ++j) {                      \
                                mma_b_f16 b;                                   \
                                mmq_deq4_f16_repacked(mm ? v1[j] : v0[j], h,   \
                                                      s2[j], m2[j], b.x);      \
                                _Pragma("unroll") for (int u = 0;              \
                                                       u < (TILES); ++u) {     \
                                    mma_f16(acc[j][u], a[u], b);               \
                                }                                              \
                            }                                                  \
                        }                                                      \
                    }                                                          \
                }                                                              \
            }                                                                  \
        }                                                                      \
                                                                               \
        MMQ_CP_ASYNC_WAIT(0);                                                  \
        __syncthreads();                                                       \
                                                                               \
        /* No epilogue: the accumulator is already the answer. */              \
        const bool whole = (kt_lo == 0) && (kt_hi == k_tiles);                 \
        _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {                   \
            const int orow = row0 + wbase + j * 8 + cc;                        \
            _Pragma("unroll") for (int u = 0; u < (TILES); ++u) {              \
                const int ot0 = tok0 + u * MMQ_M + cr;                         \
                const int ot1 = ot0 + 8;                                       \
                if (ot0 < n_tokens) {                                          \
                    if (orow < n)                                              \
                        MMQ_PUT2(whole, (size_t)ot0 * n + orow,                \
                                 acc[j][u].x[0]);                              \
                    if (orow + 1 < n)                                          \
                        MMQ_PUT2(whole, (size_t)ot0 * n + orow + 1,            \
                                 acc[j][u].x[1]);                              \
                }                                                              \
                if (ot1 < n_tokens) {                                          \
                    if (orow < n)                                              \
                        MMQ_PUT2(whole, (size_t)ot1 * n + orow,                \
                                 acc[j][u].x[2]);                              \
                    if (orow + 1 < n)                                          \
                        MMQ_PUT2(whole, (size_t)ot1 * n + orow + 1,            \
                                 acc[j][u].x[3]);                              \
                }                                                              \
            }                                                                  \
        }                                                                      \
        flat += kt_hi - kt_lo;                                                 \
    }
#define MMQ_Z_SET(SUFFIX, WARPS, NBLK, TILES, STAGES)                          \
    extern "C" __global__ __launch_bounds__((WARPS) * WARP_SIZE) void          \
    mmqz##SUFFIX##_q4_g128(float* __restrict__ out,                            \
                           const void* __restrict__ wv,                        \
                           const void* __restrict__ xv, int k, int n,          \
                           int n_tokens) {                                     \
        const block_q4_g128* wq = (const block_q4_g128*)wv;                    \
        MMQ_Z_BODY(WARPS, NBLK, TILES, STAGES)                                 \
    }

MMQ_Z_SET(1w8s2, 8, 1, 1, 2)
MMQ_Z_SET(1w8s2_2, 8, 1, 2, 2)
MMQ_Z_SET(1w8s2_4, 8, 1, 4, 2)
/* The transpose cut the weight-load instruction count fourfold, so the shape
   sweep that picked NBLK=1 and eight warps was run on a different instruction
   mix. These re-open it. */
MMQ_Z_SET(2w8s2, 8, 2, 1, 2)
MMQ_Z_SET(2w8s2_2, 8, 2, 2, 2)
MMQ_Z_SET(2w8s2_4, 8, 2, 4, 2)
MMQ_Z_SET(1w8s4, 8, 1, 1, 4)
MMQ_Z_SET(1w8s4_2, 8, 1, 2, 4)
MMQ_Z_SET(1w8s4_4, 8, 1, 4, 4)
MMQ_Z_SET(2w4s2, 4, 2, 1, 2)
MMQ_Z_SET(2w4s2_2, 4, 2, 2, 2)
MMQ_Z_SET(2w4s2_4, 4, 2, 4, 2)
MMQ_Z_SET(1w16s2, 16, 1, 1, 2)
MMQ_Z_SET(1w16s2_2, 16, 1, 2, 2)
MMQ_Z_SET(1w16s2_4, 16, 1, 4, 2)

#define MMQ_ZG_BODY(WARPS, NBLK, TILES, STAGES)                               \
    extern __shared__ __align__(16) int8_t xf[];                               \
                                                                               \
    const int tid = threadIdx.x;                                               \
    const int lane = tid % WARP_SIZE;                                          \
    const int warp = tid / WARP_SIZE;                                          \
    const int nthreads = (WARPS) * WARP_SIZE;                                  \
    const int mrows = (WARPS) * (NBLK) * 8;                                    \
    const int tok0 = blockIdx.y * (TILES) * MMQ_M;                             \
    const int nb_total = k / QK_G128;                                          \
    const int k_tiles = (k + MMQ_K - 1) / MMQ_K;                               \
    const int row_tiles = (n + mrows - 1) / mrows;                             \
    const int x_rows = (TILES) * MMQ_M;                                        \
    const int x_valid = min(x_rows, max(0, n_tokens - tok0));                  \
    const uint8_t* xbytes = (const uint8_t*)xv;                                \
                                                                               \
    const int ar = mma_a_row(lane);                                            \
    const int bc = mma_b_col(lane);                                            \
    const int cq = lane % 4;                                                   \
    const int kq = cq * 4;                                                     \
    const int cr = mma_c_row(lane);                                            \
    const int cc = mma_c_col(lane);                                            \
    const int wbase = warp * (NBLK) * 8;                                       \
                                                                               \
    /* The striped partition, which `mmqsr_*` measured worth 30% at four        \
       tokens. Same arithmetic here. */                                        \
    const int total = row_tiles * k_tiles;                                     \
    const int iters = (total + (int)gridDim.x - 1) / (int)gridDim.x;           \
    int flat = iters * (int)blockIdx.x;                                        \
    const int flat_end = min(total, flat + iters);                             \
                                                                               \
    /* Per row: `nb * 64` bytes of quants, then `nb` `__half2` of {scale,      \
       scale*zero}. Same total as the packed 68-byte blocks, and the quants     \
       are 16-byte aligned where the blocks are not — which is the whole        \
       point, since a lane's fragment is one `uint4`. Per row rather than       \
       globally so a kernel needs only the row base. */                        \
    const uint8_t* qbase = (const uint8_t*)wq;                                 \
    const __half2* sbase =                                                     \
        (const __half2*)(const void*)(qbase + (size_t)n * nb_total * 64);      \
    size_t browi[NBLK];                                                        \
    bool brow_ok[NBLK];                                                        \
    mma_c_f32 acc[NBLK][TILES];                                                \
                                                                               \
    if (x_valid < x_rows) {                                                    \
        const int per = (x_rows - x_valid) * (MMQ_XF_ROW / 4);                 \
        for (int i = tid; i < (STAGES) * per; i += nthreads) {                 \
            const int s = i / per;                                             \
            const int j = i % per;                                             \
            const int r = x_valid + j / (MMQ_XF_ROW / 4);                      \
            const int e = (j % (MMQ_XF_ROW / 4)) * 4;                          \
            *(uint32_t*)(void*)(xf + (s * x_rows + r) * MMQ_XF_STRIDE + e) = 0;\
        }                                                                      \
    }                                                                          \
                                                                               \
    while (flat < flat_end) {                                                  \
        const int nt = flat / k_tiles;                                         \
        const int kt_lo = flat % k_tiles;                                      \
        const int kt_hi = min(k_tiles, kt_lo + (flat_end - flat));             \
        const int row0 = nt * mrows;                                           \
                                                                               \
        _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {                   \
            const int r = row0 + wbase + j * 8 + bc;                           \
            brow_ok[j] = r < n;                                                \
            /* The row's byte offset, not its index: multiplying by the row  \
               stride inside the k-loop puts a 64-bit multiply on every weight \
               address and cost 13% when it was written that way. */          \
            browi[j] = brow_ok[j] ? (size_t)r * nb_total : 0;                  \
            _Pragma("unroll") for (int u = 0; u < (TILES); ++u) {              \
                _Pragma("unroll") for (int c = 0; c < 4; ++c)                  \
                    acc[j][u].x[c] = 0.0f;                                     \
            }                                                                  \
        }                                                                      \
                                                                               \
        _Pragma("unroll") for (int s = 0; s < (STAGES) - 1; ++s) {             \
            MMQ_XF_FETCH(s, kt_lo + s, kt_hi);                                 \
        }                                                                      \
                                                                               \
        for (int kt = kt_lo; kt < kt_hi; ++kt) {                               \
            const int pos = (kt - kt_lo) % (STAGES);                           \
            MMQ_CP_ASYNC_WAIT((STAGES) - 2);                                   \
            __syncthreads();                                                   \
            MMQ_XF_FETCH((pos + (STAGES) - 1) % (STAGES),                      \
                         kt + (STAGES) - 1, kt_hi);                            \
            const int8_t* xbuf = xf + pos * x_rows * MMQ_XF_STRIDE;            \
                                                                               \
            _Pragma("unroll") for (int blk = 0; blk < 2; ++blk) {              \
                const int kb = kt * 2 + blk;                                   \
                const bool kb_ok = kb < nb_total;                              \
                /* One scale and one zero per row per 128 weights, broadcast   \
                   into the halves the dequantization folds them into. */      \
                __half2 s2[NBLK], m2[NBLK];                                    \
                _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {           \
                    const int sr = row0 + wbase + j * 8 + bc;                  \
                    __half2 ds = __floats2half2_rn(0.0f, 0.0f);                \
                    if (kb_ok && brow_ok[j])                                   \
                        ds = sbase[browi[j] + kb];                             \
                    s2[j] = __float2half2_rn(__low2float(ds));                 \
                    m2[j] = __float2half2_rn(-__high2float(ds));               \
                    (void)sr;                                                  \
                }                                                              \
                /* One 16-byte request for the four words this lane used   \
                   to fetch separately: the transpose put (run 0 low, run 0   \
                   high, run 1 low, run 1 high) side by side. */              \
                uint4 wv4[NBLK];                                              \
                _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {          \
                    wv4[j] = make_uint4(0, 0, 0, 0);                          \
                    if (brow_ok[j] && kb_ok) {                                \
                        wv4[j] = *(const uint4*)(const void*)(                \
                            qbase + (browi[j] + kb) * 64 + cq * 16);          \
                    }                                                          \
                }                                                              \
                _Pragma("unroll") for (int run = 0; run < 2; ++run) {          \
                    uint32_t v0[NBLK], v1[NBLK];                               \
                    _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {       \
                        v0[j] = run ? wv4[j].z : wv4[j].x;                     \
                        v1[j] = run ? wv4[j].w : wv4[j].y;                     \
                    }                                                          \
                    _Pragma("unroll") for (int h = 0; h < 2; ++h) {            \
                        const int g = blk * 4 + run + h * 2;                   \
                        /* Two MMAs a group: k is 16 here, not 32. */          \
                        _Pragma("unroll") for (int mm = 0; mm < 2; ++mm) {     \
                            mma_a_f16 a[TILES];                                \
                            _Pragma("unroll") for (int u = 0; u < (TILES);     \
                                                   ++u) {                      \
                                const int8_t* ap =                             \
                                    xbuf + (u * MMQ_M + ar) * MMQ_XF_STRIDE    \
                                    + g * 64 + mm * 32 + cq * 8;               \
                                const int8_t* aq = ap + 8 * MMQ_XF_STRIDE;     \
                                const uint2 lo =                               \
                                    *(const uint2*)(const void*)ap;            \
                                const uint2 hi =                               \
                                    *(const uint2*)(const void*)aq;            \
                                a[u].x[0] = lo.x;                              \
                                a[u].x[1] = hi.x;                              \
                                a[u].x[2] = lo.y;                              \
                                a[u].x[3] = hi.y;                              \
                            }                                                  \
                            _Pragma("unroll") for (int j = 0; j < (NBLK);      \
                                                   ++j) {                      \
                                mma_b_f16 b;                                   \
                                mmq_deq4_f16(mm ? v1[j] : v0[j], h, s2[j],     \
                                             m2[j], b.x);                      \
                                _Pragma("unroll") for (int u = 0;              \
                                                       u < (TILES); ++u) {     \
                                    mma_f16(acc[j][u], a[u], b);               \
                                }                                              \
                            }                                                  \
                        }                                                      \
                    }                                                          \
                }                                                              \
            }                                                                  \
        }                                                                      \
                                                                               \
        MMQ_CP_ASYNC_WAIT(0);                                                  \
        __syncthreads();                                                       \
                                                                               \
        /* No epilogue: the accumulator is already the answer. */              \
        const bool whole = (kt_lo == 0) && (kt_hi == k_tiles);                 \
        _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {                   \
            const int orow = row0 + wbase + j * 8 + cc;                        \
            _Pragma("unroll") for (int u = 0; u < (TILES); ++u) {              \
                const int ot0 = tok0 + u * MMQ_M + cr;                         \
                const int ot1 = ot0 + 8;                                       \
                if (ot0 < n_tokens) {                                          \
                    if (orow < n)                                              \
                        MMQ_PUT2(whole, (size_t)ot0 * n + orow,                \
                                 acc[j][u].x[0]);                              \
                    if (orow + 1 < n)                                          \
                        MMQ_PUT2(whole, (size_t)ot0 * n + orow + 1,            \
                                 acc[j][u].x[1]);                              \
                }                                                              \
                if (ot1 < n_tokens) {                                          \
                    if (orow < n)                                              \
                        MMQ_PUT2(whole, (size_t)ot1 * n + orow,                \
                                 acc[j][u].x[2]);                              \
                    if (orow + 1 < n)                                          \
                        MMQ_PUT2(whole, (size_t)ot1 * n + orow + 1,            \
                                 acc[j][u].x[3]);                              \
                }                                                              \
            }                                                                  \
        }                                                                      \
        flat += kt_hi - kt_lo;                                                 \
    }
/* The same layout with the scales in one region at the end of the matrix
   rather than at the end of each row. Timing only: it needs the matrix width
   to find them, which the mat-vec macros do not hand their dot product, so it
   is not a layout the whole engine could take. The question it answers is
   whether the per-row split — chosen so the mat-vec would need no new plumbing
   — costs anything. */
#define MMQ_ZG_SET(SUFFIX, WARPS, NBLK, TILES, STAGES)                         \
    extern "C" __global__ __launch_bounds__((WARPS) * WARP_SIZE) void          \
    mmqzg##SUFFIX##_q4_g128(float* __restrict__ out,                           \
                            const void* __restrict__ wv,                       \
                            const void* __restrict__ xv, int k, int n,         \
                            int n_tokens) {                                    \
        const block_q4_g128* wq = (const block_q4_g128*)wv;                    \
        MMQ_ZG_BODY(WARPS, NBLK, TILES, STAGES)                                \
    }

MMQ_ZG_SET(1w8s2, 8, 1, 1, 2)
MMQ_ZG_SET(1w8s2_2, 8, 1, 2, 2)
MMQ_ZG_SET(1w8s2_4, 8, 1, 4, 2)

// A barrier with the arrival split from the wait.
//
// `__syncthreads()` is `bar.sync 0` — arrive and wait in one instruction, with
// nothing able to happen between them. PTX's named barriers separate the two,
// which lets a block announce it has finished reading the ring, do work that
// does not depend on the ring, and only then wait. Barrier 0 belongs to
// `__syncthreads`; this uses 1.
//
// The count is *twice* the thread count, and getting that wrong hangs the
// kernel rather than failing it: `bar.sync` arrives as well as waits, so each
// thread contributes two arrivals to the pair and a barrier told to expect one
// per thread releases when half of them have reached the arrive.
__device__ __forceinline__ void mmq_bar_arrive(int nthreads) {
    asm volatile("bar.arrive 1, %0;" ::"r"(2 * nthreads));
}
__device__ __forceinline__ void mmq_bar_sync(int nthreads) {
    asm volatile("bar.sync 1, %0;" ::"r"(2 * nthreads));
}

// ---- the barrier, hidden behind the weight loads ------------------------
//
// The one surviving explanation for 32 tokens running at 65% of the ceiling
// where 8 runs at 89%: two token tiles cross one `__syncthreads` per k-tile,
// and `mmqg_*` — which crosses two and has *more* resident warps — measured 6%
// slower. That is a hypothesis with one measurement behind it, and this is the
// change it predicts helps.

/* One k-tile of weight fragments and their scales, into registers. */
#define MMQ_Y_LOADW(V4, S2, M2, KT, NB)                                            \
    {                                                                          \
        _Pragma("unroll") for (int yb = 0; yb < 2; ++yb) {                     \
            const int ykb = (KT) * 2 + yb;                                     \
            const bool yok = ykb < nb_total;                                   \
            _Pragma("unroll") for (int j = 0; j < (NB); ++j) {                 \
                (V4)[yb][j] = make_uint4(0, 0, 0, 0);                          \
                __half2 yds = __floats2half2_rn(0.0f, 0.0f);                   \
                if (brow_ok[j] && yok) {                                       \
                    (V4)[yb][j] = *(const uint4*)(const void*)(                \
                        qbase + (browi[j] + ykb) * 64 + cq * 16);              \
                    yds = sbase[browi[j] + ykb];                               \
                }                                                              \
                (S2)[yb][j] = __float2half2_rn(__low2float(yds));              \
                (M2)[yb][j] = __float2half2_rn(-__high2float(yds));            \
            }                                                                  \
        }                                                                      \
    }

#define MMQ_Y_BODY(WARPS, NBLK, TILES, STAGES)                               \
    extern __shared__ __align__(16) int8_t xf[];                               \
                                                                               \
    const int tid = threadIdx.x;                                               \
    const int lane = tid % WARP_SIZE;                                          \
    const int warp = tid / WARP_SIZE;                                          \
    const int nthreads = (WARPS) * WARP_SIZE;                                  \
    const int mrows = (WARPS) * (NBLK) * 8;                                    \
    const int tok0 = blockIdx.y * (TILES) * MMQ_M;                             \
    const int nb_total = k / QK_G128;                                          \
    const int k_tiles = (k + MMQ_K - 1) / MMQ_K;                               \
    const int row_tiles = (n + mrows - 1) / mrows;                             \
    const int x_rows = (TILES) * MMQ_M;                                        \
    const int x_valid = min(x_rows, max(0, n_tokens - tok0));                  \
    const uint8_t* xbytes = (const uint8_t*)xv;                                \
                                                                               \
    const int ar = mma_a_row(lane);                                            \
    const int bc = mma_b_col(lane);                                            \
    const int cq = lane % 4;                                                   \
    const int kq = cq * 4;                                                     \
    const int cr = mma_c_row(lane);                                            \
    const int cc = mma_c_col(lane);                                            \
    const int wbase = warp * (NBLK) * 8;                                       \
                                                                               \
    /* The striped partition, which `mmqsr_*` measured worth 30% at four        \
       tokens. Same arithmetic here. */                                        \
    const int total = row_tiles * k_tiles;                                     \
    const int iters = (total + (int)gridDim.x - 1) / (int)gridDim.x;           \
    /* Laid out k-major, and measured against the transpose.                  \
                                                                              \
       The blocks resident together sit in one row group at different k, so    \
       they share the weight rows and each reads its own slice of the          \
       activations — and at 32 tokens the activations are the larger half,     \
       58.7 MiB against 31 for `ffn_gate`. Permuting the slice indices so that \
       concurrent blocks span row groups at the same k, which makes the        \
       activation slice the shared one, buys nothing: 1-block-wide shapes move \
       by less than the noise (`ffn_gate` 27.0 us either way) and the 2-wide   \
       ones get worse (`gate_up` 41.6 to 44.2). L2 does not catch it. */       \
    int flat = iters * (int)blockIdx.x;                                        \
    const int flat_end = min(total, flat + iters);                             \
                                                                               \
    /* `n * nb` blocks of 64 quant bytes, then `n * nb` scale pairs. Same      \
       total as the packed 68-byte blocks, and the quants are 16-byte aligned   \
       where the blocks are not — which is the point, since a lane's fragment   \
       is one `uint4`. */                                                      \
    const uint8_t* qbase = (const uint8_t*)wq;                                 \
    const __half2* sbase =                                                     \
        (const __half2*)(const void*)(qbase + (size_t)n * nb_total * 64);      \
    size_t browi[NBLK];                                                        \
    bool brow_ok[NBLK];                                                        \
    mma_c_f32 acc[NBLK][TILES];                                                \
                                                                               \
    if (x_valid < x_rows) {                                                    \
        const int per = (x_rows - x_valid) * (MMQ_XF_ROW / 4);                 \
        for (int i = tid; i < (STAGES) * per; i += nthreads) {                 \
            const int s = i / per;                                             \
            const int j = i % per;                                             \
            const int r = x_valid + j / (MMQ_XF_ROW / 4);                      \
            const int e = (j % (MMQ_XF_ROW / 4)) * 4;                          \
            *(uint32_t*)(void*)(xf + (s * x_rows + r) * MMQ_XF_STRIDE + e) = 0;\
        }                                                                      \
    }                                                                          \
                                                                               \
    while (flat < flat_end) {                                                  \
        const int nt = flat / k_tiles;                                         \
        const int kt_lo = flat % k_tiles;                                      \
        const int kt_hi = min(k_tiles, kt_lo + (flat_end - flat));             \
        const int row0 = nt * mrows;                                           \
                                                                               \
        _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {                   \
            const int r = row0 + wbase + j * 8 + bc;                           \
            brow_ok[j] = r < n;                                                \
            /* The row's byte offset, not its index: multiplying by the row  \
               stride inside the k-loop puts a 64-bit multiply on every weight \
               address and cost 13% when it was written that way. */          \
            browi[j] = brow_ok[j] ? (size_t)r * nb_total : 0;                  \
            _Pragma("unroll") for (int u = 0; u < (TILES); ++u) {              \
                _Pragma("unroll") for (int c = 0; c < 4; ++c)                  \
                    acc[j][u].x[c] = 0.0f;                                     \
            }                                                                  \
        }                                                                      \
                                                                               \
        _Pragma("unroll") for (int s = 0; s < (STAGES) - 1; ++s) {             \
            MMQ_XF_FETCH(s, kt_lo + s, kt_hi);                                 \
        }                                                                      \
                                                                               \
        /* One k-tile of weights ahead, held in registers.
                                                                              \
           The loads were issued between `mmq_bar_arrive` and `mmq_bar_sync`   \
           and consumed the instruction after — one barrier of latency to hide \
           a global read in. `mmq.cu`'s note that memory-level parallelism      \
           cannot be bought with registers here was measured on `mmqb_*`,       \
           which is four warps and 9.7 KB of *static* shared and so is bound    \
           by registers at 75 of them. This kernel is not: 34 KB of dynamic     \
           shared holds it to 2.9 blocks an SM, where its 50 registers would    \
           allow 5.1. The twelve registers this costs are inside that gap and   \
           buy a whole k-tile of overlap.                                       \
                                                                              \
           Issuing the loads before the `cp_async_wait` instead, which is free, \
           was measured first and is worth nothing: 98.9 ms a step against      \
           97.3. One barrier or two, the MMAs still wait on the same load. */  \
        uint4 yv4[2][NBLK], nv4[2][NBLK];                                      \
        __half2 ys2[2][NBLK], ym2[2][NBLK], ns2[2][NBLK], nm2[2][NBLK];        \
        MMQ_Y_LOADW(yv4, ys2, ym2, kt_lo, NBLK)                                      \
        for (int kt = kt_lo; kt < kt_hi; ++kt) {                               \
            const int pos = (kt - kt_lo) % (STAGES);                           \
            MMQ_CP_ASYNC_WAIT((STAGES) - 2);                                   \
            mmq_bar_arrive(nthreads);                                          \
            if (kt + 1 < kt_hi) MMQ_Y_LOADW(nv4, ns2, nm2, kt + 1, NBLK)             \
            mmq_bar_sync(nthreads);                                            \
            MMQ_XF_FETCH((pos + (STAGES) - 1) % (STAGES),                      \
                         kt + (STAGES) - 1, kt_hi);                            \
            const int8_t* xbuf = xf + pos * x_rows * MMQ_XF_STRIDE;            \
                                                                               \
            _Pragma("unroll") for (int blk = 0; blk < 2; ++blk) {              \
                const int kb = kt * 2 + blk;                                   \
                const bool kb_ok = kb < nb_total;                              \
                /* One scale and one zero per row per 128 weights, broadcast   \
                   into the halves the dequantization folds them into. */      \
                __half2 s2[NBLK], m2[NBLK];                                    \
                _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {           \
                    const int sr = row0 + wbase + j * 8 + bc;                  \
                    __half2 ds = __floats2half2_rn(0.0f, 0.0f);                \
                    (void)ds;                                                  \
                    s2[j] = ys2[blk][j];                                       \
                    m2[j] = ym2[blk][j];                                       \
                    (void)sr;                                                  \
                }                                                              \
                /* One 16-byte request for the four words this lane used   \
                   to fetch separately: the transpose put (run 0 low, run 0   \
                   high, run 1 low, run 1 high) side by side. */              \
                uint4 wv4[NBLK];                                              \
                _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {          \
                    wv4[j] = yv4[blk][j];                                     \
                }                                                              \
                _Pragma("unroll") for (int run = 0; run < 2; ++run) {          \
                    uint32_t v0[NBLK], v1[NBLK];                               \
                    _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {       \
                        v0[j] = run ? wv4[j].z : wv4[j].x;                     \
                        v1[j] = run ? wv4[j].w : wv4[j].y;                     \
                    }                                                          \
                    _Pragma("unroll") for (int h = 0; h < 2; ++h) {            \
                        const int g = blk * 4 + run + h * 2;                   \
                        /* Two MMAs a group: k is 16 here, not 32. */          \
                        _Pragma("unroll") for (int mm = 0; mm < 2; ++mm) {     \
                            mma_a_f16 a[TILES];                                \
                            _Pragma("unroll") for (int u = 0; u < (TILES);     \
                                                   ++u) {                      \
                                const int8_t* ap =                             \
                                    xbuf + (u * MMQ_M + ar) * MMQ_XF_STRIDE    \
                                    + g * 64 + mm * 32 + cq * 8;               \
                                const int8_t* aq = ap + 8 * MMQ_XF_STRIDE;     \
                                const uint2 lo =                               \
                                    *(const uint2*)(const void*)ap;            \
                                const uint2 hi =                               \
                                    *(const uint2*)(const void*)aq;            \
                                a[u].x[0] = lo.x;                              \
                                a[u].x[1] = hi.x;                              \
                                a[u].x[2] = lo.y;                              \
                                a[u].x[3] = hi.y;                              \
                            }                                                  \
                            _Pragma("unroll") for (int j = 0; j < (NBLK);      \
                                                   ++j) {                      \
                                mma_b_f16 b;                                   \
                                /* No `prmt`: the pack swapped bytes 1 and 2  \
                                   so `lop3`'s mask lands on the right pair    \
                                   and the second half is one shift away. */   \
                                mmq_deq4_f16_repacked(mm ? v1[j] : v0[j], h,   \
                                                      s2[j], m2[j], b.x);      \
                                _Pragma("unroll") for (int u = 0;              \
                                                       u < (TILES); ++u) {     \
                                    mma_f16(acc[j][u], a[u], b);               \
                                }                                              \
                            }                                                  \
                        }                                                      \
                    }                                                          \
                }                                                              \
            }                                                                  \
            /* Hand the prefetched tile over: twelve register moves at    \
               NBLK=1, against the thirty-two MMAs the iteration just ran. */  \
            _Pragma("unroll") for (int yb = 0; yb < 2; ++yb) {                 \
                _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {           \
                    yv4[yb][j] = nv4[yb][j];                                   \
                    ys2[yb][j] = ns2[yb][j];                                   \
                    ym2[yb][j] = nm2[yb][j];                                   \
                }                                                              \
            }                                                                  \
        }                                                                      \
                                                                               \
        MMQ_CP_ASYNC_WAIT(0);                                                  \
        __syncthreads();                                                       \
                                                                               \
        /* No epilogue: the accumulator is already the answer. */              \
        const bool whole = (kt_lo == 0) && (kt_hi == k_tiles);                 \
        _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {                   \
            const int orow = row0 + wbase + j * 8 + cc;                        \
            _Pragma("unroll") for (int u = 0; u < (TILES); ++u) {              \
                const int ot0 = tok0 + u * MMQ_M + cr;                         \
                const int ot1 = ot0 + 8;                                       \
                if (ot0 < n_tokens) {                                          \
                    if (orow < n)                                              \
                        MMQ_PUT2(whole, (size_t)ot0 * n + orow,                \
                                 acc[j][u].x[0]);                              \
                    if (orow + 1 < n)                                          \
                        MMQ_PUT2(whole, (size_t)ot0 * n + orow + 1,            \
                                 acc[j][u].x[1]);                              \
                }                                                              \
                if (ot1 < n_tokens) {                                          \
                    if (orow < n)                                              \
                        MMQ_PUT2(whole, (size_t)ot1 * n + orow,                \
                                 acc[j][u].x[2]);                              \
                    if (orow + 1 < n)                                          \
                        MMQ_PUT2(whole, (size_t)ot1 * n + orow + 1,            \
                                 acc[j][u].x[3]);                              \
                }                                                              \
            }                                                                  \
        }                                                                      \
        flat += kt_hi - kt_lo;                                                 \
    }
#define MMQ_Y_SET(SUFFIX, WARPS, NBLK, TILES, STAGES)                          \
    extern "C" __global__ __launch_bounds__((WARPS) * WARP_SIZE) void          \
    mmqy##SUFFIX##_q4_g128(float* __restrict__ out,                            \
                           const void* __restrict__ wv,                        \
                           const void* __restrict__ xv, int k, int n,          \
                           int n_tokens) {                                     \
        const block_q4_g128* wq = (const block_q4_g128*)wv;                    \
        MMQ_Y_BODY(WARPS, NBLK, TILES, STAGES)                                 \
    }

/* Depth three — two k-tiles of weights in flight instead of one — was written
   and measured and is not here. It costs 35 registers (66 to 101), which is two
   resident blocks against the 2.9 the 34 KB activation ring allows, and the two
   cancel: a step's matmuls take 91.6-93.4 ms against depth two's 90.7 on a
   Blackwell and 2% more on an A4000. Prefetching only the quants and reading the
   scale at its use point costs *more* — 93.2-94.2 — because the scale load lands
   on the critical path in front of the MMAs, and it does not even save the
   registers it was supposed to (64 against 66, and 100 once the third set is
   declared at all, which is what parameterizing the depth cost the default). */
MMQ_Y_SET(1w8s2, 8, 1, 1, 2)
MMQ_Y_SET(1w8s2_2, 8, 1, 2, 2)
MMQ_Y_SET(1w8s2_4, 8, 1, 4, 2)
/* Every shape constant in this file was fitted on an sm_86 card with 100 KiB
   of shared memory per SM. Blackwell has 228, so the trades that picked
   NBLK=1, eight warps and two stages are all re-openable there. */
MMQ_Y_SET(2w8s2, 8, 2, 1, 2)
MMQ_Y_SET(2w8s2_2, 8, 2, 2, 2)
MMQ_Y_SET(2w8s2_4, 8, 2, 4, 2)
MMQ_Y_SET(1w8s4, 8, 1, 1, 4)
MMQ_Y_SET(1w8s4_2, 8, 1, 2, 4)
MMQ_Y_SET(1w8s4_4, 8, 1, 4, 4)
MMQ_Y_SET(2w8s4, 8, 2, 1, 4)
MMQ_Y_SET(2w8s4_2, 8, 2, 2, 4)
MMQ_Y_SET(2w8s4_4, 8, 2, 4, 4)
MMQ_Y_SET(1w16s2, 16, 1, 1, 2)
MMQ_Y_SET(1w16s2_2, 16, 1, 2, 2)
MMQ_Y_SET(1w16s2_4, 16, 1, 4, 2)
/* Marlin's tile shape does not port to this body, and the register file says why.
   Its wide matrices run 256 rows a block with *four* warps
   (`thread_n_blocks` 16 at 256 threads); instantiated here as `8w4s2` that is
   **255 registers** — the hard cap, so the compiler spilled — and 777 us against
   `1w8s2`'s 222 on an A4000, 80 GB/s of weights against 281. `4w4s2` (128 rows,
   four warps) and `4w8s2` (256 rows, eight) both land on 215 registers, which is
   one resident block an SM, and measure 226 and 258 against 222.
 
   `1w8s2` is 100 registers and 19.5 KB of shared, which is what lets several
   blocks share an SM, and that is the shape this body is built around: many thin
   blocks, each holding one row group's accumulators. Marlin is built around the
   opposite — one fat block an SM, kept busy by a four-stage `cp.async` pipeline
   and warp-level scheduling. Its remaining 7% on `gate_up` lives in that choice,
   not in a parameter, which is the same conclusion the elimination table in
   `docs/catching-vllm.md` reaches from the other end. */
MMQ_Y_SET(4w8s2, 8, 4, 1, 2)
MMQ_Y_SET(4w8s2_2, 8, 4, 2, 2)
MMQ_Y_SET(4w8s2_4, 8, 4, 4, 2)
/* Sixteen warps, which the note above says is re-openable on a card with 228
   KiB of shared memory. Swept in the engine rather than on a warm microbench:
   at a batch of 32 `mmqy1w16s2` runs a step's matmuls in 96.1 ms against
   `mmqy1w8s2`'s 98.7 and `mmqy2w8s2`'s 99.4. */
MMQ_Y_SET(2w16s2, 16, 2, 1, 2)
MMQ_Y_SET(2w16s2_2, 16, 2, 2, 2)
MMQ_Y_SET(2w16s2_4, 16, 2, 4, 2)
MMQ_Y_SET(4w16s2, 16, 4, 1, 2)
MMQ_Y_SET(4w16s2_2, 16, 4, 2, 2)
MMQ_Y_SET(4w16s2_4, 16, 4, 4, 2)
MMQ_Y_SET(1w16s4, 16, 1, 1, 4)
MMQ_Y_SET(1w16s4_2, 16, 1, 2, 4)
MMQ_Y_SET(1w16s4_4, 16, 1, 4, 4)

// ---- the XOR swizzle, so `ldmatrix` finally pays -----------------------
//
// `ldmatrix` loads an A fragment in one instruction where the scalar gather
// takes two, and every attempt at it here measured 20-25% slower. The reason
// was the tile: a padded stride is conflict-free for one pattern at a time,
// and 544 bytes was chosen for the 8-byte gather — 8 words mod 32, which is
// exactly what two-ways `ldmatrix`.
//
// Marlin does not pad. `transform_a` (`marlin_template.h:638`) XORs the
// 16-byte chunk index within a row by `row % 8`, and the comment above it says
// the point is that *neither* reads nor writes conflict. The `cp.async` that
// fills the tile applies the same permutation, which it can because the
// destination address is ours to compute.
//
// The row is then exactly 512 bytes with no padding, and `ldmatrix` produces
// the *standard* fragment order — which is what `f32_to_f16_kperm` already
// arranges the activations for.
#define MMQ_XK_ROW (MMQ_K * 2)
#define MMQ_XK_STRIDE MMQ_XK_ROW

#define MMQ_XK_FETCH(BUF, TILE, LIMIT)                                         \
    do {                                                                       \
        const int _tl = (TILE);                                                \
        const bool _live = _tl < (LIMIT);                                      \
        for (int _i = tid; _i < x_valid * (MMQ_XK_ROW / 16); _i += nthreads) { \
            const int _r = _i / (MMQ_XK_ROW / 16);                             \
            const int _c = _i % (MMQ_XK_ROW / 16);                             \
            const bool _hit = _live && (_tl * MMQ_K + _c * 8 + 8 <= k);        \
            const size_t _off =                                                \
                ((size_t)(tok0 + _r) * k + (size_t)_tl * MMQ_K) * 2            \
                + (size_t)_c * 16;                                             \
            /* The swizzle: chunk `c` of row `r` lands at `c ^ (r % 8)`. */    \
            mmq_cp_async16(                                                    \
                xf + ((BUF) * x_rows + _r) * MMQ_XK_STRIDE                     \
                   + (_c ^ (_r & 7)) * 16,                                     \
                xbytes + (_hit ? _off : 0), _hit);                             \
        }                                                                      \
        MMQ_CP_ASYNC_FENCE();                                                  \
    } while (0)

#define MMQ_K_BODY(WARPS, NBLK, TILES, STAGES)                               \
    extern __shared__ __align__(16) int8_t xf[];                               \
                                                                               \
    const int tid = threadIdx.x;                                               \
    const int lane = tid % WARP_SIZE;                                          \
    const int warp = tid / WARP_SIZE;                                          \
    const int nthreads = (WARPS) * WARP_SIZE;                                  \
    const int mrows = (WARPS) * (NBLK) * 8;                                    \
    const int tok0 = blockIdx.y * (TILES) * MMQ_M;                             \
    const int nb_total = k / QK_G128;                                          \
    const int k_tiles = (k + MMQ_K - 1) / MMQ_K;                               \
    const int row_tiles = (n + mrows - 1) / mrows;                             \
    const int x_rows = (TILES) * MMQ_M;                                        \
    const int x_valid = min(x_rows, max(0, n_tokens - tok0));                  \
    const uint8_t* xbytes = (const uint8_t*)xv;                                \
                                                                               \
    const int ar = mma_a_row(lane);                                            \
    const int bc = mma_b_col(lane);                                            \
    const int cq = lane % 4;                                                   \
    const int kq = cq * 4;                                                     \
    const int cr = mma_c_row(lane);                                            \
    const int cc = mma_c_col(lane);                                            \
    const int wbase = warp * (NBLK) * 8;                                       \
                                                                               \
    /* The striped partition, which `mmqsr_*` measured worth 30% at four        \
       tokens. Same arithmetic here. */                                        \
    const int total = row_tiles * k_tiles;                                     \
    const int iters = (total + (int)gridDim.x - 1) / (int)gridDim.x;           \
    int flat = iters * (int)blockIdx.x;                                        \
    const int flat_end = min(total, flat + iters);                             \
                                                                               \
    /* `n * nb` blocks of 64 quant bytes, then `n * nb` scale pairs. Same      \
       total as the packed 68-byte blocks, and the quants are 16-byte aligned   \
       where the blocks are not — which is the point, since a lane's fragment   \
       is one `uint4`. */                                                      \
    const uint8_t* qbase = (const uint8_t*)wq;                                 \
    const __half2* sbase =                                                     \
        (const __half2*)(const void*)(qbase + (size_t)n * nb_total * 64);      \
    size_t browi[NBLK];                                                        \
    bool brow_ok[NBLK];                                                        \
    mma_c_f32 acc[NBLK][TILES];                                                \
                                                                               \
    if (x_valid < x_rows) {                                                    \
        const int per = (x_rows - x_valid) * (MMQ_XK_ROW / 4);                 \
        for (int i = tid; i < (STAGES) * per; i += nthreads) {                 \
            const int s = i / per;                                             \
            const int j = i % per;                                             \
            const int r = x_valid + j / (MMQ_XK_ROW / 4);                      \
            const int e = (j % (MMQ_XK_ROW / 4)) * 4;                          \
            *(uint32_t*)(void*)(xf + (s * x_rows + r) * MMQ_XK_STRIDE + e) = 0;\
        }                                                                      \
    }                                                                          \
                                                                               \
    while (flat < flat_end) {                                                  \
        const int nt = flat / k_tiles;                                         \
        const int kt_lo = flat % k_tiles;                                      \
        const int kt_hi = min(k_tiles, kt_lo + (flat_end - flat));             \
        const int row0 = nt * mrows;                                           \
                                                                               \
        _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {                   \
            const int r = row0 + wbase + j * 8 + bc;                           \
            brow_ok[j] = r < n;                                                \
            /* The row's byte offset, not its index: multiplying by the row  \
               stride inside the k-loop puts a 64-bit multiply on every weight \
               address and cost 13% when it was written that way. */          \
            browi[j] = brow_ok[j] ? (size_t)r * nb_total : 0;                  \
            _Pragma("unroll") for (int u = 0; u < (TILES); ++u) {              \
                _Pragma("unroll") for (int c = 0; c < 4; ++c)                  \
                    acc[j][u].x[c] = 0.0f;                                     \
            }                                                                  \
        }                                                                      \
                                                                               \
        _Pragma("unroll") for (int s = 0; s < (STAGES) - 1; ++s) {             \
            MMQ_XK_FETCH(s, kt_lo + s, kt_hi);                                 \
        }                                                                      \
                                                                               \
        for (int kt = kt_lo; kt < kt_hi; ++kt) {                               \
            const int pos = (kt - kt_lo) % (STAGES);                           \
            MMQ_CP_ASYNC_WAIT((STAGES) - 2);                                   \
            /* Arrive, then load this k-tile's weights — global, and needing   \
               nothing the barrier publishes — then wait. `__syncthreads` is    \
               arrive and wait in one instruction with nothing able to happen   \
               between them. */                                                \
            mmq_bar_arrive(nthreads);                                          \
            uint4 yv4[2][NBLK];                                                \
            __half2 ys2[2][NBLK], ym2[2][NBLK];                                \
            _Pragma("unroll") for (int yb = 0; yb < 2; ++yb) {                 \
                const int ykb = kt * 2 + yb;                                   \
                const bool yok = ykb < nb_total;                               \
                _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {           \
                    yv4[yb][j] = make_uint4(0, 0, 0, 0);                       \
                    __half2 yds = __floats2half2_rn(0.0f, 0.0f);               \
                    if (brow_ok[j] && yok) {                                   \
                        yv4[yb][j] = *(const uint4*)(const void*)(             \
                            qbase + (browi[j] + ykb) * 64 + cq * 16);          \
                        yds = sbase[browi[j] + ykb];                           \
                    }                                                          \
                    ys2[yb][j] = __float2half2_rn(__low2float(yds));           \
                    ym2[yb][j] = __float2half2_rn(-__high2float(yds));         \
                }                                                              \
            }                                                                  \
            mmq_bar_sync(nthreads);                                            \
            MMQ_XK_FETCH((pos + (STAGES) - 1) % (STAGES),                      \
                         kt + (STAGES) - 1, kt_hi);                            \
            const int8_t* xbuf = xf + pos * x_rows * MMQ_XK_STRIDE;            \
                                                                               \
            _Pragma("unroll") for (int blk = 0; blk < 2; ++blk) {              \
                const int kb = kt * 2 + blk;                                   \
                const bool kb_ok = kb < nb_total;                              \
                /* One scale and one zero per row per 128 weights, broadcast   \
                   into the halves the dequantization folds them into. */      \
                __half2 s2[NBLK], m2[NBLK];                                    \
                _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {           \
                    const int sr = row0 + wbase + j * 8 + bc;                  \
                    __half2 ds = __floats2half2_rn(0.0f, 0.0f);                \
                    (void)ds;                                                  \
                    s2[j] = ys2[blk][j];                                       \
                    m2[j] = ym2[blk][j];                                       \
                    (void)sr;                                                  \
                }                                                              \
                /* One 16-byte request for the four words this lane used   \
                   to fetch separately: the transpose put (run 0 low, run 0   \
                   high, run 1 low, run 1 high) side by side. */              \
                uint4 wv4[NBLK];                                              \
                _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {          \
                    wv4[j] = yv4[blk][j];                                     \
                }                                                              \
                _Pragma("unroll") for (int run = 0; run < 2; ++run) {          \
                    uint32_t v0[NBLK], v1[NBLK];                               \
                    _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {       \
                        v0[j] = run ? wv4[j].z : wv4[j].x;                     \
                        v1[j] = run ? wv4[j].w : wv4[j].y;                     \
                    }                                                          \
                    _Pragma("unroll") for (int h = 0; h < 2; ++h) {            \
                        const int g = blk * 4 + run + h * 2;                   \
                        /* Two MMAs a group: k is 16 here, not 32. */          \
                        _Pragma("unroll") for (int mm = 0; mm < 2; ++mm) {     \
                            mma_a_f16 a[TILES];                                \
                            _Pragma("unroll") for (int u = 0; u < (TILES);     \
                                                   ++u) {                      \
                                /* One instruction, conflict-free because the  \
                                   tile is swizzled rather than padded. */     \
                                mma_a_s8 t;                                    \
                                ldmatrix_a_swz(t,                              \
                                    xbuf + u * MMQ_M * MMQ_XK_STRIDE,          \
                                    MMQ_XK_STRIDE, g * 4 + mm * 2);            \
                                a[u].x[0] = (unsigned)t.x[0];                  \
                                a[u].x[1] = (unsigned)t.x[1];                  \
                                a[u].x[2] = (unsigned)t.x[2];                  \
                                a[u].x[3] = (unsigned)t.x[3];                  \
                            }                                                  \
                            _Pragma("unroll") for (int j = 0; j < (NBLK);      \
                                                   ++j) {                      \
                                mma_b_f16 b;                                   \
                                /* No `prmt`: the pack swapped bytes 1 and 2  \
                                   so `lop3`'s mask lands on the right pair    \
                                   and the second half is one shift away. */   \
                                mmq_deq4_f16_repacked(mm ? v1[j] : v0[j], h,   \
                                                      s2[j], m2[j], b.x);      \
                                _Pragma("unroll") for (int u = 0;              \
                                                       u < (TILES); ++u) {     \
                                    mma_f16(acc[j][u], a[u], b);               \
                                }                                              \
                            }                                                  \
                        }                                                      \
                    }                                                          \
                }                                                              \
            }                                                                  \
        }                                                                      \
                                                                               \
        MMQ_CP_ASYNC_WAIT(0);                                                  \
        __syncthreads();                                                       \
                                                                               \
        /* No epilogue: the accumulator is already the answer. */              \
        const bool whole = (kt_lo == 0) && (kt_hi == k_tiles);                 \
        _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {                   \
            const int orow = row0 + wbase + j * 8 + cc;                        \
            _Pragma("unroll") for (int u = 0; u < (TILES); ++u) {              \
                const int ot0 = tok0 + u * MMQ_M + cr;                         \
                const int ot1 = ot0 + 8;                                       \
                if (ot0 < n_tokens) {                                          \
                    if (orow < n)                                              \
                        MMQ_PUT2(whole, (size_t)ot0 * n + orow,                \
                                 acc[j][u].x[0]);                              \
                    if (orow + 1 < n)                                          \
                        MMQ_PUT2(whole, (size_t)ot0 * n + orow + 1,            \
                                 acc[j][u].x[1]);                              \
                }                                                              \
                if (ot1 < n_tokens) {                                          \
                    if (orow < n)                                              \
                        MMQ_PUT2(whole, (size_t)ot1 * n + orow,                \
                                 acc[j][u].x[2]);                              \
                    if (orow + 1 < n)                                          \
                        MMQ_PUT2(whole, (size_t)ot1 * n + orow + 1,            \
                                 acc[j][u].x[3]);                              \
                }                                                              \
            }                                                                  \
        }                                                                      \
        flat += kt_hi - kt_lo;                                                 \
    }
#define MMQ_K_SET(SUFFIX, WARPS, NBLK, TILES, STAGES)                          \
    extern "C" __global__ __launch_bounds__((WARPS) * WARP_SIZE) void          \
    mmqk##SUFFIX##_q4_g128(float* __restrict__ out,                            \
                           const void* __restrict__ wv,                        \
                           const void* __restrict__ xv, int k, int n,          \
                           int n_tokens) {                                     \
        const block_q4_g128* wq = (const block_q4_g128*)wv;                    \
        MMQ_K_BODY(WARPS, NBLK, TILES, STAGES)                                 \
    }

MMQ_K_SET(1w8s2, 8, 1, 1, 2)
MMQ_K_SET(1w8s2_2, 8, 1, 2, 2)
MMQ_K_SET(1w8s2_4, 8, 1, 4, 2)

// ---- the weights through `cp.async` too --------------------------------
//
// The last piece of Marlin this port had not taken. `fetch_to_shared`
// (`marlin_template.h:742`) stages B with `cp_async4` alongside A and the inner
// loop reads it as an `I4` out of shared; ours reads it straight from global.
//
// It buys no reuse — a warp's weight rows are its own — and that is why it was
// skipped: `mmqd_*` measured the shared *expansion* at 0% years of this file
// ago. What it buys is the pipeline. A global load hidden behind one split
// barrier is hidden for one barrier's worth of latency; a `cp.async` issued
// `STAGES-1` k-tiles ahead is hidden for the whole ring. Every change that won
// on the A4000 did exactly this to some operand, and the weights are the last
// one still synchronous.
//
// Two things to get right. The row stride is padded to 80 bytes rather than
// packed at 64: a lane reads 16 bytes at `row * stride + (lane % 4) * 16`, and
// at a 64-byte stride the eight rows of a warp land on two bank groups — a
// four-way conflict. 80 bytes is 20 words, which spreads them over all 32. And
// the scales stay on the direct path: four bytes per row per k-tile, whose
// global stride is `nb * 4` and so cannot be copied contiguously.
#define MMQ_BSH_STRIDE 80

#define MMQ_CB_FETCH(NBLK, WARPS, BUF, TILE, LIMIT)                            \
    do {                                                                       \
        const int _tl = (TILE);                                                \
        const bool _live = _tl < (LIMIT);                                      \
        const int _rows = (WARPS) * (NBLK) * 8;                                \
        /* Two 128-weight blocks a k-tile, four 16-byte chunks a row. */       \
        for (int _i = tid; _i < _rows * 8; _i += nthreads) {                   \
            const int _r = _i / 8;                                             \
            const int _b = (_i % 8) / 4;                                       \
            const int _c = _i % 4;                                             \
            const int _gr = row0 + _r;                                         \
            const int _kb = _tl * 2 + _b;                                      \
            const bool _hit = _live && _gr < n && _kb < nb_total;              \
            const size_t _off =                                                \
                ((size_t)_gr * nb_total + _kb) * 64 + (size_t)_c * 16;         \
            mmq_cp_async16(                                                    \
                bsh + ((BUF) * 2 + _b) * _rows * MMQ_BSH_STRIDE                \
                    + _r * MMQ_BSH_STRIDE + _c * 16,                           \
                qbase + (_hit ? _off : 0), _hit);                              \
        }                                                                      \
        MMQ_CP_ASYNC_FENCE();                                                  \
    } while (0)

#define MMQ_C_BODY(WARPS, NBLK, TILES, STAGES)                               \
    extern __shared__ __align__(16) int8_t xf[];                               \
    /* The weight ring, after the activation ring in the same allocation. */   \
    int8_t* bsh = xf + (STAGES) * (TILES) * MMQ_M * MMQ_XF_STRIDE;             \
                                                                               \
    const int tid = threadIdx.x;                                               \
    const int lane = tid % WARP_SIZE;                                          \
    const int warp = tid / WARP_SIZE;                                          \
    const int nthreads = (WARPS) * WARP_SIZE;                                  \
    const int mrows = (WARPS) * (NBLK) * 8;                                    \
    const int tok0 = blockIdx.y * (TILES) * MMQ_M;                             \
    const int nb_total = k / QK_G128;                                          \
    const int k_tiles = (k + MMQ_K - 1) / MMQ_K;                               \
    const int row_tiles = (n + mrows - 1) / mrows;                             \
    const int x_rows = (TILES) * MMQ_M;                                        \
    const int x_valid = min(x_rows, max(0, n_tokens - tok0));                  \
    const uint8_t* xbytes = (const uint8_t*)xv;                                \
                                                                               \
    const int ar = mma_a_row(lane);                                            \
    const int bc = mma_b_col(lane);                                            \
    const int cq = lane % 4;                                                   \
    const int kq = cq * 4;                                                     \
    const int cr = mma_c_row(lane);                                            \
    const int cc = mma_c_col(lane);                                            \
    const int wbase = warp * (NBLK) * 8;                                       \
                                                                               \
    /* The striped partition, which `mmqsr_*` measured worth 30% at four        \
       tokens. Same arithmetic here. */                                        \
    const int total = row_tiles * k_tiles;                                     \
    const int iters = (total + (int)gridDim.x - 1) / (int)gridDim.x;           \
    int flat = iters * (int)blockIdx.x;                                        \
    const int flat_end = min(total, flat + iters);                             \
                                                                               \
    /* `n * nb` blocks of 64 quant bytes, then `n * nb` scale pairs. Same      \
       total as the packed 68-byte blocks, and the quants are 16-byte aligned   \
       where the blocks are not — which is the point, since a lane's fragment   \
       is one `uint4`. */                                                      \
    const uint8_t* qbase = (const uint8_t*)wq;                                 \
    const __half2* sbase =                                                     \
        (const __half2*)(const void*)(qbase + (size_t)n * nb_total * 64);      \
    size_t browi[NBLK];                                                        \
    bool brow_ok[NBLK];                                                        \
    mma_c_f32 acc[NBLK][TILES];                                                \
                                                                               \
    if (x_valid < x_rows) {                                                    \
        const int per = (x_rows - x_valid) * (MMQ_XF_ROW / 4);                 \
        for (int i = tid; i < (STAGES) * per; i += nthreads) {                 \
            const int s = i / per;                                             \
            const int j = i % per;                                             \
            const int r = x_valid + j / (MMQ_XF_ROW / 4);                      \
            const int e = (j % (MMQ_XF_ROW / 4)) * 4;                          \
            *(uint32_t*)(void*)(xf + (s * x_rows + r) * MMQ_XF_STRIDE + e) = 0;\
        }                                                                      \
    }                                                                          \
                                                                               \
    while (flat < flat_end) {                                                  \
        const int nt = flat / k_tiles;                                         \
        const int kt_lo = flat % k_tiles;                                      \
        const int kt_hi = min(k_tiles, kt_lo + (flat_end - flat));             \
        const int row0 = nt * mrows;                                           \
                                                                               \
        _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {                   \
            const int r = row0 + wbase + j * 8 + bc;                           \
            brow_ok[j] = r < n;                                                \
            /* The row's byte offset, not its index: multiplying by the row  \
               stride inside the k-loop puts a 64-bit multiply on every weight \
               address and cost 13% when it was written that way. */          \
            browi[j] = brow_ok[j] ? (size_t)r * nb_total : 0;                  \
            _Pragma("unroll") for (int u = 0; u < (TILES); ++u) {              \
                _Pragma("unroll") for (int c = 0; c < 4; ++c)                  \
                    acc[j][u].x[c] = 0.0f;                                     \
            }                                                                  \
        }                                                                      \
                                                                               \
        _Pragma("unroll") for (int s = 0; s < (STAGES) - 1; ++s) {             \
            MMQ_XF_FETCH(s, kt_lo + s, kt_hi);                                 \
            MMQ_CB_FETCH(NBLK, WARPS, s, kt_lo + s, kt_hi);                    \
        }                                                                      \
                                                                               \
        for (int kt = kt_lo; kt < kt_hi; ++kt) {                               \
            const int pos = (kt - kt_lo) % (STAGES);                           \
            MMQ_CP_ASYNC_WAIT((STAGES) - 2);                                   \
            /* Arrive, then load this k-tile's weights — global, and needing   \
               nothing the barrier publishes — then wait. `__syncthreads` is    \
               arrive and wait in one instruction with nothing able to happen   \
               between them. */                                                \
            /* The scales are the only operand still coming from global —   \
               four bytes a row a k-tile, whose stride is `nb * 4` and so      \
               cannot be copied contiguously — so they are what the split      \
               barrier now has left to hide. */                                \
            mmq_bar_arrive(nthreads);                                          \
            uint4 yv4[2][NBLK];                                                \
            __half2 ys2[2][NBLK], ym2[2][NBLK];                                \
            _Pragma("unroll") for (int yb = 0; yb < 2; ++yb) {                 \
                const int ykb = kt * 2 + yb;                                   \
                const bool yok = ykb < nb_total;                               \
                _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {           \
                    __half2 yds = __floats2half2_rn(0.0f, 0.0f);               \
                    if (brow_ok[j] && yok) yds = sbase[browi[j] + ykb];        \
                    ys2[yb][j] = __float2half2_rn(__low2float(yds));           \
                    ym2[yb][j] = __float2half2_rn(-__high2float(yds));         \
                }                                                              \
            }                                                                  \
            mmq_bar_sync(nthreads);                                            \
            /* Only now: these bytes were written by other threads' copies,    \
               and `cp.async.wait` is a per-thread guarantee. */               \
            _Pragma("unroll") for (int yb = 0; yb < 2; ++yb) {                 \
                _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {           \
                    yv4[yb][j] = *(const uint4*)(const void*)(                 \
                        bsh + (pos * 2 + yb) * mrows * MMQ_BSH_STRIDE          \
                            + (wbase + j * 8 + bc) * MMQ_BSH_STRIDE            \
                            + cq * 16);                                        \
                }                                                              \
            }                                                                  \
            MMQ_XF_FETCH((pos + (STAGES) - 1) % (STAGES),                      \
                         kt + (STAGES) - 1, kt_hi);                            \
            MMQ_CB_FETCH(NBLK, WARPS, (pos + (STAGES) - 1) % (STAGES),         \
                         kt + (STAGES) - 1, kt_hi);                            \
            const int8_t* xbuf = xf + pos * x_rows * MMQ_XF_STRIDE;            \
                                                                               \
            _Pragma("unroll") for (int blk = 0; blk < 2; ++blk) {              \
                const int kb = kt * 2 + blk;                                   \
                const bool kb_ok = kb < nb_total;                              \
                /* One scale and one zero per row per 128 weights, broadcast   \
                   into the halves the dequantization folds them into. */      \
                __half2 s2[NBLK], m2[NBLK];                                    \
                _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {           \
                    const int sr = row0 + wbase + j * 8 + bc;                  \
                    __half2 ds = __floats2half2_rn(0.0f, 0.0f);                \
                    (void)ds;                                                  \
                    s2[j] = ys2[blk][j];                                       \
                    m2[j] = ym2[blk][j];                                       \
                    (void)sr;                                                  \
                }                                                              \
                /* One 16-byte request for the four words this lane used   \
                   to fetch separately: the transpose put (run 0 low, run 0   \
                   high, run 1 low, run 1 high) side by side. */              \
                uint4 wv4[NBLK];                                              \
                _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {          \
                    wv4[j] = yv4[blk][j];                                     \
                }                                                              \
                _Pragma("unroll") for (int run = 0; run < 2; ++run) {          \
                    uint32_t v0[NBLK], v1[NBLK];                               \
                    _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {       \
                        v0[j] = run ? wv4[j].z : wv4[j].x;                     \
                        v1[j] = run ? wv4[j].w : wv4[j].y;                     \
                    }                                                          \
                    _Pragma("unroll") for (int h = 0; h < 2; ++h) {            \
                        const int g = blk * 4 + run + h * 2;                   \
                        /* Two MMAs a group: k is 16 here, not 32. */          \
                        _Pragma("unroll") for (int mm = 0; mm < 2; ++mm) {     \
                            mma_a_f16 a[TILES];                                \
                            _Pragma("unroll") for (int u = 0; u < (TILES);     \
                                                   ++u) {                      \
                                const int8_t* ap =                             \
                                    xbuf + (u * MMQ_M + ar) * MMQ_XF_STRIDE    \
                                    + g * 64 + mm * 32 + cq * 8;               \
                                const int8_t* aq = ap + 8 * MMQ_XF_STRIDE;     \
                                const uint2 lo =                               \
                                    *(const uint2*)(const void*)ap;            \
                                const uint2 hi =                               \
                                    *(const uint2*)(const void*)aq;            \
                                a[u].x[0] = lo.x;                              \
                                a[u].x[1] = hi.x;                              \
                                a[u].x[2] = lo.y;                              \
                                a[u].x[3] = hi.y;                              \
                            }                                                  \
                            _Pragma("unroll") for (int j = 0; j < (NBLK);      \
                                                   ++j) {                      \
                                mma_b_f16 b;                                   \
                                /* No `prmt`: the pack swapped bytes 1 and 2  \
                                   so `lop3`'s mask lands on the right pair    \
                                   and the second half is one shift away. */   \
                                mmq_deq4_f16_repacked(mm ? v1[j] : v0[j], h,   \
                                                      s2[j], m2[j], b.x);      \
                                _Pragma("unroll") for (int u = 0;              \
                                                       u < (TILES); ++u) {     \
                                    mma_f16(acc[j][u], a[u], b);               \
                                }                                              \
                            }                                                  \
                        }                                                      \
                    }                                                          \
                }                                                              \
            }                                                                  \
        }                                                                      \
                                                                               \
        MMQ_CP_ASYNC_WAIT(0);                                                  \
        __syncthreads();                                                       \
                                                                               \
        /* No epilogue: the accumulator is already the answer. */              \
        const bool whole = (kt_lo == 0) && (kt_hi == k_tiles);                 \
        _Pragma("unroll") for (int j = 0; j < (NBLK); ++j) {                   \
            const int orow = row0 + wbase + j * 8 + cc;                        \
            _Pragma("unroll") for (int u = 0; u < (TILES); ++u) {              \
                const int ot0 = tok0 + u * MMQ_M + cr;                         \
                const int ot1 = ot0 + 8;                                       \
                if (ot0 < n_tokens) {                                          \
                    if (orow < n)                                              \
                        MMQ_PUT2(whole, (size_t)ot0 * n + orow,                \
                                 acc[j][u].x[0]);                              \
                    if (orow + 1 < n)                                          \
                        MMQ_PUT2(whole, (size_t)ot0 * n + orow + 1,            \
                                 acc[j][u].x[1]);                              \
                }                                                              \
                if (ot1 < n_tokens) {                                          \
                    if (orow < n)                                              \
                        MMQ_PUT2(whole, (size_t)ot1 * n + orow,                \
                                 acc[j][u].x[2]);                              \
                    if (orow + 1 < n)                                          \
                        MMQ_PUT2(whole, (size_t)ot1 * n + orow + 1,            \
                                 acc[j][u].x[3]);                              \
                }                                                              \
            }                                                                  \
        }                                                                      \
        flat += kt_hi - kt_lo;                                                 \
    }
/* The weight ring, measured at last, and it loses on both cards.

   The elimination table in `docs/catching-vllm.md` leaves one mechanism for the
   GEMM's missing 20%: the kernel has about 276 KB in flight per SM where
   1345 GB/s at ~700 ns of latency wants 940 KB, and the depth-three *register*
   prefetch that would buy more costs 35 registers and loses. Staging the weights
   through shared with `cp.async` buys in-flight bytes without registers, which is
   what this family does and what Marlin does.

   It was unreachable from the model until the `mmqc` prefix was accepted in
   `mmq_f16_variant_for` — measuring it before that silently ran the integer
   fallback — so these are its first numbers on real shapes (us a call, 32
   tokens):

                     A4000            Blackwell
     mmqy1w8s2       51.4 / 222.5     16.7 / 53.6     qkv / gate_up
     mmqc1w8s2       59.7 / 269.8     18.9 / --       16% and 21% slower
     mmqc1w8s4       refused          refused         110592 B of shared

   Two stages of weights cost more than the register pressure they relieve, and
   four stages do not fit at two token tiles: the per-block shared limit is 100 KB
   and the ring plus the activation ring wants 110.

   At *one* token tile the activation ring halves and every depth fits, which is
   the sweep that settles the mechanism rather than one point of it (us a call,
   16 tokens, `gate_up` / `qkv`):

                    A4000            Blackwell gate_up
     mmqy1w8s2      186.8 / 49.4     48.5    (the register path)
     mmqc1w8s2      185.4 / 57.7     50.7
     mmqc1w8s3      242.7 / 88.1     77.9
     mmqc1w8s4      254.9 / 96.3     84.4

   Depth is monotonically worse on both cards. So it is not that the ring is
   *shallow* — a deeper one does not buy the in-flight bytes the latency count
   says are missing, it buys occupancy loss. Both places one can hold bytes,
   registers and shared, are the same resource seen twice.

   Which is the whole answer to the missing 20%, from the third side. In-flight
   bytes are bought with registers or with shared, both of them are occupancy, and
   940 KB an SM is not available either way at any useful block count. TMA with
   warp specialization is the way out because it moves bytes without holding
   registers *and* lets one fat block use the SM's whole shared budget — which is
   what CUTLASS and Marlin are built around, and is a different kernel. */
#define MMQ_C_SET(SUFFIX, WARPS, NBLK, TILES, STAGES)                          \
    extern "C" __global__ __launch_bounds__((WARPS) * WARP_SIZE) void          \
    mmqc##SUFFIX##_q4_g128(float* __restrict__ out,                            \
                           const void* __restrict__ wv,                        \
                           const void* __restrict__ xv, int k, int n,          \
                           int n_tokens) {                                     \
        const block_q4_g128* wq = (const block_q4_g128*)wv;                    \
        MMQ_C_BODY(WARPS, NBLK, TILES, STAGES)                                 \
    }

MMQ_C_SET(1w8s2, 8, 1, 1, 2)
MMQ_C_SET(1w8s2_2, 8, 1, 2, 2)
MMQ_C_SET(1w8s3, 8, 1, 1, 3)
MMQ_C_SET(1w8s4, 8, 1, 1, 4)
MMQ_C_SET(1w8s4_2, 8, 1, 2, 4)
MMQ_C_SET(1w8s2_4, 8, 1, 4, 2)
MMQ_C_SET(4w8s2, 8, 4, 1, 2)
MMQ_C_SET(2w8s2, 8, 2, 1, 2)
MMQ_C_SET(2w8s4, 8, 2, 1, 4)
MMQ_C_SET(4w8s4, 8, 4, 1, 4)
MMQ_C_SET(4w8s2_2, 8, 4, 2, 2)
MMQ_C_SET(4w8s2_4, 8, 4, 4, 2)

// ---- the tensor cores are not the constraint ----------------------------
//
// `mmqnm_*` is `mmqf_*` with `mma_f16` deleted and the accumulator fed by a
// couple of ALU ops on the same operands, so every load, every dequantization
// and every barrier survives and the tensor cores do no work at all.
//
//                    ffn_gate  ffn_down  attn_q
//   mmqf1w8s2  @ 8t     332       342      262
//   mmqnm1w8s2 @ 8t     331       342      262
//   mmqf1w8s2  @32t     215       227      172
//   mmqnm1w8s2 @32t     215       227      173
//
// Identical to the digit. The MMAs are free — which settles what the 32-token
// deficit is not, and leaves one thing it can be.
//
// Per block per k-tile, this shape stages 32 tokens x 256 k x 2 bytes = 16 KB
// of activations against 64 rows x 256 k x 0.5 bytes = 8 KB of weights. The
// ratio is `4 * tokens / rows`: 0.5 at eight tokens, where the kernel sits at
// 97% of the weight-read ceiling, and 2.0 at thirty-two, where it sits at 63%.
// At 32 tokens this kernel moves twice as many activation bytes as weight
// bytes, and the weight ceiling stopped being the thing to measure against.
//
// The obvious fix — more weight rows per block, so the activation staging
// amortizes further — has been tried both ways it can be had, and both lose:
// NBLK=2 at eight warps measures 155 GB/s against 204, and sixteen warps at
// NBLK=1 measures 169. The first spends registers, the second spends blocks
// per SM. So the ratio is real and the two levers that move it are both worse
// than the ratio, which is where this stands.

#define MMQ_F16NOMMA_SET(SUFFIX, WARPS, NBLK, TILES, STAGES)                   \
    extern "C" __global__ __launch_bounds__((WARPS) * WARP_SIZE) void          \
    mmqnm##SUFFIX##_q4_g128(float* __restrict__ out,                           \
                            const void* __restrict__ wv,                       \
                            const void* __restrict__ xv, int k, int n,         \
                            int n_tokens) {                                    \
        const block_q4_g128* wq = (const block_q4_g128*)wv;                    \
        MMQ_F16NOMMA_BODY(WARPS, NBLK, TILES, STAGES)                          \
    }

MMQ_F16NOMMA_SET(1w8s2, 8, 1, 1, 2)
MMQ_F16NOMMA_SET(1w8s2_2, 8, 1, 2, 2)

#define MMQ_F16_SET(SUFFIX, WARPS, NBLK, TILES, STAGES)                        \
    extern "C" __global__ __launch_bounds__((WARPS) * WARP_SIZE) void          \
    mmqf##SUFFIX##_q4_g128(float* __restrict__ out,                            \
                           const void* __restrict__ wv,                        \
                           const void* __restrict__ xv, int k, int n,          \
                           int n_tokens) {                                     \
        const block_q4_g128* wq = (const block_q4_g128*)wv;                    \
        MMQ_F16_BODY(WARPS, NBLK, TILES, STAGES)                               \
    }

// Measured, against `mmqsr_*` — the same partition and the same pipeline with
// integer operands (tok/s, AWQ 8B, 256 tokens of history):
//
//                   batch 4    8    16    32
//   mmqsr2w4s4        167    325   583   743
//   mmqf2w4s2         173    330   564   687
//   mmqf2w4s4         126    256   498   558
//   mmqf4w4s4         146    288   517   661
//   mmqf2w2s4          85    176   367   375
//
// 3% up at four tokens, 7% down at thirty-two. The porting order calls this
// step the largest of the three and it is worth about nothing, which makes it
// the sixth structural change on this kernel to measure nothing.
//
// Half of that is occupancy and half of it is not, and the occupancy half is
// the part that can be checked. Two stages beat four here by 23% at 32 tokens,
// where in `mmqa_*`/`mmqr_*` four beat two; a half is 2 bytes against Q8_1's
// 1.125, so four stages cost 34.8 KB rather than 19.5, and `kernel_registers`
// puts that at 2 resident blocks per SM against 5.
//
// But the headline comparison is *not* occupancy, and the same probe says so:
// `mmqf2w4s2` runs 80 registers and 17.4 KB, which is 5 blocks per SM against
// `mmqsr2w4s4`'s 4 — more resident work, and still slower. What is left is
// what f16 costs per byte of weight: k per MMA halves, so twice the MMAs and
// twice the shared-memory traffic on the activation side for the same
// arithmetic. No measurement here separates those two, so this stays a
// description of the trade rather than an explanation of it.
//
// What is not in doubt is the sign. The epilogue this step removes was never
// the constraint — the same sentence as `mmqd` removing the weight staging for
// 0%, `mmqp` removing the per-group scaling for 0%, and `mmqx` widening the
// register tile for 0%.
//
// ---- and then it was not a negative result ------------------------------
//
// Everything above was measured at NBLK=2, which is the only shape this file
// had when the f16 path was written. `mmql_*` later established that NBLK=1
// wins — a wide register tile costs registers, registers cost resident blocks,
// and this kernel wants warps — and re-measuring the f16 path at the narrow
// shape reverses the conclusion (GB/s of weights on `ffn_gate`):
//
//                       8 tokens   32 tokens
//   mmql1w4s2d2 (int)      299        187
//   mmqf2w4s2 (f16)        234        132     the shape measured first
//   mmqf1w8s2 (f16)        308        204
//   mmqne1w4s2d2           338        218     the integer kernel with its
//                                             epilogue replaced by a constant
//
// So the epilogue *was* worth removing — 17% by the probe — and the first cut
// of this path spent more than that on a register tile it did not need. Eight
// warps rather than four, because halving k per MMA doubles the MMAs and the
// shape wants the issue slots back.
//
// It does not reach the probe's 218, and the gap is the other half of the
// trade the note above describes: k per MMA is 16 here against 32, so the same
// arithmetic costs twice the MMAs and twice the A-fragment traffic.
//
// The lesson is about method rather than about f16: a shape parameter that was
// never swept turned a 17% win into a 7% loss, and the conclusion sat in this
// file as settled for as long as it took to sweep it.

/* Named like the rest: mmqf<nblk>w<warps>s<stages>[_<tiles>]. */
MMQ_F16_SET(2w4s4, 4, 2, 1, 4)
MMQ_F16_SET(2w4s4_2, 4, 2, 2, 4)
MMQ_F16_SET(2w4s2, 4, 2, 1, 2)
MMQ_F16_SET(2w4s2_2, 4, 2, 2, 2)
MMQ_F16_SET(4w4s4, 4, 4, 1, 4)
MMQ_F16_SET(4w4s4_2, 4, 4, 2, 4)
MMQ_F16_SET(2w2s4, 2, 2, 1, 4)
MMQ_F16_SET(2w2s4_2, 2, 2, 2, 4)
/* The narrow shapes, added after `mmql_*` showed that NBLK=1 wins and that the
   epilogue this path removes is worth 17%. The first cut of `mmqf_*` was
   measured only at NBLK=2, where the register tile was already costing more
   than the epilogue saved. */
MMQ_F16_SET(1w4s2, 4, 1, 1, 2)
MMQ_F16_SET(1w4s2_2, 4, 1, 2, 2)
MMQ_F16_SET(1w4s4, 4, 1, 1, 4)
MMQ_F16_SET(1w4s4_2, 4, 1, 2, 4)
MMQ_F16_SET(1w8s2, 8, 1, 1, 2)
MMQ_F16_SET(1w8s2_2, 8, 1, 2, 2)
/* Four token tiles: 64 tokens to one weight pass, which is what Marlin's
   `max_thread_m_blocks = 4` buys and what this port had been leaving on the
   floor. Above 32 tokens the two-tile shape reads the weights twice and
   throughput stops scaling — 853 tok/s at batch 32 against 803 at 64. */
MMQ_F16_SET(1w8s2_4, 8, 1, 4, 2)
MMQ_F16_SET(1w4s2_4, 4, 1, 4, 2)
MMQ_F16_SET(1w16s2, 16, 1, 1, 2)
MMQ_F16_SET(1w16s2_2, 16, 1, 2, 2)
/* More weight rows per block, re-measured after the grid cap and the f16
   blocks-per-SM default changed — both of which moved under the first sweep
   that rejected these. */
MMQ_F16_SET(2w16s2, 16, 2, 1, 2)
MMQ_F16_SET(2w16s2_2, 16, 2, 2, 2)
MMQ_F16_SET(4w8s2, 8, 4, 1, 2)
MMQ_F16_SET(4w8s2_2, 8, 4, 2, 2)
MMQ_F16_SET(2w4s2b, 4, 2, 1, 2)
MMQ_F16_SET(2w4s2b_2, 4, 2, 2, 2)
MMQ_F16_SET(2w8s2, 8, 2, 1, 2)
MMQ_F16_SET(2w8s2_2, 8, 2, 2, 2)
MMQ_F16_SET(1w8s3, 8, 1, 1, 3)
MMQ_F16_SET(1w8s3_2, 8, 1, 2, 3)

// ---- attribution probe --------------------------------------------------
//
// The tile loop with the tensor cores removed. It answered the question that
// shaped this kernel — staging is 68% of the time and the MMAs 17%, and the two
// do not overlap — and stays because the next round of work on this file is
// making them overlap, which needs the same measurement.

#define MMQ_THREADS (MMQ_MAX_WARPS * WARP_SIZE)
#define MMQ_ROWS MMQ_MAX_ROWS
extern "C" __global__ __launch_bounds__(MMQ_THREADS) void mmq_stage_only_q4_K(
    float* __restrict__ out, const void* __restrict__ wv,
    const void* __restrict__ xv, int k, int n, int n_tokens) {
    __shared__ int8_t ws[MMQ_ROWS * MMQ_STRIDE];
    __shared__ int8_t xs[MMQ_M * MMQ_STRIDE];
    __shared__ float wd[MMQ_ROWS * MMQ_GROUPS];
    __shared__ float wm[MMQ_ROWS * MMQ_GROUPS];
    __shared__ float xd[MMQ_M * MMQ_GROUPS];
    __shared__ float xsum[MMQ_M * MMQ_GROUPS];

    const block_q4_K* wq = (const block_q4_K*)wv;
    const block_q8_1* xq = (const block_q8_1*)xv;
    const int tid = threadIdx.x;
    const int lane = tid % WARP_SIZE;
    const int warp = tid / WARP_SIZE;
    const int row0 = blockIdx.x * MMQ_ROWS;
    const int tok0 = blockIdx.y * MMQ_M;
    const int kb_total = k / 32;
    const int n_tiles = (kb_total + MMQ_GROUPS - 1) / MMQ_GROUPS;
    float acc = 0.0f;

    for (int kt = 0; kt < n_tiles; ++kt) {
        __syncthreads();
        mmq_load_x(xs, xd, xsum, xq, kb_total, kt * MMQ_GROUPS, tok0,
                   min(MMQ_M, max(0, n_tokens - tok0)), tid, MMQ_THREADS);
        mmq_load_w_q4_K(ws, wd, wm, wq, k / QK_K, kt, row0, n, MMQ_ROWS, tid,
                        MMQ_THREADS);
        __syncthreads();
        acc += (float)ws[warp * MMQ_STRIDE + lane] + (float)xs[lane] + wd[lane]
             + wm[lane] + xd[lane] + xsum[lane];
    }
    if (acc == 1234.5f) out[0] = acc;
}

// Does the *width* of a weight load matter, at the width this kernel uses?
//
// The whole remaining gap to Marlin is supposed to be here. It reads weights as
// `const int4*` — sixteen bytes an instruction, through `cp.async` — where this
// kernel reads `uint32_t`, four bytes, twice per fragment, because a lane's
// eight bytes are not contiguous in the AWQ pack. Making them contiguous means
// repacking, which means touching the loader, `unpack_row`, the mat-vec, the
// float path and every test that pins them.
//
// That is a day of work resting on one unmeasured assumption, so measure it
// first. Both kernels below read the same bytes, with the same grid, in the
// same order; the only difference is how many bytes an instruction carries.
//
//   `w4`  the real kernel's pattern: lane L takes bytes [4*(L%4), +4) and
//         [4*(L%4)+16, +4) of each 32-byte run, over 68-byte blocks
//   `w16` one 16-byte load per 128-weight block per row, over 64-byte blocks,
//         which is what the word transpose would make possible
//
// If these measure the same, the repack is not worth doing and the constraint
// is somewhere else entirely.

#define MMQ_BW_BODY(LOAD)                                                      \
    const int lane = threadIdx.x % WARP_SIZE;                                  \
    const int warp = threadIdx.x / WARP_SIZE;                                  \
    const int warps = blockDim.x / WARP_SIZE;                                  \
    const int c = lane % 4;                                                    \
    const int rowg = lane / 4;                                                 \
    /* Four independent accumulators. One serialises the loads behind a        \
       dependency chain and reports a ceiling 15% below what the real kernel   \
       exceeds, which is how this probe first lied. */                         \
    uint32_t acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;                           \
    /* One warp per eight weight rows, blocks striding the row grid, exactly   \
       the shape the GEMM launches with. */                                    \
    for (int rb = blockIdx.x * warps + warp; rb < n / 8;                       \
         rb += gridDim.x * warps) {                                            \
        const int row = rb * 8 + rowg;                                         \
        for (int b = 0; b < nb; ++b) {                                         \
            LOAD                                                               \
        }                                                                      \
    }                                                                          \
    if ((acc0 ^ acc1 ^ acc2 ^ acc3) == 0xdeadbeefu) out[0] = 1.0f;

extern "C" __global__ __launch_bounds__(128) void mmq_bw_probe_w4(
    float* __restrict__ out, const void* __restrict__ wv, int nb, int n) {
    const uint8_t* w = (const uint8_t*)wv;
    MMQ_BW_BODY({
        const uint8_t* qs = w + ((size_t)row * nb + b) * 68 + 4;
        const uint8_t* p0 = qs + c * 4;
        const uint8_t* p1 = qs + 32 + c * 4;
        acc0 ^= *(const uint32_t*)(const void*)p0;
        acc1 ^= *(const uint32_t*)(const void*)(p0 + 16);
        acc2 ^= *(const uint32_t*)(const void*)p1;
        acc3 ^= *(const uint32_t*)(const void*)(p1 + 16);
    })
}

// The same bytes as one contiguous 512-byte run a warp.
//
// `_w16` reads what the kernel reads: lane `l` takes row `l/4`'s sixteen bytes at
// chunk `l%4`, and consecutive rows are `nb * 64` apart, so one instruction
// issues eight 64-byte transactions two kilobytes apart. A layout that put a row
// group's eight same-`b` blocks side by side would make that one 512-byte run,
// which is the interleave `gptq_marlin_repack` does and the one piece of Marlin
// this port never took. The probe that dismissed it (`mmqfp_*`) measured load
// *width*, which the transposed layout had already fixed, not this.
//
// **Measured, and worth nothing**: 1412 GB/s coalesced against 1410 as the
// kernel reads, four buffers cycled so the 248 MB working set defeats L2. So the
// repack stays unported, now for a reason rather than by omission.
extern "C" __global__ __launch_bounds__(128) void mmq_bw_probe_c16(
    float* __restrict__ out, const void* __restrict__ wv, int nb, int n) {
    const uint8_t* w = (const uint8_t*)wv;
    MMQ_BW_BODY({
        (void)row;
        const uint8_t* qs = w + ((size_t)rb * nb + b) * 512 + lane * 16;
        const uint4 v = *(const uint4*)(const void*)qs;
        acc0 ^= v.x;
        acc1 ^= v.y;
        acc2 ^= v.z;
        acc3 ^= v.w;
    })
}

// Quants *and* scales, which is what the kernel actually reads.
//
// The probes above count only the 64 quant bytes a block. The kernel also reads
// its `__half2` of {scale, scale*zero}, and in the transposed layout those sit
// row-major in a trailing region, so the eight rows a warp covers are `nb * 4`
// apart: eight more transactions for thirty-two useful bytes. `_sc16` is the
// same two reads with that region block-major instead, so a row group's eight
// scales are one 32-byte run.
//
// **Also worth nothing.** 1343 GB/s row-major against 1345 block-major; the
// scale read costs 5% against the quants alone (1412) and its transaction
// pattern costs none of that — four lanes share each scale and L1 absorbs it.
//
// So the ceiling for everything this kernel reads is 1343 GB/s and the kernel
// reaches 1037, 77% of it. What the difference is *not*: the arithmetic
// (`mmqnm_*`, level), the activation tile (stubbing the A-fragment read is worth
// 0.8%), the barriers (stubbing both, 2.2%), the load width, the coalescing, the
// scale layout, the write stream (`_rw16`, 1%), the partition's balance (a block
// count that divides the units evenly is 10% *worse*), the prefetch depth (three
// is 1-3% down and costs 35 registers), the grid constant, the warp count, the
// row-block count, the stage count, and occupancy — the probe reads at 1345 GB/s
// whether it runs at 1504 blocks or 376, which is the kernel's own 2 an SM.
// Eleven isolations, none of them more than a few percent. The remaining 20% is
// unattributed and the next attempt should not start by re-running this list.
extern "C" __global__ __launch_bounds__(128) void mmq_bw_probe_s16(
    float* __restrict__ out, const void* __restrict__ wv, int nb, int n) {
    const uint8_t* w = (const uint8_t*)wv;
    const uint32_t* sc = (const uint32_t*)(const void*)(w + (size_t)n * nb * 64);
    MMQ_BW_BODY({
        const uint8_t* qs = w + ((size_t)row * nb + b) * 64;
        const uint4 v = *(const uint4*)(const void*)(qs + c * 16);
        acc0 ^= v.x;
        acc1 ^= v.y;
        acc2 ^= v.z;
        acc3 ^= sc[(size_t)row * nb + b];
    })
}

extern "C" __global__ __launch_bounds__(128) void mmq_bw_probe_sc16(
    float* __restrict__ out, const void* __restrict__ wv, int nb, int n) {
    const uint8_t* w = (const uint8_t*)wv;
    const uint32_t* sc = (const uint32_t*)(const void*)(w + (size_t)n * nb * 64);
    MMQ_BW_BODY({
        const uint8_t* qs = w + ((size_t)row * nb + b) * 64;
        const uint4 v = *(const uint4*)(const void*)(qs + c * 16);
        acc0 ^= v.x;
        acc1 ^= v.y;
        acc2 ^= v.z;
        /* Block-major: the row group's eight scales are adjacent. */
        acc3 ^= sc[(size_t)b * n + row];
    })
}

// The read stream with a write stream beside it, in the proportion the kernel has
// one — overstated four times over, and still worth only 4.5%.
extern "C" __global__ __launch_bounds__(128) void mmq_bw_probe_rw16(
    float* __restrict__ out, const void* __restrict__ wv, int nb, int n) {
    const uint8_t* w = (const uint8_t*)wv;
    MMQ_BW_BODY({
        const uint8_t* qs = w + ((size_t)row * nb + b) * 64;
        const uint4 v = *(const uint4*)(const void*)(qs + c * 16);
        acc0 ^= v.x;
        acc1 ^= v.y;
        acc2 ^= v.z;
        acc3 ^= v.w;
        out[((size_t)row * nb + b) % (1 << 20)] = (float)(int)v.x;
    })
}

extern "C" __global__ __launch_bounds__(128) void mmq_bw_probe_w16(
    float* __restrict__ out, const void* __restrict__ wv, int nb, int n) {
    const uint8_t* w = (const uint8_t*)wv;
    MMQ_BW_BODY({
        const uint8_t* qs = w + ((size_t)row * nb + b) * 64;
        const uint4 v = *(const uint4*)(const void*)(qs + c * 16);
        acc0 ^= v.x;
        acc1 ^= v.y;
        acc2 ^= v.z;
        acc3 ^= v.w;
    })
}

// What this card actually delivers on a pure streaming read, so the staging
// numbers above can be compared against something real rather than against the
// 448 GB/s on the spec sheet. Grid-strided uint4 loads, one reduction at the
// end, nothing else.
extern "C" __global__ void stream_read_probe(float* __restrict__ out,
                                             const void* __restrict__ src,
                                             int n_vec) {
    const uint4* p = (const uint4*)src;
    uint32_t acc = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_vec;
         i += blockDim.x * gridDim.x) {
        const uint4 v = p[i];
        acc += v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0xdeadbeefu) out[0] = 1.0f;
}

// Attribution probe: the real tile loop with the A-operand shared-memory loads
// replaced by a constant. Everything else — staging, the MMAs, the scale math,
// the stores — is identical, so the difference is exactly what the A fragment
// gather costs.
//
// The question it answers: at 32 tokens a block reads about 49 KB out of shared
// memory per k-step to feed the tensor cores, against 4.6 KB of quantized
// weights out of global memory. If that 10:1 ratio is what caps the kernel at
// 84 GB/s (against 218 GB/s at one token), removing the A loads should show it.
extern "C" __global__ __launch_bounds__(128) void mmq_noA_q4_K(
    float* __restrict__ out, const void* __restrict__ wv,
    const void* __restrict__ xv, int k, int n, int n_tokens) {
    const block_q4_K* wq = (const block_q4_K*)wv;
    const block_q8_1* xq = (const block_q8_1*)xv;
    __shared__ int8_t ws[32 * MMQ_STRIDE];
    __shared__ int8_t xs[2 * MMQ_M * MMQ_STRIDE];
    __shared__ float wd[32 * MMQ_GROUPS];
    __shared__ float wm[32 * MMQ_GROUPS];
    __shared__ float xd[2 * MMQ_M * MMQ_GROUPS];
    __shared__ float xsum[2 * MMQ_M * MMQ_GROUPS];

    const int tid = threadIdx.x;
    const int lane = tid % WARP_SIZE;
    const int warp = tid / WARP_SIZE;
    const int row0 = blockIdx.x * 32;
    const int tok0 = blockIdx.y * 2 * MMQ_M;
    const int kb_total = k / 32;
    const int n_tiles = (kb_total + MMQ_GROUPS - 1) / MMQ_GROUPS;
    const int x_valid = min(2 * MMQ_M, max(0, n_tokens - tok0));
    const int bc = mma_b_col(lane);
    const int kq = mma_k0(lane);
    const int cc = mma_c_col(lane);
    const int cr = mma_c_row(lane);
    const int wbase = warp * 8;
    float acc[2][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}};

    mmq_zero_x(xs, xd, xsum, x_valid, 2 * MMQ_M, tid, 128);
    for (int kt = 0; kt < n_tiles; ++kt) {
        __syncthreads();
        mmq_load_x(xs, xd, xsum, xq, kb_total, kt * MMQ_GROUPS, tok0, x_valid,
                   tid, 128);
        mmq_load_w_q4_K(ws, wd, wm, wq, k / QK_K, kt, row0, n, 32, tid, 128);
        __syncthreads();
#pragma unroll
        for (int g = 0; g < MMQ_GROUPS; ++g) {
            const int8_t* bp = ws + (wbase + bc) * MMQ_STRIDE + g * 32 + kq;
            mma_b_s8 b;
            b.x[0] = *(const int*)(const void*)bp;
            b.x[1] = *(const int*)(const void*)(bp + 16);
            mma_a_s8 a;  // constant, not gathered from shared
            a.x[0] = 0x01010101;
            a.x[1] = 0x01010101;
            a.x[2] = 0x01010101;
            a.x[3] = 0x01010101;
#pragma unroll
            for (int u = 0; u < 2; ++u) {
                mma_c_s32 d = {{0, 0, 0, 0}};
                mma_s8(d, a, b);
                const int r0 = (wbase + cc) * MMQ_GROUPS + g;
                const int t0 = (u * MMQ_M + cr) * MMQ_GROUPS + g;
                acc[u][0] += wd[r0] * xd[t0] * (float)d.x[0];
                acc[u][1] += wd[r0] * xd[t0] * (float)d.x[1];
                acc[u][2] += wd[r0] * xd[t0] * (float)d.x[2];
                acc[u][3] += wd[r0] * xd[t0] * (float)d.x[3];
            }
        }
    }
    if (acc[0][0] + acc[1][3] == 1234.5f) out[0] = acc[0][0];
}

extern "C" __global__ __launch_bounds__(128) void mmq_noscale_q4_K(
    float* __restrict__ out, const void* __restrict__ wv,
    const void* __restrict__ xv, int k, int n, int n_tokens) {
    const block_q4_K* wq = (const block_q4_K*)wv;
    const block_q8_1* xq = (const block_q8_1*)xv;
    __shared__ int8_t ws[32 * MMQ_STRIDE];
    __shared__ int8_t xs[2 * MMQ_M * MMQ_STRIDE];
    __shared__ float wd[32 * MMQ_GROUPS];
    __shared__ float wm[32 * MMQ_GROUPS];
    __shared__ float xd[2 * MMQ_M * MMQ_GROUPS];
    __shared__ float xsum[2 * MMQ_M * MMQ_GROUPS];

    const int tid = threadIdx.x;
    const int lane = tid % WARP_SIZE;
    const int warp = tid / WARP_SIZE;
    const int row0 = blockIdx.x * 32;
    const int tok0 = blockIdx.y * 2 * MMQ_M;
    const int kb_total = k / 32;
    const int n_tiles = (kb_total + MMQ_GROUPS - 1) / MMQ_GROUPS;
    const int x_valid = min(2 * MMQ_M, max(0, n_tokens - tok0));
    const int bc = mma_b_col(lane);
    const int kq = mma_k0(lane);
    const int cc = mma_c_col(lane);
    const int cr = mma_c_row(lane);
    const int wbase = warp * 8;
    float acc[2][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}};

    mmq_zero_x(xs, xd, xsum, x_valid, 2 * MMQ_M, tid, 128);
    for (int kt = 0; kt < n_tiles; ++kt) {
        __syncthreads();
        mmq_load_x(xs, xd, xsum, xq, kb_total, kt * MMQ_GROUPS, tok0, x_valid,
                   tid, 128);
        mmq_load_w_q4_K(ws, wd, wm, wq, k / QK_K, kt, row0, n, 32, tid, 128);
        __syncthreads();
#pragma unroll
        for (int g = 0; g < MMQ_GROUPS; ++g) {
            const int8_t* bp = ws + (wbase + bc) * MMQ_STRIDE + g * 32 + kq;
            mma_b_s8 b;
            b.x[0] = *(const int*)(const void*)bp;
            b.x[1] = *(const int*)(const void*)(bp + 16);
            const int8_t* apA = xs + (0 * MMQ_M + mma_a_row(lane)) * MMQ_STRIDE
                              + g * 32 + kq;
#pragma unroll
            for (int u = 0; u < 2; ++u) {
                const int8_t* ap = apA + u * MMQ_M * MMQ_STRIDE;
                const int8_t* aq = ap + 8 * MMQ_STRIDE;
                mma_a_s8 a;
                mma_c_s32 d = {{0, 0, 0, 0}};
                a.x[0] = *(const int*)(const void*)ap;
                a.x[1] = *(const int*)(const void*)aq;
                a.x[2] = *(const int*)(const void*)(ap + 16);
                a.x[3] = *(const int*)(const void*)(aq + 16);
                mma_s8(d, a, b);
                /* Scales replaced by constants: same MMAs, same operand loads,
                   no shared-memory lookups for wd/wm/xd/xsum. */
                acc[u][0] += 1.5f * (float)d.x[0];
                acc[u][1] += 1.5f * (float)d.x[1];
                acc[u][2] += 1.5f * (float)d.x[2];
                acc[u][3] += 1.5f * (float)d.x[3];
            }
        }
    }
    if (acc[0][0] + acc[1][3] == 1234.5f) out[0] = acc[0][0];
}


// Does the shared-memory write cost the staging its bandwidth?
//
// The mat-vec streams weights global -> registers -> dp4a and reaches 375 GB/s.
// This kernel's staging reads the same bytes and writes them to shared first,
// and measures 233. If the read alone comes back at mat-vec speed then the
// write is the gap, and the fix is a weight layout that can be read straight
// into fragments — a load-time permutation, which for a quantized block is
// lossless. If the read alone is also 233, the gap is the access pattern and
// the permutation would not help.
extern "C" __global__ void mmq_readonly_q4_K(float* __restrict__ out,
                                             const void* __restrict__ wv,
                                             const void* __restrict__ xv, int k,
                                             int n, int n_tokens) {
    const block_q4_K* w = (const block_q4_K*)wv;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int rows = 64;
    const int row0 = blockIdx.x * rows;
    const int nsb = k / QK_K;

    uint32_t acc = 0;
    for (int sb = 0; sb < nsb; ++sb) {
        // Same units and the same addresses the Q4_K stager walks, minus the
        // shared stores.
        for (int u = tid; u < rows * 4; u += nthreads) {
            const int r = u / 4;
            const int gp = u % 4;
            const int gr = row0 + r;
            if (gr >= n) continue;
            const block_q4_K* b = w + (size_t)gr * nsb + sb;
#pragma unroll
            for (int j = 0; j < 32; j += 4) {
                acc ^= *(const uint32_t*)(const void*)(b->qs + gp * 32 + j);
            }
        }
    }
    if (acc == 0xdeadbeefu) out[0] = 1.0f;
}



// Two shapes that were built, verified and measured slower than the one above,
// recorded so they are not rebuilt:
//
//   weights in the A operand (16 rows per warp, tokens in B): 397 us against
//     261 at 32 tokens. The operand-traffic arithmetic favours it — 12 bytes
//     per MMA against 20 — and it loses anyway.
//   128 rows per block, one activation fragment feeding two row tiles: 287 us
//     against 223 at 16 tokens. It halves the `ldmatrix` count and pushes
//     shared memory to 48 KiB, which drops the SM from three resident blocks to
//     two, from 24 warps to 16.
//   warp pairs splitting the quantization groups over 32 rows: 372 against 294.
//     Shared memory halves and occupancy reaches 32 warps, but the grid doubles
//     and with it the activation staging.
//
// Three ways of moving off 64 rows and 24 warps per SM, three losses. That is
// where this structure wants to sit.
