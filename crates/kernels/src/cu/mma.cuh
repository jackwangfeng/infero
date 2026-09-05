// Integer tensor-core primitive: one `mma.m16n8k32.s8` per warp.
//
// The K extent is 32 on purpose. Every ggml quantization block is 32 elements
// wide (or a multiple of it), and the Q8_1 activation blocks are too, so one
// MMA consumes exactly one block pair and the scales apply to whole fragments
// instead of straddling them.
//
// The fragment layouts below are the ones PTX documents for m16n8k32 with 8-bit
// operands. `mma_s8_probe` in this file exists so a test can prove them rather
// than have us trust a transcription: an index off by one here produces
// plausible-looking garbage everywhere downstream.
//
// Requires sm_80+ (Ampere). `arch()` gates the callers.

// A warp's slice of the operands. Register counts follow from the tile bytes
// divided by 32 lanes: A is 16x32 = 512 B = 4 regs, B is 32x8 = 256 B = 2 regs,
// the s32 accumulator is 16x8 = 4 regs.
struct mma_a_s8 { int x[4]; };
struct mma_b_s8 { int x[2]; };
struct mma_c_s32 { int x[4]; };

// Which A row this lane's registers 0/2 address (registers 1/3 address +8).
__device__ __forceinline__ int mma_a_row(int lane) { return lane / 4; }
// Which B column this lane addresses.
__device__ __forceinline__ int mma_b_col(int lane) { return lane / 4; }
// First of the four consecutive k this lane holds; registers 2/3 hold +16.
__device__ __forceinline__ int mma_k0(int lane) { return (lane % 4) * 4; }
// Which accumulator row/col this lane's d[0..1] address (d[2..3] are row +8).
__device__ __forceinline__ int mma_c_row(int lane) { return lane / 4; }
__device__ __forceinline__ int mma_c_col(int lane) { return (lane % 4) * 2; }

__device__ __forceinline__ void mma_s8(mma_c_s32& d, const mma_a_s8& a,
                                       const mma_b_s8& b) {
#if __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+r"(d.x[0]), "+r"(d.x[1]), "+r"(d.x[2]), "+r"(d.x[3])
        : "r"(a.x[0]), "r"(a.x[1]), "r"(a.x[2]), "r"(a.x[3]), "r"(b.x[0]),
          "r"(b.x[1]));
#else
    // Kept compilable for older arches so the module still loads; callers must
    // not reach here.
    (void)d; (void)a; (void)b;
#endif
}

// Gather straight from row-major tiles. Only the probe uses this — the real
// kernel builds fragments out of quantized blocks in shared memory — but it is
// what pins the layout down.
extern "C" __global__ void mma_s8_probe(const int8_t* __restrict__ A,  // 16 x 32
                                        const int8_t* __restrict__ B,  //  8 x 32
                                        int* __restrict__ D) {         // 16 x  8
    const int lane = threadIdx.x;
    const int ar = mma_a_row(lane);
    const int bc = mma_b_col(lane);
    const int k0 = mma_k0(lane);

    mma_a_s8 a;
    mma_b_s8 b;
    mma_c_s32 d = {{0, 0, 0, 0}};

    const int8_t* a_lo = A + ar * 32 + k0;
    const int8_t* a_hi = A + (ar + 8) * 32 + k0;
    a.x[0] = *(const int*)a_lo;
    a.x[1] = *(const int*)a_hi;
    a.x[2] = *(const int*)(a_lo + 16);
    a.x[3] = *(const int*)(a_hi + 16);

    const int8_t* bp = B + bc * 32 + k0;
    b.x[0] = *(const int*)bp;
    b.x[1] = *(const int*)(bp + 16);

    mma_s8(d, a, b);

    const int cr = mma_c_row(lane);
    const int cc = mma_c_col(lane);
    D[cr * 8 + cc + 0] = d.x[0];
    D[cr * 8 + cc + 1] = d.x[1];
    D[(cr + 8) * 8 + cc + 0] = d.x[2];
    D[(cr + 8) * 8 + cc + 1] = d.x[3];
}

// ---- the f16 tensor cores -----------------------------------------------
//
// `mma.m16n8k16` with f16 operands and an f32 accumulator, which is what Marlin
// runs (`marlin_template.h:88`). K is 16 rather than 32 because a half is twice
// a byte — the *tile is the same 32 bytes wide*, and that is the useful part:
//
//   s8  A fragment: 16 rows x 32 k, one byte each   = 16 rows x 32 bytes
//   f16 A fragment: 16 rows x 16 k, two bytes each  = 16 rows x 32 bytes
//
// and lane L addresses the same bytes in both. For s8 it holds k
// [(L%4)*4, +4) as four int8; for f16 it holds k [(L%4)*2, +2) as two halves,
// which is the same four bytes. Registers 2/3 are +16 bytes in both. So
// `mma_a_row`, `mma_b_col`, `mma_c_row`, `mma_c_col` carry over unchanged, and
// `ldmatrix_a_s8` loads an f16 A fragment correctly too — it is a `.b16`
// instruction that was only ever moving bytes.
//
// What does not carry over is the accumulator: f32 here against s32 there, so
// the fragment is four floats and the scales no longer have to wait for the end
// of a quantization group. That is the whole reason to want this path.

struct mma_a_f16 { unsigned x[4]; };  // 8 halves per lane
struct mma_b_f16 { unsigned x[2]; };  // 4 halves per lane
struct mma_c_f32 { float x[4]; };

// First of the two consecutive k this lane holds, in halves. `mma_k0` gives the
// same position in bytes, which is what an address needs.
__device__ __forceinline__ int mma_k0_f16(int lane) { return (lane % 4) * 2; }

__device__ __forceinline__ void mma_f16(mma_c_f32& d, const mma_a_f16& a,
                                        const mma_b_f16& b) {
#if __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d.x[0]), "+f"(d.x[1]), "+f"(d.x[2]), "+f"(d.x[3])
        : "r"(a.x[0]), "r"(a.x[1]), "r"(a.x[2]), "r"(a.x[3]), "r"(b.x[0]),
          "r"(b.x[1]));
#else
    (void)d; (void)a; (void)b;
#endif
}

// ---- the e4m3 tensor cores -----------------------------------------------
//
// `mma.m16n8k32` with e4m3 operands and an f32 accumulator: native FP8 on both
// sides, where `mma_f8_block` (see `fp8.cu`) keeps the weight in e4m3 but
// widens it to f16 before the MMA, spending an `mma.m16n8k16.f16` on it
// instead. Ada/Hopper/Blackwell's FP8 tensor cores are rated roughly double
// the f16 ones, so that widening was leaving the other factor of two on the
// table -- vLLM's own FP8 linear kernels (`cutlass_scaled_mm`, DeepGEMM) both
// quantize the activation too and run this MMA directly, not `mma_f16`.
//
// e4m3 is one byte, the same as the `s8` operand `mma_s8` above already
// speaks, and `m16n8k32` is the same shape -- so this reuses `mma_a_s8` and
// `mma_b_s8` verbatim: what changes is the instruction's element and
// accumulator type, not the fragment layout or the register count. Whatever
// `mma_a_row`/`mma_b_col`/`mma_k0`/`ldmatrix_a_s8`/`ldmatrix_b_s8` produce for
// an `s8` operand is bit-for-bit the fragment an `e4m3` operand needs too.
//
// Requires sm_89+ (Ada) — `caps().fp8` gates the callers, the same flag
// `fp8.cu`'s per-block dequant already uses.
__device__ __forceinline__ void mma_e4m3(mma_c_f32& d, const mma_a_s8& a,
                                         const mma_b_s8& b) {
#if __CUDA_ARCH__ >= 890
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d.x[0]), "+f"(d.x[1]), "+f"(d.x[2]), "+f"(d.x[3])
        : "r"(a.x[0]), "r"(a.x[1]), "r"(a.x[2]), "r"(a.x[3]), "r"(b.x[0]),
          "r"(b.x[1]));
#else
    (void)d; (void)a; (void)b;
#endif
}

// Same question as `mma_s8_probe`: does the hand-transcribed fragment layout
// actually match what the instruction wants? e4m3 encode/decode is
// `f32_to_e4m3`/`e4m3_to_f32` (`fp8.cu`), so the inputs and the expected
// output are built in plain float and only the operands cross into e4m3.
extern "C" __global__ void mma_e4m3_probe(const unsigned char* __restrict__ A,  // 16x32
                                          const unsigned char* __restrict__ B,  //  8x32
                                          float* __restrict__ D) {              // 16x8
    const int lane = threadIdx.x;
    const int ar = mma_a_row(lane);
    const int bc = mma_b_col(lane);
    const int k0 = mma_k0(lane);

    mma_a_s8 a;
    mma_b_s8 b;
    mma_c_f32 d = {{0.0f, 0.0f, 0.0f, 0.0f}};

    const unsigned char* a_lo = A + ar * 32 + k0;
    const unsigned char* a_hi = A + (ar + 8) * 32 + k0;
    a.x[0] = *(const int*)(const void*)a_lo;
    a.x[1] = *(const int*)(const void*)a_hi;
    a.x[2] = *(const int*)(const void*)(a_lo + 16);
    a.x[3] = *(const int*)(const void*)(a_hi + 16);

    const unsigned char* bp = B + bc * 32 + k0;
    b.x[0] = *(const int*)(const void*)bp;
    b.x[1] = *(const int*)(const void*)(bp + 16);

    mma_e4m3(d, a, b);

    const int cr = mma_c_row(lane);
    const int cc = mma_c_col(lane);
    D[cr * 8 + cc + 0] = d.x[0];
    D[cr * 8 + cc + 1] = d.x[1];
    D[(cr + 8) * 8 + cc + 0] = d.x[2];
    D[(cr + 8) * 8 + cc + 1] = d.x[3];
}

// The f16 counterpart of `mma_s8_probe`, and it exists for the same reason: the
// real kernel builds these fragments by hand, and a wrong index produces a
// plausible matrix rather than a loud failure.
extern "C" __global__ void mma_f16_probe(const __half* __restrict__ A,  // 16x16
                                         const __half* __restrict__ B,  //  8x16
                                         float* __restrict__ D) {       // 16x8
    const int lane = threadIdx.x;
    const int ar = mma_a_row(lane);
    const int bc = mma_b_col(lane);
    const int k0 = mma_k0_f16(lane);

    mma_a_f16 a;
    mma_b_f16 b;
    mma_c_f32 d = {{0.0f, 0.0f, 0.0f, 0.0f}};

    const __half* a_lo = A + ar * 16 + k0;
    const __half* a_hi = A + (ar + 8) * 16 + k0;
    a.x[0] = *(const unsigned*)(const void*)a_lo;
    a.x[1] = *(const unsigned*)(const void*)a_hi;
    a.x[2] = *(const unsigned*)(const void*)(a_lo + 8);
    a.x[3] = *(const unsigned*)(const void*)(a_hi + 8);

    const __half* bp = B + bc * 16 + k0;
    b.x[0] = *(const unsigned*)(const void*)bp;
    b.x[1] = *(const unsigned*)(const void*)(bp + 8);

    mma_f16(d, a, b);

    const int cr = mma_c_row(lane);
    const int cc = mma_c_col(lane);
    D[cr * 8 + cc + 0] = d.x[0];
    D[cr * 8 + cc + 1] = d.x[1];
    D[(cr + 8) * 8 + cc + 0] = d.x[2];
    D[(cr + 8) * 8 + cc + 1] = d.x[3];
}

// ---- ldmatrix -----------------------------------------------------------
//
// `ldmatrix` loads an MMA operand fragment out of shared memory cooperatively:
// one instruction where the hand-rolled gather above needs four scalar loads
// per lane. llama.cpp's MMQ uses it throughout, and their tuned Ampere config
// pairs it with 128 output rows per block — the two go together, because a wide
// block is what makes an operand load worth amortizing.
//
// The instruction loads four 8x8 tiles of 16-bit elements. An `m16n8k32.s8` A
// fragment is 16 rows of 32 bytes, which is exactly 16x16 halves, so one
// `.x4.b16` fills all four A registers.
//
// Lane L supplies the address of row `L % 8` of tile `L / 8`, and the four
// tiles cover (rows 0-7, cols 0-7), (rows 8-15, cols 0-7), (rows 0-7, cols
// 8-15), (rows 8-15, cols 8-15) in units of halves. That mapping is asserted by
// `ldmatrix_a_probe` rather than trusted: a wrong address here loads a fragment
// that is plausible and wrong, which is the failure mode this file exists to
// prevent.

__device__ __forceinline__ unsigned smem_addr(const void* p) {
    return (unsigned)__cvta_generic_to_shared(p);
}

/// Load an `m16n8k32.s8` A fragment (16 rows x 32 bytes) from a row-major
/// shared tile whose rows are `stride_bytes` apart.
__device__ __forceinline__ void ldmatrix_a_s8(mma_a_s8& a, const int8_t* base,
                                              int stride_bytes) {
#if __CUDA_ARCH__ >= 800
    const int lane = threadIdx.x % WARP_SIZE;
    const int m = lane / 8;            // which 8x8 half-tile
    const int r = lane % 8;            // row within it
    const int8_t* p = base + (r + (m & 1) * 8) * stride_bytes + (m >> 1) * 16;
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(a.x[0]), "=r"(a.x[1]), "=r"(a.x[2]), "=r"(a.x[3])
        : "r"(smem_addr(p)));
#else
    (void)a; (void)base; (void)stride_bytes;
#endif
}

/// `ldmatrix` against an XOR-swizzled tile.
///
/// A padded row stride can be conflict-free for one access pattern at a time:
/// 544 bytes suits an 8-byte gather at `8 * (lane % 4)` and is exactly the
/// stride that two-ways `ldmatrix`, which is why `ldmatrix` measured 20-25%
/// slower every time it was tried here. Marlin does not pad — it XORs the
/// 16-byte chunk index within a row by `row % 8` (`marlin_template.h:638`,
/// whose comment says the point is that *neither* reads nor writes conflict),
/// and both the `cp.async` that fills the tile and the `ldmatrix` that reads it
/// apply the same permutation.
///
/// `chunk0` is the fragment's first 16-byte chunk within the row; the fragment
/// spans that chunk and the next.
__device__ __forceinline__ void ldmatrix_a_swz(mma_a_s8& a, const int8_t* tile,
                                               int stride_bytes, int chunk0) {
#if __CUDA_ARCH__ >= 800
    const int lane = threadIdx.x % WARP_SIZE;
    const int m = lane / 8;
    const int row = (lane % 8) + (m & 1) * 8;
    const int chunk = (chunk0 + (m >> 1)) ^ (row & 7);
    const int8_t* p = tile + row * stride_bytes + chunk * 16;
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(a.x[0]), "=r"(a.x[1]), "=r"(a.x[2]), "=r"(a.x[3])
        : "r"(smem_addr(p)));
#else
    (void)a; (void)tile; (void)stride_bytes; (void)chunk0;
#endif
}

/// Same question as `mma_s8_probe`, for the cooperative load: does the fragment
/// `ldmatrix` produces match the one the hand-rolled gather produces?
extern "C" __global__ void ldmatrix_a_probe(const int8_t* __restrict__ A,
                                            int* __restrict__ out) {
    __shared__ int8_t tile[16 * 32];
    const int lane = threadIdx.x;
    for (int i = lane; i < 16 * 32 / 4; i += WARP_SIZE) {
        ((int*)tile)[i] = ((const int*)A)[i];
    }
    __syncwarp();

    mma_a_s8 got;
    ldmatrix_a_s8(got, tile, 32);

    // The gather this replaces, for comparison.
    const int ar = mma_a_row(lane);
    const int k0 = mma_k0(lane);
    const int8_t* lo = tile + ar * 32 + k0;
    const int8_t* hi = tile + (ar + 8) * 32 + k0;
    int want[4];
    want[0] = *(const int*)(const void*)lo;
    want[1] = *(const int*)(const void*)hi;
    want[2] = *(const int*)(const void*)(lo + 16);
    want[3] = *(const int*)(const void*)(hi + 16);

#pragma unroll
    for (int i = 0; i < 4; ++i) {
        out[lane * 8 + i] = got.x[i];
        out[lane * 8 + 4 + i] = want[i];
    }
}

/// Load an `m16n8k32.s8` B fragment (8 columns x 32 bytes) from a column-major
/// shared tile — that is, 8 rows of 32 bytes each, row `c` holding column `c`.
///
/// `ldmatrix.x2` covers two 8x8 half-tiles, which is exactly the 8 bytes per
/// lane a B operand needs. Lane L supplies the address of row `L % 8` of tile
/// `L / 8`, and only the first 16 lanes address anything: tile 0 is columns
/// 0-7 of the halves, tile 1 is columns 8-15.
__device__ __forceinline__ void ldmatrix_b_s8(mma_b_s8& b, const int8_t* base,
                                              int stride_bytes) {
#if __CUDA_ARCH__ >= 800
    const int lane = threadIdx.x % WARP_SIZE;
    const int m = (lane / 8) & 1;
    const int r = lane % 8;
    const int8_t* p = base + r * stride_bytes + m * 16;
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                 : "=r"(b.x[0]), "=r"(b.x[1])
                 : "r"(smem_addr(p)));
#else
    (void)b; (void)base; (void)stride_bytes;
#endif
}

/// The B-operand counterpart of `ldmatrix_a_probe`.
extern "C" __global__ void ldmatrix_b_probe(const int8_t* __restrict__ B,
                                            int* __restrict__ out) {
    __shared__ int8_t tile[8 * 32];
    const int lane = threadIdx.x;
    for (int i = lane; i < 8 * 32 / 4; i += WARP_SIZE) {
        ((int*)tile)[i] = ((const int*)B)[i];
    }
    __syncwarp();

    mma_b_s8 got;
    ldmatrix_b_s8(got, tile, 32);

    const int bc = mma_b_col(lane);
    const int k0 = mma_k0(lane);
    const int8_t* bp = tile + bc * 32 + k0;
    int want[2];
    want[0] = *(const int*)(const void*)bp;
    want[1] = *(const int*)(const void*)(bp + 16);

    out[lane * 4 + 0] = got.x[0];
    out[lane * 4 + 1] = got.x[1];
    out[lane * 4 + 2] = want[0];
    out[lane * 4 + 3] = want[1];
}

// ---- GDN kernel-2 register<->MMA-fragment bridge, validated in isolation --
//
// `gdn_chunk_state_f32` (see `gdn.cu`) keeps its per-chunk state `sc[64]`
// resident in registers across ~955 sequential chunk iterations, in a
// 256-thread layout unrelated to any MMA fragment: thread `lane` owns
// `S[i0+r][j]` for a fixed `(j, part)` pair (`j = lane/2`, `part = lane%2`,
// `i0 = part*64`), i.e. one thread per (row-half, column). Giving this state's
// own matmul-shaped steps (`pred = W @ S`, `S += Kd^T @ delta`) tensor cores
// needs it to serve as an MMA operand and receive an MMA accumulator's output
// -- neither of which uses this per-thread distribution -- so both directions
// have to round-trip through a shared-memory staging tile instead. This probe
// tests exactly that bridge, both directions, before the real kernel commits
// to it: is the layout choice below (staging `S` *transposed*, `st[j][d]`, so
// it lands in the "column-major" shape `ldmatrix`'s own B-operand convention
// already documented above expects for a `(k=d, n=j)` contraction) actually
// correct, and does an MMA accumulator's own natural per-lane output position
// write back into the same tile at the cell the ORIGINAL per-thread layout
// expects to read next?
//
// Uses a plain scalar gather from shared memory (mirroring `mma_f16_probe`'s
// existing style, sourced from `st` instead of a global tile) rather than
// `ldmatrix`, since what this probe exists to answer is whether the *data
// layout* bridge is correct, not which load instruction reads it fastest --
// that is a real, separate follow-up optimization for whichever kernel
// actually adopts this, not a question this probe needs to settle.
extern "C" __global__ __launch_bounds__(256) void gdn_state_bridge_probe(
        const float* __restrict__ s_in,      // [128][128], row d, col j
        const half* __restrict__ w_in,       // [16][16], row-major
        float* __restrict__ pred_out,        // [16][8], the mma_f16 result
        float* __restrict__ roundtrip_out) { // [128][128], s_in through the full bridge
    const int lane = threadIdx.x;   // 0..255, matches gdn_chunk_state_f32
    const int j = lane / 2;         // 0..127
    const int part = lane % 2;      // 0 or 1
    const int i0 = part * 64;       // 0 or 64

    __shared__ half st[128 * 128];  // st[j][d] -- see the comment above for why transposed

    float sc[64];
#pragma unroll
    for (int r = 0; r < 64; ++r) {
        sc[r] = s_in[(size_t)(i0 + r) * 128 + j];
        st[(size_t)j * 128 + (i0 + r)] = __float2half(sc[r]);
    }
    __syncthreads();

    if (lane < 32) {
        mma_a_f16 a;
        mma_b_f16 b;
        mma_c_f32 d = {{0.0f, 0.0f, 0.0f, 0.0f}};

        const int ar = mma_a_row(lane);
        const int bc = mma_b_col(lane);
        const int k0 = mma_k0_f16(lane);

        const half* a_lo = w_in + ar * 16 + k0;
        const half* a_hi = w_in + (ar + 8) * 16 + k0;
        a.x[0] = *(const unsigned*)(const void*)a_lo;
        a.x[1] = *(const unsigned*)(const void*)a_hi;
        a.x[2] = *(const unsigned*)(const void*)(a_lo + 8);
        a.x[3] = *(const unsigned*)(const void*)(a_hi + 8);

        const half* bp = st + bc * 128 + k0;
        b.x[0] = *(const unsigned*)(const void*)bp;
        b.x[1] = *(const unsigned*)(const void*)(bp + 8);

        // Independent thread scheduling (Volta+) does not guarantee this
        // warp's 32 lanes finish the reads above before any lane starts the
        // write-back below, even though both are straight-line code with no
        // branch between them -- `racecheck` caught exactly this the first
        // time this kernel ran (2 real hazards, passing correctness check
        // notwithstanding: a race that happens not to reorder on one run is
        // still a race). `__syncwarp()` is the fix, not a wider
        // `__syncthreads()` -- every lane touching `st` here is in this same
        // warp.
        __syncwarp();

        mma_f16(d, a, b);

        const int cr = mma_c_row(lane);
        const int cc = mma_c_col(lane);
        pred_out[cr * 8 + cc + 0] = d.x[0];
        pred_out[cr * 8 + cc + 1] = d.x[1];
        pred_out[(cr + 8) * 8 + cc + 0] = d.x[2];
        pred_out[(cr + 8) * 8 + cc + 1] = d.x[3];

        // The other direction: the accumulator's own natural per-lane output
        // cell, written back into `st` at the corresponding (j=n, d=m)
        // position -- exactly what a real state-advance step needs every
        // chunk (fold this MMA's result into the running state, in place, so
        // the same per-thread registers can read it back next chunk).
        st[(size_t)cc * 128 + cr] = __float2half(d.x[0]);
        st[(size_t)(cc + 1) * 128 + cr] = __float2half(d.x[1]);
        st[(size_t)cc * 128 + (cr + 8)] = __float2half(d.x[2]);
        st[(size_t)(cc + 1) * 128 + (cr + 8)] = __float2half(d.x[3]);
    }
    __syncthreads();

    // Every thread reads its own (i0+r, j) cell back out of `st`, exactly as
    // the real kernel's next chunk iteration would. Cells with d<16 && j<8
    // prove the MMA-output round trip; the rest prove step 1's write-then-
    // read survives undisturbed.
#pragma unroll
    for (int r = 0; r < 64; ++r) {
        roundtrip_out[(size_t)(i0 + r) * 128 + j] = __half2float(st[(size_t)j * 128 + (i0 + r)]);
    }
}
