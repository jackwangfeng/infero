// Shared device helpers. NVRTC has no include path into our source tree, so
// this file is concatenated ahead of every kernel source at build time.

#include <cuda_fp16.h>

// NVRTC compiles without the host C++ standard library, so <cstdint> and
// <cmath> are unavailable. Re-declaring these typedefs is legal as long as
// they name the same types, and they do on every platform we target.
typedef signed char int8_t;
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;

#ifndef INFINITY
#define INFINITY __int_as_float(0x7f800000)
#endif

#define WARP_SIZE 32
#define FULL_MASK 0xffffffffu

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(FULL_MASK, v, offset, WARP_SIZE);
    }
    return v;
}

__device__ __forceinline__ float warp_reduce_max(float v) {
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(FULL_MASK, v, offset, WARP_SIZE));
    }
    return v;
}

// Block-wide reduction whose result is broadcast to every thread.
// Requires blockDim.x <= 1024 and a multiple of the warp size is not assumed.
__device__ __forceinline__ float block_reduce_sum(float v) {
    __shared__ float partial[WARP_SIZE];
    __shared__ float result;

    // A __shared__ declared inside a device function is one static allocation
    // reused by every call, so a kernel that reduces twice would have the
    // second call overwrite `result` while slower threads were still reading
    // the first. Barrier on entry, not just on exit.
    __syncthreads();

    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
    const int warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

    v = warp_reduce_sum(v);
    if (lane == 0) partial[warp] = v;
    __syncthreads();

    if (warp == 0) {
        float acc = (lane < warps) ? partial[lane] : 0.0f;
        acc = warp_reduce_sum(acc);
        if (lane == 0) result = acc;
    }
    __syncthreads();
    return result;
}

__device__ __forceinline__ float block_reduce_max(float v) {
    __shared__ float partial[WARP_SIZE];
    __shared__ float result;

    __syncthreads();

    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
    const int warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

    v = warp_reduce_max(v);
    if (lane == 0) partial[warp] = v;
    __syncthreads();

    if (warp == 0) {
        float acc = (lane < warps) ? partial[lane] : -INFINITY;
        acc = warp_reduce_max(acc);
        if (lane == 0) result = acc;
    }
    __syncthreads();
    return result;
}

// ---- cp.async, global to shared without a register round trip -----------
//
// Same primitive as `mmq_cp_async16` in `mmq.cu` (kept separate there rather
// than switched to call this, since that kernel's copy is already measured
// and this one exists for a different kernel family) -- raw PTX so nothing
// here needs `cuda_pipeline.h` and the module stays reachable from NVRTC.
// `dst` must be 16-byte aligned, which is `cp.async.cg`'s own requirement,
// not an added one.
__device__ __forceinline__ void cp_async16(void* dst, const void* src,
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
#define CP_ASYNC_FENCE() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT(N) asm volatile("cp.async.wait_group %0;\n" ::"n"(N))
#else
#define CP_ASYNC_FENCE() do {} while (0)
#define CP_ASYNC_WAIT(N) do {} while (0)
#endif

// ---- ggml block layouts -------------------------------------------------
// Field order and padding must match ggml-common.h exactly; these structs are
// reinterpreted straight from the mapped GGUF file.

#define QK8_0 32
typedef struct {
    __half d;            // scale
    int8_t qs[QK8_0];
} block_q8_0;

// The legacy block-32 quants. A K-quant needs rows that are a multiple of 256,
// so models with awkward hidden sizes (Qwen2.5-0.5B's 896, for one) end up
// mostly encoded in these even in a "Q4_K_M" build.
#define QK4_0 32
typedef struct {
    __half d;
    uint8_t qs[QK4_0 / 2];
} block_q4_0;

typedef struct {
    __half d;
    __half m;
    uint8_t qs[QK4_0 / 2];
} block_q4_1;

#define QK5_0 32
typedef struct {
    __half d;
    uint8_t qh[4];        // one extra bit per weight
    uint8_t qs[QK5_0 / 2];
} block_q5_0;

typedef struct {
    __half d;
    __half m;
    uint8_t qh[4];
    uint8_t qs[QK5_0 / 2];
} block_q5_1;

// The high bits are stored as a little-endian u32 that the struct keeps
// byte-aligned only, so assemble it by hand rather than casting.
__device__ __forceinline__ uint32_t load_qh(const uint8_t* qh) {
    return (uint32_t)qh[0] | ((uint32_t)qh[1] << 8) | ((uint32_t)qh[2] << 16)
         | ((uint32_t)qh[3] << 24);
}

// AWQ, repacked. 128 weights per block with one f16 scale and one f16
// scale-times-zero, laid out output-major so a mat-vec block streams a whole
// row. Byte `b` holds weight `b` in its low nibble and weight `b + 64` in its
// high one, which is Q4_0's arrangement: it puts each 32-weight quarter — one
// Q8_1 activation block — in eight consecutive `int` loads.
//
// The zero point is stored already multiplied by the scale. The dot product
// applies it to the activation block's own sum rather than to each weight, the
// same trick Q4_1's `m` and Q4_K's `dmin` exist for.
#define QK_G128 128
typedef struct {
    __half2 ds;                  // {scale, scale * zero}
    uint8_t qs[QK_G128 / 2];
} block_q4_g128;

#define QK_K 256
#define K_SCALE_SIZE 12
typedef struct {
    __half d;                      // super-block scale for the 6-bit scales
    __half dmin;                   // super-block scale for the 6-bit mins
    uint8_t scales[K_SCALE_SIZE];  // 8 pairs of 6-bit scale/min
    uint8_t qs[QK_K / 2];          // 4-bit quants
} block_q4_K;

typedef struct {
    uint8_t ql[QK_K / 2];      // lower 4 bits
    uint8_t qh[QK_K / 4];      // upper 2 bits
    int8_t scales[QK_K / 16];  // 8-bit block scales
    __half d;                  // super-block scale
} block_q6_K;

// Unpack the 6-bit scale/min pair `j` (0..7) out of a Q4_K super-block.
__device__ __forceinline__ void q4k_scale_min(const uint8_t* q, int j,
                                              uint8_t* d, uint8_t* m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}
