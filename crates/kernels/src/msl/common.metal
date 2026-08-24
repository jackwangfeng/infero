// Shared device helpers, the Metal twin of `cu/common.cuh`.
//
// Concatenated ahead of every MSL kernel source the same way that file is: the
// runtime compiler gets a string, not an include path.
//
// One structural difference from CUDA, and it shapes every reduction below.
// A CUDA `__device__` function may declare its own `__shared__` array, so
// `block_reduce_sum(v)` needs no scratch from its caller. MSL forbids
// `threadgroup` declarations outside a kernel, so the scratch has to be
// declared in the kernel and passed down. The `BLOCK_REDUCE_SCRATCH` macro
// declares it and `BLOCK_SUM` / `BLOCK_MAX` hide the plumbing, which keeps the
// kernel bodies readable next to their CUDA originals.

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

#define WARP_SIZE 32

// `simd_shuffle_xor` is the direct analogue of `__shfl_xor_sync`, and Apple's
// SIMD groups are 32 lanes wide like a CUDA warp -- so this is a transliteration
// rather than a redesign. `simd_sum` does the whole butterfly in one call and is
// what the reductions actually use; the explicit loop is kept for the places
// that need a partial result.
inline float warp_reduce_sum(float v) {
#pragma unroll
    for (uint offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        v += simd_shuffle_xor(v, offset);
    }
    return v;
}

inline float warp_reduce_max(float v) {
#pragma unroll
    for (uint offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        v = fmax(v, simd_shuffle_xor(v, offset));
    }
    return v;
}

// Block-wide sum, broadcast to every thread.
//
// The barrier on entry is not redundant: a kernel that reduces twice would
// otherwise have the second call overwrite `result` while slower threads were
// still reading the first. Same reasoning as the CUDA version, same fix.
inline float block_reduce_sum(float v, uint tid, uint nthreads,
                              threadgroup float* partial,
                              threadgroup float* result) {
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint lane = tid % WARP_SIZE;
    const uint warp = tid / WARP_SIZE;
    const uint warps = (nthreads + WARP_SIZE - 1) / WARP_SIZE;

    v = simd_sum(v);
    if (lane == 0) partial[warp] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (warp == 0) {
        float acc = (lane < warps) ? partial[lane] : 0.0f;
        acc = simd_sum(acc);
        if (lane == 0) *result = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return *result;
}

inline float block_reduce_max(float v, uint tid, uint nthreads,
                              threadgroup float* partial,
                              threadgroup float* result) {
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint lane = tid % WARP_SIZE;
    const uint warp = tid / WARP_SIZE;
    const uint warps = (nthreads + WARP_SIZE - 1) / WARP_SIZE;

    v = simd_max(v);
    if (lane == 0) partial[warp] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (warp == 0) {
        float acc = (lane < warps) ? partial[lane] : -INFINITY;
        acc = simd_max(acc);
        if (lane == 0) *result = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return *result;
}

// Declare the scratch a block reduction needs. One 32-float array (one slot a
// SIMD group, and 1024 threads is 32 groups) plus the broadcast slot.
#define BLOCK_REDUCE_SCRATCH            \
    threadgroup float _bp[WARP_SIZE];   \
    threadgroup float _br;

#define BLOCK_SUM(v, tid, n) block_reduce_sum((v), (tid), (n), _bp, &_br)
#define BLOCK_MAX(v, tid, n) block_reduce_max((v), (tid), (n), _bp, &_br)
