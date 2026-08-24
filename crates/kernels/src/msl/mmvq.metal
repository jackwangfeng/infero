// The fused norms, and the Q8_1 activation quantiser they feed.
//
// The Metal twin of the parts of `cu/mmvq.cu` that sit on the main path. The
// integer mat-vec itself -- the `dp4a` dot products borrowed from llama.cpp --
// is not here: it needs a four-way byte dot product, and the sensible Metal
// equivalent is a different formulation rather than a transliteration. Until
// then a quantized checkpoint's decode step takes the float `gemv` family in
// `quant.metal`, which the dispatch already selects when `has_mmvq` is false.
//
// What *is* here is on every model's path regardless of quantisation:
// `rms_norm` prefers the register-resident form whenever a row fits it, which
// is every `d_model` in these models.

#define QK8_1 32
/// Registers per thread in the fused norms. Must match the host's `RMS_REGS`,
/// which sizes the block so that `blockDim * RMS_REGS >= d`.
#define RMS_REGS 8

/// Activation block: `d` scales the quants, `s` is the sum of the original
/// floats, which lets a weight format with a per-group offset fold that offset
/// in without a second pass.
typedef struct {
    half2 ds;
    char qs[QK8_1];
} block_q8_1;

/// Quantise an activation row to Q8_1. One SIMD group a block of 32.
kernel void quantize_q8_1_f32(device block_q8_1* y      [[buffer(0)]],
                              device const float* x     [[buffer(1)]],
                              constant int& n           [[buffer(2)]],
                              uint3 tgid  [[threadgroup_position_in_grid]],
                              uint3 tid   [[thread_position_in_threadgroup]],
                              uint3 tgdim [[threads_per_threadgroup]]) {
    const int i = int(tgid.x * tgdim.x + tid.x);
    const int lane = int(tid.x % WARP_SIZE);
    const int block = i / QK8_1;

    const float xi = i < n ? x[i] : 0.0f;
    const float amax = simd_max(fabs(xi));
    const float sum = simd_sum(xi);

    const float d = amax / 127.0f;
    const char q = amax == 0.0f ? 0 : char(rint(xi / d));

    if (i < n) y[block].qs[lane] = q;
    if (lane == 0 && i < n) {
        y[block].ds = half2(half(d), half(sum));
    }
}

/// RMSNorm with the row held in registers, and an optional f16 copy.
///
/// The offset has to be conditional, not the store: `hout + token * d` is
/// non-null for every row but the first, so checking the *row* pointer would
/// pass at one token and write out of bounds at two. The CUDA side's test
/// caught exactly that.
kernel void rms_norm_f16_f32(device float* out            [[buffer(0)]],
                             device half* hout            [[buffer(1)]],
                             device const float* x        [[buffer(2)]],
                             device const float* weight   [[buffer(3)]],
                             constant int& d              [[buffer(4)]],
                             constant float& eps          [[buffer(5)]],
                             uint3 tgid  [[threadgroup_position_in_grid]],
                             uint3 tid   [[thread_position_in_threadgroup]],
                             uint3 tgdim [[threads_per_threadgroup]]) {
    BLOCK_REDUCE_SCRATCH

    const int token = int(tgid.x);
    device const float* row = x + size_t(token) * d;
    device float* orow = out + size_t(token) * d;
    device half* hrow = hout ? hout + size_t(token) * d : nullptr;
    const int t = int(tid.x);

    float v[RMS_REGS];
    float acc = 0.0f;
    for (int k = 0; k < RMS_REGS; ++k) {
        const int i = k * int(tgdim.x) + t;
        v[k] = (i < d) ? row[i] : 0.0f;
        acc += v[k] * v[k];
    }
    const float scale = rsqrt(BLOCK_SUM(acc, tid.x, tgdim.x) / float(d) + eps);

    for (int k = 0; k < RMS_REGS; ++k) {
        const int i = k * int(tgdim.x) + t;
        if (i < d) {
            v[k] *= scale * weight[i];
            orow[i] = v[k];
            if (hrow) hrow[i] = half(v[k]);
        }
    }
}

/// The same, with the previous layer's residual added on the way in -- which is
/// what lets the FFN's residual be paid for by the next layer's norm.
kernel void add_rms_norm_f16_f32(device float* out            [[buffer(0)]],
                                 device half* hout            [[buffer(1)]],
                                 device float* x              [[buffer(2)]],
                                 device const float* b        [[buffer(3)]],
                                 device const float* weight   [[buffer(4)]],
                                 constant int& d              [[buffer(5)]],
                                 constant float& eps          [[buffer(6)]],
                                 uint3 tgid  [[threadgroup_position_in_grid]],
                                 uint3 tid   [[thread_position_in_threadgroup]],
                                 uint3 tgdim [[threads_per_threadgroup]]) {
    BLOCK_REDUCE_SCRATCH

    const int token = int(tgid.x);
    device float* row = x + size_t(token) * d;
    device const float* brow = b + size_t(token) * d;
    device float* orow = out + size_t(token) * d;
    device half* hrow = hout ? hout + size_t(token) * d : nullptr;
    const int t = int(tid.x);

    float v[RMS_REGS];
    float acc = 0.0f;
    for (int k = 0; k < RMS_REGS; ++k) {
        const int i = k * int(tgdim.x) + t;
        v[k] = 0.0f;
        if (i < d) {
            v[k] = row[i] + brow[i];
            row[i] = v[k];
        }
        acc += v[k] * v[k];
    }
    const float scale = rsqrt(BLOCK_SUM(acc, tid.x, tgdim.x) / float(d) + eps);

    for (int k = 0; k < RMS_REGS; ++k) {
        const int i = k * int(tgdim.x) + t;
        if (i < d) {
            v[k] *= scale * weight[i];
            orow[i] = v[k];
            if (hrow) hrow[i] = half(v[k]);
        }
    }
}

/// RMSNorm that also emits the Q8_1 activation the integer mat-vec wants.
///
/// The row stays in registers across all three phases. Reading it back from
/// device memory for the scale, and again for the quantisation, is what made
/// this cost more than the two kernels it replaced: one threadgroup walks the
/// whole row alone, so every extra pass is the full latency again.
///
/// And the layout works out: Q8_1 block `b` covers elements 32b..32b+31, and
/// with a stride of `tgdim.x` those land in one SIMD group's lanes 0..31 at
/// register slot `32b / tgdim.x`. So the strided load already produced the
/// layout the per-block scale wants -- no threadgroup memory, no barrier, no
/// re-read.
kernel void rms_norm_q8_1_f32(device float* out             [[buffer(0)]],
                              device block_q8_1* qout       [[buffer(1)]],
                              device const float* x         [[buffer(2)]],
                              device const float* weight    [[buffer(3)]],
                              constant int& d               [[buffer(4)]],
                              constant float& eps           [[buffer(5)]],
                              uint3 tgid  [[threadgroup_position_in_grid]],
                              uint3 tid   [[thread_position_in_threadgroup]],
                              uint3 tgdim [[threads_per_threadgroup]]) {
    BLOCK_REDUCE_SCRATCH

    const int token = int(tgid.x);
    device const float* row = x + size_t(token) * d;
    device float* orow = out + size_t(token) * d;
    const int t = int(tid.x);

    float v[RMS_REGS];
    float acc = 0.0f;
    for (int k = 0; k < RMS_REGS; ++k) {
        const int i = k * int(tgdim.x) + t;
        v[k] = (i < d) ? row[i] : 0.0f;
        acc += v[k] * v[k];
    }
    const float scale = rsqrt(BLOCK_SUM(acc, tid.x, tgdim.x) / float(d) + eps);

    for (int k = 0; k < RMS_REGS; ++k) {
        const int i = k * int(tgdim.x) + t;
        if (i < d) {
            v[k] *= scale * weight[i];
            orow[i] = v[k];
        }
    }

    const int lane = int(tid.x % WARP_SIZE);
    const int warp = int(tid.x / WARP_SIZE);
    const int warps = int(tgdim.x) / WARP_SIZE;
    const int n_blocks = d / QK8_1;
    device block_q8_1* qrow = qout + size_t(token) * n_blocks;
    for (int k = 0; k < RMS_REGS; ++k) {
        const int b = k * warps + warp;
        if (b >= n_blocks) continue;
        const float amax = simd_max(fabs(v[k]));
        const float sum = simd_sum(v[k]);
        const float dq = amax / 127.0f;
        qrow[b].qs[lane] = (amax == 0.0f) ? 0 : char(rint(v[k] / dq));
        if (lane == 0) {
            qrow[b].ds = half2(half(dq), half(sum));
        }
    }
}
