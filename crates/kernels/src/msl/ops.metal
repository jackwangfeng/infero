// The operator set for an F16 decoder-only forward pass, the Metal twin of the
// corresponding kernels in `cu/ops.cu` and `cu/quant.cu`.
//
// Transliterations, not reinterpretations: each kernel keeps its CUDA
// counterpart's name, parameter order and arithmetic, so that `[[buffer(n)]]`
// indices follow the `.arg()` chain the host already writes and a reader can
// diff the two bodies. Where the CUDA version reads `blockIdx.x`, this one
// reads `threadgroup_position_in_grid`; where it reads `blockDim.x`,
// `threads_per_threadgroup`. That is the whole of the difference for the
// elementwise half of the file.

// ---- elementwise ---------------------------------------------------------

kernel void add_f32(device float* out           [[buffer(0)]],
                    device const float* a       [[buffer(1)]],
                    device const float* b       [[buffer(2)]],
                    constant int& n             [[buffer(3)]],
                    uint3 tgid  [[threadgroup_position_in_grid]],
                    uint3 tid   [[thread_position_in_threadgroup]],
                    uint3 tgdim [[threads_per_threadgroup]]) {
    const int i = int(tgid.x * tgdim.x + tid.x);
    if (i < n) out[i] = a[i] + b[i];
}

kernel void add_assign_f32(device float* out          [[buffer(0)]],
                           device const float* b      [[buffer(1)]],
                           constant int& n            [[buffer(2)]],
                           uint3 tgid  [[threadgroup_position_in_grid]],
                           uint3 tid   [[thread_position_in_threadgroup]],
                           uint3 tgdim [[threads_per_threadgroup]]) {
    const int i = int(tgid.x * tgdim.x + tid.x);
    if (i < n) out[i] += b[i];
}

kernel void add_bias_f32(device float* out          [[buffer(0)]],
                         device const float* bias   [[buffer(1)]],
                         constant int& n_cols       [[buffer(2)]],
                         constant int& n_rows       [[buffer(3)]],
                         uint3 tgid  [[threadgroup_position_in_grid]],
                         uint3 tid   [[thread_position_in_threadgroup]],
                         uint3 tgdim [[threads_per_threadgroup]]) {
    const int j = int(tgid.x * tgdim.x + tid.x);
    const int t = int(tgid.y);
    if (j < n_cols && t < n_rows) out[size_t(t) * n_cols + j] += bias[j];
}

/// `out = silu(gate) * up` where the two halves sit in one `[row][2 * d_ff]`
/// buffer. Matches `silu_mul_split_f32`.
kernel void silu_mul_split_f32(device float* out         [[buffer(0)]],
                               device const float* xy    [[buffer(1)]],
                               constant int& d_ff        [[buffer(2)]],
                               constant int& total       [[buffer(3)]],
                               uint3 tgid  [[threadgroup_position_in_grid]],
                               uint3 tid   [[thread_position_in_threadgroup]],
                               uint3 tgdim [[threads_per_threadgroup]]) {
    const int i = int(tgid.x * tgdim.x + tid.x);
    if (i >= total) return;
    const int row = i / d_ff;
    const int col = i - row * d_ff;
    device const float* r = xy + size_t(row) * 2 * d_ff;
    const float g = r[col];
    out[i] = (g / (1.0f + exp(-g))) * r[d_ff + col];
}

/// Embedding lookup out of an f16 table. The CUDA path dequantises the table
/// first and then uses `take_rows_f32`; folding the conversion in here saves a
/// full-vocab copy that a batch of one token would never read.
kernel void embed_f16(device float* out           [[buffer(0)]],
                      device const half* table    [[buffer(1)]],
                      device const int* rows      [[buffer(2)]],
                      constant int& d             [[buffer(3)]],
                      uint3 tgid  [[threadgroup_position_in_grid]],
                      uint3 tid   [[thread_position_in_threadgroup]],
                      uint3 tgdim [[threads_per_threadgroup]]) {
    const int i = int(tgid.x * tgdim.x + tid.x);
    const int r = int(tgid.y);
    if (i >= d) return;
    out[size_t(r) * d + i] = float(table[size_t(rows[r]) * d + i]);
}

// ---- norms ---------------------------------------------------------------

/// One threadgroup a row, reduce, scale. Matches `rms_norm_f32`: the row is
/// read twice rather than held in registers, which is the simple variant. The
/// CUDA side also has a register-resident version it prefers whenever the row
/// fits; that is an optimisation this port has not earned yet.
kernel void rms_norm_f32(device float* out           [[buffer(0)]],
                         device const float* x       [[buffer(1)]],
                         device const float* weight  [[buffer(2)]],
                         constant int& d             [[buffer(3)]],
                         constant float& eps         [[buffer(4)]],
                         uint3 tgid  [[threadgroup_position_in_grid]],
                         uint3 tid   [[thread_position_in_threadgroup]],
                         uint3 tgdim [[threads_per_threadgroup]]) {
    BLOCK_REDUCE_SCRATCH

    const size_t row = size_t(tgid.x) * d;
    device const float* xr = x + row;
    device float* orow = out + row;

    float acc = 0.0f;
    for (int i = int(tid.x); i < d; i += int(tgdim.x)) {
        const float v = xr[i];
        acc += v * v;
    }
    const float sum = BLOCK_SUM(acc, tid.x, tgdim.x);
    const float scale = rsqrt(sum / float(d) + eps);

    for (int i = int(tid.x); i < d; i += int(tgdim.x)) {
        orow[i] = xr[i] * scale * weight[i];
    }
}

// ---- rotary --------------------------------------------------------------

/// NeoX-style rotation: element `i` pairs with `i + d_head / 2`.
///
/// The frequency uses `d_head` in the exponent, not the rotary width, and the
/// per-index `freq_factors` divisor is applied after the base -- both copied
/// from `rope_neox_f32` rather than rederived. Getting either wrong produces
/// fluent text with bad long-range retrieval, which is the failure mode this
/// port can least afford to introduce quietly.
kernel void rope_neox_f32(device float* x                    [[buffer(0)]],
                          device const int* positions        [[buffer(1)]],
                          device const float* freq_factors   [[buffer(2)]],
                          constant int& n_heads              [[buffer(3)]],
                          constant int& d_head               [[buffer(4)]],
                          constant float& theta_base         [[buffer(5)]],
                          constant float& freq_scale         [[buffer(6)]],
                          uint3 tgid  [[threadgroup_position_in_grid]],
                          uint3 tid   [[thread_position_in_threadgroup]],
                          uint3 tgdim [[threads_per_threadgroup]]) {
    const int half_d = d_head / 2;
    const int i = int(tgid.x * tgdim.x + tid.x);
    if (i >= half_d) return;

    const int head = int(tgid.y);
    const int token = int(tgid.z);

    const float pos = float(positions[token]) * freq_scale;
    const float inv_freq = pow(theta_base, -2.0f * float(i) / float(d_head));
    const float angle = pos * inv_freq / freq_factors[i];
    const float sin_a = sin(angle);
    const float cos_a = cos(angle);

    device float* row = x + (size_t(token) * n_heads + head) * d_head;
    const float a = row[i];
    const float b = row[i + half_d];
    row[i] = a * cos_a - b * sin_a;
    row[i + half_d] = a * sin_a + b * cos_a;
}

// ---- KV cache ------------------------------------------------------------

/// Append one token's K and V at `pos`, converting to f16 on the way in.
/// Layout is `[position][kv_head][d_head]`, which is what the attention kernel
/// below strides over.
kernel void store_kv_f16(device half* kcache          [[buffer(0)]],
                         device half* vcache          [[buffer(1)]],
                         device const float* k        [[buffer(2)]],
                         device const float* v        [[buffer(3)]],
                         constant int& n_kv           [[buffer(4)]],
                         constant int& d_head         [[buffer(5)]],
                         constant int& pos            [[buffer(6)]],
                         uint3 tgid  [[threadgroup_position_in_grid]],
                         uint3 tid   [[thread_position_in_threadgroup]],
                         uint3 tgdim [[threads_per_threadgroup]]) {
    const int i = int(tgid.x * tgdim.x + tid.x);
    const int total = n_kv * d_head;
    if (i >= total) return;
    const size_t dst = size_t(pos) * total + i;
    kcache[dst] = half(k[i]);
    vcache[dst] = half(v[i]);
}

// ---- attention -----------------------------------------------------------

/// Grouped-query attention for a single query token, one threadgroup a head.
///
/// Three phases with a block reduction between each: scores, softmax, weighted
/// sum. The scores live in threadgroup memory, which caps `kv_len` at
/// `MAX_KV_SCORES` -- the host checks it rather than letting the kernel walk
/// off the end. The CUDA side does not have this limit because it splits long
/// contexts across blocks and reduces afterwards (`attn_flash_reduce_f32`);
/// that split is a later step here.
#define MAX_KV_SCORES 4096

kernel void attn_decode_f32(device float* out            [[buffer(0)]],
                            device const float* q        [[buffer(1)]],
                            device const half* kcache    [[buffer(2)]],
                            device const half* vcache    [[buffer(3)]],
                            constant int& n_heads        [[buffer(4)]],
                            constant int& n_kv           [[buffer(5)]],
                            constant int& d_head         [[buffer(6)]],
                            constant int& kv_len         [[buffer(7)]],
                            constant float& scale        [[buffer(8)]],
                            uint3 tgid  [[threadgroup_position_in_grid]],
                            uint3 tid   [[thread_position_in_threadgroup]],
                            uint3 tgdim [[threads_per_threadgroup]]) {
    BLOCK_REDUCE_SCRATCH
    threadgroup float sc[MAX_KV_SCORES];

    const int head = int(tgid.x);
    // Integer division, not a shift: the ratio is 7 on this model (14 query
    // heads over 2 key/value heads) and need not be a power of two.
    const int kvh = head / (n_heads / n_kv);
    const int stride = n_kv * d_head;

    device const float* qh = q + size_t(head) * d_head;

    // Phase 1: scores.
    float local_max = -INFINITY;
    for (int j = int(tid.x); j < kv_len; j += int(tgdim.x)) {
        device const half* kj = kcache + size_t(j) * stride + size_t(kvh) * d_head;
        float dot = 0.0f;
        for (int i = 0; i < d_head; ++i) dot += qh[i] * float(kj[i]);
        const float s = dot * scale;
        sc[j] = s;
        local_max = fmax(local_max, s);
    }
    const float m = BLOCK_MAX(local_max, tid.x, tgdim.x);

    // Phase 2: exponentiate in place and sum.
    float local_sum = 0.0f;
    for (int j = int(tid.x); j < kv_len; j += int(tgdim.x)) {
        const float e = exp(sc[j] - m);
        sc[j] = e;
        local_sum += e;
    }
    const float denom = BLOCK_SUM(local_sum, tid.x, tgdim.x);
    const float inv = 1.0f / denom;

    // Phase 3: weighted sum over V. One thread a channel; the threadgroup is
    // wider than `d_head` because phase 1 wanted the parallelism, so the tail
    // threads sit out. Fixing that means splitting the phases across two
    // launches, which is a measurement this port has not made.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = int(tid.x); i < d_head; i += int(tgdim.x)) {
        float acc = 0.0f;
        for (int j = 0; j < kv_len; ++j) {
            device const half* vj = vcache + size_t(j) * stride + size_t(kvh) * d_head;
            acc += sc[j] * float(vj[i]);
        }
        out[size_t(head) * d_head + i] = acc * inv;
    }
}

// ---- Qwen3.5/3.8 additions ----------------------------------------------

/// Per-head RMSNorm over `d_head`, applied where the head lies inside a row of
/// `row_stride`. Qwen3 normalizes q and k this way *before* RoPE.
kernel void qk_norm_f32(device float* buf              [[buffer(0)]],
                        device const float* weight     [[buffer(1)]],
                        constant int& n_heads          [[buffer(2)]],
                        constant int& d_head           [[buffer(3)]],
                        constant int& row_stride       [[buffer(4)]],
                        constant int& offset           [[buffer(5)]],
                        constant float& eps            [[buffer(6)]],
                        uint3 tgid  [[threadgroup_position_in_grid]],
                        uint3 tid   [[thread_position_in_threadgroup]],
                        uint3 tgdim [[threads_per_threadgroup]]) {
    BLOCK_REDUCE_SCRATCH

    const int token = int(tgid.x) / n_heads;
    const int head = int(tgid.x) % n_heads;
    device float* h = buf + size_t(token) * row_stride + offset
                    + size_t(head) * d_head;

    float acc = 0.0f;
    for (int i = int(tid.x); i < d_head; i += int(tgdim.x)) {
        const float v = h[i];
        acc += v * v;
    }
    const float scale = rsqrt(BLOCK_SUM(acc, tid.x, tgdim.x) / float(d_head) + eps);
    for (int i = int(tid.x); i < d_head; i += int(tgdim.x)) {
        h[i] = h[i] * scale * weight[i];
    }
}

/// Partial rotary: only the first `rotary_dim` of each head rotates, the rest
/// passes through untouched.
///
/// Pairing is `(i, i + rotary_dim/2)` -- the non-interleaved `rotate_half`
/// convention, not the adjacent-pair one. The tables come from the host in f64
/// and are already duplicated across both halves, because the two obvious f32
/// formulations of `theta^(-2i/rot)` differ by an ulp that amplifies to 2.5e-3
/// in the cosine at position 130000.
kernel void rope_partial_f32(device float* x                [[buffer(0)]],
                             device const float* cos_tab    [[buffer(1)]],
                             device const float* sin_tab    [[buffer(2)]],
                             constant int& heads            [[buffer(3)]],
                             constant int& head_dim         [[buffer(4)]],
                             constant int& rotary_dim       [[buffer(5)]],
                             uint3 tgid  [[threadgroup_position_in_grid]],
                             uint3 tid   [[thread_position_in_threadgroup]],
                             uint3 tgdim [[threads_per_threadgroup]]) {
    const int half_r = rotary_dim / 2;
    const int i = int(tgid.x * tgdim.x + tid.x);
    if (i >= half_r) return;
    const int head = int(tgid.y);
    const int token = int(tgid.z);

    device float* row = x + (size_t(token) * heads + head) * head_dim;
    device const float* c = cos_tab + size_t(token) * rotary_dim;
    device const float* s = sin_tab + size_t(token) * rotary_dim;

    const float a = row[i];
    const float b = row[i + half_r];
    // rotate_half puts -x2 where x1 was, so the first half takes `-b * sin` and
    // the second takes `+a * sin`.
    row[i] = a * c[i] - b * s[i];
    row[i + half_r] = b * c[i + half_r] + a * s[i + half_r];
}
