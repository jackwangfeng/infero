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
///
/// A *contiguous* cache, `[position][kv_head][d_head]`, which is what the
/// vertical-slice examples use. The engine's paged pool is a different layout
/// and a different kernel -- `store_kv_f16` and `store_kv2_f16` below, which
/// index through a slot table.
kernel void store_kv_contig_f16(device half* kcache          [[buffer(0)]],
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

// ---- the engine's paged-pool path ---------------------------------------
//
// Everything above this line serves the vertical-slice examples, which own a
// contiguous cache. What follows is what `tuili-model` actually launches: a
// paged KV pool addressed through a per-sequence slot table, so one batch can
// hold sequences of different lengths and a freed sequence's slots return to
// the pool without moving anyone else's.

kernel void silu_mul_f32(device float* out             [[buffer(0)]],
                         device const float* gate      [[buffer(1)]],
                         device const float* up        [[buffer(2)]],
                         constant int& n               [[buffer(3)]],
                         uint3 tgid  [[threadgroup_position_in_grid]],
                         uint3 tid   [[thread_position_in_threadgroup]],
                         uint3 tgdim [[threads_per_threadgroup]]) {
    const int i = int(tgid.x * tgdim.x + tid.x);
    if (i >= n) return;
    const float g = gate[i];
    out[i] = (g / (1.0f + exp(-g))) * up[i];
}

kernel void take_rows_f32(device float* out            [[buffer(0)]],
                          device const float* in       [[buffer(1)]],
                          device const int* rows       [[buffer(2)]],
                          constant int& d              [[buffer(3)]],
                          uint3 tgid  [[threadgroup_position_in_grid]],
                          uint3 tid   [[thread_position_in_threadgroup]],
                          uint3 tgdim [[threads_per_threadgroup]]) {
    const int i = int(tgid.x * tgdim.x + tid.x);
    const int r = int(tgid.y);
    if (i >= d) return;
    out[size_t(r) * d + i] = in[size_t(rows[r]) * d + i];
}

/// Four elements a thread.
///
/// One element a thread leaves each with a single load and nothing to overlap
/// its latency against, so the kernel runs at memory latency rather than
/// bandwidth -- the CUDA side measured 167 GB/s on a card that does 1800. The
/// vector path it then takes is not ported: MSL cannot check a `device` pointer's
/// alignment portably, and four independent scalar elements already buys the
/// latency hiding that mattered.
kernel void f32_to_f16(device half* out                [[buffer(0)]],
                       device const float* in          [[buffer(1)]],
                       constant int& n                 [[buffer(2)]],
                       uint3 tgid  [[threadgroup_position_in_grid]],
                       uint3 tid   [[thread_position_in_threadgroup]],
                       uint3 tgdim [[threads_per_threadgroup]]) {
    const int base = int(tgid.x * tgdim.x + tid.x) * 4;
    if (base >= n) return;
    for (int j = base; j < base + 4 && j < n; ++j) out[j] = half(in[j]);
}

/// Interleaved-pair rotary: element `2i` pairs with `2i + 1`.
///
/// The other convention from `rope_neox_f32`, and which one a checkpoint wants
/// follows from its architecture rather than from the file -- llama, baichuan
/// and minicpm permute q and k at conversion so that this pairing reproduces
/// Hugging Face's rotate-half.
kernel void rope_norm_f32(device float* x                    [[buffer(0)]],
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
    const float a = row[2 * i];
    const float b = row[2 * i + 1];
    row[2 * i] = a * cos_a - b * sin_a;
    row[2 * i + 1] = a * sin_a + b * cos_a;
}

kernel void write_slot_table(device int* table               [[buffer(0)]],
                             device const int* seq_of        [[buffer(1)]],
                             device const int* positions     [[buffer(2)]],
                             device const int* slots         [[buffer(3)]],
                             constant int& stride            [[buffer(4)]],
                             constant int& n_tokens          [[buffer(5)]],
                             uint3 tgid  [[threadgroup_position_in_grid]],
                             uint3 tid   [[thread_position_in_threadgroup]],
                             uint3 tgdim [[threads_per_threadgroup]]) {
    const int i = int(tgid.x * tgdim.x + tid.x);
    if (i >= n_tokens) return;
    table[size_t(seq_of[i]) * stride + positions[i]] = slots[i];
}

/// One plane of the paged pool, at the slot each token was given.
kernel void store_kv_f16(device half* pool               [[buffer(0)]],
                         device const float* src         [[buffer(1)]],
                         device const int* slots         [[buffer(2)]],
                         constant int& n_kv_heads        [[buffer(3)]],
                         constant int& d_head            [[buffer(4)]],
                         constant int& n_slots           [[buffer(5)]],
                         constant int& n_tokens          [[buffer(6)]],
                         uint3 tgid  [[threadgroup_position_in_grid]],
                         uint3 tid   [[thread_position_in_threadgroup]],
                         uint3 tgdim [[threads_per_threadgroup]]) {
    const int i = int(tgid.x * tgdim.x + tid.x);
    if (i >= d_head) return;
    const int head = int(tgid.y);
    const int token = int(tgid.z);
    if (token >= n_tokens) return;

    const int slot = slots[token];
    if (slot < 0 || slot >= n_slots) return;

    const size_t dst = (size_t(head) * n_slots + slot) * d_head + i;
    const size_t s = (size_t(token) * n_kv_heads + head) * d_head + i;
    pool[dst] = half(src[s]);
}

/// Both planes in one dispatch: `y < n_kv_heads` is K, the rest is V.
kernel void store_kv2_f16(device half* k_pool            [[buffer(0)]],
                          device half* v_pool            [[buffer(1)]],
                          device const float* k_src      [[buffer(2)]],
                          device const float* v_src      [[buffer(3)]],
                          device const int* slots        [[buffer(4)]],
                          constant int& n_kv_heads       [[buffer(5)]],
                          constant int& d_head           [[buffer(6)]],
                          constant int& n_slots          [[buffer(7)]],
                          constant int& n_tokens         [[buffer(8)]],
                          uint3 tgid  [[threadgroup_position_in_grid]],
                          uint3 tid   [[thread_position_in_threadgroup]],
                          uint3 tgdim [[threads_per_threadgroup]]) {
    const int i = int(tgid.x * tgdim.x + tid.x);
    if (i >= d_head) return;
    const int y = int(tgid.y);
    const int token = int(tgid.z);
    if (token >= n_tokens) return;

    const int slot = slots[token];
    if (slot < 0 || slot >= n_slots) return;

    const bool is_k = y < n_kv_heads;
    const int head = is_k ? y : y - n_kv_heads;
    device half* pool = is_k ? k_pool : v_pool;
    device const float* src = is_k ? k_src : v_src;

    const size_t dst = (size_t(head) * n_slots + slot) * d_head + i;
    const size_t s = (size_t(token) * n_kv_heads + head) * d_head + i;
    pool[dst] = half(src[s]);
}

/// The same, reading q/k/v out of one packed projection row.
kernel void store_kv2_packed_f16(device half* k_pool          [[buffer(0)]],
                                 device half* v_pool          [[buffer(1)]],
                                 device const float* packed    [[buffer(2)]],
                                 constant int& stride          [[buffer(3)]],
                                 constant int& k_off           [[buffer(4)]],
                                 constant int& v_off           [[buffer(5)]],
                                 device const int* slots       [[buffer(6)]],
                                 constant int& n_kv_heads      [[buffer(7)]],
                                 constant int& d_head          [[buffer(8)]],
                                 constant int& n_slots         [[buffer(9)]],
                                 constant int& n_tokens        [[buffer(10)]],
                                 uint3 tgid  [[threadgroup_position_in_grid]],
                                 uint3 tid   [[thread_position_in_threadgroup]],
                                 uint3 tgdim [[threads_per_threadgroup]]) {
    const int i = int(tgid.x * tgdim.x + tid.x);
    if (i >= d_head) return;
    const int y = int(tgid.y);
    const int token = int(tgid.z);
    if (token >= n_tokens) return;

    const int slot = slots[token];
    if (slot < 0 || slot >= n_slots) return;

    const bool is_k = y < n_kv_heads;
    const int head = is_k ? y : y - n_kv_heads;
    device half* pool = is_k ? k_pool : v_pool;

    const size_t dst = (size_t(head) * n_slots + slot) * d_head + i;
    const size_t s = size_t(token) * stride + (is_k ? k_off : v_off)
                   + size_t(head) * d_head + i;
    pool[dst] = half(packed[s]);
}

/// Rotary over a packed `[q | k | v]` row, writing q out and k in place.
///
/// The tail lanes matter: `k` rotates in place so its unrotated dimensions are
/// already where they belong, but `q` is *copied* to a separate buffer, and
/// writing only the first `rotary_dim` would leave `[rotary_dim, d_head)` of
/// `q_dst` holding whatever the previous layer left there. Three quarters of
/// every query head stale rather than absent, which runs.
kernel void rope_qk_packed_f32(device float* q_dst              [[buffer(0)]],
                               device float* packed             [[buffer(1)]],
                               constant int& stride             [[buffer(2)]],
                               constant int& q_off              [[buffer(3)]],
                               constant int& k_off              [[buffer(4)]],
                               device const int* positions      [[buffer(5)]],
                               device const float* freq_factors [[buffer(6)]],
                               constant int& n_heads            [[buffer(7)]],
                               constant int& n_kv_heads         [[buffer(8)]],
                               constant int& d_head             [[buffer(9)]],
                               constant int& rotary_dim         [[buffer(10)]],
                               constant float& theta_base       [[buffer(11)]],
                               constant float& freq_scale       [[buffer(12)]],
                               constant int& interleaved        [[buffer(13)]],
                               uint3 tgid  [[threadgroup_position_in_grid]],
                               uint3 tid   [[thread_position_in_threadgroup]],
                               uint3 tgdim [[threads_per_threadgroup]]) {
    const int half_r = rotary_dim / 2;
    const int tail = d_head - rotary_dim;
    const int i = int(tgid.x * tgdim.x + tid.x);
    if (i >= half_r + tail) return;

    const int y = int(tgid.y);
    const int token = int(tgid.z);
    const bool is_q = y < n_heads;
    const int head = is_q ? y : y - n_heads;

    device const float* src = packed + size_t(token) * stride
                            + (is_q ? q_off : k_off) + size_t(head) * d_head;
    device float* dst = is_q
        ? q_dst + (size_t(token) * n_heads + head) * d_head
        : packed + size_t(token) * stride + k_off + size_t(head) * d_head;

    if (i >= half_r) {
        if (is_q) {
            const int d = rotary_dim + (i - half_r);
            dst[d] = src[d];
        }
        return;
    }

    const float pos = float(positions[token]) * freq_scale;
    const float inv_freq = pow(theta_base, -2.0f * float(i) / float(rotary_dim));
    const float angle = pos * inv_freq / freq_factors[i];
    const float sin_a = sin(angle);
    const float cos_a = cos(angle);

    const int ia = interleaved != 0 ? 2 * i : i;
    const int ib = interleaved != 0 ? 2 * i + 1 : i + half_r;
    const float a = src[ia], b = src[ib];
    dst[ia] = a * cos_a - b * sin_a;
    dst[ib] = a * sin_a + b * cos_a;
}

/// Scores against the paged pool. One SIMD group a key.
///
/// The mask is per *token*, not per batch: each token carries its own position,
/// so one batch can hold sequences of completely different lengths.
kernel void attn_scores_f32(device float* scores              [[buffer(0)]],
                            device const float* q             [[buffer(1)]],
                            device const half* k_cache        [[buffer(2)]],
                            device const int* seq_of          [[buffer(3)]],
                            device const int* positions       [[buffer(4)]],
                            device const int* slot_table      [[buffer(5)]],
                            constant int& table_stride        [[buffer(6)]],
                            constant int& n_heads             [[buffer(7)]],
                            constant int& n_kv_heads          [[buffer(8)]],
                            constant int& d_head              [[buffer(9)]],
                            constant int& n_slots             [[buffer(10)]],
                            constant int& kv_len              [[buffer(11)]],
                            constant float& scale             [[buffer(12)]],
                            uint3 tgid  [[threadgroup_position_in_grid]],
                            uint3 tid   [[thread_position_in_threadgroup]],
                            uint3 tgdim [[threads_per_threadgroup]],
                            uint3 ngrid [[threadgroups_per_grid]]) {
    const int j = int(tgid.x) * int(tgdim.x / WARP_SIZE) + int(tid.x / WARP_SIZE);
    if (j >= kv_len) return;

    const int head = int(tgid.y);
    const int token = int(tgid.z);
    const int lane = int(tid.x % WARP_SIZE);
    const int n_tok = int(ngrid.z);

    if (j > positions[token]) {
        if (lane == 0) {
            scores[(size_t(head) * n_tok + token) * kv_len + j] = -INFINITY;
        }
        return;
    }

    const int kv_head = head / (n_heads / n_kv_heads);
    const int slot = slot_table[size_t(seq_of[token]) * table_stride + j];
    device const float* qr = q + (size_t(token) * n_heads + head) * d_head;
    device const half* kr = k_cache + (size_t(kv_head) * n_slots + slot) * d_head;

    float acc = 0.0f;
    for (int i = lane; i < d_head; i += WARP_SIZE) {
        acc += qr[i] * float(kr[i]);
    }
    acc = simd_sum(acc);

    if (lane == 0) {
        scores[(size_t(head) * n_tok + token) * kv_len + j] = acc * scale;
    }
}

kernel void attn_softmax_f32(device float* scores      [[buffer(0)]],
                             constant int& kv_len      [[buffer(1)]],
                             uint3 tgid  [[threadgroup_position_in_grid]],
                             uint3 tid   [[thread_position_in_threadgroup]],
                             uint3 tgdim [[threads_per_threadgroup]],
                             uint3 ngrid [[threadgroups_per_grid]]) {
    BLOCK_REDUCE_SCRATCH

    device float* row =
        scores + (size_t(tgid.x) * ngrid.y + tgid.y) * kv_len;

    float local_max = -INFINITY;
    for (int j = int(tid.x); j < kv_len; j += int(tgdim.x)) {
        local_max = fmax(local_max, row[j]);
    }
    const float m = BLOCK_MAX(local_max, tid.x, tgdim.x);

    float local_sum = 0.0f;
    for (int j = int(tid.x); j < kv_len; j += int(tgdim.x)) {
        const float e = exp(row[j] - m);
        row[j] = e;
        local_sum += e;
    }
    const float inv = 1.0f / BLOCK_SUM(local_sum, tid.x, tgdim.x);

    for (int j = int(tid.x); j < kv_len; j += int(tgdim.x)) {
        row[j] *= inv;
    }
}

/// Weighted sum over V, gathering through the slot table.
///
/// The loop stops at the token's own position rather than running to `kv_len`:
/// masked entries are exactly zero after the softmax, but their slot-table
/// entries belong to no sequence, so multiplying whatever they address by zero
/// would be a read out of bounds that happens to be harmless.
kernel void attn_output_f32(device float* out                [[buffer(0)]],
                            device const float* scores       [[buffer(1)]],
                            device const half* v_cache       [[buffer(2)]],
                            device const int* seq_of         [[buffer(3)]],
                            device const int* positions      [[buffer(4)]],
                            device const int* slot_table     [[buffer(5)]],
                            constant int& table_stride       [[buffer(6)]],
                            constant int& n_heads            [[buffer(7)]],
                            constant int& n_kv_heads         [[buffer(8)]],
                            constant int& d_head             [[buffer(9)]],
                            constant int& n_slots            [[buffer(10)]],
                            constant int& kv_len             [[buffer(11)]],
                            uint3 tgid  [[threadgroup_position_in_grid]],
                            uint3 tid   [[thread_position_in_threadgroup]],
                            uint3 tgdim [[threads_per_threadgroup]],
                            uint3 ngrid [[threadgroups_per_grid]]) {
    const int head = int(tgid.x);
    const int token = int(tgid.y);
    const int i = int(tid.x);
    if (i >= d_head) return;

    const int kv_head = head / (n_heads / n_kv_heads);
    device const float* srow =
        scores + (size_t(head) * ngrid.y + token) * kv_len;
    device const half* vbase = v_cache + size_t(kv_head) * n_slots * d_head;
    device const int* table = slot_table + size_t(seq_of[token]) * table_stride;
    const int last = positions[token];

    float acc = 0.0f;
    for (int j = 0; j <= last && j < kv_len; ++j) {
        acc += srow[j] * float(vbase[size_t(table[j]) * d_head + i]);
    }
    out[(size_t(token) * n_heads + head) * d_head + i] = acc;
}

kernel void silu_mul_split_f16_f32(device float* out          [[buffer(0)]],
                                   device half* hout          [[buffer(1)]],
                                   device const float* xy     [[buffer(2)]],
                                   constant int& d_ff         [[buffer(3)]],
                                   constant int& total        [[buffer(4)]],
                                   uint3 tgid  [[threadgroup_position_in_grid]],
                                   uint3 tid   [[thread_position_in_threadgroup]],
                                   uint3 tgdim [[threads_per_threadgroup]]) {
    const int i = int(tgid.x * tgdim.x + tid.x);
    if (i >= total) return;
    const int row = i / d_ff;
    const int col = i - row * d_ff;
    device const float* r = xy + size_t(row) * 2 * d_ff;
    const float g = r[col];
    const float v = (g / (1.0f + exp(-g))) * r[d_ff + col];
    out[i] = v;
    hout[i] = half(v);
}

/// Combine the per-chunk partial softmaxes of a split attention.
kernel void attn_flash_reduce_f16_f32(device float* out             [[buffer(0)]],
                                      device half* hout             [[buffer(1)]],
                                      device const float* partial   [[buffer(2)]],
                                      constant int& ms_off          [[buffer(3)]],
                                      constant int& n_heads         [[buffer(4)]],
                                      constant int& d_head          [[buffer(5)]],
                                      constant int& n_tokens        [[buffer(6)]],
                                      constant int& n_chunks        [[buffer(7)]],
                                      uint3 tgid  [[threadgroup_position_in_grid]],
                                      uint3 tid   [[thread_position_in_threadgroup]],
                                      uint3 tgdim [[threads_per_threadgroup]]) {
    device const float* partial_acc = partial;
    device const float* partial_ms = partial + ms_off;
    const int i = int(tgid.x * tgdim.x + tid.x);
    const int total = n_tokens * n_heads * d_head;
    if (i >= total) return;
    const int head = (i / d_head) % n_heads;
    const int token = i / (d_head * n_heads);

    float m = -INFINITY;
    for (int c = 0; c < n_chunks; ++c) {
        m = fmax(m, partial_ms[((size_t(c) * n_tokens + token) * n_heads + head) * 2]);
    }
    if (m == -INFINITY) {
        out[i] = 0.0f;
        hout[i] = half(0.0f);
        return;
    }
    float acc = 0.0f, denom = 0.0f;
    for (int c = 0; c < n_chunks; ++c) {
        device const float* ms =
            partial_ms + ((size_t(c) * n_tokens + token) * n_heads + head) * 2;
        if (ms[0] == -INFINITY) continue;
        const float w = exp(ms[0] - m);
        denom += ms[1] * w;
        acc += partial_acc[size_t(c) * total + i] * w;
    }
    const float v = denom > 0.0f ? acc / denom : 0.0f;
    out[i] = v;
    hout[i] = half(v);
}

/// Split a fused `[q | k | v]` projection into three buffers.
kernel void split_qkv_f32(device float* q                [[buffer(0)]],
                          device float* k                [[buffer(1)]],
                          device float* v                [[buffer(2)]],
                          device const float* fused      [[buffer(3)]],
                          constant int& d                [[buffer(4)]],
                          constant int& kv_dim           [[buffer(5)]],
                          constant int& total            [[buffer(6)]],
                          uint3 tgid  [[threadgroup_position_in_grid]],
                          uint3 tid   [[thread_position_in_threadgroup]],
                          uint3 tgdim [[threads_per_threadgroup]]) {
    const int i = int(tgid.x * tgdim.x + tid.x);
    if (i >= total) return;
    const int row_w = d + 2 * kv_dim;
    const int row = i / row_w;
    const int col = i - row * row_w;
    const float x = fused[i];
    if (col < d) {
        q[size_t(row) * d + col] = x;
    } else if (col < d + kv_dim) {
        k[size_t(row) * kv_dim + (col - d)] = x;
    } else {
        v[size_t(row) * kv_dim + (col - d - kv_dim)] = x;
    }
}

/// Rotary over separate q and k buffers, either pairing convention.
kernel void rope_qk_f32(device float* q                    [[buffer(0)]],
                        device float* k                    [[buffer(1)]],
                        device const int* positions        [[buffer(2)]],
                        device const float* freq_factors   [[buffer(3)]],
                        constant int& n_heads              [[buffer(4)]],
                        constant int& n_kv_heads           [[buffer(5)]],
                        constant int& d_head               [[buffer(6)]],
                        constant int& rotary_dim           [[buffer(7)]],
                        constant float& theta_base         [[buffer(8)]],
                        constant float& freq_scale         [[buffer(9)]],
                        constant int& interleaved          [[buffer(10)]],
                        uint3 tgid  [[threadgroup_position_in_grid]],
                        uint3 tid   [[thread_position_in_threadgroup]],
                        uint3 tgdim [[threads_per_threadgroup]]) {
    const int half_r = rotary_dim / 2;
    const int i = int(tgid.x * tgdim.x + tid.x);
    if (i >= half_r) return;

    const int y = int(tgid.y);
    const int token = int(tgid.z);
    const bool is_q = y < n_heads;
    const int head = is_q ? y : y - n_heads;
    device float* base = is_q ? q : k;
    const int heads = is_q ? n_heads : n_kv_heads;

    const float pos = float(positions[token]) * freq_scale;
    const float inv_freq = pow(theta_base, -2.0f * float(i) / float(rotary_dim));
    const float angle = pos * inv_freq / freq_factors[i];
    const float sin_a = sin(angle);
    const float cos_a = cos(angle);

    device float* row = base + (size_t(token) * heads + head) * d_head;
    const int ia = interleaved != 0 ? 2 * i : i;
    const int ib = interleaved != 0 ? 2 * i + 1 : i + half_r;
    const float a = row[ia];
    const float b = row[ib];
    row[ia] = a * cos_a - b * sin_a;
    row[ib] = a * sin_a + b * cos_a;
}
