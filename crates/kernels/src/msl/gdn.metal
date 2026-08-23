// GatedDeltaNet on Metal: the linear-attention block of Qwen3.5/3.8, and the
// output gate its full-attention blocks carry.
//
// The Metal twin of `cu/gdn.cu`, kernel for kernel and name for name. Two
// things needed adapting rather than transliterating:
//
//   * `extern __shared__ float shared[]` becomes a `[[threadgroup(0)]]`
//     parameter, which the host sizes with `setThreadgroupMemoryLength`.
//   * `block_reduce_sum` takes its scratch as arguments -- see `common.metal`.
//
// The register-resident recurrence (`gdn_delta_rule_reg128_f32` on the CUDA
// side, 58x faster at prefill) is deliberately NOT ported yet. At the 27B's
// shape the state moves 288 MiB a decode token while the weights move 17.5 GB,
// so the recurrence is under 2% of the step here -- the CUDA version needed it
// because its weights were FP8 and eight times cheaper to read. Port it when
// the weights stop dominating, not before.
//
// Every layout choice below has a second reading that runs to completion and
// produces fluent nonsense. They are checked against `tuili_model::qwen35`,
// which is checked against a capture of the reference implementation on the
// real checkpoint.

/// Depthwise causal convolution over time with a carried window, plus SiLU.
///
/// Direction is the thing to get right: output `t` reads `t-(k-1) ..= t` and
/// weight `j` pairs with input `t - (k-1) + j`, so `w[k-1]` multiplies the
/// current token. Reversing it runs fine and shifts the model one token into
/// the future.
kernel void gdn_conv_f32(device float* out                [[buffer(0)]],
                         device const float* x            [[buffer(1)]],
                         device float* state              [[buffer(2)]],
                         device const float* w            [[buffer(3)]],
                         device const int* first_token    [[buffer(4)]],
                         device const int* n_tok          [[buffer(5)]],
                         constant int& channels           [[buffer(6)]],
                         constant int& k                  [[buffer(7)]],
                         uint3 tgid  [[threadgroup_position_in_grid]],
                         uint3 tid   [[thread_position_in_threadgroup]],
                         uint3 tgdim [[threads_per_threadgroup]]) {
    const int c = int(tgid.x * tgdim.x + tid.x);
    if (c >= channels) return;
    const int seq = int(tgid.y);
    const int nt = n_tok[seq];
    if (nt <= 0) return;              // a slot not in this batch
    const int t0 = first_token[seq];
    const int hist = k - 1;

    device float* st = state + (size_t(seq) * channels + c) * hist;
    // k is 4 in this checkpoint, so the window is three floats. The bound keeps
    // the array a compile-time size.
    float win[8];
    for (int j = 0; j < hist; ++j) win[j] = st[j];

    device const float* wc = w + size_t(c) * k;
    for (int n = 0; n < nt; ++n) {
        const float cur = x[size_t(t0 + n) * channels + c];
        float acc = wc[hist] * cur;
        for (int j = 0; j < hist; ++j) acc += wc[j] * win[j];
        // SiLU, fused: the reference applies it to the convolution output
        // before the split into q, k and v.
        out[size_t(t0 + n) * channels + c] = acc / (1.0f + exp(-acc));
        for (int j = 0; j + 1 < hist; ++j) win[j] = win[j + 1];
        if (hist > 0) win[hist - 1] = cur;
    }
    for (int j = 0; j < hist; ++j) st[j] = win[j];
}

/// `beta = sigmoid(b)` and `g = -exp(A_log) * softplus(a + dt_bias)`.
///
/// `g` is non-positive by construction so `exp(g)` is a decay; losing the sign
/// makes the state grow without bound and surfaces as NaN several layers later.
kernel void gdn_gate_decay_f32(device float* beta_out       [[buffer(0)]],
                               device float* g_out          [[buffer(1)]],
                               device const float* a        [[buffer(2)]],
                               device const float* b        [[buffer(3)]],
                               device const float* a_log    [[buffer(4)]],
                               device const float* dt_bias  [[buffer(5)]],
                               constant int& n_tokens       [[buffer(6)]],
                               constant int& heads          [[buffer(7)]],
                               constant int& stride         [[buffer(8)]],
                               uint3 tgid  [[threadgroup_position_in_grid]],
                               uint3 tid   [[thread_position_in_threadgroup]],
                               uint3 tgdim [[threads_per_threadgroup]]) {
    const int idx = int(tgid.x * tgdim.x + tid.x);
    if (idx >= n_tokens * heads) return;
    const int h = idx % heads;
    const int src = (idx / heads) * stride + h;
    beta_out[idx] = 1.0f / (1.0f + exp(-b[src]));
    const float z = a[src] + dt_bias[h];
    // log1p(exp(z)) with the large-z branch taken directly: dt_bias reaches +19
    // in this checkpoint and exp of a large sum overflows before the log can
    // bring it back.
    const float sp = z > 20.0f ? z : log(1.0f + exp(z));
    g_out[idx] = -exp(a_log[h]) * sp;
}

/// L2-normalize each head's row of q and k in place, then scale q by
/// `1/sqrt(dk)`.
///
/// `eps` is added to the sum of squares, not to the norm, matching the FLA
/// convention the reference cites. q and k are normalized where they lie inside
/// the packed `[q | k | v]` row.
kernel void gdn_qk_l2norm_f32(device float* qkv        [[buffer(0)]],
                              constant int& key_heads  [[buffer(1)]],
                              constant int& dk         [[buffer(2)]],
                              constant int& stride     [[buffer(3)]],
                              constant int& q_off      [[buffer(4)]],
                              constant int& k_off      [[buffer(5)]],
                              constant float& eps      [[buffer(6)]],
                              constant float& q_scale  [[buffer(7)]],
                              uint3 tgid  [[threadgroup_position_in_grid]],
                              uint3 tid   [[thread_position_in_threadgroup]],
                              uint3 tgdim [[threads_per_threadgroup]]) {
    BLOCK_REDUCE_SCRATCH

    const int token = int(tgid.x) / key_heads;
    const int head = int(tgid.x) % key_heads;
    device float* base = qkv + size_t(token) * stride + size_t(head) * dk;
    device float* qr = base + q_off;
    device float* kr = base + k_off;

    float qa = 0.0f, ka = 0.0f;
    for (int i = int(tid.x); i < dk; i += int(tgdim.x)) {
        qa += qr[i] * qr[i];
        ka += kr[i] * kr[i];
    }
    const float qs = rsqrt(BLOCK_SUM(qa, tid.x, tgdim.x) + eps) * q_scale;
    const float ks = rsqrt(BLOCK_SUM(ka, tid.x, tgdim.x) + eps);
    for (int i = int(tid.x); i < dk; i += int(tgdim.x)) {
        qr[i] *= qs;
        kr[i] *= ks;
    }
}

/// The gated delta rule.
///
///   S      *= exp(g_t)
///   kv_mem  = k^T S                 contracting the key axis
///   delta   = (v_t - kv_mem) * beta_t
///   S      += k_t (x) delta
///   o_t     = q^T S                 the same contraction, with q
///
/// `state` is `[n_seqs, heads, dk, dv]`; `qkv` holds q, k and v in one row per
/// token, `stride` wide, with q and k already normalized and q already scaled.
///
/// The head expansion happens here rather than by materializing a wider q and
/// k, and *which* expansion depends on how the checkpoint was written.
///
/// A Hugging Face checkpoint stores V heads grouped by key head --
/// `[G0_v0..v2, G1_v0..v2, ...]` -- so value head `h` reads key head
/// `h / (heads / key_heads)`. That is `repeat_interleave`, and it is what
/// `v_tiled = 0` does.
///
/// A GGUF does not. llama.cpp reorders the V heads to *tiled* order at
/// conversion time -- `[G0_v0, G1_v0, ..., G0_v1, ...]` -- so that ggml's
/// binary broadcast can use `ggml_repeat` instead of an interleaved repeat, and
/// then value head `h` reads key head `h % key_heads`. That is `v_tiled = 1`.
/// The same permutation is applied to `in_proj_z`, `in_proj_a`, `in_proj_b`,
/// `A_log`, `dt_bias`, the V channels of `conv1d` and the columns of
/// `out_proj`, so everything indexed by a value head agrees and only this
/// lookup has to know.
///
/// Both expansions run to completion and give different models. Reading a GGUF
/// with the grouped rule produces grammatical, fluent, content-free text: the
/// prompt's own words and the commonest function words, with the answer absent
/// from the top ten.
///
/// One threadgroup a (head, sequence); thread `j` owns column `j` of S, so the
/// reads `S[i * dv + j]` are contiguous across the group. The two passes cannot
/// merge: `delta` needs the whole `kv_mem` reduction before the update starts.
kernel void gdn_delta_rule_f32(device float* out             [[buffer(0)]],
                               device float* state           [[buffer(1)]],
                               device const float* qkv       [[buffer(2)]],
                               device const float* g         [[buffer(3)]],
                               device const float* beta      [[buffer(4)]],
                               device const int* first_token [[buffer(5)]],
                               device const int* n_tok       [[buffer(6)]],
                               constant int& heads           [[buffer(7)]],
                               constant int& key_heads       [[buffer(8)]],
                               constant int& dk              [[buffer(9)]],
                               constant int& dv              [[buffer(10)]],
                               constant int& stride          [[buffer(11)]],
                               constant int& q_off           [[buffer(12)]],
                               constant int& k_off           [[buffer(13)]],
                               constant int& v_off           [[buffer(14)]],
                               constant int& v_tiled         [[buffer(15)]],
                               threadgroup float* shared     [[threadgroup(0)]],
                               uint3 tgid  [[threadgroup_position_in_grid]],
                               uint3 tid   [[thread_position_in_threadgroup]],
                               uint3 tgdim [[threads_per_threadgroup]]) {
    threadgroup float* qs = shared;         // dk
    threadgroup float* ks = shared + dk;    // dk

    const int head = int(tgid.x);
    const int seq = int(tgid.y);
    const int nt = n_tok[seq];
    // A sequence not in this batch has no tokens and its group exits here,
    // which is what lets one dispatch cover the whole pool of slots without the
    // caller compacting anything.
    if (nt <= 0) return;
    const int t0 = first_token[seq];
    const int j = int(tid.x);
    const int khead = v_tiled != 0 ? (head % key_heads)
                                   : (head / (heads / key_heads));

    device float* S = state + (size_t(seq) * heads + head) * size_t(dk) * dv;

    for (int n = 0; n < nt; ++n) {
        const int t = t0 + n;
        device const float* row = qkv + size_t(t) * stride;
        device const float* qsrc = row + q_off + size_t(khead) * dk;
        device const float* ksrc = row + k_off + size_t(khead) * dk;
        for (int i = int(tid.x); i < dk; i += int(tgdim.x)) {
            qs[i] = qsrc[i];
            ks[i] = ksrc[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float decay = exp(g[size_t(t) * heads + head]);
        const float b = beta[size_t(t) * heads + head];

        if (j < dv) {
            const float v_tj = row[v_off + size_t(head) * dv + j];
            float kv = 0.0f;
            for (int i = 0; i < dk; ++i) {
                const float s = S[size_t(i) * dv + j] * decay;
                S[size_t(i) * dv + j] = s;
                kv += s * ks[i];
            }
            const float delta = (v_tj - kv) * b;
            float o = 0.0f;
            for (int i = 0; i < dk; ++i) {
                const float s = S[size_t(i) * dv + j] + ks[i] * delta;
                S[size_t(i) * dv + j] = s;
                o += s * qs[i];
            }
            out[(size_t(t) * heads + head) * dv + j] = o;
        }
        // The next token overwrites the shared q and k, and reads the S this
        // token wrote.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

/// RMSNorm over `dv` with a SiLU'd gate folded into the scale. The output
/// projection of a linear-attention block reads this.
kernel void gdn_gated_rmsnorm_f32(device float* out          [[buffer(0)]],
                                  device const float* x      [[buffer(1)]],
                                  device const float* z      [[buffer(2)]],
                                  device const float* weight [[buffer(3)]],
                                  constant int& dv           [[buffer(4)]],
                                  constant float& eps        [[buffer(5)]],
                                  uint3 tgid  [[threadgroup_position_in_grid]],
                                  uint3 tid   [[thread_position_in_threadgroup]],
                                  uint3 tgdim [[threads_per_threadgroup]]) {
    BLOCK_REDUCE_SCRATCH

    const size_t row = size_t(tgid.x) * dv;
    device const float* xr = x + row;
    device const float* zr = z + row;
    device float* orow = out + row;

    float acc = 0.0f;
    for (int i = int(tid.x); i < dv; i += int(tgdim.x)) {
        const float val = xr[i];
        acc += val * val;
    }
    const float scale = rsqrt(BLOCK_SUM(acc, tid.x, tgdim.x) / float(dv) + eps);
    for (int i = int(tid.x); i < dv; i += int(tgdim.x)) {
        const float zi = zr[i];
        orow[i] = weight[i] * (xr[i] * scale) * (zi / (1.0f + exp(-zi)));
    }
}

/// `x *= sigmoid(gate)`. The output gate the full-attention blocks apply
/// *before* `o_proj`, and sigmoid rather than SiLU.
kernel void sigmoid_gate_f32(device float* x            [[buffer(0)]],
                             device const float* gate   [[buffer(1)]],
                             constant long& n           [[buffer(2)]],
                             uint3 tgid  [[threadgroup_position_in_grid]],
                             uint3 tid   [[thread_position_in_threadgroup]],
                             uint3 tgdim [[threads_per_threadgroup]]) {
    const long i = long(tgid.x) * tgdim.x + tid.x;
    if (i >= n) return;
    x[i] *= 1.0f / (1.0f + exp(-gate[i]));
}

/// Split `q_proj`'s output into the query and its gate.
///
/// The output is `[t_len, heads * 2 * head_dim]` and the reference reads it as
/// `[t_len, heads, 2 * head_dim]` before splitting the last axis -- so within
/// one head's `2 * head_dim` values the query comes first and the gate second.
/// Reading it as `[all queries | all gates]` is the other plausible layout, it
/// runs to completion, and it is wrong.
kernel void split_interleaved_f32(device float* q            [[buffer(0)]],
                                  device float* gate         [[buffer(1)]],
                                  device const float* src    [[buffer(2)]],
                                  constant int& heads        [[buffer(3)]],
                                  constant int& head_dim     [[buffer(4)]],
                                  constant long& n           [[buffer(5)]],
                                  uint3 tgid  [[threadgroup_position_in_grid]],
                                  uint3 tid   [[thread_position_in_threadgroup]],
                                  uint3 tgdim [[threads_per_threadgroup]]) {
    const long i = long(tgid.x) * tgdim.x + tid.x;
    if (i >= n) return;
    const int lane = int(i % head_dim);
    const long head_ix = i / head_dim;              // token * heads + head
    const int head = int(head_ix % heads);
    const long token = head_ix / heads;
    const long row = token * long(heads) * 2 * head_dim
                   + long(head) * 2 * head_dim;
    q[i] = src[row + lane];
    gate[i] = src[row + head_dim + lane];
}
