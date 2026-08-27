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
// produces fluent nonsense. They are checked against `infero_model::qwen35`,
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

/// A verification pass's rollback bookkeeping, staged/recorded in one launch
/// a group instead of two/four separate copies. See the note on
/// `gdn_rollback_stage2_f32` in `crates/kernels/src/cu/gdn.cu` for why this
/// exists and why it is not where the real fix (`GdnRollback::stage` only
/// copying the armed sequence's own slot) lives.

/// Two segments: the conv window, then the recurrent state.
kernel void gdn_rollback_stage2_f32(device float* dst0        [[buffer(0)]],
                                    device const float* src0  [[buffer(1)]],
                                    constant long& n0         [[buffer(2)]],
                                    device float* dst1        [[buffer(3)]],
                                    device const float* src1  [[buffer(4)]],
                                    constant long& n1         [[buffer(5)]],
                                    uint3 tgid  [[threadgroup_position_in_grid]],
                                    uint3 tid   [[thread_position_in_threadgroup]],
                                    uint3 tgdim [[threads_per_threadgroup]]) {
    const long i = long(tgid.x) * tgdim.x + tid.x;
    if (i < n0) {
        dst0[i] = src0[i];
    } else if (i < n0 + n1) {
        dst1[i - n0] = src1[i - n0];
    }
}

/// Four segments: the journal's pre-conv, post-conv, gate and beta taps.
kernel void gdn_rollback_record4_f32(device float* dst0        [[buffer(0)]],
                                     device const float* src0  [[buffer(1)]],
                                     constant long& n0         [[buffer(2)]],
                                     device float* dst1        [[buffer(3)]],
                                     device const float* src1  [[buffer(4)]],
                                     constant long& n1         [[buffer(5)]],
                                     device float* dst2        [[buffer(6)]],
                                     device const float* src2  [[buffer(7)]],
                                     constant long& n2         [[buffer(8)]],
                                     device float* dst3        [[buffer(9)]],
                                     device const float* src3  [[buffer(10)]],
                                     constant long& n3         [[buffer(11)]],
                                     uint3 tgid  [[threadgroup_position_in_grid]],
                                     uint3 tid   [[thread_position_in_threadgroup]],
                                     uint3 tgdim [[threads_per_threadgroup]]) {
    long i = long(tgid.x) * tgdim.x + tid.x;
    if (i < n0) {
        dst0[i] = src0[i];
        return;
    }
    i -= n0;
    if (i < n1) {
        dst1[i] = src1[i];
        return;
    }
    i -= n1;
    if (i < n2) {
        dst2[i] = src2[i];
        return;
    }
    i -= n2;
    if (i < n3) {
        dst3[i] = src3[i];
    }
}

// Register-resident gated delta rule, ported from `cu/gdn.cu`'s
// `gdn_delta_rule_reg_body<128, 128, 2, 4>` (see that file for the full
// derivation and the four measured design decisions this mirrors).
//
// `gdn_delta_rule_f32` streams the whole `dk x dv` state to and from device
// memory every token -- 128 KiB a head a token at this checkpoint's shape,
// which the traffic table in `cu/gdn.cu` shows is 2x what a register-
// resident version needs to move. This is that version for Metal: one
// simdgroup pair of lanes owns a column of `S` for the whole chunk, loaded
// once on entry and stored once on exit, and what actually crosses to device
// memory each token is q, k, v and the output -- about 1.5 KiB a head instead
// of 128 KiB.
//
// R = 2 threads a column, matching the CUDA kernel: a single thread holding
// all 128 rows of its column spills (documented on the CUDA side at 255
// registers with 88 bytes of spill for that shape), and splitting the column
// across two lanes of one simdgroup -- adjacent lanes, so `simd_shuffle_xor`
// finishes the reduction without a barrier -- is what keeps it register-
// resident instead. This is the load-bearing part of the port, not a
// refinement on top of it.
#define REG_DK 128
#define REG_DV 128
#define REG_R 2
#define REG_RB (REG_DK / REG_R)   // rows of S a thread owns
#define REG_ACC 4

kernel void gdn_delta_rule_reg128_f32(
        device float* out             [[buffer(0)]],
        device float* state           [[buffer(1)]],
        device const float* qkv       [[buffer(2)]],
        device const float* g         [[buffer(3)]],
        device const float* beta      [[buffer(4)]],
        device const int* first_token [[buffer(5)]],
        device const int* n_tok       [[buffer(6)]],
        constant int& heads           [[buffer(7)]],
        constant int& key_heads       [[buffer(8)]],
        constant int& dk_unused       [[buffer(9)]],
        constant int& dv_unused       [[buffer(10)]],
        constant int& stride          [[buffer(11)]],
        constant int& q_off           [[buffer(12)]],
        constant int& k_off           [[buffer(13)]],
        constant int& v_off           [[buffer(14)]],
        constant int& v_tiled         [[buffer(15)]],
        threadgroup float* smem       [[threadgroup(0)]],
        uint3 tgid  [[threadgroup_position_in_grid]],
        uint3 tid   [[thread_position_in_threadgroup]]) {
    (void)dk_unused;
    (void)dv_unused;

    const int head = int(tgid.x);
    const int seq = int(tgid.y);
    const int nt = n_tok[seq];
    if (nt <= 0) return;
    const int t0 = first_token[seq];
    const int lane = int(tid.x);
    const int j = lane / REG_R;
    const int part = lane % REG_R;
    const int i0 = part * REG_RB;
    const int khead = v_tiled != 0 ? (head % key_heads)
                                   : (head / (heads / key_heads));

    threadgroup float* qs = smem;                    // 2 * REG_DK
    threadgroup float* ks = smem + 2 * REG_DK;        // 2 * REG_DK
    device float* S = state + (size_t(seq) * heads + head) * size_t(REG_DK) * REG_DV;

    float sc[REG_RB];
#pragma unroll
    for (int r = 0; r < REG_RB; ++r) sc[r] = S[size_t(i0 + r) * REG_DV + j];

    device const float* row0 = qkv + size_t(t0) * stride;
    float qn = 0.0f, kn = 0.0f;
    if (lane < REG_DK) {
        qn = row0[q_off + size_t(khead) * REG_DK + lane];
        kn = row0[k_off + size_t(khead) * REG_DK + lane];
        qs[lane] = qn;
        ks[lane] = kn;
    }
    float vn = row0[v_off + size_t(head) * REG_DV + j];
    float gn = g[size_t(t0) * heads + head];
    float bn = beta[size_t(t0) * heads + head];

    for (int n = 0; n < nt; ++n) {
        const int t = t0 + n;
        const int cur = n & 1;
        const float v_tj = vn;
        const float decay = exp(gn);
        const float b = bn;

        if (n + 1 < nt) {
            device const float* rn = qkv + size_t(t + 1) * stride;
            if (lane < REG_DK) {
                qn = rn[q_off + size_t(khead) * REG_DK + lane];
                kn = rn[k_off + size_t(khead) * REG_DK + lane];
            }
            vn = rn[v_off + size_t(head) * REG_DV + j];
            gn = g[size_t(t + 1) * heads + head];
            bn = beta[size_t(t + 1) * heads + head];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup float* qc = qs + cur * REG_DK;
        threadgroup float* kc = ks + cur * REG_DK;

        float kv[REG_ACC];
#pragma unroll
        for (int a = 0; a < REG_ACC; ++a) kv[a] = 0.0f;
#pragma unroll
        for (int r = 0; r < REG_RB; ++r) {
            sc[r] *= decay;
            kv[r % REG_ACC] += sc[r] * kc[i0 + r];
        }
#pragma unroll
        for (int a = 1; a < REG_ACC; ++a) kv[0] += kv[a];
        float kvt = kv[0];
#pragma unroll
        for (int m = 1; m < REG_R; m <<= 1) kvt += simd_shuffle_xor(kvt, uint(m));

        const float delta = (v_tj - kvt) * b;

        float o[REG_ACC];
#pragma unroll
        for (int a = 0; a < REG_ACC; ++a) o[a] = 0.0f;
#pragma unroll
        for (int r = 0; r < REG_RB; ++r) {
            sc[r] += kc[i0 + r] * delta;
            o[r % REG_ACC] += sc[r] * qc[i0 + r];
        }
#pragma unroll
        for (int a = 1; a < REG_ACC; ++a) o[0] += o[a];
        float ot = o[0];
#pragma unroll
        for (int m = 1; m < REG_R; m <<= 1) ot += simd_shuffle_xor(ot, uint(m));
        if (part == 0) out[(size_t(t) * heads + head) * REG_DV + j] = ot;

        if (n + 1 < nt && lane < REG_DK) {
            qs[(cur ^ 1) * REG_DK + lane] = qn;
            ks[(cur ^ 1) * REG_DK + lane] = kn;
        }
    }

#pragma unroll
    for (int r = 0; r < REG_RB; ++r) S[size_t(i0 + r) * REG_DV + j] = sc[r];
}
