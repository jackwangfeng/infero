// Sampling, on the device -- the Metal twin of `cu/sample.cu`.
//
// Read that file first. Every decision about *what* is computed is there: why
// the logits stay on the device, why the repetition penalty has two
// non-agreeing paths that both have to be reproduced, why the tie-break is
// lowest-index, why the top-k is split across the vocabulary rather than
// scanned once per survivor. None of that is restated here. What follows is the
// same computation and only the two places where Apple's GPU forces a
// difference.
//
// **There is no fp64.** The CUDA original accumulates the softmax total, the
// nucleus prefix and the draw itself in `double`, and takes the random draw as
// a `double` per row. Metal has no double at all -- not slow, absent -- so:
//
//   * the accumulators are `float`. Over at most `SAMPLE_MAX_TOP_K` = 256
//     already-exponentiated terms the relative error is around 1e-6, which
//     moves the picked token only when two cumulative-probability boundaries
//     fall within that of the draw. The tokens are drawn from the same
//     distribution; a seeded run is reproducible against itself and is *not*
//     bit-comparable with a CUDA run. That is the honest statement of it.
//
//   * `rnd` keeps its 8-bytes-a-row layout so the host binding is unchanged,
//     and `samp_draw` narrows the IEEE-754 double to float in the kernel. A
//     uniform in [0, 1) has nothing below f32's 24 bits that a multinomial cares
//     about, so this loses no behaviour -- unlike changing the host to upload
//     f32, which would have changed `StdRng`'s consumption and with it every
//     seeded sequence on both backends.
//
// **Threadgroup memory is 32 KiB, not 48.** `sample_rows_f32` wants a bitset
// over the whole vocabulary, which is `vocab/32` words: 19 KiB at Qwen2.5's
// 151936 and 31 KiB at Qwen3.8's 248320, and with the reduction scratch on top
// the second one does not fit. `Kernels::can_sample_on_device` carries the
// smaller limit on this backend and the split path -- whose bitset covers one
// sixty-fourth of the vocabulary -- serves the vocabularies it excludes. Both
// kernels are here; which one runs is the host's decision and it already knows
// how to make it.

#define SAMPLE_BLOCK 256
#define SAMPLE_SPLITS 64

/// A row's parameters, mirroring `SamplingParams`. Field order and types match
/// `cu/sample.cu` exactly: the host uploads one `[f32; 4]` a row and reads it as
/// this on both backends.
struct SampleParams {
    float temperature;
    float top_p;
    int top_k;
    float rep_penalty;
};

/// (value, index) under "higher value first, lower index first".
inline bool samp_better(float av, int ai, float bv, int bi) {
    return av > bv || (av == bv && ai < bi);
}

inline void samp_reduce(threadgroup float* sv, threadgroup int* si, int tid) {
    for (int s = SAMPLE_BLOCK / 2; s > 0; s >>= 1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < s && samp_better(sv[tid + s], si[tid + s], sv[tid], si[tid])) {
            sv[tid] = sv[tid + s];
            si[tid] = si[tid + s];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

inline int samp_count(device const int* toks, device const int* cnts, int len, int tok) {
    int lo = 0, hi = len - 1;
    while (lo <= hi) {
        const int mid = (lo + hi) >> 1;
        if (toks[mid] == tok) return cnts[mid];
        if (toks[mid] < tok) lo = mid + 1;
        else hi = mid - 1;
    }
    return 0;
}

inline float samp_penalize(float l, int count, float p, bool once) {
    if (count <= 0 || p == 1.0f) return l;
    const int n = once ? 1 : count;
    for (int i = 0; i < n; ++i) l = l > 0.0f ? l / p : l * p;
    return l;
}

/// This row's uniform draw, narrowed from the host's `f64`.
///
/// Read as a pair of words rather than as a `double`, because the type does not
/// exist here. The exponent bias goes 1023 -> 127 and the mantissa keeps its top
/// 23 bits; a double below f32's smallest normal becomes zero, which for a draw
/// in [0, 1) means "take the first survivor" and is what an exact zero does too.
inline float samp_draw(device const uint2* rnd, int row) {
    const uint2 w = rnd[row];
    const uint hi = w.y;
    const int e = int((hi >> 20) & 0x7FFu) - 1023;
    if (e < -126) return 0.0f;
    const uint mant = ((hi & 0xFFFFFu) << 3) | (w.x >> 29);
    const uint bits = (hi & 0x80000000u) | (uint(e + 127) << 23) | mant;
    return as_type<float>(bits);
}

/// The shared tail of `sample_rows_f32` and `sample_rows_topk_f32`: softmax at
/// temperature over the survivors, the nucleus cut, the draw, and optionally the
/// distribution the draw came from.
///
/// A function rather than the copy the CUDA file carries. There it is duplicated
/// because the two kernels reached it with different scratch in registers; here
/// both hand over the same two threadgroup arrays, and one copy means the two
/// cannot drift -- which for the arithmetic that speculation's acceptance ratio
/// divides by is worth more than matching the original's shape.
inline void samp_tail(threadgroup float* kv, threadgroup int* ki, int k,
                      SampleParams p, float draw,
                      device uint* out, int row,
                      device uint* surv_id, device float* surv_p,
                      device int* surv_len, int surv_stride) {
    const float inv_t = 1.0f / fmax(p.temperature, 1e-5f);
    const float mx = kv[0];
    float total = 0.0f;
    for (int j = 0; j < k; ++j) {
        const float q = exp((kv[j] - mx) * inv_t);
        kv[j] = q;
        total += q;
    }
    int keep = k;
    if (p.top_p < 1.0f) {
        const float target = total * fmin(fmax(p.top_p, 1e-4f), 1.0f);
        float acc = 0.0f;
        keep = 0;
        for (int j = 0; j < k; ++j) {
            acc += kv[j];
            ++keep;
            if (acc >= target) break;
        }
        if (keep < 1) keep = 1;
        total = 0.0f;
        for (int j = 0; j < keep; ++j) total += kv[j];
    }
    float r = draw * total;
    uint pick = uint(ki[keep - 1]);
    for (int j = 0; j < keep; ++j) {
        r -= kv[j];
        if (r <= 0.0f) {
            pick = uint(ki[j]);
            break;
        }
    }
    out[row] = pick;
    if (surv_len) {
        surv_len[row] = keep;
        const float inv = total > 0.0f ? 1.0f / total : 0.0f;
        for (int j = 0; j < keep && j < surv_stride; ++j) {
            surv_id[size_t(row) * size_t(surv_stride) + j] = uint(ki[j]);
            surv_p[size_t(row) * size_t(surv_stride) + j] = kv[j] * inv;
        }
    }
}

/// Set one bit of a dynamically-sized threadgroup bitset.
///
/// The array arrives as `threadgroup uint*` -- a kernel parameter, so its type
/// is fixed by the host's `setThreadgroupMemoryLength` and cannot be declared
/// `atomic_uint`. Several threads may touch the same word, so the OR has to be
/// atomic, and the cast is the only way to say so. It is the standard idiom for
/// dynamic threadgroup memory on this platform.
inline void samp_set_bit(threadgroup uint* bits, int b) {
    atomic_fetch_or_explicit(
        (threadgroup atomic_uint*)&bits[b >> 5], 1u << (b & 31), memory_order_relaxed);
}

kernel void argmax_partial_f32(
    device float* pv [[buffer(0)]],
    device int* pi [[buffer(1)]],
    device const float* logits [[buffer(2)]],
    device const SampleParams* params [[buffer(3)]],
    device const int* pen_tok [[buffer(4)]],
    device const int* pen_cnt [[buffer(5)]],
    device const int* pen_len [[buffer(6)]],
    constant int& vocab [[buffer(7)]],
    constant int& pen_stride [[buffer(8)]],
    constant int& splits [[buffer(9)]],
    threadgroup uint* smem [[threadgroup(0)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]]) {
    const int s = int(tgid.x);
    const int row = int(tgid.y);
    const int t = int(tid.x);
    const int chunk = (vocab + splits - 1) / splits;
    const int lo = s * chunk;
    const int hi = min(vocab, lo + chunk);
    const int words = (chunk + 31) / 32 + 1;

    threadgroup uint* bits = smem;
    threadgroup float* rv = (threadgroup float*)(smem + words);
    threadgroup int* ri = (threadgroup int*)(rv + SAMPLE_BLOCK);

    const SampleParams p = params[row];
    device const float* row_logits = logits + size_t(row) * size_t(vocab);
    const int plen = pen_len[row];
    device const int* ptok = pen_tok + size_t(row) * size_t(pen_stride);
    device const int* pcnt = pen_cnt + size_t(row) * size_t(pen_stride);

    for (int i = t; i < words; i += SAMPLE_BLOCK) bits[i] = 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = t; i < plen; i += SAMPLE_BLOCK) {
        const int tk = ptok[i];
        if (tk >= lo && tk < hi) samp_set_bit(bits, tk - lo);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float best = -INFINITY;
    int besti = 0x7fffffff;
    for (int i = lo + t; i < hi; i += SAMPLE_BLOCK) {
        float v = row_logits[i];
        const int b = i - lo;
        if (bits[b >> 5] & (1u << (b & 31))) {
            v = samp_penalize(v, samp_count(ptok, pcnt, plen, i), p.rep_penalty, true);
        }
        if (samp_better(v, i, best, besti)) {
            best = v;
            besti = i;
        }
    }
    rv[t] = best;
    ri[t] = besti;
    samp_reduce(rv, ri, t);
    if (t == 0) {
        pv[size_t(row) * size_t(splits) + s] = rv[0];
        pi[size_t(row) * size_t(splits) + s] = ri[0];
    }
}

kernel void argmax_combine_f32(
    device uint* out [[buffer(0)]],
    device const float* pv [[buffer(1)]],
    device const int* pi [[buffer(2)]],
    constant int& splits [[buffer(3)]],
    threadgroup uint* smem [[threadgroup(0)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]]) {
    threadgroup float* rv = (threadgroup float*)smem;
    threadgroup int* ri = (threadgroup int*)(rv + SAMPLE_BLOCK);

    const int row = int(tgid.x);
    const int t = int(tid.x);
    float best = -INFINITY;
    int besti = 0x7fffffff;
    for (int i = t; i < splits; i += SAMPLE_BLOCK) {
        const float v = pv[size_t(row) * size_t(splits) + i];
        const int idx = pi[size_t(row) * size_t(splits) + i];
        if (samp_better(v, idx, best, besti)) {
            best = v;
            besti = idx;
        }
    }
    rv[t] = best;
    ri[t] = besti;
    samp_reduce(rv, ri, t);
    if (t == 0) out[row] = uint(ri[0]);
}

kernel void sample_rows_f32(
    device uint* out [[buffer(0)]],
    device const float* logits [[buffer(1)]],
    device const SampleParams* params [[buffer(2)]],
    device const int* pen_tok [[buffer(3)]],
    device const int* pen_cnt [[buffer(4)]],
    device const int* pen_len [[buffer(5)]],
    device const uint2* rnd [[buffer(6)]],
    constant int& vocab [[buffer(7)]],
    constant int& pen_stride [[buffer(8)]],
    device uint* surv_id [[buffer(9)]],
    device float* surv_p [[buffer(10)]],
    device int* surv_len [[buffer(11)]],
    constant int& surv_stride [[buffer(12)]],
    threadgroup uint* smem [[threadgroup(0)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]]) {
    const int row = int(tgid.x);
    const int t = int(tid.x);
    const int words = (vocab + 31) / 32;

    threadgroup uint* bits = smem;
    threadgroup float* rv = (threadgroup float*)(smem + words);
    threadgroup int* ri = (threadgroup int*)(rv + SAMPLE_BLOCK);
    threadgroup float* kv = (threadgroup float*)(ri + SAMPLE_BLOCK);
    threadgroup int* ki = (threadgroup int*)(kv + SAMPLE_BLOCK);

    const SampleParams p = params[row];
    device const float* row_logits = logits + size_t(row) * size_t(vocab);
    const int plen = pen_len[row];
    device const int* ptok = pen_tok + size_t(row) * size_t(pen_stride);
    device const int* pcnt = pen_cnt + size_t(row) * size_t(pen_stride);

    const bool greedy = p.temperature <= 0.0f || p.top_k == 1;

    for (int i = t; i < words; i += SAMPLE_BLOCK) bits[i] = 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = t; i < plen; i += SAMPLE_BLOCK) {
        const int tk = ptok[i];
        if (tk >= 0 && tk < vocab) samp_set_bit(bits, tk);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float best = -INFINITY;
    int besti = 0;
    for (int i = t; i < vocab; i += SAMPLE_BLOCK) {
        float v = row_logits[i];
        if (bits[i >> 5] & (1u << (i & 31))) {
            v = samp_penalize(v, samp_count(ptok, pcnt, plen, i), p.rep_penalty, greedy);
        }
        if (samp_better(v, i, best, besti)) {
            best = v;
            besti = i;
        }
    }
    rv[t] = best;
    ri[t] = besti;
    samp_reduce(rv, ri, t);

    if (greedy) {
        if (t == 0) out[row] = uint(ri[0]);
        return;
    }

    const int k = min(max(p.top_k, 1), vocab);
    if (t == 0) {
        kv[0] = rv[0];
        ki[0] = ri[0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int j = 1; j < k; ++j) {
        const float lastv = kv[j - 1];
        const int lasti = ki[j - 1];
        float bv = -INFINITY;
        int bi = 0;
        bool have = false;
        for (int i = t; i < vocab; i += SAMPLE_BLOCK) {
            float v = row_logits[i];
            if (bits[i >> 5] & (1u << (i & 31))) {
                v = samp_penalize(v, samp_count(ptok, pcnt, plen, i), p.rep_penalty, false);
            }
            if (!samp_better(lastv, lasti, v, i)) continue;
            if (!have || samp_better(v, i, bv, bi)) {
                bv = v;
                bi = i;
                have = true;
            }
        }
        rv[t] = have ? bv : -INFINITY;
        ri[t] = have ? bi : 0x7fffffff;
        samp_reduce(rv, ri, t);
        if (t == 0) {
            kv[j] = rv[0];
            ki[j] = ri[0];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (t == 0) {
        samp_tail(kv, ki, k, p, samp_draw(rnd, row), out, row,
                  surv_id, surv_p, surv_len, surv_stride);
    }
}

kernel void sample_topk_partial_f32(
    device float* cand_v [[buffer(0)]],
    device int* cand_i [[buffer(1)]],
    device const float* logits [[buffer(2)]],
    device const SampleParams* params [[buffer(3)]],
    device const int* pen_tok [[buffer(4)]],
    device const int* pen_cnt [[buffer(5)]],
    device const int* pen_len [[buffer(6)]],
    constant int& vocab [[buffer(7)]],
    constant int& pen_stride [[buffer(8)]],
    constant int& cand_k [[buffer(9)]],
    threadgroup uint* smem [[threadgroup(0)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]]) {
    const int row = int(tgid.x);
    const int split = int(tgid.y);
    const int t = int(tid.x);

    const int per = (vocab + SAMPLE_SPLITS - 1) / SAMPLE_SPLITS;
    const int lo = split * per;
    const int hi = min(lo + per, vocab);

    const int words = (per + 31) / 32;
    threadgroup uint* bits = smem;
    threadgroup float* rv = (threadgroup float*)(smem + words);
    threadgroup int* ri = (threadgroup int*)(rv + SAMPLE_BLOCK);

    const SampleParams p = params[row];
    device const float* row_logits = logits + size_t(row) * size_t(vocab);
    const int plen = pen_len[row];
    device const int* ptok = pen_tok + size_t(row) * size_t(pen_stride);
    device const int* pcnt = pen_cnt + size_t(row) * size_t(pen_stride);
    const bool greedy = p.temperature <= 0.0f || p.top_k == 1;
    const int k = min(max(p.top_k, 1), vocab);

    device float* out_v = cand_v + (size_t(row) * SAMPLE_SPLITS + split) * size_t(cand_k);
    device int* out_i = cand_i + (size_t(row) * SAMPLE_SPLITS + split) * size_t(cand_k);

    for (int i = t; i < words; i += SAMPLE_BLOCK) bits[i] = 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = t; i < plen; i += SAMPLE_BLOCK) {
        const int tk = ptok[i];
        if (tk >= lo && tk < hi) samp_set_bit(bits, tk - lo);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float lastv = INFINITY;
    int lasti = -1;
    for (int j = 0; j < k; ++j) {
        float bv = -INFINITY;
        int bi = 0;
        bool have = false;
        for (int i = lo + t; i < hi; i += SAMPLE_BLOCK) {
            float v = row_logits[i];
            const int r = i - lo;
            if (bits[r >> 5] & (1u << (r & 31))) {
                v = samp_penalize(v, samp_count(ptok, pcnt, plen, i), p.rep_penalty, greedy);
            }
            if (j > 0 && !samp_better(lastv, lasti, v, i)) continue;
            if (!have || samp_better(v, i, bv, bi)) {
                bv = v;
                bi = i;
                have = true;
            }
        }
        rv[t] = have ? bv : -INFINITY;
        ri[t] = have ? bi : 0x7fffffff;
        samp_reduce(rv, ri, t);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        lastv = rv[0];
        lasti = ri[0];
        if (t == 0) {
            out_v[j] = lastv;
            out_i[j] = (lasti == 0x7fffffff) ? vocab : lasti;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lasti == 0x7fffffff) {
            for (int t2 = j + 1 + t; t2 < k; t2 += SAMPLE_BLOCK) {
                out_v[t2] = -INFINITY;
                out_i[t2] = vocab;
            }
            break;
        }
    }
}

kernel void sample_rows_topk_f32(
    device uint* out [[buffer(0)]],
    device const float* cand_v [[buffer(1)]],
    device const int* cand_i [[buffer(2)]],
    device const SampleParams* params [[buffer(3)]],
    device const uint2* rnd [[buffer(4)]],
    constant int& vocab [[buffer(5)]],
    constant int& cand_k [[buffer(6)]],
    device uint* surv_id [[buffer(7)]],
    device float* surv_p [[buffer(8)]],
    device int* surv_len [[buffer(9)]],
    constant int& surv_stride [[buffer(10)]],
    threadgroup uint* smem [[threadgroup(0)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]]) {
    const int row = int(tgid.x);
    const int t = int(tid.x);
    threadgroup float* rv = (threadgroup float*)smem;
    threadgroup int* ri = (threadgroup int*)(rv + SAMPLE_BLOCK);
    threadgroup float* kv = (threadgroup float*)(ri + SAMPLE_BLOCK);
    threadgroup int* ki = (threadgroup int*)(kv + SAMPLE_BLOCK);

    const SampleParams p = params[row];
    const int k = min(max(p.top_k, 1), vocab);
    const int total_cand = SAMPLE_SPLITS * cand_k;
    device const float* cv = cand_v + size_t(row) * size_t(total_cand);
    device const int* ci = cand_i + size_t(row) * size_t(total_cand);

    float lastv = INFINITY;
    int lasti = -1;
    for (int j = 0; j < k; ++j) {
        float bv = -INFINITY;
        int bi = 0;
        bool have = false;
        for (int i = t; i < total_cand; i += SAMPLE_BLOCK) {
            const float v = cv[i];
            const int id = ci[i];
            if (id >= vocab) continue;
            if (j > 0 && !samp_better(lastv, lasti, v, id)) continue;
            if (!have || samp_better(v, id, bv, bi)) {
                bv = v;
                bi = id;
                have = true;
            }
        }
        rv[t] = have ? bv : -INFINITY;
        ri[t] = have ? bi : 0x7fffffff;
        samp_reduce(rv, ri, t);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        lastv = rv[0];
        lasti = ri[0];
        if (t == 0) {
            kv[j] = lastv;
            ki[j] = lasti;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (t == 0) {
        samp_tail(kv, ki, k, p, samp_draw(rnd, row), out, row,
                  surv_id, surv_p, surv_len, surv_stride);
    }
}
