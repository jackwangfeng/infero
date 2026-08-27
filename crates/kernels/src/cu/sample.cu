// Sampling, on the device.
//
// The logits used to come back to the host in full: one row per sequence, the
// whole vocabulary, every step. At a batch of 32 with Llama-3.1's 128256-entry
// vocabulary that is a 16 MiB device-to-host copy, and it measured 2.19 ms of a
// 12.18 ms step — more than the attention output projection and the six
// elementwise kernels put together. vLLM never pays it: `v1/sample/sampler.py`
// keeps the logits on the device through penalties, temperature, top-k/top-p
// and the multinomial draw, and only `sampled_token_ids` — one int per row —
// crosses back.
//
// This is that, with infero's parameter set and, importantly, infero's exact
// semantics. `crates/model/src/sampling.rs` has two paths that do not agree
// with each other about the repetition penalty, and both have to be reproduced:
//
//   * `greedy()` marks the window in a bitset and penalizes each distinct
//     token **once**.
//   * `apply_repetition_penalty()` walks the window and rewrites the candidate
//     in place, so a token appearing three times is penalized **three times**.
//
// The sign check happens per application in both, and since a penalty above one
// never moves a value across zero, iterating is the same as one divide by
// `p^count` — except in float rounding, which is why the loop below iterates
// rather than calling `powf`.
//
// The random draw stays on the host: one `f64` per row from that sequence's own
// `StdRng`, uploaded with the batch. The generator, its seeding, and its
// sequence are therefore untouched — a seeded run gives the same tokens it gave
// before, which a device-side RNG would not have preserved.

#define SAMPLE_BLOCK 256

/// A row's parameters, mirroring `SamplingParams`.
struct SampleParams {
    float temperature;
    float top_p;
    int top_k;
    float rep_penalty;
};

/// (value, index) under "higher value first, lower index first".
///
/// The tie-break is not decoration: the host scans with a strict `>` from
/// negative infinity, so among equal logits it keeps the lowest index, and a
/// reduction that kept an arbitrary one would disagree with it on exactly the
/// inputs a test would use.
__device__ __forceinline__ bool samp_better(float av, int ai, float bv, int bi) {
    return av > bv || (av == bv && ai < bi);
}

__device__ __forceinline__ void samp_reduce(float* sv, int* si, int tid) {
    for (int s = SAMPLE_BLOCK / 2; s > 0; s >>= 1) {
        __syncthreads();
        if (tid < s && samp_better(sv[tid + s], si[tid + s], sv[tid], si[tid])) {
            sv[tid] = sv[tid + s];
            si[tid] = si[tid + s];
        }
    }
    __syncthreads();
}

/// How many times `tok` appears in this row's window, by binary search over the
/// sorted unique ids the host uploaded. Only ever called on a bitset hit, so it
/// runs a few hundred times per row rather than once per vocabulary entry.
__device__ __forceinline__ int samp_count(const int* toks, const int* cnts,
                                          int len, int tok) {
    int lo = 0, hi = len - 1;
    while (lo <= hi) {
        const int mid = (lo + hi) >> 1;
        if (toks[mid] == tok) return cnts[mid];
        if (toks[mid] < tok) lo = mid + 1;
        else hi = mid - 1;
    }
    return 0;
}

/// The penalized logit. `once` picks between the two host paths.
__device__ __forceinline__ float samp_penalize(float l, int count, float p,
                                               bool once) {
    if (count <= 0 || p == 1.0f) return l;
    const int n = once ? 1 : count;
    for (int i = 0; i < n; ++i) l = l > 0.0f ? l / p : l * p;
    return l;
}

/// The greedy path, split across the device instead of one block a row.
///
/// `sample_rows_f32` gives a row to a block, which at a batch of 32 is 32 blocks
/// of 256 threads — 2% of a 188-SM card — and 128256 logits a row then take
/// 175 us a step at 94 GB/s where the bytes alone are 16.4 MB. The scan is a
/// reduction, so it splits: every block takes a slice of the vocabulary, and a
/// second kernel picks the winner among the slices. Nothing about the answer
/// changes — `samp_better` breaks ties by lowest index in both passes, so the
/// token is the same one the single-block kernel would have chosen.
///
/// Only greedy rows. Above one candidate the survivors have to be selected
/// across the *whole* row in descending order, which is what the top-k loop in
/// `sample_rows_f32` does and is not a per-slice reduction.
///
/// The penalty bitmap covers this slice only — `chunk/32` words rather than the
/// vocabulary's 4008 — which is what keeps the shared memory small enough for
/// many blocks an SM.
extern "C" __global__ void argmax_partial_f32(
    float* __restrict__ pv, int* __restrict__ pi,
    const float* __restrict__ logits, const SampleParams* __restrict__ params,
    const int* __restrict__ pen_tok, const int* __restrict__ pen_cnt,
    const int* __restrict__ pen_len, int vocab, int pen_stride, int splits) {
    extern __shared__ __align__(16) unsigned int smem[];

    const int s = blockIdx.x;
    const int row = blockIdx.y;
    const int tid = threadIdx.x;
    const int chunk = (vocab + splits - 1) / splits;
    const int lo = s * chunk;
    const int hi = min(vocab, lo + chunk);
    const int words = (chunk + 31) / 32 + 1;

    unsigned int* bits = smem;
    float* rv = (float*)(void*)(smem + words);
    int* ri = (int*)(void*)(rv + SAMPLE_BLOCK);

    const SampleParams p = params[row];
    const float* row_logits = logits + (size_t)row * vocab;
    const int plen = pen_len[row];
    const int* ptok = pen_tok + (size_t)row * pen_stride;
    const int* pcnt = pen_cnt + (size_t)row * pen_stride;

    for (int i = tid; i < words; i += SAMPLE_BLOCK) bits[i] = 0u;
    __syncthreads();
    for (int i = tid; i < plen; i += SAMPLE_BLOCK) {
        const int t = ptok[i];
        if (t >= lo && t < hi) {
            const int b = t - lo;
            atomicOr(&bits[b >> 5], 1u << (b & 31));
        }
    }
    __syncthreads();

    float best = -INFINITY;
    int besti = 0x7fffffff;
    for (int i = lo + tid; i < hi; i += SAMPLE_BLOCK) {
        float v = row_logits[i];
        const int b = i - lo;
        if (bits[b >> 5] & (1u << (b & 31))) {
            v = samp_penalize(v, samp_count(ptok, pcnt, plen, i), p.rep_penalty,
                              true);
        }
        if (samp_better(v, i, best, besti)) {
            best = v;
            besti = i;
        }
    }
    rv[tid] = best;
    ri[tid] = besti;
    samp_reduce(rv, ri, tid);
    if (tid == 0) {
        pv[(size_t)row * splits + s] = rv[0];
        pi[(size_t)row * splits + s] = ri[0];
    }
}

/// One block a row over the slice winners, which is a few hundred values.
extern "C" __global__ void argmax_combine_f32(unsigned int* __restrict__ out,
                                             const float* __restrict__ pv,
                                             const int* __restrict__ pi,
                                             int splits) {
    extern __shared__ __align__(16) unsigned int smem[];
    float* rv = (float*)(void*)smem;
    int* ri = (int*)(void*)(rv + SAMPLE_BLOCK);

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    float best = -INFINITY;
    int besti = 0x7fffffff;
    for (int i = tid; i < splits; i += SAMPLE_BLOCK) {
        const float v = pv[(size_t)row * splits + i];
        const int idx = pi[(size_t)row * splits + i];
        if (samp_better(v, idx, best, besti)) {
            best = v;
            besti = idx;
        }
    }
    rv[tid] = best;
    ri[tid] = besti;
    samp_reduce(rv, ri, tid);
    if (tid == 0) out[row] = (unsigned int)ri[0];
}

/// One block per row.
///
/// `pen_tok` / `pen_cnt` hold each row's window as sorted unique ids with
/// counts, `pen_len` how many; `rnd` is that row's uniform draw. `out` takes
/// the sampled id. Dynamic shared memory is the vocabulary bitset followed by
/// the reduction scratch and the top-k survivors.
//
// `surv_id` / `surv_p` / `surv_len` are optional and, when given, take the
// surviving distribution the draw was made from: the ids the nucleus kept, in
// descending order, with their normalized probabilities. Nothing extra is
// computed for them — the kernel already has all three at the end.
//
// Speculative decoding is why. The acceptance test needs `q(x)` for the token
// the drafter sampled, and a rejection needs the whole of `q` to subtract from
// `p`; both are over the truncated support, a few dozen entries. Reading them
// back is a kilobyte, where having the host redo the sampling means copying
// 248320 logits — 993 KB and a full-vocabulary pass — which measured 0.709 ms of
// a 2.249 ms draft, a third of it, and 1.42 ms of a round at k = 2.
extern "C" __global__ void sample_rows_f32(
    unsigned int* __restrict__ out, const float* __restrict__ logits,
    const SampleParams* __restrict__ params, const int* __restrict__ pen_tok,
    const int* __restrict__ pen_cnt, const int* __restrict__ pen_len,
    const double* __restrict__ rnd, int vocab, int pen_stride,
    unsigned int* __restrict__ surv_id, float* __restrict__ surv_p,
    int* __restrict__ surv_len, int surv_stride) {
    extern __shared__ __align__(16) unsigned int smem[];

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int words = (vocab + 31) / 32;

    unsigned int* bits = smem;
    float* rv = (float*)(void*)(smem + words);
    int* ri = (int*)(void*)(rv + SAMPLE_BLOCK);
    // The survivors, in descending order. `top_k` is bounded by the host.
    float* kv = (float*)(void*)(ri + SAMPLE_BLOCK);
    int* ki = (int*)(void*)(kv + SAMPLE_BLOCK);

    const SampleParams p = params[row];
    const float* row_logits = logits + (size_t)row * vocab;
    const int plen = pen_len[row];
    const int* ptok = pen_tok + (size_t)row * pen_stride;
    const int* pcnt = pen_cnt + (size_t)row * pen_stride;

    // `is_greedy()`: zero temperature, or a top-k of one.
    const bool greedy = p.temperature <= 0.0f || p.top_k == 1;

    for (int i = tid; i < words; i += SAMPLE_BLOCK) bits[i] = 0u;
    __syncthreads();
    for (int i = tid; i < plen; i += SAMPLE_BLOCK) {
        const int t = ptok[i];
        if (t >= 0 && t < vocab) atomicOr(&bits[t >> 5], 1u << (t & 31));
    }
    __syncthreads();

    // The whole vocabulary, penalized on the fly. Materializing it would be the
    // megabyte of writes per row the host path was rewritten to avoid.
    float best = -INFINITY;
    int besti = 0;
    for (int i = tid; i < vocab; i += SAMPLE_BLOCK) {
        float v = row_logits[i];
        if (bits[i >> 5] & (1u << (i & 31))) {
            v = samp_penalize(v, samp_count(ptok, pcnt, plen, i), p.rep_penalty,
                              greedy);
        }
        if (samp_better(v, i, best, besti)) {
            best = v;
            besti = i;
        }
    }
    rv[tid] = best;
    ri[tid] = besti;
    samp_reduce(rv, ri, tid);

    if (greedy) {
        if (tid == 0) out[row] = (unsigned int)ri[0];
        return;
    }

    const int k = min(max(p.top_k, 1), vocab);
    // Pass one already found the maximum; each further pass takes the best pair
    // strictly below the last one taken, which leaves the survivors sorted —
    // which is what the nucleus prefix below needs anyway.
    if (tid == 0) {
        kv[0] = rv[0];
        ki[0] = ri[0];
    }
    __syncthreads();
    for (int j = 1; j < k; ++j) {
        const float lastv = kv[j - 1];
        const int lasti = ki[j - 1];
        float bv = -INFINITY;
        int bi = 0;
        bool have = false;
        for (int i = tid; i < vocab; i += SAMPLE_BLOCK) {
            float v = row_logits[i];
            if (bits[i >> 5] & (1u << (i & 31))) {
                v = samp_penalize(v, samp_count(ptok, pcnt, plen, i),
                                  p.rep_penalty, false);
            }
            // Strictly worse than the last one taken, in the same order the
            // reduction uses. Testing "not better" instead would keep the last
            // one itself — and everything above it — so pass two would come
            // back empty and pass three would hand back the global maximum a
            // second time.
            if (!samp_better(lastv, lasti, v, i)) continue;
            if (!have || samp_better(v, i, bv, bi)) {
                bv = v;
                bi = i;
                have = true;
            }
        }
        rv[tid] = have ? bv : -INFINITY;
        ri[tid] = have ? bi : 0x7fffffff;
        samp_reduce(rv, ri, tid);
        if (tid == 0) {
            kv[j] = rv[0];
            ki[j] = ri[0];
        }
        __syncthreads();
    }

    // Softmax at temperature over the survivors, the nucleus cut, then the
    // draw — all of it serial over at most a few hundred entries, and all of it
    // in the order and the precision the host used.
    if (tid == 0) {
        const float inv_t = 1.0f / fmaxf(p.temperature, 1e-5f);
        const float mx = kv[0];
        double total = 0.0;
        for (int j = 0; j < k; ++j) {
            const double q = exp((double)((kv[j] - mx) * inv_t));
            kv[j] = (float)q;
            total += q;
        }
        int keep = k;
        if (p.top_p < 1.0f) {
            const double target = total * (double)fminf(fmaxf(p.top_p, 1e-4f), 1.0f);
            double acc = 0.0;
            keep = 0;
            for (int j = 0; j < k; ++j) {
                acc += (double)kv[j];
                ++keep;
                if (acc >= target) break;
            }
            if (keep < 1) keep = 1;
            total = 0.0;
            for (int j = 0; j < keep; ++j) total += (double)kv[j];
        }
        double r = rnd[row] * total;
        unsigned int pick = (unsigned int)ki[keep - 1];
        for (int j = 0; j < keep; ++j) {
            r -= (double)kv[j];
            if (r <= 0.0) {
                pick = (unsigned int)ki[j];
                break;
            }
        }
        out[row] = pick;
        // The distribution the draw came from, for a caller that has to compose
        // with it. Normalized here rather than on the host so that `q` sums to
        // one in the same arithmetic that picked from it.
        if (surv_len) {
            surv_len[row] = keep;
            const double inv = total > 0.0 ? 1.0 / total : 0.0;
            for (int j = 0; j < keep && j < surv_stride; ++j) {
                surv_id[(size_t)row * surv_stride + j] = (unsigned int)ki[j];
                surv_p[(size_t)row * surv_stride + j] = (float)((double)kv[j] * inv);
            }
        }
    }
}

// ---- top-k in one pass over the vocabulary -----------------------------------
//
// `sample_rows_f32` above finds its survivors by scanning the vocabulary once
// per survivor: pass `j` takes the best entry strictly below pass `j-1`. That is
// fine at the vocabularies it was written for and quadratic in the wrong place
// at 248320 tokens — at `top_k = 40` it is forty passes and ten million reads,
// all in one block, and it measured 5.99 ms against the host's 0.71 for a
// speculative draft.
//
// So: split the vocabulary across blocks, have each emit its own slice's top-k,
// and merge. A token in the global top-k is in its own slice's top-k — at most
// `k-1` tokens beat it anywhere, so at most that many beat it in its slice — so
// `k` candidates a slice is exactly enough and nothing is lost.
//
// This is the shape `argmax_partial_f32` / `argmax_combine_f32` already use for
// `k = 1`, generalized. The penalties are applied in the first stage and the
// penalized value travels with the candidate, so the second stage and the tail
// see exactly what a single-block scan would have.
#define SAMPLE_SPLITS 64

/// One block per (row, slice). Emits the slice's top-`k` as (value, id) pairs.
///
/// The penalty bitset covers the slice only — 122 words at this vocabulary and
/// this split count, against 7760 for the whole of it — which is what lets many
/// of these blocks be resident at once.
extern "C" __global__ void sample_topk_partial_f32(
    float* __restrict__ cand_v, int* __restrict__ cand_i,
    const float* __restrict__ logits, const SampleParams* __restrict__ params,
    const int* __restrict__ pen_tok, const int* __restrict__ pen_cnt,
    const int* __restrict__ pen_len, int vocab, int pen_stride, int cand_k) {
    extern __shared__ __align__(16) unsigned int smem[];

    const int row = blockIdx.x;
    const int split = blockIdx.y;
    const int tid = threadIdx.x;

    const int per = (vocab + SAMPLE_SPLITS - 1) / SAMPLE_SPLITS;
    const int lo = split * per;
    const int hi = min(lo + per, vocab);

    const int words = (per + 31) / 32;
    unsigned int* bits = smem;
    float* rv = (float*)(void*)(smem + words);
    int* ri = (int*)(void*)(rv + SAMPLE_BLOCK);

    const SampleParams p = params[row];
    const float* row_logits = logits + (size_t)row * vocab;
    const int plen = pen_len[row];
    const int* ptok = pen_tok + (size_t)row * pen_stride;
    const int* pcnt = pen_cnt + (size_t)row * pen_stride;
    const bool greedy = p.temperature <= 0.0f || p.top_k == 1;
    const int k = min(max(p.top_k, 1), vocab);

    float* out_v = cand_v + ((size_t)row * SAMPLE_SPLITS + split) * cand_k;
    int* out_i = cand_i + ((size_t)row * SAMPLE_SPLITS + split) * cand_k;

    // Slice-local bitset: only the window entries that land in [lo, hi).
    for (int i = tid; i < words; i += SAMPLE_BLOCK) bits[i] = 0u;
    __syncthreads();
    for (int i = tid; i < plen; i += SAMPLE_BLOCK) {
        const int t = ptok[i];
        if (t >= lo && t < hi) {
            const int r = t - lo;
            atomicOr(&bits[r >> 5], 1u << (r & 31));
        }
    }
    __syncthreads();

    // The slice's own top-`k`, by the same rule the single-block version uses,
    // over `per` entries instead of `vocab`.
    float lastv = INFINITY;
    int lasti = -1;
    for (int j = 0; j < k; ++j) {
        float bv = -INFINITY;
        int bi = 0;
        bool have = false;
        for (int i = lo + tid; i < hi; i += SAMPLE_BLOCK) {
            float v = row_logits[i];
            const int r = i - lo;
            if (bits[r >> 5] & (1u << (r & 31))) {
                v = samp_penalize(v, samp_count(ptok, pcnt, plen, i),
                                  p.rep_penalty, greedy);
            }
            if (j > 0 && !samp_better(lastv, lasti, v, i)) continue;
            if (!have || samp_better(v, i, bv, bi)) {
                bv = v;
                bi = i;
                have = true;
            }
        }
        rv[tid] = have ? bv : -INFINITY;
        ri[tid] = have ? bi : 0x7fffffff;
        samp_reduce(rv, ri, tid);
        __syncthreads();
        lastv = rv[0];
        lasti = ri[0];
        if (tid == 0) {
            // A slice narrower than `k` runs out; `-INFINITY` with an
            // unreachable id makes the merge ignore the padding.
            out_v[j] = lastv;
            out_i[j] = (lasti == 0x7fffffff) ? vocab : lasti;
        }
        __syncthreads();
        if (lasti == 0x7fffffff) {
            // Nothing left in this slice: pad the rest and stop.
            for (int t2 = j + 1 + tid; t2 < k; t2 += SAMPLE_BLOCK) {
                out_v[t2] = -INFINITY;
                out_i[t2] = vocab;
            }
            break;
        }
    }
}

/// One block per row. Merges the slice candidates, then samples exactly as
/// `sample_rows_f32` does from that point on.
///
/// The candidates are read from global rather than staged: `SAMPLE_SPLITS * k`
/// is a few thousand entries, they are L2-hot, and staging them would want 32 KB
/// of shared memory that the survivors and the reduction also need.
extern "C" __global__ void sample_rows_topk_f32(
    unsigned int* __restrict__ out, const float* __restrict__ cand_v,
    const int* __restrict__ cand_i, const SampleParams* __restrict__ params,
    const double* __restrict__ rnd, int vocab, int cand_k,
    unsigned int* __restrict__ surv_id, float* __restrict__ surv_p,
    int* __restrict__ surv_len, int surv_stride) {
    extern __shared__ __align__(16) unsigned int smem[];

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    float* rv = (float*)(void*)smem;
    int* ri = (int*)(void*)(rv + SAMPLE_BLOCK);
    float* kv = (float*)(void*)(ri + SAMPLE_BLOCK);
    int* ki = (int*)(void*)(kv + SAMPLE_BLOCK);

    const SampleParams p = params[row];
    const int k = min(max(p.top_k, 1), vocab);
    const int total_cand = SAMPLE_SPLITS * cand_k;
    const float* cv = cand_v + (size_t)row * total_cand;
    const int* ci = cand_i + (size_t)row * total_cand;

    float lastv = INFINITY;
    int lasti = -1;
    for (int j = 0; j < k; ++j) {
        float bv = -INFINITY;
        int bi = 0;
        bool have = false;
        for (int i = tid; i < total_cand; i += SAMPLE_BLOCK) {
            const float v = cv[i];
            const int id = ci[i];
            if (id >= vocab) continue;  // padding from a short slice
            if (j > 0 && !samp_better(lastv, lasti, v, id)) continue;
            if (!have || samp_better(v, id, bv, bi)) {
                bv = v;
                bi = id;
                have = true;
            }
        }
        rv[tid] = have ? bv : -INFINITY;
        ri[tid] = have ? bi : 0x7fffffff;
        samp_reduce(rv, ri, tid);
        __syncthreads();
        lastv = rv[0];
        lasti = ri[0];
        if (tid == 0) {
            kv[j] = lastv;
            ki[j] = lasti;
        }
        __syncthreads();
    }

    // Identical to the tail of `sample_rows_f32`, in the same order and the same
    // precision — the survivors are the same values a single-block scan would
    // have put there.
    if (tid == 0) {
        const float inv_t = 1.0f / fmaxf(p.temperature, 1e-5f);
        const float mx = kv[0];
        double total = 0.0;
        for (int j = 0; j < k; ++j) {
            const double q = exp((double)((kv[j] - mx) * inv_t));
            kv[j] = (float)q;
            total += q;
        }
        int keep = k;
        if (p.top_p < 1.0f) {
            const double target = total * (double)fminf(fmaxf(p.top_p, 1e-4f), 1.0f);
            double acc = 0.0;
            keep = 0;
            for (int j = 0; j < k; ++j) {
                acc += (double)kv[j];
                ++keep;
                if (acc >= target) break;
            }
            if (keep < 1) keep = 1;
            total = 0.0;
            for (int j = 0; j < keep; ++j) total += (double)kv[j];
        }
        double r = rnd[row] * total;
        unsigned int pick = (unsigned int)ki[keep - 1];
        for (int j = 0; j < keep; ++j) {
            r -= (double)kv[j];
            if (r <= 0.0) {
                pick = (unsigned int)ki[j];
                break;
            }
        }
        out[row] = pick;
        if (surv_len) {
            surv_len[row] = keep;
            const double inv = total > 0.0 ? 1.0 / total : 0.0;
            for (int j = 0; j < keep && j < surv_stride; ++j) {
                surv_id[(size_t)row * surv_stride + j] = (unsigned int)ki[j];
                surv_p[(size_t)row * surv_stride + j] = (float)((double)kv[j] * inv);
            }
        }
    }
}
