// Mixture-of-experts routing and the per-expert mat-vec.
//
// Two things live here that the dense path has no equivalent of. The router
// turns one row of logits into `k` expert ids and the weights to combine their
// outputs with, and `mmvq_moe` runs the same integer mat-vec the dense decode
// step uses against `k` different weight matrices — the active experts —
// selected by an index read on the device rather than by the host.
//
// The expert blocks are concatenated with a constant byte stride, so an
// expert's weights are the dense layout at an offset. That is the whole reason
// this file is short: `tq_dot_q4_g128t` addresses quants and scales from a base
// pointer, a row and the matrix's own row count, and none of those change when
// the base moves. See `weights::Experts`.

// Softmax over every expert, then the top `k` of it.
//
// One block per token, `blockDim.x` at least `n_experts`. The reference is
// `F.softmax(logits, -1)` followed by `topk` followed by an optional
// renormalization, and that order matters: with `norm_topk_prob` the global
// denominator cancels and the weights are a softmax over the selected logits
// alone, but without it they are not, so both sums are computed.
//
// Ties go to the lower index, which is what `torch.topk` does. Selecting the
// higher one instead routes to a different expert on exactly the rows where two
// are equally good, which is rare, correct-looking, and untraceable.
extern "C" __global__ void moe_topk_f32(int* __restrict__ ids,
                                        float* __restrict__ weights,
                                        const float* __restrict__ logits,
                                        int n_experts, int k, int norm) {
    extern __shared__ char smem[];
    float* val = (float*)smem;                       // n_experts
    const int t = blockIdx.x;
    const float* row = logits + (size_t)t * n_experts;

    for (int e = threadIdx.x; e < n_experts; e += blockDim.x) val[e] = row[e];
    __syncthreads();

    // Thread 0 does the selection. `k` is 8 and `n_experts` 128 on the
    // checkpoint this was written for, so this is 1024 comparisons against a
    // block reduction's synchronization overhead, and the whole kernel is one
    // launch per layer either way.
    if (threadIdx.x != 0) return;

    float max_all = -INFINITY;
    for (int e = 0; e < n_experts; ++e) max_all = fmaxf(max_all, val[e]);
    float sum_all = 0.0f;
    for (int e = 0; e < n_experts; ++e) sum_all += __expf(val[e] - max_all);

    float sum_top = 0.0f;
    for (int a = 0; a < k; ++a) {
        int best = 0;
        float bv = -INFINITY;
        for (int e = 0; e < n_experts; ++e) {
            if (val[e] > bv) {
                bv = val[e];
                best = e;
            }
        }
        ids[(size_t)t * k + a] = best;
        const float p = __expf(bv - max_all);
        weights[(size_t)t * k + a] = p;
        sum_top += p;
        val[best] = -INFINITY;
    }
    const float denom = norm ? sum_top : sum_all;
    for (int a = 0; a < k; ++a) weights[(size_t)t * k + a] /= denom;
}

// The dense mat-vec, run once per (token, active expert) pair.
//
// `grid.y` is that pair — `slot` — rather than a token group, which is the axis
// `mmvq_batch` cannot express. `out` is `[n_slots, n]` and `expert_ids`
// `[n_slots]`, both indexed by it.
//
// `y_group` is how many consecutive slots share one activation row, and it is
// what lets decode and prefill be the same launch. The `gate` and `up`
// projections read the token's normalized residual, so a token's `k` slots all
// want the same row: `y_group = k_active`, and at one token that is every slot
// reading row zero. `down` reads the SwiGLU product, which is per slot:
// `y_group = 1`. Getting this backwards gives every expert the first expert's
// activation, which is a plausible-looking answer and a wrong one.
#define MMVQ_MOE_KERNEL(NAME, DOT, SLICES, IQS_STEP, BLOCK_ELEMS)              \
    extern "C" __global__ void NAME(float* __restrict__ out,                   \
                                    const void* __restrict__ w_all,            \
                                    const int* __restrict__ expert_ids,        \
                                    const block_q8_1* __restrict__ y, int k,   \
                                    int n, long long stride, int y_group) {    \
        const int row = blockIdx.x;                                            \
        const int slot = blockIdx.y;                                           \
        if (row >= n) return;                                                  \
        const int nb = k / (BLOCK_ELEMS);                                      \
        const int ny = k / QK8_1; /* Q8_1 blocks per activation row */          \
        const int wn = n;                                                      \
        const char* wbase = (const char*)w_all + (size_t)expert_ids[slot] * stride; \
        const char* wr = wbase + (size_t)row * nb * WEIGHT_STRIDE;             \
        const block_q8_1* yr = y + (size_t)(slot / y_group) * ny;               \
                                                                               \
        float acc = 0.0f;                                                      \
        for (int c = threadIdx.x; c < nb * (SLICES); c += blockDim.x) {         \
            const int kbx = c / (SLICES);                                      \
            const int iqs = (c % (SLICES)) * (IQS_STEP);                       \
            acc += DOT(wr, yr + kbx * ((BLOCK_ELEMS) / QK8_1), kbx, iqs);      \
        }                                                                      \
                                                                               \
        const float total = block_reduce_sum(acc);                             \
        if (threadIdx.x == 0) out[(size_t)slot * n + row] = total;             \
    }

// `TQ_DOT_G128T` binds `wbase`, `nb` and `wn` from the enclosing scope, which
// the macro above defines exactly as the dense one does — so the transposed
// layout's scale block lands at `wbase + wn * nb * 64` and the offset that
// selected the expert carries through it.
#define WEIGHT_STRIDE (int)sizeof(block_q4_g128)
MMVQ_MOE_KERNEL(mmvq_moe_q4_g128, tq_dot_q4_g128, 4, 1, 128)
MMVQ_MOE_KERNEL(mmvq_moe_q4_g128t, TQ_DOT_G128T, 4, 1, 128)
#undef WEIGHT_STRIDE

#define WEIGHT_STRIDE (int)sizeof(block_q8_0)
MMVQ_MOE_KERNEL(mmvq_moe_q8_0, tq_dot_q8_0, 4, 2, 32)
#undef WEIGHT_STRIDE

// Weighted sum of the active experts' outputs.
//
// `partials` is `[n_tokens, k, d]` and `weights` `[n_tokens, k]`; the result is
// one row of `d` per token. Accumulating in f32 in registers rather than with
// atomics keeps this independent of the order the experts finished in, which is
// what makes a MoE step reproducible at a fixed batch width.
extern "C" __global__ void moe_combine_f32(float* __restrict__ out,
                                           const float* __restrict__ partials,
                                           const float* __restrict__ weights,
                                           int d, int k, int total) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int token = i / d;
    const int col = i - token * d;
    const float* p = partials + (size_t)token * k * d + col;
    const float* w = weights + (size_t)token * k;
    float acc = 0.0f;
    for (int a = 0; a < k; ++a) acc += w[a] * p[(size_t)a * d];
    out[i] = acc;
}
