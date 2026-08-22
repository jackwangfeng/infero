// GatedDeltaNet: the linear-attention block of Qwen3.5, and the output gate its
// full-attention blocks carry.
//
// Kept as its own translation unit rather than appended to ops.cu because it is
// a self-contained mechanism with its own state discipline: the recurrent state
// is updated in place and persists across calls, which nothing else in this
// engine does.
//
// Activations are f32 and the state is f32 — the checkpoint says
// `mamba_ssm_dtype: float32`, and the state is a running accumulation over the
// whole sequence, so narrowing it is not the same trade as narrowing a weight.
//
// Every layout choice here has a second reading that runs to completion and
// produces fluent nonsense, so all of it is checked against
// `tuili_model::qwen35`, which is in turn checked against a capture of the
// reference implementation on the real checkpoint. See tests/gated_delta.rs.

//
// Qwen3.5 runs linear attention in 48 of its 64 layers. The mechanism is a
// gated delta rule: a per-head state matrix `S[dk, dv]` that decays, absorbs a
// correction toward the current value, and is read out by the query. Unlike a
// KV cache it does not grow with the sequence, and unlike a KV cache it is
// updated *in place*, which is what makes it awkward for a graph-replay engine.
//
// Every layout choice below has a second reading that runs to completion and
// produces fluent nonsense, so all of it is checked against
// `tuili_model::qwen35`, which is in turn checked against a capture of the
// reference implementation on the real checkpoint. See tests/gated_delta.rs.

// Depthwise causal convolution over time with a carried window, plus SiLU.
//
// `x` and `out` are `[n_tokens, channels]`; `state` is
// `[n_seqs, channels, k - 1]`, oldest tap first. Tokens of one sequence must be
// contiguous and in order, which `first_token`/`n_tok` assert by construction.
//
// The direction is the thing to get right: output `t` reads inputs
// `t-(k-1) ..= t`, and weight `j` pairs with input `t - (k-1) + j`, so `w[k-1]`
// multiplies the current token. Reversing that runs fine and shifts the whole
// model one token into the future.
//
// One thread a channel; the loop over tokens is serial because the window is.
extern "C" __global__ void gdn_conv_f32(float* __restrict__ out,
                                        const float* __restrict__ x,
                                        float* __restrict__ state,
                                        const float* __restrict__ w,
                                        const int* __restrict__ first_token,
                                        const int* __restrict__ n_tok,
                                        int channels, int k) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    const int seq = blockIdx.y;
    const int nt = n_tok[seq];
    if (nt <= 0) return;   // a slot not in this batch
    const int t0 = first_token[seq];
    const int hist = k - 1;

    float* st = state + ((size_t)seq * channels + c) * hist;
    // The window, oldest first, held in registers across the sequence. k is 4
    // in this checkpoint, so this is three floats; the bound keeps the array a
    // compile-time size.
    float win[8];
    for (int j = 0; j < hist; ++j) win[j] = st[j];

    const float* wc = w + (size_t)c * k;
    for (int n = 0; n < nt; ++n) {
        const float cur = x[(size_t)(t0 + n) * channels + c];
        float acc = wc[hist] * cur;
        for (int j = 0; j < hist; ++j) acc += wc[j] * win[j];
        // SiLU, fused: the reference applies it to the convolution output
        // before the split into q, k and v.
        out[(size_t)(t0 + n) * channels + c] = acc / (1.0f + __expf(-acc));
        for (int j = 0; j + 1 < hist; ++j) win[j] = win[j + 1];
        if (hist > 0) win[hist - 1] = cur;
    }
    for (int j = 0; j < hist; ++j) st[j] = win[j];
}

// beta = sigmoid(b);  g = -exp(A_log) * softplus(a + dt_bias)
//
// `g` is non-positive by construction, so `exp(g)` in the recurrence is a
// decay. Losing the sign makes the state grow without bound, which shows up as
// NaN several layers later rather than here.
//
// One thread a (token, head) pair.
extern "C" __global__ void gdn_gate_decay_f32(float* __restrict__ beta_out,
                                              float* __restrict__ g_out,
                                              const float* __restrict__ a,
                                              const float* __restrict__ b,
                                              const float* __restrict__ a_log,
                                              const float* __restrict__ dt_bias,
                                              int n_tokens, int heads) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_tokens * heads) return;
    const int h = idx % heads;
    beta_out[idx] = 1.0f / (1.0f + __expf(-b[idx]));
    const float z = a[idx] + dt_bias[h];
    // log1p(exp(z)) with the large-z branch taken directly: dt_bias reaches +19
    // in this checkpoint, and exp of a large sum overflows before the log can
    // bring it back.
    const float sp = z > 20.0f ? z : log1pf(__expf(z));
    g_out[idx] = -__expf(a_log[h]) * sp;
}

// L2-normalize each head's row of q and k in place, then scale q by 1/sqrt(dk).
//
// The reference does both inside its kernel (`use_qk_l2norm_in_kernel=True`),
// and the scale lands on q only. `eps` is added to the sum of squares, not to
// the norm, matching the FLA convention the reference cites.
//
// q and k are normalized where they lie, inside the packed
// `[q | k | v]` row that the input projection produced — `stride` is the row's
// width and `q_off`/`k_off` locate them. Copying them out first would cost a
// pass over 4 KiB a token for nothing.
//
// One block a (token, key head) pair; the block reduces over dk.
extern "C" __global__ void gdn_qk_l2norm_f32(float* __restrict__ qkv,
                                             int key_heads, int dk,
                                             int stride, int q_off, int k_off,
                                             float eps, float q_scale) {
    const int token = blockIdx.x / key_heads;
    const int head = blockIdx.x % key_heads;
    float* base = qkv + (size_t)token * stride + (size_t)head * dk;
    float* qr = base + q_off;
    float* kr = base + k_off;

    float qa = 0.0f, ka = 0.0f;
    for (int i = threadIdx.x; i < dk; i += blockDim.x) {
        qa += qr[i] * qr[i];
        ka += kr[i] * kr[i];
    }
    const float qs = rsqrtf(block_reduce_sum(qa) + eps) * q_scale;
    __syncthreads();
    const float ks = rsqrtf(block_reduce_sum(ka) + eps);
    for (int i = threadIdx.x; i < dk; i += blockDim.x) {
        qr[i] *= qs;
        kr[i] *= ks;
    }
}

// The gated delta rule.
//
//   S      *= exp(g_t)
//   kv_mem  = kᵀ S                  contracting the key axis
//   delta   = (v_t - kv_mem) * beta_t
//   S      += k_t ⊗ delta
//   o_t     = qᵀ S                  the same contraction, with q
//
// Layout: `state` is `[n_seqs, heads, dk, dv]`; `qkv` holds q, k and v in one
// row per token, `stride` wide, at `q_off`, `k_off` and `v_off`, with q and k
// already normalized and q already scaled. `out` is `[n_tokens, heads, dv]`.
//
// The head expansion happens here rather than by materializing a wider q and k.
// There are `key_heads` key heads against `heads` value heads — 16 against 48 in
// this checkpoint — and the reference widens them with `repeat_interleave`, so
// value head `h` reads key head `h / (heads / key_heads)`. Expanding modularly
// (`h % key_heads`) also runs, and gives a different model.
//
// One block a (head, sequence); thread `j` owns column `j` of S, so the reads
// `S[i * dv + j]` are contiguous across the block. Two passes over S per token:
// the first cannot be merged with the second because `delta` needs the whole
// `kv_mem` reduction before the update can start. Holding S in registers across
// the whole chunk would remove the traffic entirely and is the obvious next
// step; this version keeps it in global memory because it is the one that can be
// read against a reference.
extern "C" __global__ void gdn_delta_rule_f32(float* __restrict__ out,
                                              float* __restrict__ state,
                                              const float* __restrict__ qkv,
                                              const float* __restrict__ g,
                                              const float* __restrict__ beta,
                                              const int* __restrict__ first_token,
                                              const int* __restrict__ n_tok,
                                              int heads, int key_heads,
                                              int dk, int dv,
                                              int stride, int q_off, int k_off,
                                              int v_off) {
    extern __shared__ float shared[];
    float* qs = shared;          // dk
    float* ks = shared + dk;     // dk

    const int head = blockIdx.x;
    const int seq = blockIdx.y;
    const int nt = n_tok[seq];
    // A sequence not in this batch has no tokens, and its block exits here.
    // That is what lets one launch cover the whole pool of sequence slots
    // without the caller compacting anything.
    if (nt <= 0) return;
    const int t0 = first_token[seq];
    const int j = threadIdx.x;
    const int khead = head / (heads / key_heads);

    float* S = state + ((size_t)seq * heads + head) * (size_t)dk * dv;

    for (int n = 0; n < nt; ++n) {
        const int t = t0 + n;
        const float* row = qkv + (size_t)t * stride;
        const float* qsrc = row + q_off + (size_t)khead * dk;
        const float* ksrc = row + k_off + (size_t)khead * dk;
        for (int i = threadIdx.x; i < dk; i += blockDim.x) {
            qs[i] = qsrc[i];
            ks[i] = ksrc[i];
        }
        __syncthreads();

        const float decay = __expf(g[(size_t)t * heads + head]);
        const float b = beta[(size_t)t * heads + head];

        if (j < dv) {
            const float v_tj = row[v_off + (size_t)head * dv + j];
            float kv = 0.0f;
            for (int i = 0; i < dk; ++i) {
                const float s = S[(size_t)i * dv + j] * decay;
                S[(size_t)i * dv + j] = s;
                kv += s * ks[i];
            }
            const float delta = (v_tj - kv) * b;
            float o = 0.0f;
            for (int i = 0; i < dk; ++i) {
                const float s = S[(size_t)i * dv + j] + ks[i] * delta;
                S[(size_t)i * dv + j] = s;
                o += s * qs[i];
            }
            out[((size_t)t * heads + head) * dv + j] = o;
        }
        // The next token overwrites the shared q and k, and reads S that this
        // token wrote.
        __syncthreads();
    }
}

// RMSNorm over each head's `dv` lane with a learned gain, then multiply by
// silu(z).
//
// Normalize first, gate second. Gating first and normalizing after is the other
// plausible order; it runs, and it is a different model. Note also that the
// gate is silu here, while the *attention* layers' output gate is sigmoid.
//
// One block a row of `dv`.
extern "C" __global__ void gdn_gated_rmsnorm_f32(float* __restrict__ out,
                                                 const float* __restrict__ x,
                                                 const float* __restrict__ z,
                                                 const float* __restrict__ weight,
                                                 int dv, float eps) {
    const size_t row = (size_t)blockIdx.x * dv;
    const float* xr = x + row;
    const float* zr = z + row;
    float* orow = out + row;

    float acc = 0.0f;
    for (int i = threadIdx.x; i < dv; i += blockDim.x) {
        const float val = xr[i];
        acc += val * val;
    }
    const float scale = rsqrtf(block_reduce_sum(acc) / (float)dv + eps);
    for (int i = threadIdx.x; i < dv; i += blockDim.x) {
        const float zi = zr[i];
        orow[i] = weight[i] * (xr[i] * scale) * (zi / (1.0f + __expf(-zi)));
    }
}

// Multiply an attention output by sigmoid of its gate, in place.
//
// The full-attention layers of Qwen3.5 carry an output gate applied before
// o_proj. It is sigmoid, not silu — config's `output_gate_type: "swish"` is not
// what the reference implementation reads.
extern "C" __global__ void sigmoid_gate_f32(float* __restrict__ x,
                                            const float* __restrict__ gate,
                                            long long n) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] *= 1.0f / (1.0f + __expf(-gate[i]));
}

// De-interleave a projection that produced a value and its gate per head.
//
// Qwen3.5's `q_proj` emits `heads * 2 * head_dim` columns, read as
// `[heads, 2 * head_dim]` with the query first and the gate second *within each
// head*. So the query's heads are strided by `2 * head_dim`, which no existing
// kernel's layout covers — `qk_norm` takes an offset and a row stride but
// assumes a head's lanes are contiguous from there.
//
// The other reading, `[all queries | all gates]`, is a plain split with no
// kernel at all. It runs, and it is wrong: for every head past the first it
// reads the wrong columns. See `the_split_is_per_head_not_per_half` in
// tests/gated_delta.rs.
//
// One thread an output element of `q`; `gate` gets the matching one.
extern "C" __global__ void split_interleaved_f32(float* __restrict__ q,
                                                 float* __restrict__ gate,
                                                 const float* __restrict__ src,
                                                 int heads, int head_dim,
                                                 long long n) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int lane = (int)(i % head_dim);
    const long long head_ix = i / head_dim;          // token * heads + head
    const int head = (int)(head_ix % heads);
    const long long token = head_ix / heads;
    const long long row = token * (long long)heads * 2 * head_dim
                        + (long long)head * 2 * head_dim;
    q[i] = src[row + lane];
    gate[i] = src[row + head_dim + lane];
}
