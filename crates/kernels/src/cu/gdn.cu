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
// `kv_mem` reduction before the update can start.
//
// This is the reference version and it is bandwidth-bound to a standstill: S is
// 64 KiB a head at dk = dv = 128, both passes read and write it, so a token
// moves 128 KiB a head — 288 MiB for one decode token across 48 heads and 48
// linear layers, which is a few tens of tokens a second on its own. It survives
// as the fallback for shapes the register version below does not cover, and as
// the thing that version is checked against. See `gdn_delta_rule_reg128_f32`.
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

// The same recurrence with the state held in registers for the whole chunk.
//
// The traffic above is the whole cost, and it is structural: S never changes
// size, so a chunk of tokens rereads and rewrites the same 64 KiB a head every
// token for nothing. Here the block holds all of S in registers — loaded once
// on entry, stored once on exit — and what a token moves is q, k, v and the
// output: about 1.5 KiB a head instead of 128 KiB. Microseconds a launch at the
// 27B's shape — 48 value heads, 16 key heads, dk = dv = 128 — against the
// global version, on the sm_86 this was developed on and the sm_120 the engine
// targets:
//
//                     1 tok 1 seq    1 tok 32 seqs    512 tok 1 seq
//   A4000  global          73.4            1047             17814
//          register        18.6             522               588
//   RTX PRO 6000 (sm_120, 188 SMs)
//          global          75.6             180             21798
//          register         8.0             137               378
//
// The 32-sequence column on sm_120 is the one to be honest about: 180 us for
// the global version is 2245 GB/s of *apparent* traffic, which is past that
// card's DRAM. Its L2 is 128 MB and the 32-sequence state is 96 MiB, so the
// rereads within a launch are absorbed by cache and the traffic argument stops
// applying at that one shape. The register version still wins it, by 1.3x
// rather than 2x, and wins the shape where the reread count is 512 by 58x.
//
// Four decisions, each of which was measured against its alternative because
// each has a plausible-looking version that is slower:
//
//  1. **Two threads a column, not one.** The obvious blocking gives thread `j`
//     all of `S[:, j]`, 128 floats. It works, and it costs 255 registers with
//     88 bytes of spill, because on top of the state the compiler wants all
//     `dk` of the shared `k` in registers — `k` is read in both passes, so
//     hoisting it is the right call locally and ruinous globally. Declaring the
//     shared pointers `volatile` stops the hoist and removes the spill, and
//     then each pass reloads `k` from shared: no spill, 167 registers, and
//     prefill goes the wrong way (1.12 ms against 0.86). Splitting the column
//     across `R = 2` threads is the way out of that trade rather than a
//     position on it — a thread owns 64 rows, so hoisting its slice of `k`
//     costs 64 registers instead of 128, and the whole body fits in 128
//     registers with nothing spilled and no reload.
//
//     R = 4 and R = 8 keep going down that road and lose: 1.97 ms and 2.21 ms
//     at prefill. Two reasons, both about the barrier. A 512-thread block is
//     one block an SM where a 256-thread block is two, so `__syncthreads()`
//     stalls every warp on the SM instead of half of them; and the fixed
//     per-token cost — two barriers, the partner reduction, the scalar loads
//     — is amortized over a quarter as much arithmetic a thread.
//
//  2. **The partner reduction is a shuffle, not shared memory.** Lane order is
//     `R * j + part`, so a column's `R` threads are adjacent lanes of one warp
//     and `__shfl_xor_sync` finishes the `kv` and `o` reductions without a
//     barrier. Putting `part` in the high bits instead would coalesce the
//     state load marginally better and put the partners in different warps,
//     which costs a barrier a token.
//
//  3. **q and k are double-buffered.** With one buffer a token needs two
//     barriers, one after filling shared and one before the next token
//     overwrites it. With two, the next token's fill goes to the buffer nobody
//     is reading, so there is one barrier a token and the next token's global
//     loads are issued *before* it — the ~500-cycle latency overlaps this
//     token's arithmetic instead of standing in front of it. Worth 840 -> 597
//     us at prefill, where a token's work is small enough for that latency to
//     show, and about 4% at decode.
//
//  4. **Four partial sums, not one.** Each pass is a reduction over the
//     thread's rows, and one accumulator makes it a chain of 64 dependent
//     FMAs. Two accumulators and eight both measure within 4% of four, so this
//     is not a delicate number, but one is not among the options.
//
// What did *not* matter: occupancy. On sm_86 this is 128 registers at 256
// threads, two blocks an SM against the global version's twelve; on sm_120 the
// same source gets 161 registers and therefore *one* block, and that is where
// it is 2.2x faster than here. It wins at every shape measured on both,
// including the 1536-block one where there is plenty of other work to hide
// behind. This kernel was never short of parallelism, and the register budget
// it spends is not worth defending — do not trade traffic back for occupancy
// here.
//
// The one invariant a rewrite here can quietly break: every loop over the row
// index must be fully unrolled. A local array with a dynamic index does not
// live in registers, it lives in local memory — the same DRAM the version above
// streams, with worse coalescing — and it compiles and runs and produces the
// right answer. `the_register_state_does_not_spill` in tests/gated_delta.rs
// asserts CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES is zero for exactly that reason.
// `__launch_bounds__` is the other half of it: without it ptxas assumes a
// 1024-thread block and caps the body at 64 registers, which spills whatever
// the unrolling achieved.
template <int DK_C, int DV_C, int R, int ACC>
__device__ __forceinline__ void gdn_delta_rule_reg_body(
        float* __restrict__ out, float* __restrict__ state,
        const float* __restrict__ qkv, const float* __restrict__ g,
        const float* __restrict__ beta, const int* __restrict__ first_token,
        const int* __restrict__ n_tok, int heads, int key_heads, int stride,
        int q_off, int k_off, int v_off, float* smem) {
    // A thread is both a lane of q and k and a piece of a column of S, which
    // is one index only when the two head dims agree.
    static_assert(DK_C == DV_C, "the register block needs dk == dv");
    // The partner lanes of a column must sit in one warp for the shuffle.
    static_assert(R >= 1 && R <= 32 && (R & (R - 1)) == 0, "R must be 1..32, a power of two");
    constexpr int RB = DK_C / R;   // rows of S a thread owns

    const int head = blockIdx.x;
    const int seq = blockIdx.y;
    const int nt = n_tok[seq];
    // Before any access to `state`. Falling through would still leave an idle
    // slot bit-identical — the store below writes back exactly what the load
    // read, so `two_sequences_in_one_batch_keep_separate_state` could not tell
    // — and it would cost 64 KiB each way per idle slot per layer, which is the
    // traffic this whole kernel exists to not spend.
    if (nt <= 0) return;
    const int t0 = first_token[seq];
    const int lane = threadIdx.x;
    const int j = lane / R;        // the column this thread contributes to
    const int part = lane % R;     // which slice of it, in adjacent lanes
    const int i0 = part * RB;
    // repeat_interleave, not modular — see the note on the version above.
    const int khead = head / (heads / key_heads);

    // Double-buffered: buffer `n & 1` is the token being consumed.
    float* qs = smem;                  // 2 * DK_C
    float* ks = smem + 2 * DK_C;       // 2 * DK_C
    float* S = state + ((size_t)seq * heads + head) * (size_t)DK_C * DV_C;

    // Rows [i0, i0 + RB) of column j. Across a warp these reads are 32 / R
    // consecutive floats per row, so a row costs whole sectors and wastes no
    // bytes; it happens once for the chunk either way.
    float sc[RB];
#pragma unroll
    for (int r = 0; r < RB; ++r) sc[r] = S[(size_t)(i0 + r) * DV_C + j];

    // The first token, into buffer 0. `vn`, `gn`, `bn` are this thread's
    // per-token scalars, carried in registers one token ahead of their use.
    const float* row0 = qkv + (size_t)t0 * stride;
    float qn = 0.0f, kn = 0.0f;
    if (lane < DK_C) {
        qn = row0[q_off + (size_t)khead * DK_C + lane];
        kn = row0[k_off + (size_t)khead * DK_C + lane];
        qs[lane] = qn;
        ks[lane] = kn;
    }
    float vn = row0[v_off + (size_t)head * DV_C + j];
    float gn = g[(size_t)t0 * heads + head];
    float bn = beta[(size_t)t0 * heads + head];

    for (int n = 0; n < nt; ++n) {
        const int t = t0 + n;
        const int cur = n & 1;
        const float v_tj = vn;
        const float decay = __expf(gn);
        const float b = bn;

        // The next token's loads, issued before the barrier so their latency
        // overlaps this token's arithmetic rather than preceding it.
        if (n + 1 < nt) {
            const float* rn = qkv + (size_t)(t + 1) * stride;
            if (lane < DK_C) {
                qn = rn[q_off + (size_t)khead * DK_C + lane];
                kn = rn[k_off + (size_t)khead * DK_C + lane];
            }
            vn = rn[v_off + (size_t)head * DV_C + j];
            gn = g[(size_t)(t + 1) * heads + head];
            bn = beta[(size_t)(t + 1) * heads + head];
        }
        // Buffer `cur` is filled: by the prologue on the first pass, by the
        // tail of the previous iteration after that.
        __syncthreads();

        const float* qc = qs + cur * DK_C;
        const float* kc = ks + cur * DK_C;

        // S *= exp(g);  kv_mem = kᵀ S, this thread's rows then the partner's.
        float kv[ACC];
#pragma unroll
        for (int a = 0; a < ACC; ++a) kv[a] = 0.0f;
#pragma unroll
        for (int r = 0; r < RB; ++r) {
            sc[r] *= decay;
            kv[r % ACC] += sc[r] * kc[i0 + r];
        }
#pragma unroll
        for (int a = 1; a < ACC; ++a) kv[0] += kv[a];
        float kvt = kv[0];
#pragma unroll
        for (int m = 1; m < R; m <<= 1) kvt += __shfl_xor_sync(0xffffffffu, kvt, m);

        const float delta = (v_tj - kvt) * b;

        // S += k ⊗ delta;  o = qᵀ S — with the *updated* S. Reading out before
        // the update also runs, and is a different model.
        float o[ACC];
#pragma unroll
        for (int a = 0; a < ACC; ++a) o[a] = 0.0f;
#pragma unroll
        for (int r = 0; r < RB; ++r) {
            sc[r] += kc[i0 + r] * delta;
            o[r % ACC] += sc[r] * qc[i0 + r];
        }
#pragma unroll
        for (int a = 1; a < ACC; ++a) o[0] += o[a];
        float ot = o[0];
#pragma unroll
        for (int m = 1; m < R; m <<= 1) ot += __shfl_xor_sync(0xffffffffu, ot, m);
        if (part == 0) out[((size_t)t * heads + head) * DV_C + j] = ot;

        // Into the buffer nobody read this iteration, so no barrier separates
        // the reads above from this write; the one at the top of the next
        // iteration is what publishes it.
        if (n + 1 < nt && lane < DK_C) {
            qs[(cur ^ 1) * DK_C + lane] = qn;
            ks[(cur ^ 1) * DK_C + lane] = kn;
        }
    }

#pragma unroll
    for (int r = 0; r < RB; ++r) S[(size_t)(i0 + r) * DV_C + j] = sc[r];
}

// dk = dv = 128, which is this checkpoint's `linear_key_head_dim` and
// `linear_value_head_dim`. Launched with `2 * dv` threads and `4 * dk` floats
// of dynamic shared; `Kernels::gdn_delta_rule` sizes both and sends every
// other shape to the global version.
extern "C" __global__ __launch_bounds__(256) void gdn_delta_rule_reg128_f32(
        float* __restrict__ out, float* __restrict__ state,
        const float* __restrict__ qkv, const float* __restrict__ g,
        const float* __restrict__ beta, const int* __restrict__ first_token,
        const int* __restrict__ n_tok, int heads, int key_heads, int dk, int dv,
        int stride, int q_off, int k_off, int v_off) {
    // `dk` and `dv` are template parameters here; they stay in the signature so
    // that the launcher pushes one argument list for any of the three kernels.
    (void)dk;
    (void)dv;
    extern __shared__ float smem[];
    gdn_delta_rule_reg_body<128, 128, 2, 4>(out, state, qkv, g, beta,
                                            first_token, n_tok, heads, key_heads,
                                            stride, q_off, k_off, v_off, smem);
}

// The middle ground: S in dynamic shared memory rather than registers.
//
// Same traffic saving — loaded once, stored once — but the per-token passes
// read shared instead of registers, and it works for any `dk`/`dv` whose
// `dk * dv * 4` bytes fit the opt-in dynamic limit. It costs one bank-conflict
// -free shared read per element per pass against a register read, and it buys
// back the registers.
//
// It loses to the register version at every shape measured — 58.7 against 18.0
// us for one decode token, 1726 against 521 for 32, 3.25 ms against 0.60 for a
// 512-token prefill — and it loses to the *global* version at 32 sequences,
// which is the number that says what is wrong. It is not shared-memory
// bandwidth: 65 KiB a block is one resident block an SM against the global
// version's twelve, so every `__syncthreads()` stalls the entire SM and there
// is no second block to run in the gap. A shared-memory state buys the traffic
// back and spends the occupancy that was hiding the barrier.
//
// That verdict is the one number here that is architecture-specific rather
// than structural: 65 KiB against sm_86's 100 KiB per block is one resident
// block, and against a part with a larger unified store it would be more. If
// this is ever revisited on such a part, the thing to re-measure is the 32-
// sequence column, where the barrier stall is what it loses on.
//
// Kept, and kept tested, for two reasons: it is the only version that would
// cover a `dk` past what 255 registers hold, and it is the third independent
// implementation of a recurrence where every layout choice has a plausible
// wrong reading.
// No `__launch_bounds__`: unlike the register version this one is launched with
// `dv` threads for whatever `dv` is, up to the 1024 the block limit allows, and
// it has no register pressure to declare.
extern "C" __global__ void gdn_delta_rule_smem_f32(
        float* __restrict__ out, float* __restrict__ state,
        const float* __restrict__ qkv, const float* __restrict__ g,
        const float* __restrict__ beta, const int* __restrict__ first_token,
        const int* __restrict__ n_tok, int heads, int key_heads, int dk, int dv,
        int stride, int q_off, int k_off, int v_off) {
    extern __shared__ float smem[];

    const int head = blockIdx.x;
    const int seq = blockIdx.y;
    const int nt = n_tok[seq];
    if (nt <= 0) return;   // idle slot: no read, no write
    const int t0 = first_token[seq];
    const int j = threadIdx.x;
    const int khead = head / (heads / key_heads);

    float* qs = smem;               // dk
    float* ks = smem + dk;          // dk
    float* Ss = smem + 2 * dk;      // dk * dv

    float* S = state + ((size_t)seq * heads + head) * (size_t)dk * dv;
    const int n_state = dk * dv;
    for (int idx = threadIdx.x; idx < n_state; idx += blockDim.x) Ss[idx] = S[idx];

    for (int n = 0; n < nt; ++n) {
        const int t = t0 + n;
        const float* row = qkv + (size_t)t * stride;
        const float* qsrc = row + q_off + (size_t)khead * dk;
        const float* ksrc = row + k_off + (size_t)khead * dk;
        for (int i = threadIdx.x; i < dk; i += blockDim.x) {
            qs[i] = qsrc[i];
            ks[i] = ksrc[i];
        }
        // Also covers the state load above on the first pass.
        __syncthreads();

        const float decay = __expf(g[(size_t)t * heads + head]);
        const float b = beta[(size_t)t * heads + head];

        if (j < dv) {
            const float v_tj = row[v_off + (size_t)head * dv + j];
            // Thread j walks column j with stride dv, so a warp reads dv
            // consecutive floats: one bank each, no conflict.
            float kv = 0.0f;
            for (int i = 0; i < dk; ++i) {
                const float s = Ss[i * dv + j] * decay;
                Ss[i * dv + j] = s;
                kv += s * ks[i];
            }
            const float delta = (v_tj - kv) * b;
            float o = 0.0f;
            for (int i = 0; i < dk; ++i) {
                const float s = Ss[i * dv + j] + ks[i] * delta;
                Ss[i * dv + j] = s;
                o += s * qs[i];
            }
            out[((size_t)t * heads + head) * dv + j] = o;
        }
        __syncthreads();
    }

    for (int idx = threadIdx.x; idx < n_state; idx += blockDim.x) S[idx] = Ss[idx];
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
