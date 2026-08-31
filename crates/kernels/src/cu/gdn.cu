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
// `infero_model::qwen35`, which is in turn checked against a capture of the
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
// `infero_model::qwen35`, which is in turn checked against a capture of the
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

    // wc doesn't vary with n, so it's loaded into registers once rather than
    // re-read from global memory every one of the (up to thousands of)
    // iterations below -- that redundant per-step load was also badly
    // strided across threads (consecutive c's are k floats apart), which is
    // what made it show up as an uncoalesced-access hotspot in profiling.
    const float* wc_g = w + (size_t)c * k;
    float wc[8];
    for (int j = 0; j <= hist; ++j) wc[j] = wc_g[j];
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

// `gdn_conv_f32` grids `(channels/BLOCK, n_seqs)` -- for decode (n_seqs many,
// nt=1 each) that's already plenty of blocks, but a real prefill has n_seqs=1
// and channels alone (10240 at this checkpoint / 128 = 80 blocks) against
// this GPU's 188 SMs, leaving most of it idle for the whole, sequential-over-
// nt (up to 1024) token loop -- confirmed by `ncu`: 8.28% achieved occupancy
// against 100% theoretical, "Est. Local Speedup: 91.72%". The token loop
// looks sequential but isn't really: every read is from `x` (the raw input,
// fully resident before this kernel starts) and the carried window is only
// `k - 1` values, not a full recurrent state -- unlike GDN's chunk-parallel
// delta rule, there is no hidden state threading chunk to chunk except the
// `state` buffer at the very start of a call. So a chunk past the first can
// bootstrap its own window by re-reading the `hist` raw inputs immediately
// before it, with no cross-block dependency at all; only the chunk covering
// the sequence's last token needs to write the carried-forward `state`.
extern "C" __global__ void gdn_conv_chunked_f32(float* __restrict__ out,
                                                const float* __restrict__ x,
                                                float* __restrict__ state,
                                                const float* __restrict__ w,
                                                const int* __restrict__ first_token,
                                                const int* __restrict__ n_tok,
                                                int channels, int k, int chunk_len) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    const int seq = blockIdx.z;
    const int nt = n_tok[seq];
    if (nt <= 0) return;
    const int t0 = first_token[seq];
    const int hist = k - 1;

    const int start = blockIdx.y * chunk_len;
    if (start >= nt) return;
    const int len = min(chunk_len, nt - start);

    float win[8];
    if (start == 0) {
        const float* st = state + ((size_t)seq * channels + c) * hist;
        for (int j = 0; j < hist; ++j) win[j] = st[j];
    } else {
        for (int j = 0; j < hist; ++j) {
            const int pos = start - hist + j;
            win[j] = (pos >= 0) ? x[(size_t)(t0 + pos) * channels + c] : 0.0f;
        }
    }

    const float* wc_g = w + (size_t)c * k;
    float wc[8];
    for (int j = 0; j <= hist; ++j) wc[j] = wc_g[j];

    for (int n = 0; n < len; ++n) {
        const float cur = x[(size_t)(t0 + start + n) * channels + c];
        float acc = wc[hist] * cur;
        for (int j = 0; j < hist; ++j) acc += wc[j] * win[j];
        out[(size_t)(t0 + start + n) * channels + c] = acc / (1.0f + __expf(-acc));
        for (int j = 0; j + 1 < hist; ++j) win[j] = win[j + 1];
        if (hist > 0) win[hist - 1] = cur;
    }
    if (start + len == nt) {
        float* st = state + ((size_t)seq * channels + c) * hist;
        for (int j = 0; j < hist; ++j) st[j] = win[j];
    }
}

// beta = sigmoid(b);  g = -exp(A_log) * softplus(a + dt_bias)
//
// `g` is non-positive by construction, so `exp(g)` in the recurrence is a
// decay. Losing the sign makes the state grow without bound, which shows up as
// NaN several layers later rather than here.
//
// One thread a (token, head) pair.
// `stride` is the row pitch of `a` and `b`, which is not always `heads`.
//
// `in_proj_a` and `in_proj_b` are 48 rows over a 5120 contraction — bytes that
// want 0.34 us where a launch costs 14.2, twice a layer and 96 times a decode
// step. Stacked into one projection they come out interleaved, `a` then `b` per
// token, so the caller passes one buffer twice with `b` offset by `heads` and a
// stride of `2 * heads`. Unstacked it passes two buffers and `stride = heads`.
extern "C" __global__ void gdn_gate_decay_f32(float* __restrict__ beta_out,
                                              float* __restrict__ g_out,
                                              const float* __restrict__ a,
                                              const float* __restrict__ b,
                                              const float* __restrict__ a_log,
                                              const float* __restrict__ dt_bias,
                                              int n_tokens, int heads,
                                              int stride) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_tokens * heads) return;
    const int h = idx % heads;
    const int src = (idx / heads) * stride + h;
    beta_out[idx] = 1.0f / (1.0f + __expf(-b[src]));
    const float z = a[src] + dt_bias[h];
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
// value head `h` reads key head `h / (heads / key_heads)` when the checkpoint
// stores V heads *grouped* by key head, which a Hugging Face one does. A GGUF
// does not: llama.cpp reorders them to tiled order so that ggml's broadcast can
// use `repeat` instead of an interleaved repeat, and then the mapping is
// `h % key_heads`. `v_tiled` selects between the two, because both run to
// completion and give different models. Expanding modularly on a grouped
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
                                              int v_off, int v_tiled) {
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
    const int khead = v_tiled ? (head % key_heads)
                              : (head / (heads / key_heads));

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
// `G` splits `DV_C` columns across `G` blocks a head, so a launch can use
// more of the device than `heads * n_seqs` blocks when that alone leaves
// most SMs idle (48 heads on a 188-SM part is 25.5% launch-wide, independent
// of anything per-block occupancy can fix -- see the note above
// `gdn_delta_rule_reg128_f32`). `G == 1` reproduces the original kernel
// exactly, column for column; this is a strict generalization, not a second
// algorithm, and a compile-time one (`GCOLS`, `GTHREADS`, `Q_LOADS` below
// are all `constexpr`) for the same register-spill reason every other loop
// bound in this kernel is a template parameter, not a runtime one -- see the
// note above this function's non-templated call sites. Splitting costs a
// redundant per-group reload of this token's q/k (v, g and beta already
// varied by column and were never shared), traded for `GCOLS` threads a
// block instead of `DV_C` -- fewer registers a block at the same registers a
// thread, so more blocks fit a launch, both by using idle SMs and by
// letting more than one block's worth of this head's own columns share one
// SM and hide each other's stalls.
template <int DK_C, int DV_C, int R, int ACC, int G>
__device__ __forceinline__ void gdn_delta_rule_reg_body(
        float* __restrict__ out, float* __restrict__ state,
        const float* __restrict__ qkv, const float* __restrict__ g,
        const float* __restrict__ beta, const int* __restrict__ first_token,
        const int* __restrict__ n_tok, int heads, int key_heads, int stride,
        int q_off, int k_off, int v_off, int v_tiled, float* smem) {
    // A thread is both a lane of q and k and a piece of a column of S, which
    // is one index only when the two head dims agree.
    static_assert(DK_C == DV_C, "the register block needs dk == dv");
    // The partner lanes of a column must sit in one warp for the shuffle.
    static_assert(R >= 1 && R <= 32 && (R & (R - 1)) == 0, "R must be 1..32, a power of two");
    constexpr int RB = DK_C / R;   // rows of S a thread owns
    // `RB`'s region in the q/k shared buffer starts at bank `(part*RB) % 32`.
    // When RB is itself a multiple of 32 (128/2 here), every part restarts at
    // bank 0, so two parts reading the same unrolled `r` this iteration hit
    // one bank from two different addresses -- a real, measured conflict
    // (ncu: ~27% est. speedup on this kernel's shared loads, 4-way, half the
    // wavefronts). A gap between each part's region, sized so the region
    // stride is no longer a multiple of 32, breaks the alignment.
    constexpr int PAD = (RB % 32 == 0) ? 4 : 0;
    constexpr int ROW = RB + PAD;     // one part's region, with its gap
    constexpr int BUF = R * ROW;      // one cur-buffer of q or k, padded

    // This block's share of `DV_C` columns, and how many threads that takes:
    // `G == 1` gives `GCOLS == DV_C`, `GTHREADS == 2 * DV_C` -- the original
    // kernel's shape exactly. `Q_LOADS` covers the q/k prologue and lookahead
    // below needing more than one pass to cover `DK_C` once `GTHREADS < DK_C`
    // (true from `G == 4` on here); it is 1, unrolled to nothing extra, right
    // up to there.
    static_assert(DV_C % G == 0, "G must divide the column count evenly");
    constexpr int GCOLS = DV_C / G;
    constexpr int GTHREADS = R * GCOLS;
    constexpr int Q_LOADS = (DK_C + GTHREADS - 1) / GTHREADS;

    const int head = blockIdx.x;
    const int seq = blockIdx.y;
    const int col_group = blockIdx.z;
    const int nt = n_tok[seq];
    // Before any access to `state`. Falling through would still leave an idle
    // slot bit-identical — the store below writes back exactly what the load
    // read, so `two_sequences_in_one_batch_keep_separate_state` could not tell
    // — and it would cost 64 KiB each way per idle slot per layer, which is the
    // traffic this whole kernel exists to not spend.
    if (nt <= 0) return;
    const int t0 = first_token[seq];
    const int lane = threadIdx.x;
    const int j = lane / R;        // the column this thread contributes to, within its group
    const int jg = col_group * GCOLS + j;  // ... and across the whole head
    const int part = lane % R;     // which slice of it, in adjacent lanes
    const int i0 = part * RB;
    const int i0s = part * ROW;    // same rows, in the padded shared layout
    // repeat_interleave, not modular — see the note on the version above.
    const int khead = v_tiled ? (head % key_heads)
                              : (head / (heads / key_heads));

    // Double-buffered: buffer `n & 1` is the token being consumed.
    float* qs = smem;                  // 2 * BUF
    float* ks = smem + 2 * BUF;        // 2 * BUF
    float* S = state + ((size_t)seq * heads + head) * (size_t)DK_C * DV_C;

    // Rows [i0, i0 + RB) of column jg. Across a warp these reads are 32 / R
    // consecutive floats per row, so a row costs whole sectors and wastes no
    // bytes; it happens once for the chunk either way.
    float sc[RB];
#pragma unroll
    for (int r = 0; r < RB; ++r) sc[r] = S[(size_t)(i0 + r) * DV_C + jg];

    // The first token, into buffer 0. `vn`, `gn`, `bn` are this thread's
    // per-token scalars, carried in registers one token ahead of their use;
    // `qn`/`kn` are `Q_LOADS`-wide because a thread covers more than one
    // `dk` index once `GTHREADS < DK_C` (see `Q_LOADS`'s note above).
    const float* row0 = qkv + (size_t)t0 * stride;
    float qn[Q_LOADS], kn[Q_LOADS];
#pragma unroll
    for (int it = 0; it < Q_LOADS; ++it) {
        const int d = lane + it * GTHREADS;
        if (d < DK_C) {
            qn[it] = row0[q_off + (size_t)khead * DK_C + d];
            kn[it] = row0[k_off + (size_t)khead * DK_C + d];
            const int off = d + PAD * (d / RB);
            qs[off] = qn[it];
            ks[off] = kn[it];
        }
    }
    float vn = row0[v_off + (size_t)head * DV_C + jg];
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
#pragma unroll
            for (int it = 0; it < Q_LOADS; ++it) {
                const int d = lane + it * GTHREADS;
                if (d < DK_C) {
                    qn[it] = rn[q_off + (size_t)khead * DK_C + d];
                    kn[it] = rn[k_off + (size_t)khead * DK_C + d];
                }
            }
            vn = rn[v_off + (size_t)head * DV_C + jg];
            gn = g[(size_t)(t + 1) * heads + head];
            bn = beta[(size_t)(t + 1) * heads + head];
        }
        // Buffer `cur` is filled: by the prologue on the first pass, by the
        // tail of the previous iteration after that.
        __syncthreads();

        const float* qc = qs + cur * BUF;
        const float* kc = ks + cur * BUF;

        // S *= exp(g);  kv_mem = kᵀ S, this thread's rows then the partner's.
        float kv[ACC];
#pragma unroll
        for (int a = 0; a < ACC; ++a) kv[a] = 0.0f;
#pragma unroll
        for (int r = 0; r < RB; ++r) {
            sc[r] *= decay;
            kv[r % ACC] += sc[r] * kc[i0s + r];
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
            sc[r] += kc[i0s + r] * delta;
            o[r % ACC] += sc[r] * qc[i0s + r];
        }
#pragma unroll
        for (int a = 1; a < ACC; ++a) o[0] += o[a];
        float ot = o[0];
#pragma unroll
        for (int m = 1; m < R; m <<= 1) ot += __shfl_xor_sync(0xffffffffu, ot, m);
        if (part == 0) out[((size_t)t * heads + head) * DV_C + jg] = ot;

        // Into the buffer nobody read this iteration, so no barrier separates
        // the reads above from this write; the one at the top of the next
        // iteration is what publishes it.
        if (n + 1 < nt) {
#pragma unroll
            for (int it = 0; it < Q_LOADS; ++it) {
                const int d = lane + it * GTHREADS;
                if (d < DK_C) {
                    const int off = (cur ^ 1) * BUF + d + PAD * (d / RB);
                    qs[off] = qn[it];
                    ks[off] = kn[it];
                }
            }
        }
    }

#pragma unroll
    for (int r = 0; r < RB; ++r) S[(size_t)(i0 + r) * DV_C + jg] = sc[r];
}

// dk = dv = 128, which is this checkpoint's `linear_key_head_dim` and
// `linear_value_head_dim`. Launched with `2 * dv` threads and `4 * dk + 32`
// floats of dynamic shared -- the `+ 32` is the bank-conflict padding above
// (`4 * R * PAD` = `4 * 2 * 4`, R and PAD fixed for this dk); `Kernels::
// gdn_delta_rule` sizes both and sends every other shape to the global
// version.
extern "C" __global__ __launch_bounds__(256) void gdn_delta_rule_reg128_f32(
        float* __restrict__ out, float* __restrict__ state,
        const float* __restrict__ qkv, const float* __restrict__ g,
        const float* __restrict__ beta, const int* __restrict__ first_token,
        const int* __restrict__ n_tok, int heads, int key_heads, int dk, int dv,
        int stride, int q_off, int k_off, int v_off, int v_tiled) {
    // `dk` and `dv` are template parameters here; they stay in the signature so
    // that the launcher pushes one argument list for any of the three kernels.
    (void)dk;
    (void)dv;
    extern __shared__ float smem[];
    gdn_delta_rule_reg_body<128, 128, 2, 4, 1>(out, state, qkv, g, beta,
                                            first_token, n_tok, heads, key_heads,
                                            stride, q_off, k_off, v_off, v_tiled, smem);
}

// Same recurrence, same per-thread register shape (`RB`, `PAD`, `ROW`, `BUF`
// are all unchanged -- see the note above `gdn_delta_rule_reg_body`), but
// each of `heads * n_seqs * 4` blocks owns a quarter of `dv`'s 128 columns
// instead of one block owning all of them. `heads = 48` on this checkpoint,
// so a solo long prefill (the common case this exists for: `n_seqs == 1`)
// launches 48 blocks on a 188-SM part either way -- 25.5% of the device,
// occupancy notwithstanding, before this. `Kernels::gdn_delta_rule` picks
// between the two by measured shape, not by feel; see its own call site.
// Launched with `64` threads (`2 * dv / 4`) and the same dynamic shared as
// the undivided kernel, since `BUF` does not depend on `G`.
extern "C" __global__ __launch_bounds__(256) void gdn_delta_rule_reg128_split4_f32(
        float* __restrict__ out, float* __restrict__ state,
        const float* __restrict__ qkv, const float* __restrict__ g,
        const float* __restrict__ beta, const int* __restrict__ first_token,
        const int* __restrict__ n_tok, int heads, int key_heads, int dk, int dv,
        int stride, int q_off, int k_off, int v_off, int v_tiled) {
    (void)dk;
    (void)dv;
    extern __shared__ float smem[];
    gdn_delta_rule_reg_body<128, 128, 2, 4, 4>(out, state, qkv, g, beta,
                                            first_token, n_tok, heads, key_heads,
                                            stride, q_off, k_off, v_off, v_tiled, smem);
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
        int stride, int q_off, int k_off, int v_off, int v_tiled) {
    extern __shared__ float smem[];

    const int head = blockIdx.x;
    const int seq = blockIdx.y;
    const int nt = n_tok[seq];
    if (nt <= 0) return;   // idle slot: no read, no write
    const int t0 = first_token[seq];
    const int j = threadIdx.x;
    const int khead = v_tiled ? (head % key_heads)
                              : (head / (heads / key_heads));

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

// A verification pass's rollback bookkeeping (`crates::spec::GdnRollback`)
// used to be two and four independent `memcpy_dtod` calls a layer — one to
// stage the conv window and recurrent state before the pass runs, another
// four to tap its inputs for the journal after. Merging each group into one
// launch turned out not to be where the round's missing milliseconds were —
// that was `stage` copying all `max_seqs` sequences' state instead of just
// the armed one, a 31x-narrower fix that lives on `GdnRollback::stage`
// itself. This is kept anyway: fewer nodes in the captured graph, and one
// thread a destination element across every segment in the group is no
// slower than the memcpy it replaces.

// Two segments: the conv window, then the recurrent state.
extern "C" __global__ void gdn_rollback_stage2_f32(
        float* __restrict__ dst0, const float* __restrict__ src0, long long n0,
        float* __restrict__ dst1, const float* __restrict__ src1, long long n1) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n0) {
        dst0[i] = src0[i];
    } else if (i < n0 + n1) {
        dst1[i - n0] = src1[i - n0];
    }
}

// Four segments: the journal's pre-conv, post-conv, gate and beta taps.
extern "C" __global__ void gdn_rollback_record4_f32(
        float* __restrict__ dst0, const float* __restrict__ src0, long long n0,
        float* __restrict__ dst1, const float* __restrict__ src1, long long n1,
        float* __restrict__ dst2, const float* __restrict__ src2, long long n2,
        float* __restrict__ dst3, const float* __restrict__ src3, long long n3) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
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

// Chunk-parallel gated delta rule. Same recurrence as `gdn_delta_rule_reg128_f32`
// (`S *= exp(g_t)` before the rank-1 update, per the comment on that function),
// reassociated so a block of `GDN_CHUNK` tokens is processed with block-wide
// parallel matrix ops instead of one token at a time -- trading
// `gdn_delta_rule_reg128_f32`'s O(seq_len) *sequential* steps for
// O(seq_len / GDN_CHUNK) sequential chunk steps.
//
// **Measured, not just implemented, and the measurement is still a loss --
// though a much smaller one than the first attempt, after real bugs got
// found and fixed rather than accepted.** Correct (matches
// `gdn_delta_rule_reg128_f32` to reference precision, see
// `the_three_delta_rule_kernels_agree_with_each_other_and_the_reference` in
// `tests/gated_delta.rs`), but still slower at the checkpoint's real prefill
// shape (30552 tokens, 1 sequence, `examples/gdn_delta_bench.rs`):
//
//   212.5 ms  first working version (correct, unexamined for performance)
//   121.1 ms  -- staged `q` in shared memory: the intra-chunk score loop
//               read it from global once per (i, kk) pair, re-fetching the
//               same row up to `GDN_CHUNK` times over (1.76x)
//   112.4 ms  -- padded every 2-D shared buffer's row stride past a multiple
//               of the 32-way bank count (`GDN_ROW_PAD`/`GDN_A_STRIDE`),
//               removing worst-case bank conflicts on every cross-row access
//               (1.08x)
//   111.6 ms  -- `__syncwarp()` not `__syncthreads()` in the forward-sub
//               loop, dividing/modding by the compile-time `GDN_CHUNK` not
//               the runtime `C`, merging the load/cumsum barrier, hoisting
//               `beta_k * exp(gc_k)` out of a loop that recomputed it
//               `GDN_DK` times over -- four plausible-sounding fixes, each
//               measured at approximately zero (this is the sum of all
//               four, not four separate meaningful wins)
//
// against `gdn_delta_rule_reg128_f32`'s steady 27.8 ms -- a real, if
// incomplete, 1.9x recovery from the fixes that worked, not the whole gap.
// `ncu` (not a guess) on both kernels found neither the barrier count nor
// occupancy explains the remainder: both launch the same 48-block grid
// (one per head; this checkpoint's `n_kv_heads`-shaped ceiling, not
// something either kernel chose) against this GPU's 188 SMs, and both cap
// out at exactly 1 block/SM, 8 warps/SM, 16.67% theoretical occupancy --
// `reg128` on its 161 registers/thread, this kernel on its ~89 KiB of
// dynamic shared memory. Same grid-underutilization ceiling, same
// occupancy, and `reg128` is still ~4x faster: the gap is genuine per-SM
// throughput, not a synchronization or launch-configuration artifact --
// `reg128` earns it the way its own comment documents (double-buffering,
// four independent partial sums instead of one dependent chain, R=2's
// register/occupancy trade, each individually measured), and this kernel
// has none of that tuning yet.
//
// The one lever `reg128`'s architecture cannot use, that this one's
// algorithm *could*: `reg128` is a pure sequential recurrence, permanently
// bound to that same 48-block grid. This kernel's `W`/`U`/`A_intra`
// decomposition exists specifically to separate a chunk's *state transition*
// (genuinely sequential, but only `GDN_CHUNK`-cheap now) from its *output*
// (an embarrassingly parallel function of that chunk's own incoming state,
// once known) -- splitting those into two kernel launches (a cheap
// sequential pass computing and storing every chunk's incoming state, then
// an output pass gridded over `chunks * heads` instead of just `heads`)
// could use far more than 48 blocks and escape this ceiling entirely,
// something `reg128` structurally cannot do. Untried: it needs storing every
// chunk's state (`nt / GDN_CHUNK` copies of a `dk * dv` matrix a head, real
// VRAM, not free) and is a materially larger rewrite (two kernels, new
// buffer lifetime) than a tuning pass, not something to start under a "just
// push harder" mandate without saying so explicitly first.
//
// Not reachable through `DeltaVariant::Auto` -- correct is necessary, not
// sufficient, and this doesn't clear the second bar yet. Kept (like
// `gdn_delta_rule_smem_f32` above it) as a working, tested, independent
// implementation checked against a real external reference, not as a
// candidate for the fast path.
//
// The algorithm (chunk-local cumsum, the K K^T system matrix, its unit-lower-
// triangular inverse via forward substitution, the WY-style `W`/`U`
// reconstruction, and how the carried state combines with intra-chunk causal
// terms) is checked line-for-line against the vendored `flash-linear-attention`
// reference inside vLLM (`chunk.py`, `cumsum.py`, `chunk_scaled_dot_kkt.py`,
// `solve_tril.py`, `wy_fast.py`, `chunk_delta_h.py`, `chunk_o.py`) rather than
// re-derived from the paper from memory -- see the perf-gap memory doc for the
// exact file/line citations this was built from. That reference's own
// `FLA_CHUNK_SIZE` default is 64, but everything here is `float`, not `half`:
// a first attempt staged `k`/`v`/`W` at half precision (same tradeoff this
// engine's attention kernels make) and measured a real, if small, accuracy
// loss against the reference -- not a reassociation artifact, an f16-storage
// one, because this algorithm's forward-substitution inverse is *recursive*
// (row `i` depends on every earlier row), so any rounding baked into `A` a
// few rows in gets multiplied back in at every later row rather than staying
// a fixed, bounded error the way it would in a single matmul. `GDN_CHUNK =
// 32` (half the reference's) buys back the shared-memory room `float`
// everywhere costs.
//
// State layout matches `gdn_delta_rule_reg128_f32` exactly (register-resident,
// `R = 2` threads a column, thread `(j, part)` owns `S[j][part*64 .. part*64+63]`)
// so the two kernels read/write the same state-buffer format; only what
// happens to a chunk's worth of tokens between load and store differs. Unlike
// that kernel this one is only instantiated for `dk = dv = 128`, hardcoded
// (not templated) -- the shared-memory layout below is sized for exactly that.
#define GDN_CHUNK 32
#define GDN_DK 128
#define GDN_DV 128
// Padded row strides for every 2-D shared buffer below: GDN_DK/GDN_DV and
// GDN_CHUNK are both exact multiples of the 32-way shared-memory bank count,
// so an unpadded stride puts every row at the same bank -- worst-case,
// 32-way conflicts on every cross-row access these loops do (A/A_intra's
// k_i.k_j and q_i.k_j sweeps, W/U's and the state update's accumulation
// over rows). This is the same failure mode ATTN_MMA_KPAD/ATTN_MMA_VPAD
// exist to avoid in ops.cu's attention kernels, applied here after a
// benchmark (not a hunch) showed it mattered: reintroducing `q` staging
// alone cut this kernel's measured time on the real 30552-token shape by
// 1.76x, and this bank-conflict fix on top of that is worth re-measuring
// against, not assuming.
#define GDN_ROW_PAD (GDN_DK + 4)
#define GDN_A_STRIDE (GDN_CHUNK + 1)

extern "C" __global__ __launch_bounds__(256) void gdn_chunk_delta_rule_f32(
        float* __restrict__ out, float* __restrict__ state,
        const float* __restrict__ qkv, const float* __restrict__ g,
        const float* __restrict__ beta, const int* __restrict__ first_token,
        const int* __restrict__ n_tok, int heads, int key_heads, int dk, int dv,
        int stride, int q_off, int k_off, int v_off, int v_tiled) {
    // `dk`/`dv` stay in the signature so the launcher pushes one argument
    // list for every delta-rule kernel; this one is only ever built for 128.
    (void)dk;
    (void)dv;
    const int head = blockIdx.x;
    const int seq = blockIdx.y;
    const int nt = n_tok[seq];
    if (nt <= 0) return;
    const int t0 = first_token[seq];
    const int lane = threadIdx.x;  // 0..255
    const int j = lane / 2;        // this thread's S column (value dim)
    const int part = lane % 2;     // which half of the key dim it owns
    const int i0 = part * 64;
    const int khead = v_tiled ? (head % key_heads)
                              : (head / (heads / key_heads));

    // `q` IS staged, unlike an earlier version of this kernel's reasoning --
    // it's read once per (i, kk) pair in the intra-chunk score loop below,
    // and a fixed `i` is paired with up to `GDN_CHUNK` different `kk`, so
    // reading it from global there re-fetches the same row up to `GDN_CHUNK`
    // times over (measured as most of a real 7.6x slowdown against
    // `gdn_delta_rule_reg128_f32`, together with the barrier count noted on
    // the kernel comment -- caching it here was the fix, not accepting the
    // loss). `GDN_CHUNK` dropped from the reference's 64 to 32 for `sk`/`sv`/
    // `sW`/`sD`'s `float` precision (see the top comment) leaves the room.
    extern __shared__ char gdn_chunk_smem[];
    float* sk = (float*)gdn_chunk_smem;                // [GDN_CHUNK][GDN_ROW_PAD]
    float* sq = sk + GDN_CHUNK * GDN_ROW_PAD;          // [GDN_CHUNK][GDN_ROW_PAD]
    float* sv = sq + GDN_CHUNK * GDN_ROW_PAD;          // [GDN_CHUNK][GDN_ROW_PAD]
    float* sgc = sv + GDN_CHUNK * GDN_ROW_PAD;         // [GDN_CHUNK], cumsum g
    float* sbeta = sgc + GDN_CHUNK;                    // [GDN_CHUNK]
    // beta_k * exp(gc_k), precomputed once below rather than inside the W/U
    // loop's inner sum, which would otherwise recompute it GDN_DK (128)
    // times over for every k -- once per output channel `d`, none of which
    // it depends on. `__expf` isn't free, and this was a bigger cost than
    // the barrier or bank-conflict fixes above combined.
    float* sbg = sbeta + GDN_CHUNK;                    // [GDN_CHUNK]
    // A, then (I+A)^-1 in place, then reused for the intra-chunk Q.K scores.
    float* sA = sbg + GDN_CHUNK;                       // [GDN_CHUNK][GDN_A_STRIDE]
    // W, then reused as the intra-chunk score matrix once W is dead.
    float* sW = sA + GDN_CHUNK * GDN_A_STRIDE;           // [GDN_CHUNK][GDN_ROW_PAD]
    // U, then the residual (`delta`) in place once U is dead.
    float* sD = sW + GDN_CHUNK * GDN_ROW_PAD;            // [GDN_CHUNK][GDN_ROW_PAD]

    float* S = state + ((size_t)seq * heads + head) * (size_t)GDN_DK * GDN_DV;
    float sc[64];
#pragma unroll
    for (int r = 0; r < 64; ++r) sc[r] = S[(size_t)(i0 + r) * GDN_DV + j];

    for (int c0 = 0; c0 < nt; c0 += GDN_CHUNK) {
        const int C = min(GDN_CHUNK, nt - c0);

        for (int idx = lane; idx < C * GDN_DK; idx += blockDim.x) {
            const int r = idx / GDN_DK, d = idx % GDN_DK;
            const float* row = qkv + (size_t)(t0 + c0 + r) * stride;
            sk[r * GDN_ROW_PAD + d] = row[k_off + (size_t)khead * GDN_DK + d];
            sq[r * GDN_ROW_PAD + d] = row[q_off + (size_t)khead * GDN_DK + d];
        }
        for (int idx = lane; idx < C * GDN_DV; idx += blockDim.x) {
            const int r = idx / GDN_DV, d = idx % GDN_DV;
            const float* row = qkv + (size_t)(t0 + c0 + r) * stride;
            sv[r * GDN_ROW_PAD + d] = row[v_off + (size_t)head * GDN_DV + d];
        }
        if (lane < C) {
            sbeta[lane] = beta[(size_t)(t0 + c0 + lane) * heads + head];
        }
        // Chunk-local inclusive cumsum of g. Serial, but at most GDN_CHUNK
        // adds -- and independent of the K/V/Q/beta loading above (reads
        // `g` fresh from global, writes only `sgc`, which nothing above
        // touches), so it runs concurrently with it rather than after a
        // barrier of its own; one `__syncthreads()` below covers both.
        if (lane == 0) {
            float acc = 0.0f;
            for (int r = 0; r < C; ++r) {
                acc += g[(size_t)(t0 + c0 + r) * heads + head];
                sgc[r] = acc;
            }
        }
        __syncthreads();

        if (lane < C) {
            sbg[lane] = sbeta[lane] * __expf(sgc[lane]);
        }

        // A[i][k] = beta_i * exp(gc_i - gc_k) * (k_i . k_k), strictly i > k;
        // the system matrix `chunk_scaled_dot_kkt_fwd` builds (kkt.py:97-112).
        //
        // Divides and mods by `GDN_CHUNK`, not the runtime `C`: `C == GDN_CHUNK`
        // for every chunk but a possibly-ragged last one, and a compile-time
        // power-of-two divisor is a shift, where the same op against a
        // runtime `C` is a genuine (slow) integer division -- paid on every
        // one of `GDN_CHUNK * GDN_CHUNK` iterations, every chunk. The trip
        // count grows to the fixed `GDN_CHUNK * GDN_CHUNK` too (previously
        // `C * C`, shrinking on that same last chunk); the `i >= C || kk >=
        // C` guard skips the now-out-of-range tail there, at the cost of a
        // few wasted iterations on at most one chunk out of the whole run.
        for (int idx = lane; idx < GDN_CHUNK * GDN_CHUNK; idx += blockDim.x) {
            const int i = idx / GDN_CHUNK, kk = idx % GDN_CHUNK;
            if (i >= C || kk >= C) continue;
            float v = 0.0f;
            if (i > kk) {
                float dot = 0.0f;
#pragma unroll
                for (int d = 0; d < GDN_DK; ++d) {
                    dot += sk[i * GDN_ROW_PAD + d] * sk[kk * GDN_ROW_PAD + d];
                }
                v = sbeta[i] * __expf(sgc[i] - sgc[kk]) * dot;
            }
            sA[i * GDN_A_STRIDE + kk] = v;
        }
        __syncthreads();

        // Forward substitution for Ai = (I + A)^-1 (`solve_tril.py`'s actual
        // recurrence, generalized from its 16x16 base case rather than its
        // block-recursive merge, which is a Triton parallelism trick this
        // doesn't need -- see the perf-gap memory doc). `I + A` is unit lower
        // triangular (A's diagonal is exactly 0 by the `i > kk` mask above),
        // so this never divides and is never singular. Row `i` depends on
        // every earlier row, already finalized in a prior iteration; within
        // a row, every column is independent (one thread each), so the
        // barrier is once a row, not once a cell.
        //
        // `GDN_CHUNK` is exactly the warp size, and `lane < C` (`C <=
        // GDN_CHUNK`) means only warp 0 ever does anything here -- so the
        // per-row barrier only has to keep warp 0's own 32 lanes in step,
        // not the whole block. `__syncwarp()` does that at a fraction of
        // `__syncthreads()`'s cost (it doesn't wait for the seven idle
        // warps, and doesn't cross the whole SM to do it); one
        // `__syncthreads()` after the loop is enough to let those seven
        // warps -- which raced through every iteration's `if` doing nothing
        // -- catch up before anyone reads the finalized `sA`. This was worth
        // finding: for the ~1000-chunk 30552-token shape, up to `GDN_CHUNK`
        // block-wide barriers a chunk here alone was a meaningful share of
        // why this kernel measured slower than `gdn_delta_rule_reg128_f32`
        // rather than faster (see the top-of-kernel comment).
        //
        // A race hidden here from `gdn_two_phase_state_f32`'s single-chunk
        // isolation down: `sA[i][m]` for `m` in `[kk, i)` reads OTHER lanes'
        // own targets in this same iteration (`m` can equal another active
        // lane's `kk`), so every lane's read of row `i` must finish before
        // any lane's write to row `i` starts. The one `__syncwarp()` at the
        // end of each `i` only orders writer against the *next* iteration's
        // readers, not against this iteration's own -- racecheck confirms
        // it (192-11232 hazards depending on shape) though the answer comes
        // out right anyway on this hardware, which is exactly the class of
        // bug `--tool racecheck` exists to catch that a numerical diff
        // can't. Splitting read and write into their own phases with a
        // `__syncwarp()` between them costs one more of the same
        // near-zero-cost barrier, not a `__syncthreads()`.
        for (int i = 0; i < C; ++i) {
            const bool active = (lane < C && lane <= i);
            float acc = 0.0f;
            if (active) {
                const int kk = lane;
                acc = (i == kk) ? 1.0f : 0.0f;
                for (int m = kk; m < i; ++m) {
                    acc -= sA[i * GDN_A_STRIDE + m] * sA[m * GDN_A_STRIDE + kk];
                }
            }
            __syncwarp();
            if (active) {
                sA[i * GDN_A_STRIDE + lane] = acc;
            }
            __syncwarp();
        }
        __syncthreads();

        // W/U reconstruction (`wy_fast.py:91-116`): W folds the decay into K,
        // U does not.
        for (int idx = lane; idx < C * GDN_DK; idx += blockDim.x) {
            const int i = idx / GDN_DK, d = idx % GDN_DK;
            float wacc = 0.0f, uacc = 0.0f;
            for (int kk = 0; kk <= i; ++kk) {
                const float aik = sA[i * GDN_A_STRIDE + kk];
                wacc += aik * sbg[kk] * sk[kk * GDN_ROW_PAD + d];
                uacc += aik * sbeta[kk] * sv[kk * GDN_ROW_PAD + d];
            }
            sW[i * GDN_ROW_PAD + d] = wacc;
            sD[i * GDN_ROW_PAD + d] = uacc;  // sD holds U for now
        }
        __syncthreads();

        // History contribution against the OLD (pre-chunk) state `sc`:
        // pred = W @ S_before^T (chunk_delta_h.py:174-197) and
        // o_hist = exp(gc_i) * (Q @ S_before^T) (chunk_o.py:111-119), written
        // straight to `out` as this token's first term. `delta` overwrites U
        // in place -- kept UNDECAYED here (just `U - pred`), unlike
        // `chunk_delta_h.py:216-221`'s `Δ *= exp(ḡ_last − ḡ_j)`: that forward
        // -to-chunk-end decay only belongs in the *state* update below, which
        // combines every token's delta at one common reference point. Folding
        // it in here too (this kernel's first, wrong attempt) double-applies
        // it wherever `A_intra` below also carries a `ḡ_i − ḡ_k` factor,
        // overweighting every history/intra split — caught by
        // `the_three_delta_rule_kernels_agree_with_each_other_and_the_reference`
        // (13-token, single-chunk case) and confirmed against a from-scratch
        // numpy re-derivation before touching this file again, not just
        // patched until the one failing assertion went quiet.
        for (int i = 0; i < C; ++i) {
            float pp = 0.0f, oh = 0.0f;
#pragma unroll
            for (int r = 0; r < 64; ++r) {
                pp += sc[r] * sW[i * GDN_ROW_PAD + i0 + r];
                oh += sc[r] * sq[i * GDN_ROW_PAD + i0 + r];
            }
            pp += __shfl_xor_sync(0xffffffffu, pp, 1, 32);
            oh += __shfl_xor_sync(0xffffffffu, oh, 1, 32);
            if (part == 0) {
                const float u = sD[i * GDN_ROW_PAD + j];
                sD[i * GDN_ROW_PAD + j] = u - pp;
                out[((size_t)(t0 + c0 + i) * heads + head) * GDN_DV + j] =
                    __expf(sgc[i]) * oh;
            }
        }
        __syncthreads();

        // Intra-chunk causal scores, i >= k inclusive this time (unlike the
        // system matrix above) -- `chunk_o.py:113,120,124-126`. `sW` (dead:
        // read for the last time above) is reused for this, same byte size.
        float* sAi2 = sW;
        // Same fixed-trip-count, compile-time-divisor rewrite as the system
        // matrix above, same reason.
        for (int idx = lane; idx < GDN_CHUNK * GDN_CHUNK; idx += blockDim.x) {
            const int i = idx / GDN_CHUNK, kk = idx % GDN_CHUNK;
            if (i >= C || kk >= C) continue;
            float v = 0.0f;
            if (i >= kk) {
                float dot = 0.0f;
#pragma unroll
                for (int d = 0; d < GDN_DK; ++d) {
                    dot += sq[i * GDN_ROW_PAD + d] * sk[kk * GDN_ROW_PAD + d];
                }
                v = __expf(sgc[i] - sgc[kk]) * dot;
            }
            sAi2[i * GDN_A_STRIDE + kk] = v;
        }
        __syncthreads();

        // O += (intra scores) @ delta (chunk_o.py:137), added onto the
        // history term already sitting in `out`. No extra `scale` here: `q`
        // arrives pre-scaled by the caller (see the comment on
        // `gdn_delta_rule_f32` above) exactly like `gdn_delta_rule_reg128_f32`
        // relies on, so the `1/sqrt(dk)` `chunk_o.py:160` applies is already
        // folded into every dot product against `q` above.
        for (int i = 0; i < C; ++i) {
            if (part == 0) {
                float acc = 0.0f;
                for (int kk = 0; kk <= i; ++kk) {
                    acc += sAi2[i * GDN_A_STRIDE + kk] * sD[kk * GDN_ROW_PAD + j];
                }
                out[((size_t)(t0 + c0 + i) * heads + head) * GDN_DV + j] += acc;
            }
        }
        __syncthreads();

        // State update for the next chunk: decay the whole chunk forward,
        // then fold in this chunk's delta, each token's forwarded to the
        // chunk's end first (chunk_delta_h.py:216-228,276-298) -- the one
        // place `exp(ḡ_last − ḡ_t)` belongs (see the comment above `sD`'s
        // history-term write). `S` is `[dv, dk]`; this is
        // `S_before*exp(gc_last) + (delta .* exp(gc_last - gc))ᵀ @ K`.
        //
        const float decay_whole = __expf(sgc[C - 1]);
#pragma unroll
        for (int r = 0; r < 64; ++r) {
            float acc = sc[r] * decay_whole;
            for (int t = 0; t < C; ++t) {
                const float dt = sD[t * GDN_ROW_PAD + j] * __expf(sgc[C - 1] - sgc[t]);
                acc += dt * sk[t * GDN_ROW_PAD + i0 + r];
            }
            sc[r] = acc;
        }
        __syncthreads();  // before the next chunk overwrites sk/sv/sA/sW/sD
    }

#pragma unroll
    for (int r = 0; r < 64; ++r) S[(size_t)(i0 + r) * GDN_DV + j] = sc[r];
}

// Isolated toy probe -- NOT the real kernel, NOT launched by any real
// caller. `gdn_delta_rule_reg128_f32`'s own doc history already
// established this kernel "was never short of parallelism" (256 threads,
// 8 warps, all resident) -- so unlike `ws4`'s attention family, adding
// more warps for occupancy is not the lever here. The real lever, traced
// through the actual per-timestep dependency graph in
// `gdn_delta_rule_reg_body` above: state(t) has exactly two consumers --
// timestep t+1's decay step (the true recurrence, must be sequential) and
// timestep t's own output computation (`o = qᵀS`, a leaf that does NOT
// feed back into the recurrence at all). Today both consumers run in the
// SAME threads, back to back, inside the SAME `__syncthreads()`-bounded
// iteration -- meaning the output computation (a real, non-trivial chunk
// of per-timestep work, roughly the same size as the state update itself:
// the same RB-element loop, the same one-shuffle reduction) sits on the
// critical path for no structural reason. It could instead trail one
// timestep behind on a SEPARATE physical warp while the state-advancing
// warp races ahead uninterrupted -- real inter-warp concurrency (like the
// attention ping-pong probe), but this time on a workload this session's
// own prior investigation already confirmed is not occupancy-starved, and
// with a REAL, traced, unconditional (not merely hoped-for) reason the
// two stages don't depend on each other in the direction that would block
// this.
//
// R=2, ACC=4, RB=64 -- this checkpoint's own `gdn_delta_rule_reg128_f32`
// shape (`DK_C=DV_C=128`) -- with GCOLS scaled down to 16 (one warp's
// worth via R=2 partnering) instead of the real kernel's 128, since this
// probe exists to validate the cross-warp handoff mechanism and its
// timing, not to reproduce the real kernel's exact memory layout.
#define GDN_PP_ITERS 4096
#define GDN_PP_STATE_READY0 1
#define GDN_PP_STATE_READY1 2
#define GDN_PP_STATE_FREE0 3
#define GDN_PP_STATE_FREE1 4

__device__ __forceinline__ float gdn_pp_state_advance(float* sc, const float* kc,
                                                       const float* qc, float decay,
                                                       float v, float b, int lane) {
    float kv[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int r = 0; r < 64; ++r) {
        sc[r] *= decay;
        kv[r % 4] += sc[r] * kc[r];
    }
#pragma unroll
    for (int a = 1; a < 4; ++a) kv[0] += kv[a];
    float kvt = kv[0] + __shfl_xor_sync(0xffffffffu, kv[0], 1);
    const float delta = (v - kvt) * b;
#pragma unroll
    for (int r = 0; r < 64; ++r) sc[r] += kc[r] * delta;
    (void)lane;
    return delta;
}

__device__ __forceinline__ float gdn_pp_output(const float* sc, const float* qc) {
    float o[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int r = 0; r < 64; ++r) o[r % 4] += sc[r] * qc[r];
#pragma unroll
    for (int a = 1; a < 4; ++a) o[0] += o[a];
    return o[0] + __shfl_xor_sync(0xffffffffu, o[0], 1);
}

// Three-kernel split of `gdn_chunk_delta_rule_f32`'s own algorithm, following
// the real dependency structure traced through that kernel's body: forming
// the system matrix, inverting it, and the WY (`W`/`U`) reconstruction all
// depend ONLY on a chunk's own K/V/beta/g -- not on the carried state --
// while only the "history" output term and the state carry-forward itself
// are genuinely sequential across chunks. Splitting those apart lets the
// state-independent, expensive part run on a grid of `chunks * heads`
// blocks instead of being serialized inside a `heads`-only loop (48 blocks
// on this checkpoint, the same hard ceiling `gdn_delta_rule_reg128_f32`
// also has) -- exactly the same 3-kernel decomposition SGLang's real
// production GDN kernel uses for Blackwell (`gdn_blackwell/kernel_h.py`,
// `kernel_kkt_inv_uw.py`, `kernel_o.py`), independently re-derived from this
// codebase's own already-reference-checked fused kernel rather than copied
// -- SGLang's actual implementation is unusable here regardless (it's built
// entirely on `tcgen05`, Blackwell *datacenter* sm_100/101's tensor-memory
// MMA, which sm_120a does not have at all, on top of not having the older
// `wgmma` either).
//
// Scoped to a SINGLE sequence for this first pass (`first_token`/`n_tok`
// still take the same layout as the fused kernel for calling-convention
// parity, but only index 0 is read) -- proving the architecture is correct
// and measuring its real cost is the goal here, not yet matching
// `gdn_chunk_delta_rule_f32`'s full multi-sequence/incremental-call
// generality.
//
// Kernel 1 of 3: parallel over (head, chunk), no state dependency at all.
// Computes the system matrix `A` (`chunk_scaled_dot_kkt.py`), its unit-
// lower-triangular inverse via forward substitution (`solve_tril.py`), and
// the WY reconstruction `W`/`U` (`wy_fast.py`) -- byte-for-byte the same
// per-chunk math as `gdn_chunk_delta_rule_f32`'s own loop body, just without
// the state (`sc`) or the intra-chunk causal/output steps, which belong to
// kernels 2 and 3 below. Needs K, V, beta, g -- NOT Q (`W` folds decay into
// K, `U` does not touch K at all; neither touches Q), matching
// `kkt_inv_uw_cutedsl`'s own real argument list exactly (checked against
// `gdn_blackwell/__init__.py`'s call site).
extern "C" __global__ __launch_bounds__(256) void gdn_chunk_uw_f32(
        float* __restrict__ w_out, float* __restrict__ u_out,
        const float* __restrict__ qkv, const float* __restrict__ g,
        const float* __restrict__ beta, const int* __restrict__ first_token,
        const int* __restrict__ n_tok, int heads, int key_heads, int dk, int dv,
        int stride, int k_off, int v_off, int v_tiled) {
    (void)dk;
    (void)dv;
    const int head = blockIdx.x;
    const int chunk = blockIdx.y;
    const int seq = 0;
    const int nt = n_tok[seq];
    const int c0 = chunk * GDN_CHUNK;
    if (c0 >= nt) return;
    const int t0 = first_token[seq];
    const int C = min(GDN_CHUNK, nt - c0);
    const int lane = threadIdx.x;
    const int khead = v_tiled ? (head % key_heads) : (head / (heads / key_heads));

    extern __shared__ char gdn_uw_smem[];
    float* sk = (float*)gdn_uw_smem;                    // [GDN_CHUNK][GDN_ROW_PAD]
    float* sv = sk + GDN_CHUNK * GDN_ROW_PAD;          // [GDN_CHUNK][GDN_ROW_PAD]
    float* sgc = sv + GDN_CHUNK * GDN_ROW_PAD;         // [GDN_CHUNK]
    float* sbeta = sgc + GDN_CHUNK;                    // [GDN_CHUNK]
    float* sbg = sbeta + GDN_CHUNK;                    // [GDN_CHUNK]
    float* sA = sbg + GDN_CHUNK;                       // [GDN_CHUNK][GDN_A_STRIDE]

    for (int idx = lane; idx < C * GDN_DK; idx += blockDim.x) {
        const int r = idx / GDN_DK, d = idx % GDN_DK;
        const float* row = qkv + (size_t)(t0 + c0 + r) * stride;
        sk[r * GDN_ROW_PAD + d] = row[k_off + (size_t)khead * GDN_DK + d];
    }
    for (int idx = lane; idx < C * GDN_DV; idx += blockDim.x) {
        const int r = idx / GDN_DV, d = idx % GDN_DV;
        const float* row = qkv + (size_t)(t0 + c0 + r) * stride;
        sv[r * GDN_ROW_PAD + d] = row[v_off + (size_t)head * GDN_DV + d];
    }
    if (lane < C) {
        sbeta[lane] = beta[(size_t)(t0 + c0 + lane) * heads + head];
    }
    if (lane == 0) {
        float acc = 0.0f;
        for (int r = 0; r < C; ++r) {
            acc += g[(size_t)(t0 + c0 + r) * heads + head];
            sgc[r] = acc;
        }
    }
    __syncthreads();

    if (lane < C) {
        sbg[lane] = sbeta[lane] * __expf(sgc[lane]);
    }

    for (int idx = lane; idx < GDN_CHUNK * GDN_CHUNK; idx += blockDim.x) {
        const int i = idx / GDN_CHUNK, kk = idx % GDN_CHUNK;
        if (i >= C || kk >= C) continue;
        float v = 0.0f;
        if (i > kk) {
            float dot = 0.0f;
#pragma unroll
            for (int d = 0; d < GDN_DK; ++d) {
                dot += sk[i * GDN_ROW_PAD + d] * sk[kk * GDN_ROW_PAD + d];
            }
            v = sbeta[i] * __expf(sgc[i] - sgc[kk]) * dot;
        }
        sA[i * GDN_A_STRIDE + kk] = v;
    }
    __syncthreads();

    for (int i = 0; i < C; ++i) {
        const bool active = (lane < C && lane <= i);
        float acc = 0.0f;
        if (active) {
            const int kk = lane;
            acc = (i == kk) ? 1.0f : 0.0f;
            for (int m = kk; m < i; ++m) {
                acc -= sA[i * GDN_A_STRIDE + m] * sA[m * GDN_A_STRIDE + kk];
            }
        }
        __syncwarp();
        if (active) {
            sA[i * GDN_A_STRIDE + lane] = acc;
        }
        __syncwarp();
    }
    __syncthreads();

    float* w_chunk = w_out + ((size_t)chunk * heads + head) * GDN_CHUNK * GDN_DK;
    float* u_chunk = u_out + ((size_t)chunk * heads + head) * GDN_CHUNK * GDN_DV;
    for (int idx = lane; idx < C * GDN_DK; idx += blockDim.x) {
        const int i = idx / GDN_DK, d = idx % GDN_DK;
        float wacc = 0.0f, uacc = 0.0f;
        for (int kk = 0; kk <= i; ++kk) {
            const float aik = sA[i * GDN_A_STRIDE + kk];
            wacc += aik * sbg[kk] * sk[kk * GDN_ROW_PAD + d];
            uacc += aik * sbeta[kk] * sv[kk * GDN_ROW_PAD + d];
        }
        w_chunk[i * GDN_DK + d] = wacc;
        u_chunk[i * GDN_DV + d] = uacc;  // GDN_DV == GDN_DK == 128 here
    }
}

// Kernel 2 of 3: sequential over chunks, gridded over `heads` only (the same
// ceiling `gdn_delta_rule_reg128_f32` has) -- but now touching only the
// CHEAP part of each chunk: `pred = W @ S_before`, `delta = U - pred`
// (`chunk_delta_h.py:174-197`), and the state carry-forward itself
// (`chunk_delta_h.py:216-228,276-298`). No forward substitution here at
// all -- that already happened, once, in kernel 1 -- so this kernel's own
// per-chunk work has none of `gdn_chunk_delta_rule_f32`'s O(GDN_CHUNK)
// sequential `__syncwarp()` micro-barriers, only O(1) barriers a chunk.
// Writes `S_before` (the incoming state, BEFORE this chunk's update) for
// kernel 3's history term -- the real, accepted VRAM cost this design pays
// (`n_chunks` copies of a full `[GDN_DK, GDN_DV]` state matrix a head) for
// letting kernel 3 run with no cross-chunk dependency at all. Needs K again
// (for the state-advance step) -- a real, small, accepted redundant load
// against kernel 1's own K read.
extern "C" __global__ __launch_bounds__(256) void gdn_chunk_state_f32(
        float* __restrict__ delta_out, float* __restrict__ s_before_out,
        float* __restrict__ state, const float* __restrict__ w_in,
        const float* __restrict__ u_in, const float* __restrict__ qkv,
        const float* __restrict__ g, const int* __restrict__ first_token,
        const int* __restrict__ n_tok, int heads, int key_heads, int dk, int dv,
        int stride, int k_off, int v_tiled, int n_chunks) {
    (void)dk;
    (void)dv;
    const int head = blockIdx.x;
    const int seq = 0;
    const int nt = n_tok[seq];
    if (nt <= 0) return;
    const int t0 = first_token[seq];
    const int lane = threadIdx.x;  // 0..255
    const int j = lane / 2;
    const int part = lane % 2;
    const int i0 = part * 64;
    const int khead = v_tiled ? (head % key_heads) : (head / (heads / key_heads));

    extern __shared__ char gdn_state_smem[];
    float* sk = (float*)gdn_state_smem;               // [GDN_CHUNK][GDN_ROW_PAD]
    float* sgc = sk + GDN_CHUNK * GDN_ROW_PAD;         // [GDN_CHUNK]
    // `W` staged here too, unlike an earlier version of this kernel that
    // read it straight from global for every (i, r) pair in the pred loop
    // below -- up to C*64 real global-memory instructions a thread a
    // chunk, each paying real per-instruction overhead even on an L1 hit,
    // not just the one-time bulk copy this staging costs instead. Real,
    // measured cost of the unstaged version: most of a 5.6x slowdown
    // against `gdn_delta_rule_reg128_f32` (`gdn_split3_bench`, 30552-token
    // real shape) that a plain "read straight from global" implementation
    // did not pay for K/V in kernel 1 (which already stages them) or for Q/K
    // in kernel 3 (same) -- only this kernel skipped it, and only this
    // kernel was anomalously slow.
    float* sw = sgc + GDN_CHUNK;                       // [GDN_CHUNK][GDN_ROW_PAD]

    float* S = state + (size_t)head * GDN_DK * GDN_DV;
    float sc[64];
#pragma unroll
    for (int r = 0; r < 64; ++r) sc[r] = S[(size_t)(i0 + r) * GDN_DV + j];

    for (int chunk = 0; chunk < n_chunks; ++chunk) {
        const int c0 = chunk * GDN_CHUNK;
        if (c0 >= nt) break;
        const int C = min(GDN_CHUNK, nt - c0);

        float* sbefore_chunk = s_before_out + ((size_t)chunk * heads + head) * GDN_DK * GDN_DV;
#pragma unroll
        for (int r = 0; r < 64; ++r) {
            sbefore_chunk[(size_t)(i0 + r) * GDN_DV + j] = sc[r];
        }

        const float* w_chunk = w_in + ((size_t)chunk * heads + head) * GDN_CHUNK * GDN_DK;
        const float* u_chunk = u_in + ((size_t)chunk * heads + head) * GDN_CHUNK * GDN_DV;
        float* delta_chunk = delta_out + ((size_t)chunk * heads + head) * GDN_CHUNK * GDN_DV;

        for (int idx = lane; idx < C * GDN_DK; idx += blockDim.x) {
            const int r = idx / GDN_DK, d = idx % GDN_DK;
            const float* row = qkv + (size_t)(t0 + c0 + r) * stride;
            sk[r * GDN_ROW_PAD + d] = row[k_off + (size_t)khead * GDN_DK + d];
            sw[r * GDN_ROW_PAD + d] = w_chunk[r * GDN_DK + d];
        }
        if (lane == 0) {
            float acc = 0.0f;
            for (int r = 0; r < C; ++r) {
                acc += g[(size_t)(t0 + c0 + r) * heads + head];
                sgc[r] = acc;
            }
        }
        __syncthreads();

        for (int i = 0; i < C; ++i) {
            float pp = 0.0f;
#pragma unroll
            for (int r = 0; r < 64; ++r) {
                pp += sc[r] * sw[i * GDN_ROW_PAD + i0 + r];
            }
            pp += __shfl_xor_sync(0xffffffffu, pp, 1, 32);
            if (part == 0) {
                delta_chunk[i * GDN_DV + j] = u_chunk[i * GDN_DV + j] - pp;
            }
        }
        __syncthreads();

        // Precompute `dt[t]` ONCE a chunk (GDN_CHUNK global reads of
        // `delta_chunk`), not once per (r, t) pair (up to 64x that) --
        // `delta_chunk` is GLOBAL memory here, unlike `gdn_chunk_delta_rule_f32`'s
        // own state-advance loop this was adapted from, where the
        // equivalent `sD` is shared memory and the same re-read-inside-the-
        // r-loop pattern is cheap. Measured, not assumed: this exact bug
        // was 77.7% of this kernel's own real, measured end-to-end cost
        // before this fix (`gdn_split3_bench`, 30552-token real shape).
        const float decay_whole = __expf(sgc[C - 1]);
        float dt_cache[GDN_CHUNK];
#pragma unroll
        for (int t = 0; t < GDN_CHUNK; ++t) {
            dt_cache[t] = (t < C) ? delta_chunk[t * GDN_DV + j] * __expf(sgc[C - 1] - sgc[t]) : 0.0f;
        }
#pragma unroll
        for (int r = 0; r < 64; ++r) {
            float acc = sc[r] * decay_whole;
            for (int t = 0; t < C; ++t) {
                acc += dt_cache[t] * sk[t * GDN_ROW_PAD + i0 + r];
            }
            sc[r] = acc;
        }
        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < 64; ++r) S[(size_t)(i0 + r) * GDN_DV + j] = sc[r];
}

// Kernel 3 of 3: parallel over (head, chunk) again, no cross-chunk
// dependency at all -- consumes `S_before`/`delta` from kernel 2 and
// recomputes the intra-chunk causal scores (`chunk_o.py:113,120,124-126`,
// a pure function of this chunk's own Q/K/g) to produce the full output:
// the history term (`chunk_o.py:111-119`) plus the intra-chunk term
// (`chunk_o.py:137`) -- exactly `gdn_chunk_delta_rule_f32`'s own two output
// writes, just reading `S_before`/`delta` from global instead of carrying
// them in registers/shared across the sequential loop.
extern "C" __global__ __launch_bounds__(256) void gdn_chunk_output_f32(
        float* __restrict__ out, const float* __restrict__ delta_in,
        const float* __restrict__ s_before_in, const float* __restrict__ qkv,
        const float* __restrict__ g, const int* __restrict__ first_token,
        const int* __restrict__ n_tok, int heads, int key_heads, int dk, int dv,
        int stride, int q_off, int k_off, int v_tiled) {
    (void)dk;
    (void)dv;
    const int head = blockIdx.x;
    const int chunk = blockIdx.y;
    const int seq = 0;
    const int nt = n_tok[seq];
    const int c0 = chunk * GDN_CHUNK;
    if (c0 >= nt) return;
    const int t0 = first_token[seq];
    const int C = min(GDN_CHUNK, nt - c0);
    const int lane = threadIdx.x;  // 0..255
    const int j = lane / 2;
    const int part = lane % 2;
    const int i0 = part * 64;
    const int khead = v_tiled ? (head % key_heads) : (head / (heads / key_heads));

    extern __shared__ char gdn_out_smem[];
    float* sk = (float*)gdn_out_smem;                   // [GDN_CHUNK][GDN_ROW_PAD]
    float* sq = sk + GDN_CHUNK * GDN_ROW_PAD;          // [GDN_CHUNK][GDN_ROW_PAD]
    float* sgc = sq + GDN_CHUNK * GDN_ROW_PAD;         // [GDN_CHUNK]
    float* sAi2 = sgc + GDN_CHUNK;                     // [GDN_CHUNK][GDN_A_STRIDE]

    for (int idx = lane; idx < C * GDN_DK; idx += blockDim.x) {
        const int r = idx / GDN_DK, d = idx % GDN_DK;
        const float* row = qkv + (size_t)(t0 + c0 + r) * stride;
        sk[r * GDN_ROW_PAD + d] = row[k_off + (size_t)khead * GDN_DK + d];
        sq[r * GDN_ROW_PAD + d] = row[q_off + (size_t)khead * GDN_DK + d];
    }
    if (lane == 0) {
        float acc = 0.0f;
        for (int r = 0; r < C; ++r) {
            acc += g[(size_t)(t0 + c0 + r) * heads + head];
            sgc[r] = acc;
        }
    }
    __syncthreads();

    const float* sbefore = s_before_in + ((size_t)chunk * heads + head) * GDN_DK * GDN_DV;
    const float* delta_chunk = delta_in + ((size_t)chunk * heads + head) * GDN_CHUNK * GDN_DV;

    for (int i = 0; i < C; ++i) {
        float oh = 0.0f;
#pragma unroll
        for (int r = 0; r < 64; ++r) {
            oh += sbefore[(size_t)(i0 + r) * GDN_DV + j] * sq[i * GDN_ROW_PAD + i0 + r];
        }
        oh += __shfl_xor_sync(0xffffffffu, oh, 1, 32);
        if (part == 0) {
            out[((size_t)(t0 + c0 + i) * heads + head) * GDN_DV + j] = __expf(sgc[i]) * oh;
        }
    }
    __syncthreads();

    for (int idx = lane; idx < GDN_CHUNK * GDN_CHUNK; idx += blockDim.x) {
        const int i = idx / GDN_CHUNK, kk = idx % GDN_CHUNK;
        if (i >= C || kk >= C) continue;
        float v = 0.0f;
        if (i >= kk) {
            float dot = 0.0f;
#pragma unroll
            for (int d = 0; d < GDN_DK; ++d) {
                dot += sq[i * GDN_ROW_PAD + d] * sk[kk * GDN_ROW_PAD + d];
            }
            v = __expf(sgc[i] - sgc[kk]) * dot;
        }
        sAi2[i * GDN_A_STRIDE + kk] = v;
    }
    __syncthreads();

    for (int i = 0; i < C; ++i) {
        if (part == 0) {
            float acc = 0.0f;
            for (int kk = 0; kk <= i; ++kk) {
                acc += sAi2[i * GDN_A_STRIDE + kk] * delta_chunk[kk * GDN_DV + j];
            }
            out[((size_t)(t0 + c0 + i) * heads + head) * GDN_DV + j] += acc;
        }
    }
}

// Sequential reference: one warp, both stages, every iteration -- exactly
// today's real kernel's own per-timestep order, just without the real
// memory layout.
extern "C" __global__ void gdn_pp_sequential_ref(float* __restrict__ out_checksum) {
    const int lane = threadIdx.x % WARP_SIZE;
    float sc[64], kc[64], qc[64];
#pragma unroll
    for (int r = 0; r < 64; ++r) {
        sc[r] = 0.0f;
        kc[r] = 0.01f * ((lane * 64 + r) % 7 - 3);
        qc[r] = 0.01f * ((lane * 64 + r) % 5 - 2);
    }
    float out_sum = 0.0f;
    for (int n = 0; n < GDN_PP_ITERS; ++n) {
        const float decay = 0.999f;
        const float v = 0.1f + 0.001f * (n % 13);
        const float b = 0.5f;
        gdn_pp_state_advance(sc, kc, qc, decay, v, b, lane);
        const float ot = gdn_pp_output(sc, qc);
        out_sum += ot;
    }
    if (lane == 0) out_checksum[0] = out_sum;
    float s = 0.0f;
#pragma unroll
    for (int r = 0; r < 64; ++r) s += sc[r];
    if (lane == 0) out_checksum[1] = s;
}

// Pipelined: warp 0 advances state for every timestep, uninterrupted by
// the other warp at all; warp 1 computes output for timestep n-1 while
// warp 0 is already on timestep n. Handoff: warp 0 publishes its just-
// updated `sc[64]` (this thread's own rows) to a double-buffered shared
// slot; warp 1 reads it and computes output independently.
extern "C" __global__ void gdn_pp_pipelined_probe(float* __restrict__ out_checksum) {
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
    extern __shared__ float gdn_pp_smem[];
    // [2 stages][32 lanes][64 rows]
    float* sbuf0 = gdn_pp_smem;
    float* sbuf1 = gdn_pp_smem + 32 * 64;

    if (warp == 0) {
        float sc[64], kc[64];
#pragma unroll
        for (int r = 0; r < 64; ++r) {
            sc[r] = 0.0f;
            kc[r] = 0.01f * ((lane * 64 + r) % 7 - 3);
        }
        for (int n = 0; n < GDN_PP_ITERS; ++n) {
            const int stage = n & 1;
            if (n >= 2) {
                const int bar = (stage == 0) ? GDN_PP_STATE_FREE0 : GDN_PP_STATE_FREE1;
                asm volatile("bar.sync %0, %1;" ::"r"(bar), "r"(64) : "memory");
            }
            const float decay = 0.999f;
            const float v = 0.1f + 0.001f * (n % 13);
            const float b = 0.5f;
            // `qc` isn't needed for the state-advance stage at all (see
            // `gdn_pp_state_advance`) -- only the output stage reads Q.
            gdn_pp_state_advance(sc, kc, kc, decay, v, b, lane);
            float* dst = (stage == 0) ? sbuf0 : sbuf1;
#pragma unroll
            for (int r = 0; r < 64; ++r) dst[lane * 64 + r] = sc[r];
            const int bar = (stage == 0) ? GDN_PP_STATE_READY0 : GDN_PP_STATE_READY1;
            asm volatile("bar.arrive %0, %1;" ::"r"(bar), "r"(64) : "memory");
        }
    } else {
        float qc[64];
#pragma unroll
        for (int r = 0; r < 64; ++r) qc[r] = 0.01f * ((lane * 64 + r) % 5 - 2);
        float out_sum = 0.0f;
        float last_sc[64];
        for (int n = 0; n < GDN_PP_ITERS; ++n) {
            const int stage = n & 1;
            const int bar = (stage == 0) ? GDN_PP_STATE_READY0 : GDN_PP_STATE_READY1;
            asm volatile("bar.sync %0, %1;" ::"r"(bar), "r"(64) : "memory");
            const float* src = (stage == 0) ? sbuf0 : sbuf1;
#pragma unroll
            for (int r = 0; r < 64; ++r) last_sc[r] = src[lane * 64 + r];
            {
                const int fbar = (stage == 0) ? GDN_PP_STATE_FREE0 : GDN_PP_STATE_FREE1;
                asm volatile("bar.arrive %0, %1;" ::"r"(fbar), "r"(64) : "memory");
            }
            const float ot = gdn_pp_output(last_sc, qc);
            out_sum += ot;
        }
        if (lane == 0) out_checksum[0] = out_sum;
        float s = 0.0f;
#pragma unroll
        for (int r = 0; r < 64; ++r) s += last_sc[r];
        if (lane == 0) out_checksum[1] = s;
    }
}

// Same pipeline, coarser handoff: `GDN_PP_BATCH` (4) timesteps' worth of
// state per barrier round instead of 1. The per-timestep real work
// (~128 FMA-ish ops, 2 shuffles) is far smaller than one named-barrier
// round-trip's own fixed cost -- the exact result above (0.269x, a real
// 3.7x regression, despite a correct handoff) -- so this checks whether
// amortizing that fixed cost over more real work per round recovers any
// of it, before concluding the whole direction is dead rather than just
// mis-sized.
#define GDN_PP_BATCH 4
extern "C" __global__ void gdn_pp_pipelined_batched_probe(float* __restrict__ out_checksum) {
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
    extern __shared__ float gdn_pp_smem[];
    // [2 stages][32 lanes][GDN_PP_BATCH timesteps][64 rows]
    const int stage_floats = 32 * GDN_PP_BATCH * 64;
    float* sbuf0 = gdn_pp_smem;
    float* sbuf1 = gdn_pp_smem + stage_floats;

    const int n_batches = GDN_PP_ITERS / GDN_PP_BATCH;

    if (warp == 0) {
        float sc[64], kc[64];
#pragma unroll
        for (int r = 0; r < 64; ++r) {
            sc[r] = 0.0f;
            kc[r] = 0.01f * ((lane * 64 + r) % 7 - 3);
        }
        for (int batch = 0; batch < n_batches; ++batch) {
            const int stage = batch & 1;
            if (batch >= 2) {
                const int bar = (stage == 0) ? GDN_PP_STATE_FREE0 : GDN_PP_STATE_FREE1;
                asm volatile("bar.sync %0, %1;" ::"r"(bar), "r"(64) : "memory");
            }
            float* dst = (stage == 0) ? sbuf0 : sbuf1;
#pragma unroll
            for (int i = 0; i < GDN_PP_BATCH; ++i) {
                const int n = batch * GDN_PP_BATCH + i;
                const float decay = 0.999f;
                const float v = 0.1f + 0.001f * (n % 13);
                const float b = 0.5f;
                gdn_pp_state_advance(sc, kc, kc, decay, v, b, lane);
#pragma unroll
                for (int r = 0; r < 64; ++r) dst[(lane * GDN_PP_BATCH + i) * 64 + r] = sc[r];
            }
            const int bar = (stage == 0) ? GDN_PP_STATE_READY0 : GDN_PP_STATE_READY1;
            asm volatile("bar.arrive %0, %1;" ::"r"(bar), "r"(64) : "memory");
        }
    } else {
        float qc[64];
#pragma unroll
        for (int r = 0; r < 64; ++r) qc[r] = 0.01f * ((lane * 64 + r) % 5 - 2);
        float out_sum = 0.0f;
        float last_sc[64];
        for (int batch = 0; batch < n_batches; ++batch) {
            const int stage = batch & 1;
            const int bar = (stage == 0) ? GDN_PP_STATE_READY0 : GDN_PP_STATE_READY1;
            asm volatile("bar.sync %0, %1;" ::"r"(bar), "r"(64) : "memory");
            const float* src = (stage == 0) ? sbuf0 : sbuf1;
#pragma unroll
            for (int i = 0; i < GDN_PP_BATCH; ++i) {
#pragma unroll
                for (int r = 0; r < 64; ++r) last_sc[r] = src[(lane * GDN_PP_BATCH + i) * 64 + r];
                const float ot = gdn_pp_output(last_sc, qc);
                out_sum += ot;
            }
            {
                const int fbar = (stage == 0) ? GDN_PP_STATE_FREE0 : GDN_PP_STATE_FREE1;
                asm volatile("bar.arrive %0, %1;" ::"r"(fbar), "r"(64) : "memory");
            }
        }
        if (lane == 0) out_checksum[0] = out_sum;
        float s = 0.0f;
#pragma unroll
        for (int r = 0; r < 64; ++r) s += last_sc[r];
        if (lane == 0) out_checksum[1] = s;
    }
}
