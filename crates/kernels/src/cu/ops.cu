// Elementwise, normalization, positional and attention kernels.
//
// Activations are f32 throughout; only weights and the KV cache are narrower.
// That costs bandwidth a llama.cpp-style engine would rather keep, but it makes
// every intermediate directly comparable against a CPU reference, which is what
// finding a wrong RoPE convention actually requires.

// ---- normalization ------------------------------------------------------

// Per-head RMS norm, in place, over one head's `d_head` lane of a row.
//
// Qwen3 normalizes each attention head of q and k on its own, with a learned
// `[d_head]` weight, before the rotary. Applying it after would rotate the
// unnormalized vector and give a different answer, and skipping it entirely
// gives fluent nonsense rather than an error — the same failure shape as the
// dropped QKV biases, so this is checked against a CPU reference in
// `tests/qk_norm.rs` rather than against itself.
//
// `row_stride` and `offset` exist because the fused QKV path leaves k inside
// the packed `[q | k | v]` row rather than in a buffer of its own: q is
// contiguous at stride `n_heads * d_head`, k sits at `offset = d` with the
// packed row's stride. One block a (token, head) pair.
extern "C" __global__ void qk_norm_f32(float* __restrict__ buf,
                                       const float* __restrict__ weight,
                                       int n_heads, int d_head, int row_stride,
                                       int offset, float eps) {
    const int token = blockIdx.x / n_heads;
    const int head = blockIdx.x % n_heads;
    float* h = buf + (size_t)token * row_stride + offset + (size_t)head * d_head;

    float acc = 0.0f;
    for (int i = threadIdx.x; i < d_head; i += blockDim.x) {
        const float v = h[i];
        acc += v * v;
    }
    const float scale = rsqrtf(block_reduce_sum(acc) / (float)d_head + eps);

    for (int i = threadIdx.x; i < d_head; i += blockDim.x) {
        h[i] = h[i] * scale * weight[i];
    }
}

// out[t, :] = x[t, :] * rsqrt(mean(x[t, :]^2) + eps) * weight
extern "C" __global__ void rms_norm_f32(float* __restrict__ out,
                                        const float* __restrict__ x,
                                        const float* __restrict__ weight,
                                        int d, float eps) {
    const size_t row = (size_t)blockIdx.x * d;
    const float* xr = x + row;
    float* orow = out + row;

    float acc = 0.0f;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        const float v = xr[i];
        acc += v * v;
    }
    const float scale = rsqrtf(block_reduce_sum(acc) / (float)d + eps);

    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        orow[i] = xr[i] * scale * weight[i];
    }
}

// ---- elementwise --------------------------------------------------------

extern "C" __global__ void add_f32(float* __restrict__ out,
                                   const float* __restrict__ a,
                                   const float* __restrict__ b, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

// out += b, for folding a sublayer back into the residual stream without
// needing a second buffer.
extern "C" __global__ void add_assign_f32(float* __restrict__ out,
                                          const float* __restrict__ b, int n) {
    // Four elements a thread, for the reason spelled out on `f32_to_f16`: at a
    // batch of 32 the scalar form ran this at 349 GB/s.
    const int base = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (base >= n) return;
    const bool aligned =
        ((unsigned long long)out % 16 == 0) && ((unsigned long long)b % 16 == 0);
    if (aligned && base + 3 < n) {
        float4 o = *(const float4*)(const void*)(out + base);
        const float4 v = *(const float4*)(const void*)(b + base);
        o.x += v.x;
        o.y += v.y;
        o.z += v.z;
        o.w += v.w;
        *(float4*)(void*)(out + base) = o;
        return;
    }
    for (int j = base; j < base + 4 && j < n; ++j) out[j] += b[j];
}

// Broadcast a per-column bias over rows: out[t, j] += bias[j]
extern "C" __global__ void add_bias_f32(float* __restrict__ out,
                                        const float* __restrict__ bias,
                                        int n_cols, int n_rows) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (j < n_cols && t < n_rows) out[(size_t)t * n_cols + j] += bias[j];
}

// SwiGLU: out = silu(gate) * up, elementwise.
extern "C" __global__ void silu_mul_f32(float* __restrict__ out,
                                        const float* __restrict__ gate,
                                        const float* __restrict__ up, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = gate[i];
    out[i] = (g / (1.0f + __expf(-g))) * up[i];
}

// Scatter one fused `q ++ k ++ v` row back into three tensors.
//
// Fusing the three projections is worth about 15 us a layer — 31.7 us of
// separate matmuls against 16.7 for one 4096x6144, because the narrow ones
// cannot fill the device — and the alternative to this copy is a row stride
// threaded through `rope_qk`, `store_kv` and `attn_scores`. At a batch of 32
// the copy is 1.5 MiB and the trade is comfortably positive.
extern "C" __global__ void split_qkv_f32(float* __restrict__ q,
                                         float* __restrict__ k,
                                         float* __restrict__ v,
                                         const float* __restrict__ fused,
                                         int d, int kv_dim, int total) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int row_w = d + 2 * kv_dim;
    const int row = i / row_w;
    const int col = i - row * row_w;
    const float x = fused[i];
    if (col < d) {
        q[(size_t)row * d + col] = x;
    } else if (col < d + kv_dim) {
        k[(size_t)row * kv_dim + (col - d)] = x;
    } else {
        v[(size_t)row * kv_dim + (col - d - kv_dim)] = x;
    }
}

// `silu_mul_f32` over the two halves of one fused row.
//
// Running gate and up as one matmul makes a row `2 * d_ff` wide with gate in
// the low half and up in the high half, so the two operands are `d_ff` apart
// within a row and `2 * d_ff` apart between rows rather than being two separate
// tensors. Same arithmetic, one weight matrix.
extern "C" __global__ void silu_mul_split_f32(float* __restrict__ out,
                                              const float* __restrict__ xy,
                                              int d_ff, int total) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int row = i / d_ff;
    const int col = i - row * d_ff;
    const float* r = xy + (size_t)row * 2 * d_ff;
    const float g = r[col];
    out[i] = (g / (1.0f + __expf(-g))) * r[d_ff + col];
}

// The same, also writing the f16 copy `down_proj` is about to read.
//
// This product feeds exactly one matmul, and above `GEMM_THRESHOLD` tokens that
// matmul takes f16 — so the f32 result was being written, read back and
// converted by a separate `f32_to_f16` launch. The value is already in a
// register here. Same numbers as the two-step form: one `__float2half` of the
// same f32, just without the round trip. The f32 copy stays because the narrow
// batches still take the mat-vec path.
//
// `add_rms_norm_f16_f32` does this for the norms, and the two `to_f16` launches
// this removes are what the trace left after it: 2.4 us a layer.
extern "C" __global__ void silu_mul_split_f16_f32(float* __restrict__ out,
                                                  __half* __restrict__ hout,
                                                  const float* __restrict__ xy,
                                                  int d_ff, int total) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int row = i / d_ff;
    const int col = i - row * d_ff;
    const float* r = xy + (size_t)row * 2 * d_ff;
    const float g = r[col];
    const float v = (g / (1.0f + __expf(-g))) * r[d_ff + col];
    out[i] = v;
    hout[i] = __float2half(v);
}

// out[r, :] = in[rows[r], :]
//
// Picks the rows a batch actually needs logits for out of the residual stream,
// so the vocab projection runs once over a handful of rows instead of over
// every token in the batch.
extern "C" __global__ void take_rows_f32(float* __restrict__ out,
                                         const float* __restrict__ in,
                                         const int* __restrict__ rows, int d) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int r = blockIdx.y;
    if (i >= d) return;
    out[(size_t)r * d + i] = in[(size_t)rows[r] * d + i];
}

extern "C" __global__ void f32_to_f16(__half* __restrict__ out,
                                      const float* __restrict__ in, int n) {
    // Four elements a thread.
    //
    // One element a thread leaves each thread with a single load and nothing to
    // overlap its latency against, and the arithmetic is one convert — so the
    // kernel runs at memory latency rather than memory bandwidth. At a batch of
    // 32 it moved 768 KiB in 4.6 us, which is 167 GB/s on a card that does
    // about 1800; `silu_mul`, which is fourteen times the bytes, reaches 1019.
    // Small kernels here are latency-bound, not launch-bound, and four
    // independent elements is the same fix the weight probe and
    // `attn_output_v4_f32` needed.
    //
    // The vector path wants 16 bytes in and 8 out. Every buffer this is called
    // on is a whole allocation or a zero offset into one, so the test passes,
    // but it is uniform across the block and costs nothing to make sure.
    const int base = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (base >= n) return;
    const bool aligned = ((unsigned long long)in % 16 == 0)
                         && ((unsigned long long)out % 8 == 0);
    if (aligned && base + 3 < n) {
        const float4 v = *(const float4*)(const void*)(in + base);
        __half2 h[2];
        h[0] = __floats2half2_rn(v.x, v.y);
        h[1] = __floats2half2_rn(v.z, v.w);
        *(uint2*)(void*)(out + base) = *(const uint2*)(const void*)h;
        return;
    }
    for (int j = base; j < base + 4 && j < n; ++j) out[j] = __float2half(in[j]);
}

// The same conversion, writing k in the order an `m16n8k16` A fragment wants
// when the B fragment comes straight out of an AWQ pack.
//
// `ldmatrix` produces the *standard* fragment: lane L holds k `2c, 2c+1` in
// its first register pair and `2c+8, 2c+9` in its second, with `c = L % 4`. A
// lane's weight word, read at byte `4c` of a 32-byte run, holds weights
// `4c..4c+3`. Those two orders disagree, and the usual fix is to repack the
// weights — measured worthless here, see `mmqfp_*` in `mmq.cu`.
//
// So permute the *activations* instead. They are one matmul's worth of tokens
// against a whole weight matrix — a thousandth of the bytes — and they are
// rewritten every step anyway, so the reordering is free where repacking the
// weights is not. Position p of each sixteen holds activation `j`:
//
//   p =  2c + r   ->  j = 4c + r          (the first register pair)
//   p =  2c+8 + r ->  j = 4c + 2 + r      (the second)
//
// which is an involution, and `tests/ops.rs` pins it against the CPU.
__device__ __forceinline__ int f16_perm16(int p) {
    const int c = (p % 8) / 2;
    const int r = p % 2;
    return p < 8 ? (4 * c + r) : (4 * c + 2 + r);
}

extern "C" __global__ void f32_to_f16_kperm(__half* __restrict__ out,
                                            const float* __restrict__ in,
                                            int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int base = i & ~15;
    out[i] = __float2half(in[base + f16_perm16(i & 15)]);
}

// ---- rotary embeddings --------------------------------------------------

// NeoX / rotate-half convention: element i pairs with i + d_head/2. This is
// what Qwen2 and Llama-family GGUFs expect; the interleaved variant pairs
// 2i with 2i+1 and would silently produce fluent-but-wrong text.
//
// `freq_factors` divides the per-dimension frequency, which is how Llama 3.1
// stretches its low-frequency dimensions for a 128k context. GGUF carries it
// precomputed as `rope_freqs.weight`; models without it pass all ones. Ignoring
// it costs nothing at position zero and progressively more further along, which
// reads as output that starts fine and drifts.
//
// x is [n_tokens, n_heads, d_head], positions is [n_tokens].
extern "C" __global__ void rope_neox_f32(float* __restrict__ x,
                                         const int* __restrict__ positions,
                                         const float* __restrict__ freq_factors,
                                         int n_heads, int d_head,
                                         float theta_base, float freq_scale) {
    const int half = d_head / 2;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= half) return;

    const int head = blockIdx.y;
    const int token = blockIdx.z;

    const float pos = (float)positions[token] * freq_scale;
    const float inv_freq = __powf(theta_base, -2.0f * (float)i / (float)d_head);
    const float angle = pos * inv_freq / freq_factors[i];
    float sin_a, cos_a;
    __sincosf(angle, &sin_a, &cos_a);

    float* row = x + ((size_t)token * n_heads + head) * d_head;
    const float a = row[i];
    const float b = row[i + half];
    row[i] = a * cos_a - b * sin_a;
    row[i + half] = a * sin_a + b * cos_a;
}

// The interleaved convention: element 2i pairs with 2i+1.
//
// Which one a model wants is not a detail it announces — it follows from the
// architecture. llama.cpp permutes Q and K during conversion for llama-family
// files so that this pairing reproduces Hugging Face's rotate-half, and leaves
// qwen2 alone for the NeoX pairing above. Using the wrong one gives fluent
// text that is subtly and increasingly wrong, which is worse than a crash.
extern "C" __global__ void rope_norm_f32(float* __restrict__ x,
                                         const int* __restrict__ positions,
                                         const float* __restrict__ freq_factors,
                                         int n_heads, int d_head,
                                         float theta_base, float freq_scale) {
    const int half = d_head / 2;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= half) return;

    const int head = blockIdx.y;
    const int token = blockIdx.z;

    const float pos = (float)positions[token] * freq_scale;
    const float inv_freq = __powf(theta_base, -2.0f * (float)i / (float)d_head);
    const float angle = pos * inv_freq / freq_factors[i];
    float sin_a, cos_a;
    __sincosf(angle, &sin_a, &cos_a);

    float* row = x + ((size_t)token * n_heads + head) * d_head;
    const float a = row[2 * i];
    const float b = row[2 * i + 1];
    row[2 * i] = a * cos_a - b * sin_a;
    row[2 * i + 1] = a * sin_a + b * cos_a;
}

// Q and K in one launch.
//
// The two calls are identical but for the tensor and its head count, and at a
// batch of one each is eight or thirty-two blocks of sixty-four threads — far
// too little work to be worth its own launch. `blockIdx.y` runs over both head
// sets: the first `n_heads` are Q's, the rest are K's.
//
// `interleaved` picks the pairing at runtime rather than through two kernels.
// The branch is uniform across the whole grid, so it costs nothing beyond the
// predicate.
//
// `rotary_dim` is how many of each head's dimensions rotate — `d_head` for
// every model before Qwen3.5, 64 of 256 for that one. Two things follow, and
// both of them run to completion if got wrong:
//
//  * the pairing is over the *rotary* half, `(i, i + rotary_dim/2)`, so the
//    partner of dimension 0 is dimension 32 and not dimension 128;
//  * the frequency exponent is divided by `rotary_dim`, not by `d_head`. The
//    partial table is the same frequency span compressed into fewer
//    dimensions, not the leading slice of the wide one. Dividing by `d_head`
//    leaves the whole table too high-frequency at the low end, which costs
//    long-range retrieval and nothing else.
//
// Dimensions at or past `rotary_dim` are never addressed here, so they keep
// their bits by construction: this kernel rotates in place.
extern "C" __global__ void rope_qk_f32(float* __restrict__ q,
                                       float* __restrict__ k,
                                       const int* __restrict__ positions,
                                       const float* __restrict__ freq_factors,
                                       int n_heads, int n_kv_heads, int d_head,
                                       int rotary_dim,
                                       float theta_base, float freq_scale,
                                       int interleaved) {
    const int half = rotary_dim / 2;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= half) return;

    const int y = blockIdx.y;
    const int token = blockIdx.z;
    const bool is_q = y < n_heads;
    const int head = is_q ? y : y - n_heads;
    float* base = is_q ? q : k;
    const int heads = is_q ? n_heads : n_kv_heads;

    const float pos = (float)positions[token] * freq_scale;
    const float inv_freq = __powf(theta_base, -2.0f * (float)i / (float)rotary_dim);
    const float angle = pos * inv_freq / freq_factors[i];
    float sin_a, cos_a;
    __sincosf(angle, &sin_a, &cos_a);

    float* row = base + ((size_t)token * heads + head) * d_head;
    const int ia = interleaved ? 2 * i : i;
    const int ib = interleaved ? 2 * i + 1 : i + half;
    const float a = row[ia];
    const float b = row[ib];
    row[ia] = a * cos_a - b * sin_a;
    row[ib] = a * sin_a + b * cos_a;
}

// ---- KV cache -----------------------------------------------------------

// Append this step's keys/values to the pool.
//
// src is [n_tokens, n_kv_heads, d_head] f32; the pool is
// [n_kv_heads, n_slots, d_head] f16 and is shared by every sequence in
// flight. `slots[token]` is the physical slot this token was allocated, which
// is what decouples a sequence's logical position from where its history
// actually lives.
extern "C" __global__ void store_kv_f16(__half* __restrict__ pool,
                                        const float* __restrict__ src,
                                        const int* __restrict__ slots,
                                        int n_kv_heads, int d_head,
                                        int n_slots, int n_tokens) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d_head) return;
    const int head = blockIdx.y;
    const int token = blockIdx.z;
    if (token >= n_tokens) return;

    const int slot = slots[token];
    if (slot < 0 || slot >= n_slots) return;

    const size_t dst = ((size_t)head * n_slots + slot) * d_head + i;
    const size_t s = ((size_t)token * n_kv_heads + head) * d_head + i;
    pool[dst] = __float2half(src[s]);
}

// Both halves of the cache in one launch.
//
// K and V are the same shape and land in the same slots, so the only thing the
// two calls differ in is which pool they write. `blockIdx.y` covers both:
// below `n_kv_heads` is K, above it is V. At a batch of one each call was
// eight blocks of a hundred and twenty-eight threads, which is less work than
// the launch that carries it.
// RoPE straight out of the fused projection's output, which removes the copy
// that used to unpack it.
//
// The stacked `qkv` matmul writes one row per token — `q` then `k` then `v`, a
// `stride` apart — and `split_qkv` existed to scatter that into three
// contiguous buffers so this kernel and `store_kv2` could index them the easy
// way. That is a 1.5 MB round trip a layer for nothing: the indexing is the
// same arithmetic with a stride in it.
//
// `q` is written out to its own buffer because attention reads it contiguously;
// `k` is rotated in place, where `store_kv2_packed_f16` picks it up next to the
// `v` it never had to move at all.
extern "C" __global__ void rope_qk_packed_f32(float* __restrict__ q_dst,
                                             float* __restrict__ packed,
                                             int stride, int q_off, int k_off,
                                             const int* __restrict__ positions,
                                             const float* __restrict__ freq_factors,
                                             int n_heads, int n_kv_heads,
                                             int d_head, int rotary_dim,
                                             float theta_base,
                                             float freq_scale, int interleaved) {
    const int half = rotary_dim / 2;
    const int tail = d_head - rotary_dim;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    // `half` lanes rotate; the `tail` lanes after them exist only to carry q's
    // unrotated dimensions across to `q_dst`. When `rotary_dim == d_head` there
    // are none, and this is the grid the full-width version always launched.
    if (i >= half + tail) return;

    const int y = blockIdx.y;
    const int token = blockIdx.z;
    const bool is_q = y < n_heads;
    const int head = is_q ? y : y - n_heads;

    const float* src = packed + (size_t)token * stride
                     + (is_q ? q_off : k_off) + (size_t)head * d_head;
    float* dst = is_q ? q_dst + ((size_t)token * n_heads + head) * d_head
                      : packed + (size_t)token * stride + k_off
                            + (size_t)head * d_head;

    if (i >= half) {
        // The unrotated tail. `k` is rotated in place, so its tail is already
        // where it belongs and there is nothing to do; `q` is *copied* into a
        // separate buffer, and a partial rotation that only writes the first
        // `rotary_dim` would leave dimensions [rotary_dim, d_head) of `q_dst`
        // holding whatever the previous layer left there. That is the one new
        // way this kernel can go wrong and still run: three quarters of every
        // query head would be stale rather than absent.
        if (is_q) {
            const int d = rotary_dim + (i - half);
            dst[d] = src[d];
        }
        return;
    }

    const float pos = (float)positions[token] * freq_scale;
    const float inv_freq = __powf(theta_base, -2.0f * (float)i / (float)rotary_dim);
    const float angle = pos * inv_freq / freq_factors[i];
    float sin_a, cos_a;
    __sincosf(angle, &sin_a, &cos_a);

    const int ia = interleaved ? 2 * i : i;
    const int ib = interleaved ? 2 * i + 1 : i + half;
    const float a = src[ia], b = src[ib];
    dst[ia] = a * cos_a - b * sin_a;
    dst[ib] = a * sin_a + b * cos_a;
}

// `store_kv2_f16` reading `k` and `v` out of the fused projection's row.
extern "C" __global__ void store_kv2_packed_f16(__half* __restrict__ k_pool,
                                               __half* __restrict__ v_pool,
                                               const float* __restrict__ packed,
                                               int stride, int k_off, int v_off,
                                               const int* __restrict__ slots,
                                               int n_kv_heads, int d_head,
                                               int n_slots, int n_tokens) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d_head) return;
    const int y = blockIdx.y;
    const int token = blockIdx.z;
    if (token >= n_tokens) return;

    const int slot = slots[token];
    if (slot < 0 || slot >= n_slots) return;

    const bool is_k = y < n_kv_heads;
    const int head = is_k ? y : y - n_kv_heads;
    __half* pool = is_k ? k_pool : v_pool;

    const size_t dst = ((size_t)head * n_slots + slot) * d_head + i;
    const size_t s = (size_t)token * stride + (is_k ? k_off : v_off)
                   + (size_t)head * d_head + i;
    pool[dst] = __float2half(packed[s]);
}

extern "C" __global__ void store_kv2_f16(__half* __restrict__ k_pool,
                                         __half* __restrict__ v_pool,
                                         const float* __restrict__ k_src,
                                         const float* __restrict__ v_src,
                                         const int* __restrict__ slots,
                                         int n_kv_heads, int d_head,
                                         int n_slots, int n_tokens) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d_head) return;
    const int y = blockIdx.y;
    const int token = blockIdx.z;
    if (token >= n_tokens) return;

    const int slot = slots[token];
    if (slot < 0 || slot >= n_slots) return;

    const bool is_k = y < n_kv_heads;
    const int head = is_k ? y : y - n_kv_heads;
    __half* pool = is_k ? k_pool : v_pool;
    const float* src = is_k ? k_src : v_src;

    const size_t dst = ((size_t)head * n_slots + slot) * d_head + i;
    const size_t s = ((size_t)token * n_kv_heads + head) * d_head + i;
    pool[dst] = __float2half(src[s]);
}

// slot_table[seq_of[i] * stride + positions[i]] = slots[i]
//
// The batch already carries all three arrays, so recording where each new
// token landed costs one kernel rather than one host-to-device copy per
// sequence — and a pageable copy drains the stream, which at thirty-odd
// sequences per step is most of a decode.
extern "C" __global__ void write_slot_table(int* __restrict__ table,
                                            const int* __restrict__ seq_of,
                                            const int* __restrict__ positions,
                                            const int* __restrict__ slots,
                                            int stride, int n_tokens) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_tokens) return;
    table[(size_t)seq_of[i] * stride + positions[i]] = slots[i];
}

// ---- attention ----------------------------------------------------------

// scores[h, t, j] = dot(q[t, h, :], k_cache[h / gqa, j, :]) * scale,
// with -inf where j is in the future of token t.
//
// One warp per score. Grid is (kv positions, heads, tokens).
extern "C" __global__ void attn_scores_f32(float* __restrict__ scores,
                                           const float* __restrict__ q,
                                           const __half* __restrict__ k_cache,
                                           const int* __restrict__ seq_of,
                                           const int* __restrict__ positions,
                                           const int* __restrict__ slot_table,
                                           int table_stride, int n_heads,
                                           int n_kv_heads, int d_head,
                                           int n_slots, int kv_len,
                                           float scale) {
    const int j = blockIdx.x * (blockDim.x / WARP_SIZE) + (threadIdx.x / WARP_SIZE);
    if (j >= kv_len) return;

    const int head = blockIdx.y;
    const int token = blockIdx.z;
    const int lane = threadIdx.x % WARP_SIZE;

    // Each token carries its own position, so one batch can hold sequences of
    // completely different lengths: the mask is per token, not per batch.
    if (j > positions[token]) {
        if (lane == 0) {
            scores[((size_t)head * gridDim.z + token) * kv_len + j] = -INFINITY;
        }
        return;
    }

    const int kv_head = head / (n_heads / n_kv_heads);
    const int slot = slot_table[(size_t)seq_of[token] * table_stride + j];
    const float* qr = q + ((size_t)token * n_heads + head) * d_head;
    const __half* kr = k_cache + ((size_t)kv_head * n_slots + slot) * d_head;

    // Left scalar on purpose. A `half2` version of this loop was written and
    // measured at 275 ms against 210 for the same work: the compiler already
    // vectorizes the strided form, and the runtime width check needed to keep
    // 64-wide heads working stopped it doing so.
    float acc = 0.0f;
    for (int i = lane; i < d_head; i += WARP_SIZE) {
        acc += qr[i] * __half2float(kr[i]);
    }
    acc = warp_reduce_sum(acc);

    if (lane == 0) {
        scores[((size_t)head * gridDim.z + token) * kv_len + j] = acc * scale;
    }
}

// In-place softmax over the kv axis. One block per (head, token).
extern "C" __global__ void attn_softmax_f32(float* __restrict__ scores,
                                            int kv_len) {
    float* row = scores + ((size_t)blockIdx.x * gridDim.y + blockIdx.y) * kv_len;

    float local_max = -INFINITY;
    for (int j = threadIdx.x; j < kv_len; j += blockDim.x) {
        local_max = fmaxf(local_max, row[j]);
    }
    const float m = block_reduce_max(local_max);

    float local_sum = 0.0f;
    for (int j = threadIdx.x; j < kv_len; j += blockDim.x) {
        // A fully masked row (possible only if kv_len is 0) would produce
        // nan; every real row sees at least its own position.
        const float e = __expf(row[j] - m);
        row[j] = e;
        local_sum += e;
    }
    const float inv = 1.0f / block_reduce_sum(local_sum);

    for (int j = threadIdx.x; j < kv_len; j += blockDim.x) {
        row[j] *= inv;
    }
}

// out[t, h, :] = sum_j scores[h, t, j] * v_cache[h / gqa, j, :]
// One block per (head, token); threads split the head dimension.
extern "C" __global__ void attn_output_f32(float* __restrict__ out,
                                           const float* __restrict__ scores,
                                           const __half* __restrict__ v_cache,
                                           const int* __restrict__ seq_of,
                                           const int* __restrict__ positions,
                                           const int* __restrict__ slot_table,
                                           int table_stride, int n_heads,
                                           int n_kv_heads, int d_head,
                                           int n_slots, int kv_len) {
    const int head = blockIdx.x;
    const int token = blockIdx.y;
    const int i = threadIdx.x;
    if (i >= d_head) return;

    const int kv_head = head / (n_heads / n_kv_heads);
    const float* srow = scores + ((size_t)head * gridDim.y + token) * kv_len;
    const __half* vbase = v_cache + (size_t)kv_head * n_slots * d_head;
    const int* table = slot_table + (size_t)seq_of[token] * table_stride;
    // Masked entries are exactly zero after the softmax, but their slot table
    // entries belong to no sequence, so the loop stops rather than multiplying
    // whatever they happen to address by zero.
    const int last = positions[token];

    float acc = 0.0f;
    for (int j = 0; j <= last && j < kv_len; ++j) {
        acc += srow[j] * __half2float(vbase[(size_t)table[j] * d_head + i]);
    }
    out[((size_t)token * n_heads + head) * d_head + i] = acc;
}

// The same sum, read eight halves at a time and sliced over the key range.
//
// `attn_output_f32` gives one thread one element of `d_head`, so a thread's V
// load is two bytes and a warp's is sixty-four — a fraction of a sector per
// instruction — and the `j` loop is a serial chain with one load in flight at a
// time. It measured 85.3 us per launch at a batch of 32 against 42.8 for
// `attn_scores`, which moves the same 33.5 MiB of cache: 393 GB/s on a card
// that does about 1800.
//
// Both halves of that are addressed here. A row of `d_head` halves is 256 bytes
// at the usual 128, so sixteen lanes cover it with one `uint4` each, and the
// remaining threads take a stride of the key range with their own accumulators.
// Sixteen slices is sixteen independent load chains per block instead of one —
// the same fix, and the same reason, as the four accumulators the bandwidth
// probe needed before it stopped measuring its own latency.
extern "C" __global__ void attn_output_v4_f32(
    float* __restrict__ out, const float* __restrict__ scores,
    const __half* __restrict__ v_cache, const int* __restrict__ seq_of,
    const int* __restrict__ positions, const int* __restrict__ slot_table,
    int table_stride, int n_heads, int n_kv_heads, int d_head, int n_slots,
    int kv_len) {
    extern __shared__ float red[];

    const int head = blockIdx.x;
    const int token = blockIdx.y;
    const int lanes = d_head / 8;
    const int lane = threadIdx.x % lanes;
    const int slice = threadIdx.x / lanes;
    const int slices = blockDim.x / lanes;

    const int kv_head = head / (n_heads / n_kv_heads);
    const float* srow = scores + ((size_t)head * gridDim.y + token) * kv_len;
    const __half* vbase = v_cache + (size_t)kv_head * n_slots * d_head;
    const int* table = slot_table + (size_t)seq_of[token] * table_stride;
    const int last = positions[token];

    float acc[8];
#pragma unroll
    for (int c = 0; c < 8; ++c) acc[c] = 0.0f;

    // One row an iteration, and measured against two.
    //
    // Issuing two `uint4` loads before consuming either is what took
    // `attn_scores_gqa` from 43.4 us a layer to 33.9, and here it goes the
    // other way: 36.6 us to 44.7. This loop already carries eight accumulators
    // in registers, and the second row's operands cost more occupancy than the
    // extra load in flight buys.
    for (int j = slice; j <= last && j < kv_len; j += slices) {
        const float s = srow[j];
        const uint4 raw = *(const uint4*)(const void*)(vbase
                                                       + (size_t)table[j] * d_head
                                                       + lane * 8);
        const __half2* hv = (const __half2*)(const void*)&raw;
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const float2 f = __half22float2(hv[c]);
            acc[c * 2] += s * f.x;
            acc[c * 2 + 1] += s * f.y;
        }
    }

    float* mine = red + (size_t)slice * d_head + lane * 8;
#pragma unroll
    for (int c = 0; c < 8; ++c) mine[c] = acc[c];
    __syncthreads();
    for (int st = slices >> 1; st > 0; st >>= 1) {
        if (slice < st) {
            const float* other = mine + (size_t)st * d_head;
#pragma unroll
            for (int c = 0; c < 8; ++c) mine[c] += other[c];
        }
        __syncthreads();
    }
    if (slice == 0) {
        float* dst = out + ((size_t)token * n_heads + head) * d_head + lane * 8;
#pragma unroll
        for (int c = 0; c < 8; ++c) dst[c] = mine[c];
    }
}

// Split-K attention output, for the case the plain kernel handles badly.
//
// `attn_output_f32` gives one block to each (head, token) pair. At a batch of
// one that is 32 blocks of 128 threads on a 48-SM device: two thirds of the
// device idle, the rest at 8% occupancy, and every thread walking the whole KV
// range serially. The same kernel at a batch of 32 has 1024 blocks and is 10x
// more efficient per token — the work was never the problem, the grid was.
//
// So chunk the KV range as well, one block per (head, token, chunk), and reduce
// the partial sums afterwards. The reduction is a separate pass rather than an
// atomic so the result does not depend on the order blocks happen to finish.
extern "C" __global__ void attn_output_split_f32(
    float* __restrict__ partial, const float* __restrict__ scores,
    const __half* __restrict__ v_cache, const int* __restrict__ seq_of,
    const int* __restrict__ positions, const int* __restrict__ slot_table,
    int table_stride, int n_heads, int n_kv_heads, int d_head, int n_slots,
    int kv_len, int n_chunks, int chunk) {
    const int head = blockIdx.x;
    const int token = blockIdx.y;
    const int c = blockIdx.z;
    const int i = threadIdx.x;
    if (i >= d_head) return;

    const int kv_head = head / (n_heads / n_kv_heads);
    const float* srow = scores + ((size_t)head * gridDim.y + token) * kv_len;
    const __half* vbase = v_cache + (size_t)kv_head * n_slots * d_head;
    const int* table = slot_table + (size_t)seq_of[token] * table_stride;
    const int last = positions[token];

    const int j0 = c * chunk;
    int j1 = j0 + chunk;
    if (j1 > kv_len) j1 = kv_len;
    if (j1 > last + 1) j1 = last + 1;

    float acc = 0.0f;
    for (int j = j0; j < j1; ++j) {
        acc += srow[j] * __half2float(vbase[(size_t)table[j] * d_head + i]);
    }
    // [chunk][token][head][d_head]
    partial[(((size_t)c * gridDim.y + token) * n_heads + head) * d_head + i] = acc;
}

// `attn_output_split_f32` with the V row read once for the whole query group.
//
// The two halves of this existed separately and neither could be used. The GQA
// value kernel reads each V row once per KV head instead of once per query
// head — a fourfold cut on Llama-3.1's 32-over-8 — but one block per (KV head,
// token) is 64 blocks at a batch of eight, a quarter of the grid, and the
// launcher disabled it for exactly that. The split kernel restores the grid by
// chunking the key range but reads V per query head.
//
// Together they are 256 blocks at a batch of eight — the same as the plain
// kernel — moving a quarter of the bytes. Each block holds `group` running
// sums across the chunk, so the V row is fetched once and used four times.
extern "C" __global__ void attn_output_gqa_split_f32(
    float* __restrict__ partial, const float* __restrict__ scores,
    const __half* __restrict__ v_cache, const int* __restrict__ seq_of,
    const int* __restrict__ positions, const int* __restrict__ slot_table,
    int table_stride, int n_heads, int n_kv_heads, int d_head, int n_slots,
    int kv_len, int n_chunks, int chunk, int group) {
    const int kv_head = blockIdx.x;
    const int token = blockIdx.y;
    const int c = blockIdx.z;
    const int i = threadIdx.x;
    if (i >= d_head) return;

    const __half* vbase = v_cache + (size_t)kv_head * n_slots * d_head;
    const int* table = slot_table + (size_t)seq_of[token] * table_stride;
    const int last = positions[token];
    const int n_tokens = gridDim.y;

    const int j0 = c * chunk;
    int j1 = j0 + chunk;
    if (j1 > kv_len) j1 = kv_len;
    if (j1 > last + 1) j1 = last + 1;

    // Four is the group Llama-3.1 uses; the launcher only takes this path when
    // the group fits, so the array is sized for the largest it accepts.
    float acc[8];
#pragma unroll
    for (int g = 0; g < 8; ++g) acc[g] = 0.0f;

    for (int j = j0; j < j1; ++j) {
        const float v = __half2float(vbase[(size_t)table[j] * d_head + i]);
        for (int g = 0; g < group; ++g) {
            const int head = kv_head * group + g;
            const float* srow = scores + ((size_t)head * n_tokens + token) * kv_len;
            acc[g] += srow[j] * v;
        }
    }

#pragma unroll
    for (int g = 0; g < 8; ++g) {
        if (g >= group) break;
        const int head = kv_head * group + g;
        partial[(((size_t)c * n_tokens + token) * n_heads + head) * d_head + i] =
            acc[g];
    }
}

extern "C" __global__ void attn_output_reduce_f32(
    float* __restrict__ out, const float* __restrict__ partial, int n_heads,
    int d_head, int n_tokens, int n_chunks) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n_tokens * n_heads * d_head;
    if (idx >= total) return;
    float acc = 0.0f;
    for (int c = 0; c < n_chunks; ++c) {
        acc += partial[(size_t)c * total + idx];
    }
    out[idx] = acc;
}

// GQA-aware scores: one K row, all the query heads that share it.
//
// `attn_scores_gqa_f32` with each lane's slice of the row made contiguous.
//
// The grouped kernel reads `kr[lane + c * WARP_SIZE]`: a lane's two or four
// elements sit 32 apart, so each is its own two-byte load and a warp moves 64
// bytes per instruction. Giving each lane a *contiguous* run instead makes it
// one eight-byte load, and the query row alongside it one sixteen-byte load.
//
// Which run a lane gets changes, so the terms reach `warp_reduce_sum` in a
// different order and the sum differs in the last bits. That is the only
// behavioural difference; the reduction is over the same products.
//
// The width test is hoisted out of the loop deliberately. An earlier `half2`
// attempt on the ungrouped kernel measured worse and the comment there blamed
// the compiler; the cost was the runtime check it left *inside* the loop.
extern "C" __global__ void attn_scores_gqa_v4_f32(
    float* __restrict__ scores, const float* __restrict__ q,
    const __half* __restrict__ k_cache, const int* __restrict__ seq_of,
    const int* __restrict__ positions, const int* __restrict__ slot_table,
    int table_stride, int n_heads, int n_kv_heads, int d_head, int n_slots,
    int kv_len, float scale, int group) {
    // Two keys a warp.
    //
    // `attn_scores_gqa_f32` gives a warp one key: it reads that row, 256 bytes,
    // and spends it on the group. The row is the only thing in flight, and the
    // kernel measures 44.0 us a layer for 33.55 MiB of K — 762 GB/s, where
    // vLLM's fused attention moves K and V at 1156. Two rows doubles what is in
    // flight without changing the instruction count per key: the reduction is
    // still one per (key, head).
    //
    // Lane access stays strided rather than contiguous. Contiguous was tried
    // and is slower here — 45.1 us against 43.6 — because a warp's strided read
    // of the query is already 128 consecutive floats, and the K row is held in
    // registers for the whole group either way. See the note in `attn_scores`
    // dispatch.
    const int warps = blockDim.x / WARP_SIZE;
    const int w = blockIdx.x * warps + (threadIdx.x / WARP_SIZE);
    const int lane = threadIdx.x % WARP_SIZE;
    const int kv_head = blockIdx.y;
    const int token = blockIdx.z;
    const int last = positions[token];
    const int per_lane = d_head / WARP_SIZE;

    const int* table = slot_table + (size_t)seq_of[token] * table_stride;
    const __half* kbase = k_cache + (size_t)kv_head * n_slots * d_head;

    float kv[2][4];
    int js[2];
    bool live[2];
#pragma unroll
    for (int u = 0; u < 2; ++u) {
        js[u] = w * 2 + u;
        live[u] = js[u] < kv_len && js[u] <= last;
#pragma unroll
        for (int c = 0; c < 4; ++c) kv[u][c] = 0.0f;
        if (live[u]) {
            const __half* kr = kbase + (size_t)table[js[u]] * d_head;
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                if (c < per_lane) kv[u][c] = __half2float(kr[lane + c * WARP_SIZE]);
            }
        }
    }

    for (int g = 0; g < group; ++g) {
        const int head = kv_head * group + g;
        const float* qr = q + ((size_t)token * n_heads + head) * d_head;
        float qv[4];
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            qv[c] = (c < per_lane) ? qr[lane + c * WARP_SIZE] : 0.0f;
        }
#pragma unroll
        for (int u = 0; u < 2; ++u) {
            if (js[u] >= kv_len) continue;
            float acc = 0.0f;
#pragma unroll
            for (int c = 0; c < 4; ++c) acc += qv[c] * kv[u][c];
            acc = warp_reduce_sum(acc);
            if (lane == 0) {
                scores[((size_t)head * gridDim.z + token) * kv_len + js[u]] =
                    live[u] ? acc * scale : -INFINITY;
            }
        }
    }
}

// `attn_scores_f32` with four contiguous halves per lane.
//
// The scalar loop above walks `d_head` at a stride of the warp, so a lane's
// four elements are 32 apart and each is its own two-byte load: a warp moves 64
// bytes per instruction and needs four instructions for one K row. Four
// *contiguous* halves per lane is one instruction moving 256 bytes.
//
// A `half2` version of the loop was tried before and measured worse, which the
// comment above blames on the compiler already vectorizing the strided form. It
// does not — strided-by-32 cannot be merged — and the real cost was named in
// the same sentence: a runtime width check inside the loop. That decision lives
// on the host here, so the inner loop has no branch to defeat it. The same fix
// on `attn_output_f32`, whose defect was identical, took it from 86.2 us to
// 36.0.
extern "C" __global__ void attn_scores_v4_f32(
    float* __restrict__ scores, const float* __restrict__ q,
    const __half* __restrict__ k_cache, const int* __restrict__ seq_of,
    const int* __restrict__ positions, const int* __restrict__ slot_table,
    int table_stride, int n_heads, int n_kv_heads, int d_head, int n_slots,
    int kv_len, float scale) {
    const int j = blockIdx.x * (blockDim.x / WARP_SIZE) + (threadIdx.x / WARP_SIZE);
    if (j >= kv_len) return;

    const int head = blockIdx.y;
    const int token = blockIdx.z;
    const int lane = threadIdx.x % WARP_SIZE;

    if (j > positions[token]) {
        if (lane == 0) {
            scores[((size_t)head * gridDim.z + token) * kv_len + j] = -INFINITY;
        }
        return;
    }

    const int kv_head = head / (n_heads / n_kv_heads);
    const int slot = slot_table[(size_t)seq_of[token] * table_stride + j];
    const float* qr = q + ((size_t)token * n_heads + head) * d_head;
    const __half* kr = k_cache + ((size_t)kv_head * n_slots + slot) * d_head;

    // The host only picks this kernel when `d_head` is a multiple of four, so
    // there is no tail and no check.
    const int quads = d_head / 4;
    float acc = 0.0f;
    for (int i = lane; i < quads; i += WARP_SIZE) {
        const uint2 raw = *(const uint2*)(const void*)(kr + i * 4);
        const __half2* h = (const __half2*)(const void*)&raw;
        const float2 a = __half22float2(h[0]);
        const float2 b = __half22float2(h[1]);
        const float* qq = qr + i * 4;
        acc += qq[0] * a.x + qq[1] * a.y + qq[2] * b.x + qq[3] * b.y;
    }
    acc = warp_reduce_sum(acc);

    if (lane == 0) {
        scores[((size_t)head * gridDim.z + token) * kv_len + j] = acc * scale;
    }
}

// `attn_scores_f32` gives each (query head, token, key) its own warp, so with
// grouped-query attention the same K row is fetched once per query head in the
// group — four times over for Llama-3.1's 32 heads over 8 KV heads. Per layer
// at a batch of 32 that is 100 MiB of K traffic against 25 MiB of distinct K,
// and the measured 0.33 ms per layer sits right at the no-reuse bound.
//
// Here a warp owns a (KV head, token, key) and loops the query heads inside,
// holding the K row in registers across the group. The arithmetic is unchanged;
// only the fetch count is.
extern "C" __global__ void attn_scores_gqa_f32(
    float* __restrict__ scores, const float* __restrict__ q,
    const __half* __restrict__ k_cache, const int* __restrict__ seq_of,
    const int* __restrict__ positions, const int* __restrict__ slot_table,
    int table_stride, int n_heads, int n_kv_heads, int d_head, int n_slots,
    int kv_len, float scale, int group) {
    const int j = blockIdx.x * (blockDim.x / WARP_SIZE) + (threadIdx.x / WARP_SIZE);
    if (j >= kv_len) return;

    const int kv_head = blockIdx.y;
    const int token = blockIdx.z;
    const int lane = threadIdx.x % WARP_SIZE;
    const int n_tokens = gridDim.z;

    if (j > positions[token]) {
        if (lane == 0) {
            for (int g = 0; g < group; ++g) {
                const int head = kv_head * group + g;
                scores[((size_t)head * n_tokens + token) * kv_len + j] = -INFINITY;
            }
        }
        return;
    }

    const int slot = slot_table[(size_t)seq_of[token] * table_stride + j];
    const __half* kr = k_cache + ((size_t)kv_head * n_slots + slot) * d_head;

    // The K row, once, held across the whole group. `d_head` is 64 or 128, so
    // this is two or four values per lane.
    float kv[4];
    const int per_lane = d_head / WARP_SIZE;
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        kv[c] = (c < per_lane) ? __half2float(kr[lane + c * WARP_SIZE]) : 0.0f;
    }

    for (int g = 0; g < group; ++g) {
        const int head = kv_head * group + g;
        const float* qr = q + ((size_t)token * n_heads + head) * d_head;
        float acc = 0.0f;
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            if (c < per_lane) acc += qr[lane + c * WARP_SIZE] * kv[c];
        }
        acc = warp_reduce_sum(acc);
        if (lane == 0) {
            scores[((size_t)head * n_tokens + token) * kv_len + j] = acc * scale;
        }
    }
}

// The same reuse for the value side.
//
// V rows are shared by every query head in a group exactly as K rows are, and
// `attn_output_f32` re-reads them per head for the same reason. One block per
// (KV head, token) here, with `group` accumulators per thread — the V element
// is fetched once and spent on all of them.
extern "C" __global__ void attn_output_gqa_f32(
    float* __restrict__ out, const float* __restrict__ scores,
    const __half* __restrict__ v_cache, const int* __restrict__ seq_of,
    const int* __restrict__ positions, const int* __restrict__ slot_table,
    int table_stride, int n_heads, int n_kv_heads, int d_head, int n_slots,
    int kv_len, int group) {
    const int kv_head = blockIdx.x;
    const int token = blockIdx.y;
    const int i = threadIdx.x;
    if (i >= d_head) return;
    const int n_tokens = gridDim.y;

    const __half* vbase = v_cache + (size_t)kv_head * n_slots * d_head;
    const int* table = slot_table + (size_t)seq_of[token] * table_stride;
    const int last = positions[token];

    // Up to eight query heads per KV head covers every grouped-query model in
    // range here; wider groups fall back to the per-head kernel.
    float acc[8];
#pragma unroll
    for (int g = 0; g < 8; ++g) acc[g] = 0.0f;

    for (int j = 0; j <= last && j < kv_len; ++j) {
        const float v = __half2float(vbase[(size_t)table[j] * d_head + i]);
        for (int g = 0; g < group; ++g) {
            const int head = kv_head * group + g;
            acc[g] += scores[((size_t)head * n_tokens + token) * kv_len + j] * v;
        }
    }

    for (int g = 0; g < group; ++g) {
        const int head = kv_head * group + g;
        out[((size_t)token * n_heads + head) * d_head + i] = acc[g];
    }
}

// Fused attention for decode: scores, softmax and the weighted sum in one pass.
//
// The three-kernel path writes the whole score matrix to HBM, reads it back to
// normalize, and reads it a third time to weight the values. At a batch of one
// that round trip is small in bytes and expensive in latency: three dependent
// launches per layer, 96 per step, and at that size every kernel is latency
// rather than bandwidth. Keeping the scores in shared memory removes all of it.
//
// Split over the key range like `attn_output_split_f32`, because one block per
// (head, token) is 32 blocks on a 48-SM device. Each block reduces its own
// chunk and hands back a partial sum with the running max and denominator, in
// the usual flash-attention form, for `attn_flash_reduce_f32` to combine.
//
//   partial_acc[chunk][token][head][d_head]  the unnormalized weighted sum
//   partial_ms [chunk][token][head][2]       {max, denominator}
extern "C" __global__ void attn_flash_f32(
    float* __restrict__ partial, int ms_off, const float* __restrict__ q,
    const __half* __restrict__ k_cache, const __half* __restrict__ v_cache,
    const int* __restrict__ seq_of, const int* __restrict__ positions,
    const int* __restrict__ slot_table, int table_stride, int n_heads,
    int n_kv_heads, int d_head, int n_slots, float scale, int chunk) {
    // [chunk] scores, [chunk] slot indices, then [subs][d_head] partial sums.
    extern __shared__ float sh[];
    float* __restrict__ partial_acc = partial;
    float* __restrict__ partial_ms = partial + ms_off;

    const int head = blockIdx.x;
    const int token = blockIdx.y;
    const int c = blockIdx.z;
    const int n_tokens = gridDim.y;
    const int tid = threadIdx.x;
    const int lane = tid % WARP_SIZE;
    const int warp = tid / WARP_SIZE;
    const int warps = blockDim.x / WARP_SIZE;
    // The value loop below derives its own split of the block; see `vsubs`.

    const int kv_head = head / (n_heads / n_kv_heads);
    const int last = positions[token];
    const int* table = slot_table + (size_t)seq_of[token] * table_stride;
    const float* qr = q + ((size_t)token * n_heads + head) * d_head;
    const __half* kbase = k_cache + (size_t)kv_head * n_slots * d_head;
    const __half* vbase = v_cache + (size_t)kv_head * n_slots * d_head;

    const int j0 = c * chunk;
    int j1 = j0 + chunk;
    if (j1 > last + 1) j1 = last + 1;
    const int n = j1 - j0;

    // Stage this chunk's slots first. Read straight from global they cost a
    // dependent load before every K and V fetch — the address of the row is
    // itself in memory — and that pointer chase is what held this kernel to a
    // fraction of the card's bandwidth. From shared memory the K/V loads have
    // their addresses immediately and pipeline.
    int* slots = (int*)(sh + chunk);
    float* red = sh + 2 * chunk;
    for (int j = tid; j < n; j += blockDim.x) slots[j] = table[j0 + j];
    __syncthreads();

    if (n <= 0) {
        if (tid < d_head) {
            partial_acc[(((size_t)c * n_tokens + token) * n_heads + head) * d_head
                        + tid] = 0.0f;
        }
        if (tid == 0) {
            float* ms = partial_ms
                      + (((size_t)c * n_tokens + token) * n_heads + head) * 2;
            ms[0] = -INFINITY;
            ms[1] = 0.0f;
        }
        return;
    }

    // Scores for this chunk, one warp per key.
    //
    // Four contiguous halves a lane rather than a stride of the warp: one
    // eight-byte load where the strided form issues four two-byte ones, and one
    // sixteen-byte load of the query beside it. `d_head` is a multiple of four
    // in every model this path serves; the tail keeps the rest correct.
    const int quads = d_head / 4;
    for (int j = warp; j < n; j += warps) {
        const __half* kr = kbase + (size_t)slots[j] * d_head;
        float acc = 0.0f;
        for (int i = lane; i < quads; i += WARP_SIZE) {
            const uint2 raw = *(const uint2*)(const void*)(kr + i * 4);
            const __half2* h = (const __half2*)(const void*)&raw;
            const float2 a = __half22float2(h[0]);
            const float2 b = __half22float2(h[1]);
            const float4 qv = *(const float4*)(const void*)(qr + i * 4);
            acc += qv.x * a.x + qv.y * a.y + qv.z * b.x + qv.w * b.y;
        }
        for (int i = quads * 4 + lane; i < d_head; i += WARP_SIZE) {
            acc += qr[i] * __half2float(kr[i]);
        }
        acc = warp_reduce_sum(acc);
        if (lane == 0) sh[j] = acc * scale;
    }
    __syncthreads();

    // Chunk max and denominator. Every thread ends up with both.
    float m = -INFINITY;
    for (int j = tid; j < n; j += blockDim.x) m = fmaxf(m, sh[j]);
    m = block_reduce_max(m);
    float l = 0.0f;
    for (int j = tid; j < n; j += blockDim.x) l += __expf(sh[j] - m);
    l = block_reduce_sum(l);

    // Exponentiate once per key rather than once per key and thread: the
    // accumulation below had every one of `d_head` threads evaluating the same
    // exponential.
    for (int j = tid; j < n; j += blockDim.x) sh[j] = __expf(sh[j] - m);
    __syncthreads();

    // Weighted sum, still unnormalized: the reduce kernel divides once it has
    // seen every chunk's max.
    // Eight halves a thread, which frees the rest of the block to take more of
    // the key range: sixteen lanes cover a 128-wide row, so a 128-thread block
    // walks eight slices of `j` at once instead of one. Two bytes a thread is
    // what held this loop, and `attn_output_f32`'s identical loop, to a
    // fraction of the card's bandwidth — 85.6 us against 35.5 once fixed there.
    const int vlanes = d_head / 8;
    const int vlane = tid % vlanes;
    const int vsub = tid / vlanes;
    const int vsubs = blockDim.x / vlanes;

    float acc8[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) acc8[e] = 0.0f;
    for (int j = vsub; j < n; j += vsubs) {
        const float s = sh[j];
        const uint4 raw = *(const uint4*)(const void*)(
            vbase + (size_t)slots[j] * d_head + vlane * 8);
        const __half2* h = (const __half2*)(const void*)&raw;
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const float2 f = __half22float2(h[e]);
            acc8[e * 2] += s * f.x;
            acc8[e * 2 + 1] += s * f.y;
        }
    }

    // `red` is `vsubs * d_head` floats, which the host sizes.
    float* mine = red + (size_t)vsub * d_head + vlane * 8;
#pragma unroll
    for (int e = 0; e < 8; ++e) mine[e] = acc8[e];
    __syncthreads();
    for (int st = vsubs >> 1; st > 0; st >>= 1) {
        if (vsub < st) {
            const float* other = mine + (size_t)st * d_head;
#pragma unroll
            for (int e = 0; e < 8; ++e) mine[e] += other[e];
        }
        __syncthreads();
    }
    if (vsub == 0) {
        float* dst = partial_acc
                   + (((size_t)c * n_tokens + token) * n_heads + head) * d_head
                   + vlane * 8;
#pragma unroll
        for (int e = 0; e < 8; ++e) dst[e] = mine[e];
    }
    if (tid == 0) {
        float* ms = partial_ms + (((size_t)c * n_tokens + token) * n_heads + head) * 2;
        ms[0] = m;
        ms[1] = l;
    }
}

/// Combine the chunks: rescale each by its own max against the global one.
// The combine, also writing the f16 copy the output projection reads.
//
// `hout` may be null, which is the batch-1 and mat-vec case: those paths take
// the f32. When it is not, this is the last kernel to touch the attention
// output before `o_proj`, so the conversion belongs here rather than in a
// `f32_to_f16` launch that reads the f32 back — the same trade
// `silu_mul_split_f16_f32` makes for the SwiGLU product, and the second of the
// two conversions the fused norm left behind.
extern "C" __global__ void attn_flash_reduce_f16_f32(
    float* __restrict__ out, __half* __restrict__ hout,
    const float* __restrict__ partial, int ms_off,
    int n_heads, int d_head, int n_tokens, int n_chunks) {
    const float* __restrict__ partial_acc = partial;
    const float* __restrict__ partial_ms = partial + ms_off;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n_tokens * n_heads * d_head;
    if (i >= total) return;
    const int head = (i / d_head) % n_heads;
    const int token = i / (d_head * n_heads);

    float m = -INFINITY;
    for (int c = 0; c < n_chunks; ++c) {
        m = fmaxf(m, partial_ms[(((size_t)c * n_tokens + token) * n_heads + head) * 2]);
    }
    if (m == -INFINITY) {
        out[i] = 0.0f;
        hout[i] = __float2half(0.0f);
        return;
    }
    float acc = 0.0f, denom = 0.0f;
    for (int c = 0; c < n_chunks; ++c) {
        const float* ms =
            partial_ms + (((size_t)c * n_tokens + token) * n_heads + head) * 2;
        if (ms[0] == -INFINITY) continue;
        const float w = __expf(ms[0] - m);
        denom += ms[1] * w;
        acc += partial_acc[(size_t)c * total + i] * w;
    }
    const float v = denom > 0.0f ? acc / denom : 0.0f;
    out[i] = v;
    hout[i] = __float2half(v);
}

extern "C" __global__ void attn_flash_reduce_f32(
    float* __restrict__ out, const float* __restrict__ partial, int ms_off,
    int n_heads, int d_head, int n_tokens, int n_chunks) {
    const float* __restrict__ partial_acc = partial;
    const float* __restrict__ partial_ms = partial + ms_off;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n_tokens * n_heads * d_head;
    if (i >= total) return;
    const int head = (i / d_head) % n_heads;
    const int token = i / (d_head * n_heads);

    float m = -INFINITY;
    for (int c = 0; c < n_chunks; ++c) {
        m = fmaxf(m, partial_ms[(((size_t)c * n_tokens + token) * n_heads + head) * 2]);
    }
    if (m == -INFINITY) {
        out[i] = 0.0f;
        return;
    }
    float acc = 0.0f, denom = 0.0f;
    for (int c = 0; c < n_chunks; ++c) {
        const float* ms =
            partial_ms + (((size_t)c * n_tokens + token) * n_heads + head) * 2;
        if (ms[0] == -INFINITY) continue;
        const float w = __expf(ms[0] - m);
        denom += ms[1] * w;
        acc += partial_acc[(size_t)c * total + i] * w;
    }
    out[i] = denom > 0.0f ? acc / denom : 0.0f;
}

// Fused decode attention, one block per KV head rather than per query head.
//
// Grouped-query attention has four query heads share each key/value pair, and
// `attn_flash_f32` pays for that three times over: it fetches V once per query
// head, it fetches K once per query head, and — because the weight is applied
// inside the accumulation loop — every one of its `d_head` threads evaluates
// the same exponential. Giving the group to one block fixes all three. The
// block holds one accumulator per query head in registers, so a value row is
// read once and used four times, the four score rows are softmaxed once by
// four warps, and the weights reach the accumulation loop already exponentiated
// through shared memory.
//
// The cost is grid width: eight KV heads instead of thirty-two query heads.
// Each block does four times the work, so the total is unchanged, but there are
// fewer concurrent memory streams to hide latency with — which is why this is
// a variant to measure rather than an obvious replacement.
//
// Requires `group == warps` and `d_head == blockDim.x`.
extern "C" __global__ void attn_flash_gqa_f32(
    float* __restrict__ partial, int ms_off, const float* __restrict__ q,
    const __half* __restrict__ k_cache, const __half* __restrict__ v_cache,
    const int* __restrict__ seq_of, const int* __restrict__ positions,
    const int* __restrict__ slot_table, int table_stride, int n_heads,
    int n_kv_heads, int d_head, int n_slots, float scale, int chunk) {
    // [group][chunk] weights, then [chunk] slot indices.
    extern __shared__ float sh[];

    const int kv_head = blockIdx.x;
    const int token = blockIdx.y;
    const int c = blockIdx.z;
    const int n_tokens = gridDim.y;
    const int tid = threadIdx.x;
    const int lane = tid % WARP_SIZE;
    const int warp = tid / WARP_SIZE;
    const int group = n_heads / n_kv_heads;

    const int last = positions[token];
    const int* table = slot_table + (size_t)seq_of[token] * table_stride;
    const __half* kbase = k_cache + (size_t)kv_head * n_slots * d_head;
    const __half* vbase = v_cache + (size_t)kv_head * n_slots * d_head;

    const int j0 = c * chunk;
    int j1 = j0 + chunk;
    if (j1 > last + 1) j1 = last + 1;
    const int n = j1 - j0;

    int* slots = (int*)(sh + (size_t)group * chunk);
    for (int j = tid; j < n; j += blockDim.x) slots[j] = table[j0 + j];
    __syncthreads();

    if (n <= 0) {
        for (int g = 0; g < group; ++g) {
            const int head = kv_head * group + g;
            partial[(((size_t)c * n_tokens + token) * n_heads + head) * d_head + tid] =
                0.0f;
            if (tid == 0) {
                float* ms = partial + ms_off
                          + (((size_t)c * n_tokens + token) * n_heads + head) * 2;
                ms[0] = -INFINITY;
                ms[1] = 0.0f;
            }
        }
        return;
    }

    // One warp per *key*, holding that key's row across the whole group.
    //
    // A warp per query head — which is what this was — has every warp in the
    // block read every K row, so K crosses the bus `group` times and only V is
    // shared. That is the half of the grouping this kernel was not doing, and
    // it is why the unfused `attn_scores_gqa_f32`, which holds a K row in
    // registers for all four heads, beat it on the score side.
    const int head = kv_head * group + warp;
    const int warps = blockDim.x / WARP_SIZE;
    float m = -INFINITY, l = 0.0f;
    const int per_lane = d_head / WARP_SIZE;
    const int koff = lane * per_lane;

    for (int j = tid; j < group * chunk; j += blockDim.x) {
        if (j % chunk >= n) sh[j] = -INFINITY;
    }
    __syncthreads();

    for (int j = warp; j < n; j += warps) {
        const __half* kr = kbase + (size_t)slots[j] * d_head;
        float kv[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) kv[e] = 0.0f;
        if (per_lane == 4) {
            const uint2 raw = *(const uint2*)(const void*)(kr + koff);
            const __half2* h = (const __half2*)(const void*)&raw;
            const float2 a = __half22float2(h[0]);
            const float2 b = __half22float2(h[1]);
            kv[0] = a.x;
            kv[1] = a.y;
            kv[2] = b.x;
            kv[3] = b.y;
        } else {
            const unsigned raw = *(const unsigned*)(const void*)(kr + koff);
            const float2 a = __half22float2(*(const __half2*)(const void*)&raw);
            kv[0] = a.x;
            kv[1] = a.y;
        }
        for (int g = 0; g < group; ++g) {
            const float* qq =
                q + ((size_t)token * n_heads + kv_head * group + g) * d_head + koff;
            float acc;
            if (per_lane == 4) {
                const float4 qv = *(const float4*)(const void*)qq;
                acc = qv.x * kv[0] + qv.y * kv[1] + qv.z * kv[2] + qv.w * kv[3];
            } else {
                const float2 qv = *(const float2*)(const void*)qq;
                acc = qv.x * kv[0] + qv.y * kv[1];
            }
            acc = warp_reduce_sum(acc);
            if (lane == 0) sh[(size_t)g * chunk + j] = acc * scale;
        }
    }
    __syncthreads();

    if (warp < group) {
        for (int j = lane; j < chunk; j += WARP_SIZE) {
            m = fmaxf(m, sh[(size_t)warp * chunk + j]);
        }
        m = warp_reduce_max(m);
        for (int j = lane; j < chunk; j += WARP_SIZE) {
            const float p = (j < n) ? __expf(sh[(size_t)warp * chunk + j] - m) : 0.0f;
            sh[(size_t)warp * chunk + j] = p;
            l += p;
        }
        l = warp_reduce_sum(l);
        if (lane == 0) {
            float* ms = partial + ms_off
                      + (((size_t)c * n_tokens + token) * n_heads + head) * 2;
            ms[0] = m;
            ms[1] = l;
        }
    }
    __syncthreads();

    // One pass over V for the whole group — which is the point of this kernel,
    // and what `attn_flash_f32` cannot do: there K and V cross the bus once per
    // *query* head, four times over for Llama-3.1's 32 heads on 8 KV heads.
    //
    // Written scalar this loop was `n` two-byte loads with nothing in flight
    // but the current one, and it dominated the kernel. Eight halves a thread
    // covers `d_head` in sixteen lanes, which leaves the block's other threads
    // to take their own stride of the key range: eight independent chains
    // instead of one, and sixteen-byte loads instead of two.
    const int vlanes = d_head / 8;
    const int vlane = tid % vlanes;
    const int vsub = tid / vlanes;
    const int vsubs = blockDim.x / vlanes;
    // `group` is capped at eight by the host.
    float acc[8][8];
#pragma unroll
    for (int g = 0; g < 8; ++g) {
#pragma unroll
        for (int e = 0; e < 8; ++e) acc[g][e] = 0.0f;
    }
    for (int j = vsub; j < n; j += vsubs) {
        const uint4 raw = *(const uint4*)(const void*)(
            vbase + (size_t)slots[j] * d_head + vlane * 8);
        const __half2* h = (const __half2*)(const void*)&raw;
        float v[8];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const float2 f = __half22float2(h[e]);
            v[e * 2] = f.x;
            v[e * 2 + 1] = f.y;
        }
        for (int g = 0; g < group; ++g) {
            const float w = sh[(size_t)g * chunk + j];
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[g][e] += w * v[e];
        }
    }

    // One group at a time through a `vsubs x d_head` scratch, so the reduction
    // costs one buffer rather than one per query head.
    float* red = (float*)(void*)(slots + chunk);
    for (int g = 0; g < group; ++g) {
        __syncthreads();
        float* mine = red + (size_t)vsub * d_head + vlane * 8;
#pragma unroll
        for (int e = 0; e < 8; ++e) mine[e] = acc[g][e];
        // Bottom-up, so `vsubs` need not be a power of two: a group of seven
        // query heads on 64-wide keys gives 28 slices, and a halving tree
        // would drop four of them.
        for (int st = 1; st < vsubs; st <<= 1) {
            __syncthreads();
            if ((vsub & ((st << 1) - 1)) == 0 && vsub + st < vsubs) {
                const float* other = mine + (size_t)st * d_head;
#pragma unroll
                for (int e = 0; e < 8; ++e) mine[e] += other[e];
            }
        }
        __syncthreads();
        if (vsub == 0) {
            const int h = kv_head * group + g;
            float* dst = partial
                       + (((size_t)c * n_tokens + token) * n_heads + h) * d_head
                       + vlane * 8;
#pragma unroll
            for (int e = 0; e < 8; ++e) dst[e] = mine[e];
        }
    }
}

// Fused decode attention that tiles the key range instead of holding it.
//
// The three-kernel path is 47.5 us a layer at the shape a decode step actually
// runs — batch 32, 384 of history, 32 query heads over 8 KV heads of 128 —
// against `vllm_flash_attn`'s 33.3 us measured inside its own engine. Both
// numbers took some finding: the 85.8 us this file's comments quote for tuili
// was taken under `TUILI_PROFILE`, which serializes and puts an event pair
// around every launch, and the 58.1 us they quote for vLLM came from a Python
// harness whose own overhead is 28 us — it reports the same 58 us at a history
// of 128, where the kernel cannot possibly be doing that much work.
//
// So the gap is real and it is 1.4x. Where it comes from:
//
//   * The score matrix is written to HBM, read back to normalize, and read a
//     third time to weight the values — 6.3 MB a layer against 50 MB of KV.
//   * `attn_output_v4_f32` fetches V once per *query* head, four times over.
//     L2 absorbs it at this size, but it is still four times the requests.
//   * Three launches where one would do, each with its own tail.
//
// `attn_flash_f32` already fuses and loses anyway; the comment above
// `attn_flash_split` records seven shapes of it. Its problem is that a block
// holds a whole chunk's scores in shared memory, which caps how many blocks an
// SM can hold, and its phases leave most of the block idle in turn. This one
// keeps nothing but a 32-key tile:
//
//   * A block owns a (KV head, token, chunk) and `group * 32` threads.
//   * Warp `g` owns query head `g` for the whole kernel — in phase one lane `j`
//     scores key `j`, in phase two the warp softmaxes its own 32 scores with
//     shuffles, and in phase three lane `c` owns dimensions `4c..4c+4` of the
//     accumulator. Every phase uses every thread.
//   * The scores never reach shared memory, let alone HBM: they are one float
//     per lane, and phase three reads lane `j`'s with `__shfl_sync`.
//   * K and V cross the bus once for the whole group, which is what the
//     unfused path only manages on K.
//
// Both halves are staged, and V is staged even though it looks as though it
// need not be: phase three has a warp's 32 lanes read one contiguous 256-byte
// V row between them, which is a perfectly coalesced global read. Reading it
// there directly was measured — 59.6 us against 56.6 at a history of 512, and
// worse at every other shape — because the four warps of a group each want the
// same row, so it crosses L1 four times instead of being fetched once by the
// tile load. Shared is 2 KB of query rows and one 32-key tile of each, 19.5 KB
// at Llama-3.1's shape, which is what lets five blocks share an SM.
//
// Partials in `attn_flash_reduce_f32`'s layout, so the combine pass is shared
// with the older fused path.
/* The `__hfma2` score loop: a third of the instructions, an f16 accumulator
   flushed every eight products, and an answer that is close rather than
   identical — 2.1e-4 absolute on outputs of order 0.26, which is 0.08% and
   which `attn_decode_matches_the_three_kernels` rejects at its 2e-4 bound.
 
   Measured and left off, because the instructions were not what the engine was
   waiting for. The probe says 55.3 us a layer against 59.2 at history 512 — but
   that harness varies 3% run to run (the same build measured 57.4 earlier
   today), and in the served engine it is *nothing*: 5009 tok/s against 5012,
   `layers_ms` 5.114 against 5.120.
 
   That is the third way of making this kernel's arithmetic cheaper, after
   `m16n8k16` and breaking the FMA chain, and the third that does not move the
   engine. The 7.3 us a layer that *deleting* the arithmetic removes is real and
   does not convert: what is left when the multiply gets cheaper is the latency
   the multiply was hiding. Whatever attention is short of here, it is not
   instruction issue. */
#define ATTN_DECODE_H2 0
#define ATTN_DECODE_TILE 16
#define ATTN_DECODE_LPK (WARP_SIZE / ATTN_DECODE_TILE)
#define ATTN_DECODE_PAD 8

extern "C" __global__ void attn_decode_gqa_f32(
    float* __restrict__ partial, int ms_off, float* __restrict__ out,
    int single, const float* __restrict__ q, const __half* __restrict__ k_cache,
    const __half* __restrict__ v_cache, const int* __restrict__ seq_of,
    const int* __restrict__ positions, const int* __restrict__ slot_table,
    int table_stride, int n_heads, int n_kv_heads, int d_head, int n_slots,
    float scale, int chunk, int kv_len, int group) {
    const int kv_head = blockIdx.x;
    const int token = blockIdx.y;
    const int c = blockIdx.z;
    const int n_tokens = gridDim.y;
    const int lane = threadIdx.x % WARP_SIZE;
    const int g = threadIdx.x / WARP_SIZE;

    const int row = d_head + ATTN_DECODE_PAD;
    extern __shared__ char attn_smem[];
    float* sq = (float*)attn_smem;
    __half* sk = (__half*)(sq + group * d_head);
    __half* sv = sk + ATTN_DECODE_TILE * row;
#if ATTN_DECODE_H2
    /* Q again as halves, for the `__hfma2` score loop below: one instruction a
       pair of products instead of a conversion and two FMAs. A kilobyte at
       Llama-3.1's shape, written once a block. */
    __half* sqh = sv + ATTN_DECODE_TILE * row;
#endif

    for (int i = threadIdx.x; i < group * d_head; i += blockDim.x) {
        const float qv =
            q[((size_t)token * n_heads + kv_head * group + i / d_head) * d_head
              + i % d_head];
        sq[i] = qv;
#if ATTN_DECODE_H2
        sqh[i] = __float2half(qv);
#endif
    }

    const int last = positions[token];
    const int* table = slot_table + (size_t)seq_of[token] * table_stride;
    const __half* kbase = k_cache + (size_t)kv_head * n_slots * d_head;
    const __half* vbase = v_cache + (size_t)kv_head * n_slots * d_head;

    const int begin = c * chunk;
    int end = begin + chunk;
    if (end > kv_len) end = kv_len;
    // The mask is per token, so a block whose whole chunk is in the future of
    // this token has nothing to do — but it still has to record that, or the
    // combine pass would read whatever was in the partial buffer.
    if (end > last + 1) end = last + 1;

    float acc[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[e] = 0.0f;
    float m_run = -INFINITY, l_run = 0.0f;

    const int per_lane = d_head / WARP_SIZE;  // dimensions a lane accumulates
    const int quads = d_head / 8;             // 16-byte loads across a row

    // A tile's loads are issued before the tile before it is spent, and wait
    // in registers until there is somewhere to put them.
    //
    // Without this a block spends about 2% of its time with loads outstanding
    // and the rest on arithmetic that has nothing in flight behind it, which is
    // why the kernel sat at 1188 GB/s of KV where reading the same bytes and
    // discarding them reaches 1440. Double buffering in *shared* fixes the same
    // problem and costs more than it saves — 36.9 KB a block against 19.5 drops
    // an SM from five resident blocks to two, and the kernel from 56.9 us to
    // 76.6. Registers are what this card has spare.
    //
    // Four slots covers a tile when `group * 32` threads make at most four
    // passes over `TILE * quads` loads, which is every grouped-query shape with
    // a group of four or wider. Narrower groups load straight to shared.
    const bool pre = ATTN_DECODE_TILE * quads <= 4 * (int)blockDim.x;
    uint4 pk[4], pv[4];
    bool live[4];

#define ATTN_DECODE_TO_REG(base_)                                              \
    {                                                                          \
        const int n_ = min(ATTN_DECODE_TILE, end - (base_));                   \
        _Pragma("unroll") for (int i_ = 0; i_ < 4; ++i_) {                     \
            const int e_ = threadIdx.x + i_ * blockDim.x;                      \
            live[i_] = e_ < n_ * quads;                                        \
            if (live[i_]) {                                                    \
                const size_t off_ =                                            \
                    (size_t)table[(base_) + e_ / quads] * d_head               \
                    + (e_ % quads) * 8;                                        \
                pk[i_] = *(const uint4*)(const void*)(kbase + off_);           \
                pv[i_] = *(const uint4*)(const void*)(vbase + off_);           \
            }                                                                  \
        }                                                                      \
    }

#define ATTN_DECODE_TO_SHARED()                                                \
    {                                                                          \
        _Pragma("unroll") for (int i_ = 0; i_ < 4; ++i_) {                     \
            if (live[i_]) {                                                    \
                const int e_ = threadIdx.x + i_ * blockDim.x;                  \
                const int at_ = (e_ / quads) * row + (e_ % quads) * 8;         \
                *(uint4*)(void*)(sk + at_) = pk[i_];                           \
                *(uint4*)(void*)(sv + at_) = pv[i_];                           \
            }                                                                  \
        }                                                                      \
    }

#define ATTN_DECODE_DIRECT(base_)                                              \
    {                                                                          \
        const int n_ = min(ATTN_DECODE_TILE, end - (base_));                   \
        for (int e_ = threadIdx.x; e_ < n_ * quads; e_ += blockDim.x) {        \
            const int r_ = e_ / quads, w_ = e_ % quads;                        \
            const size_t off_ = (size_t)table[(base_) + r_] * d_head + w_ * 8; \
            const int at_ = r_ * row + w_ * 8;                                 \
            *(uint4*)(void*)(sk + at_) =                                       \
                *(const uint4*)(const void*)(kbase + off_);                    \
            *(uint4*)(void*)(sv + at_) =                                       \
                *(const uint4*)(const void*)(vbase + off_);                    \
        }                                                                      \
    }

    if (begin < end) {
        if (pre) {
            ATTN_DECODE_TO_REG(begin)
            ATTN_DECODE_TO_SHARED()
        } else {
            ATTN_DECODE_DIRECT(begin)
        }
    }
    __syncthreads();

    for (int base = begin; base < end; base += ATTN_DECODE_TILE) {
        const int n = min(ATTN_DECODE_TILE, end - base);
        const int next = base + ATTN_DECODE_TILE;
        // Issued here, consumed after this tile's arithmetic.
        if (pre && next < end) ATTN_DECODE_TO_REG(next)

        // Phase one: a *pair* of lanes scores one key, each taking half the
        // head, and the halves meet through one shuffle.
        //
        // A lane to a key would want a 32-key tile, and the tile is what sets
        // occupancy: 19.5 KB a block holds five blocks to an SM where the
        // probe, which stages nothing, holds sixteen and reads the same bytes
        // 35% faster. A narrower tile costs shuffles — `log2(LPK)` to add the
        // partial dots, one to move key `j`'s score into lane `j`, where phase
        // two and phase three expect it — and buys resident blocks. Measured
        // per layer at the engine's median history of 384, against the three
        // kernels' 47.6 us: 32 keys 49.7, 16 keys 45.9, 8 keys below.
        const int kj = lane / ATTN_DECODE_LPK;
        float s = -INFINITY;
        {
            const __half* kr = sk + kj * row;
            const int part = lane % ATTN_DECODE_LPK;
            const float* qr = sq + g * d_head + part * (d_head / ATTN_DECODE_LPK);
            const __half* kh = kr + part * (d_head / ATTN_DECODE_LPK);
            // One accumulator on purpose. This is a chain of 64 dependent FMAs
            // and breaking it into four independent ones — which is the textbook
            // fix, and which is where the 7.3 us a layer that deleting the
            // arithmetic removes appears to live — measures *worse* on both
            // cards: 65.5 us a layer against 57.4 on a Blackwell, 299.9 against
            // 289.2 on an A4000. Holding four accumulators and four `float2`
            // temporaries live costs registers, and this kernel is waiting on
            // memory latency rather than on the chain, so occupancy is what it
            // spends them on. The arithmetic is exposed, but not because of the
            // chain.
            float dot = 0.0f;
#if ATTN_DECODE_H2
            /* `__hfma2` retires two products an instruction where the f32 form
               spends a conversion and two FMAs on the same pair — three
               instructions to one. The f16 accumulator is flushed to f32 every
               eight products so the summation depth stays short: the score is
               O(10) and f16 resolves 0.004 there, so eight of them is about
               0.016 before the 0.088 scale, which moves an attention weight by
               a part in a thousand. Not bit-identical, which is why this is a
               switch and its output is checked rather than assumed. */
            const __half2* qh2 =
                (const __half2*)(const void*)(sqh + g * d_head
                                              + part * (d_head / ATTN_DECODE_LPK));
            __half2 acc2 = __floats2half2_rn(0.0f, 0.0f);
            for (int w = 0; w < quads / ATTN_DECODE_LPK; ++w) {
                const uint4 raw = *(const uint4*)(const void*)(kh + w * 8);
                const __half2* h2 = (const __half2*)(const void*)&raw;
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    acc2 = __hfma2(qh2[w * 4 + u], h2[u], acc2);
                }
                {
                    const float2 f = __half22float2(acc2);
                    dot += f.x + f.y;
                    acc2 = __floats2half2_rn(0.0f, 0.0f);
                }
            }
            {
                const float2 f = __half22float2(acc2);
                dot += f.x + f.y;
            }
#else
            for (int w = 0; w < quads / ATTN_DECODE_LPK; ++w) {
                const uint4 raw = *(const uint4*)(const void*)(kh + w * 8);
                const __half2* h2 = (const __half2*)(const void*)&raw;
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    const float2 f = __half22float2(h2[u]);
                    dot += qr[w * 8 + 2 * u] * f.x + qr[w * 8 + 2 * u + 1] * f.y;
                }
            }
#endif
#pragma unroll
            for (int st = 1; st < ATTN_DECODE_LPK; st <<= 1) {
                dot += __shfl_xor_sync(0xffffffff, dot, st, WARP_SIZE);
            }
            s = dot * scale;
        }
        // Lane `j` takes key `j`'s score; the upper half of the warp has no key
        // and stays masked, which the softmax below already handles.
        s = __shfl_sync(0xffffffff, s, (lane % ATTN_DECODE_TILE) * ATTN_DECODE_LPK,
                        WARP_SIZE);
        if (lane >= n) s = -INFINITY;

        // Phase two: the warp's own 32 scores, softmaxed in registers.
        const float m_tile = warp_reduce_max(s);
        const float m_new = fmaxf(m_run, m_tile);
        const float p = (s == -INFINITY) ? 0.0f : __expf(s - m_new);
        const float sum = warp_reduce_sum(p);
        // `m_run` starts at -inf and `exp(-inf - m_new)` is 0, which is the
        // right correction for an empty running sum — but only when `m_new` is
        // finite. A tile entirely in the future leaves both at -inf.
        const float corr = (m_run == -INFINITY) ? 0.0f : __expf(m_run - m_new);
        if (m_new > -INFINITY) {
            l_run = l_run * corr + sum;
            m_run = m_new;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                if (e < per_lane) acc[e] *= corr;
            }

            // Phase three: lane `c` owns dimensions `per_lane*c ..`, and reads
            // lane `j`'s weight straight out of its register.
            //
            // The width matters more here than anywhere else in the kernel.
            // A lane touches `per_lane` values of every key in the tile, so an
            // element-at-a-time loop is `n * per_lane` two-byte shared loads —
            // 128 of them a tile at Llama-3.1's shape, against 16 sixteen-byte
            // loads for the whole of phase one. Reading the lane's slice as one
            // eight-byte word took the kernel from 854 GB/s of KV to 1188.
            if (per_lane == 4) {
                for (int j = 0; j < n; ++j) {
                    const float w = __shfl_sync(0xffffffff, p, j, WARP_SIZE);
                    const uint2 raw =
                        *(const uint2*)(const void*)(sv + j * row + lane * 4);
                    const __half2* h2 = (const __half2*)(const void*)&raw;
                    const float2 a = __half22float2(h2[0]);
                    const float2 b = __half22float2(h2[1]);
                    acc[0] += w * a.x;
                    acc[1] += w * a.y;
                    acc[2] += w * b.x;
                    acc[3] += w * b.y;
                }
            } else {
                for (int j = 0; j < n; ++j) {
                    const float w = __shfl_sync(0xffffffff, p, j, WARP_SIZE);
                    const __half* vr = sv + j * row + lane * per_lane;
#pragma unroll
                    for (int e = 0; e < 8; ++e) {
                        if (e < per_lane) acc[e] += w * __half2float(vr[e]);
                    }
                }
            }
        }
        __syncthreads();
        if (next < end) {
            if (pre) {
                ATTN_DECODE_TO_SHARED()
            } else {
                ATTN_DECODE_DIRECT(next)
            }
            __syncthreads();
        }
    }

    const int head = kv_head * group + g;
    // One chunk covers the whole range, so there is nothing to combine and the
    // normalized answer can go straight out — which saves the partial buffer
    // its 3.2 MB a layer and the combine pass its launch.
    if (single) {
        float* dst = out + ((size_t)token * n_heads + head) * d_head + lane * per_lane;
        const float inv = l_run > 0.0f ? 1.0f / l_run : 0.0f;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            if (e < per_lane) dst[e] = acc[e] * inv;
        }
        return;
    }
    float* dst = partial
               + (((size_t)c * n_tokens + token) * n_heads + head) * d_head
               + lane * per_lane;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        if (e < per_lane) dst[e] = acc[e];
    }
    if (lane == 0) {
        float* ms = partial + ms_off
                  + (((size_t)c * n_tokens + token) * n_heads + head) * 2;
        ms[0] = m_run;
        ms[1] = l_run;
    }
}

// What the KV cache can be read at, with nothing else happening.
//
// Both the three-kernel path and `attn_decode_gqa_f32` sit around 1100 GB/s at
// a batch of 32, and that is either a defect in both or the shape of the
// problem. This reads exactly what they read — a slot table, then one 256-byte
// row per key from each of K and V, over the same grid — and does nothing with
// it beyond enough arithmetic to keep the loads from being elided. Whatever it
// measures is the ceiling; anything at it is finished.
extern "C" __global__ void attn_kv_probe_f32(float* __restrict__ sink,
                                             const __half* __restrict__ k_cache,
                                             const __half* __restrict__ v_cache,
                                             const int* __restrict__ seq_of,
                                             const int* __restrict__ positions,
                                             const int* __restrict__ slot_table,
                                             int table_stride, int n_kv_heads,
                                             int d_head, int n_slots, int kv_len,
                                             int chunk) {
    const int kv_head = blockIdx.x;
    const int token = blockIdx.y;
    const int begin = blockIdx.z * chunk;
    int end = begin + chunk;
    if (end > kv_len) end = kv_len;
    const int last = positions[token];
    if (end > last + 1) end = last + 1;

    const int* table = slot_table + (size_t)seq_of[token] * table_stride;
    const __half* kbase = k_cache + (size_t)kv_head * n_slots * d_head;
    const __half* vbase = v_cache + (size_t)kv_head * n_slots * d_head;
    const int quads = d_head / 8;

    float acc = 0.0f;
    for (int e = begin * quads + threadIdx.x; e < end * quads; e += blockDim.x) {
        const size_t off = (size_t)table[e / quads] * d_head + (e % quads) * 8;
        const uint4 kk = *(const uint4*)(const void*)(kbase + off);
        const uint4 vv = *(const uint4*)(const void*)(vbase + off);
        acc += (float)(kk.x ^ vv.x) + (float)(kk.w ^ vv.w);
    }
    // Never true, and the compiler cannot know that.
    if (acc == 1.2345e-30f) sink[0] = acc;
}

// Decode attention on the tensor cores, in FlashAttention-2's decomposition.
//
// `attn_decode_gqa_f32` above spends 7.3 of its 46.1 us a layer on arithmetic —
// measured by deleting it, which leaves 38.8 — and every one of those FMAs is
// preceded by a `__half22float2`. `m16n8k16` takes f16 operands and accumulates
// in f32, so it removes the conversion and the multiply together: sixteen MMA
// instructions where a warp ran two hundred and thirty.
//
// The decomposition is vLLM's, which its trace names
// `Flash_fwd_kernel_traits<128, 64, 128, 4>` — a 64-key tile, four warps, and
// the query group packed into the MMA's `m`. Four of sixteen rows are useful at
// Llama-3.1's group of four, so three quarters of the tensor core is wasted;
// `mmqnm_*` established that the tensor core is free, and vLLM wastes it the
// same way.
//
// What each warp owns:
//
//   * Sixteen of the tile's sixty-four keys, and the *whole* query group. So a
//     warp is an independent split of the key range with its own running max,
//     denominator and output — no cross-warp softmax, one combine at the end.
//   * `S = Q Kᵀ` as two `m16n8k16` MMAs a k-step: A is the group's query rows,
//     B is `sk[key][dim]` *unchanged*, because a B fragment is indexed
//     `[col][k]` and the key tile already is.
//   * `O += P V` likewise, except a B fragment now wants `[dim][key]`, so V is
//     transposed on the way into shared.
//
// It is off by default, and the reason is precision rather than speed. A tensor
// core takes f16 operands, so the softmax weights go through half on their way
// into the value product where the scalar kernel keeps them in f32 —
// FlashAttention makes the same trade. That is about 6e-4 relative on an output
// element, which is small until it meets `tests/batching.rs`: the chunk count
// depends on the batch width, so the summation order does too, and an error ten
// times larger than the scalar path's is enough to flip a greedy token between a
// batched and a solo decode. The engine documents that invariance, so this path
// stays opt-in until the split is made batch-independent or the invariance is
// renegotiated.
//
// The one gift in the layout: the `S` accumulator lands in exactly the registers
// the `P` A-fragment wants. Lane `l` holds rows `l/4` and columns `(l%4)*2+{0,1}`
// of each 8-key tile, and an A fragment wants rows `l/4` and k `(l%4)*2+{0,1}`
// and `+8` — which is the first n-tile and the second. No shuffle, no round trip
// through shared; pack two floats into two halves and go.
#define ATTN_MMA_TILE 64
#define ATTN_MMA_WK 16
#define ATTN_MMA_KPAD 8
/* Two rather than eight: the transposed V is read four bytes at a time at a
   stride of this, and 66 halves puts consecutive lanes two banks apart where 72
   would put them on the same one. */
#define ATTN_MMA_VPAD 2

extern "C" __global__ void attn_decode_mma_f32(
    float* __restrict__ partial, int ms_off, float* __restrict__ out, int single,
    const float* __restrict__ q, const __half* __restrict__ k_cache,
    const __half* __restrict__ v_cache, const int* __restrict__ seq_of,
    const int* __restrict__ positions, const int* __restrict__ slot_table,
    int table_stride, int n_heads, int n_kv_heads, int d_head, int n_slots,
    float scale, int chunk, int kv_len, int group) {
    const int kv_head = blockIdx.x;
    const int token = blockIdx.y;
    const int c = blockIdx.z;
    const int n_tokens = gridDim.y;
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
    const int nwarps = blockDim.x / WARP_SIZE;

    const int krow = d_head + ATTN_MMA_KPAD;
    const int vrow = ATTN_MMA_TILE + ATTN_MMA_VPAD;
    const int quads = d_head / 8;
    const int ksteps = d_head / 16;
    const int ntiles = d_head / 8;  // n-tiles of the PV product

    extern __shared__ char attn_mma_smem[];
    __half* sq = (__half*)attn_mma_smem;
    __half* sk = sq + 16 * krow;
    __half* svt = sk + ATTN_MMA_TILE * krow;
    /* The combine buffer aliases the key tile: nothing reads a tile after the
       loop, and `nwarps * group * d_head` floats is smaller than it. */
    float* sacc = (float*)sk;
    float* sml = (float*)svt;  // only after the loop, likewise

    // The group's query rows as f16, the rest of the MMA's sixteen zeroed.
    for (int i = threadIdx.x; i < 16 * d_head; i += blockDim.x) {
        const int r = i / d_head, d = i % d_head;
        float v = 0.0f;
        if (r < group) {
            v = q[((size_t)token * n_heads + kv_head * group + r) * d_head + d];
        }
        sq[r * krow + d] = __float2half(v);
    }
    __syncthreads();

    const int ar = mma_a_row(lane);
    const int bc = mma_b_col(lane);
    const int k0 = mma_k0_f16(lane);
    const int cr = mma_c_row(lane);
    const int cc = mma_c_col(lane);

    // The query fragments, once for the whole kernel.
    mma_a_f16 qa[8];
#pragma unroll
    for (int t = 0; t < 8; ++t) {
        if (t >= ksteps) break;
        const __half* lo = sq + ar * krow + t * 16 + k0;
        const __half* hi = sq + (ar + 8) * krow + t * 16 + k0;
        qa[t].x[0] = *(const unsigned*)(const void*)lo;
        qa[t].x[1] = *(const unsigned*)(const void*)hi;
        qa[t].x[2] = *(const unsigned*)(const void*)(lo + 8);
        qa[t].x[3] = *(const unsigned*)(const void*)(hi + 8);
    }

    const int last = positions[token];
    const int* table = slot_table + (size_t)seq_of[token] * table_stride;
    const __half* kbase = k_cache + (size_t)kv_head * n_slots * d_head;
    const __half* vbase = v_cache + (size_t)kv_head * n_slots * d_head;

    const int begin = c * chunk;
    int end = begin + chunk;
    if (end > kv_len) end = kv_len;
    if (end > last + 1) end = last + 1;

    mma_c_f32 o[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        o[i].x[0] = 0.0f;
        o[i].x[1] = 0.0f;
        o[i].x[2] = 0.0f;
        o[i].x[3] = 0.0f;
    }
    float m_run = -INFINITY, l_run = 0.0f;

    for (int base = begin; base < end; base += ATTN_MMA_TILE) {
        const int n = min(ATTN_MMA_TILE, end - base);

        // K straight, V transposed. Sixteen threads cover a row of either.
        for (int e = threadIdx.x; e < n * quads; e += blockDim.x) {
            const int r = e / quads, w8 = e % quads;
            const size_t off = (size_t)table[base + r] * d_head + w8 * 8;
            *(uint4*)(void*)(sk + r * krow + w8 * 8) =
                *(const uint4*)(const void*)(kbase + off);
            uint4 vv = *(const uint4*)(const void*)(vbase + off);
            const __half* vh = (const __half*)(const void*)&vv;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                svt[(w8 * 8 + i) * vrow + r] = vh[i];
            }
        }
        __syncthreads();

        const int k_lo = warp * ATTN_MMA_WK;
        if (k_lo < n) {
            // S = Q Kᵀ for this warp's sixteen keys, two 8-key tiles.
            mma_c_f32 s2[2];
#pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                s2[nt].x[0] = 0.0f;
                s2[nt].x[1] = 0.0f;
                s2[nt].x[2] = 0.0f;
                s2[nt].x[3] = 0.0f;
                const int key0 = k_lo + nt * 8;
#pragma unroll
                for (int t = 0; t < 8; ++t) {
                    if (t >= ksteps) break;
                    mma_b_f16 b;
                    const __half* bp = sk + (key0 + bc) * krow + t * 16 + k0;
                    b.x[0] = *(const unsigned*)(const void*)bp;
                    b.x[1] = *(const unsigned*)(const void*)(bp + 8);
                    mma_f16(s2[nt], qa[t], b);
                }
            }

            // This lane's two scores per tile belong to head `cr`, keys
            // `k_lo + nt*8 + cc + {0,1}`. Past the tile's live keys they are
            // whatever the last tile left in shared, so mask them.
            float sv[4];
#pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const int key = k_lo + nt * 8 + cc + e;
                    sv[nt * 2 + e] =
                        (cr < group && key < n) ? s2[nt].x[e] * scale : -INFINITY;
                }
            }

            // Softmax over this warp's own keys, per head. A head's sixteen
            // scores live in the four lanes with the same `cr`, so the
            // reduction is two shuffles inside the warp.
            float m_own = fmaxf(fmaxf(sv[0], sv[1]), fmaxf(sv[2], sv[3]));
            m_own = fmaxf(m_own, __shfl_xor_sync(0xffffffff, m_own, 1, WARP_SIZE));
            m_own = fmaxf(m_own, __shfl_xor_sync(0xffffffff, m_own, 2, WARP_SIZE));
            const float m_new = fmaxf(m_run, m_own);
            float p[4], psum = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                p[i] = (sv[i] == -INFINITY) ? 0.0f : __expf(sv[i] - m_new);
                psum += p[i];
            }
            psum += __shfl_xor_sync(0xffffffff, psum, 1, WARP_SIZE);
            psum += __shfl_xor_sync(0xffffffff, psum, 2, WARP_SIZE);

            // No guard on `m_new` here, and that is not an oversight:
            // `mma.sync.aligned` needs every lane of the warp, and `m_new` is
            // *not* warp-uniform — the lanes holding the MMA's padding rows
            // (`cr >= group`) never see a live score, so a
            // `if (m_new > -INFINITY)` around the MMA deadlocks the warp. It
            // does not need one. A dead lane has `p == 0` and `corr == 0`, so
            // the rescale is a no-op on an accumulator that is already zero and
            // the MMA adds zero.
            {
                const float corr =
                    (m_run == -INFINITY) ? 0.0f : __expf(m_run - m_new);
                l_run = l_run * corr + psum;
                m_run = m_new;
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    if (i >= ntiles) break;
                    o[i].x[0] *= corr;
                    o[i].x[1] *= corr;
                    o[i].x[2] *= corr;
                    o[i].x[3] *= corr;
                }

                // P as an A fragment: the S accumulator already has it.
                mma_a_f16 pa;
                const __half2 p01 = __floats2half2_rn(p[0], p[1]);
                const __half2 p23 = __floats2half2_rn(p[2], p[3]);
                pa.x[0] = *(const unsigned*)(const void*)&p01;
                pa.x[2] = *(const unsigned*)(const void*)&p23;
                pa.x[1] = 0;
                pa.x[3] = 0;
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    if (i >= ntiles) break;
                    mma_b_f16 b;
                    const __half* bp = svt + (i * 8 + bc) * vrow + k_lo + k0;
                    b.x[0] = *(const unsigned*)(const void*)bp;
                    b.x[1] = *(const unsigned*)(const void*)(bp + 8);
                    mma_f16(o[i], pa, b);
                }
            }
        }
        __syncthreads();
    }

    // Combine the warps' splits. Each is an independent flash partial.
    for (int i = threadIdx.x; i < nwarps * group * d_head; i += blockDim.x) {
        sacc[i] = 0.0f;
    }
    __syncthreads();
    if (cr < group) {
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            if (i >= ntiles) break;
            float* dst = sacc + ((size_t)warp * group + cr) * d_head + i * 8 + cc;
            dst[0] = o[i].x[0];
            dst[1] = o[i].x[1];
        }
        if (lane % 4 == 0) {
            sml[(warp * group + cr) * 2] = m_run;
            sml[(warp * group + cr) * 2 + 1] = l_run;
        }
    }
    __syncthreads();

    for (int i = threadIdx.x; i < group * d_head; i += blockDim.x) {
        const int h = i / d_head, d = i % d_head;
        float m = -INFINITY;
        for (int w = 0; w < nwarps; ++w) {
            m = fmaxf(m, sml[(w * group + h) * 2]);
        }
        float acc = 0.0f, den = 0.0f;
        if (m > -INFINITY) {
            for (int w = 0; w < nwarps; ++w) {
                const float mw = sml[(w * group + h) * 2];
                if (mw == -INFINITY) continue;
                const float wt = __expf(mw - m);
                den += sml[(w * group + h) * 2 + 1] * wt;
                acc += sacc[((size_t)w * group + h) * d_head + d] * wt;
            }
        }
        const int head = kv_head * group + h;
        if (single) {
            out[((size_t)token * n_heads + head) * d_head + d] =
                den > 0.0f ? acc / den : 0.0f;
        } else {
            partial[(((size_t)c * n_tokens + token) * n_heads + head) * d_head + d] =
                acc;
            if (d == 0) {
                float* ms = partial + ms_off
                          + (((size_t)c * n_tokens + token) * n_heads + head) * 2;
                ms[0] = m;
                ms[1] = den;
            }
        }
    }
}
