// The Qwen3.5 vision tower: preprocessing, the 27 blocks' interior, the merger.
//
// Its own translation unit, like gdn.cu, and for a sharper reason than build
// times: this tower reverses nearly every convention the text-side kernels in
// ops.cu are built around, so sharing a file with them would put two opposite
// readings of the same word ("qkv", "rope", "norm") side by side.
//
//   | | ops.cu (text) | here (vision) |
//   |---|---|---|
//   | normalization | RMSNorm, no bias, no centring | LayerNorm, bias, centred |
//   | q/k/v packing | q interleaves its gate per head | [all q | all k | all v] |
//   | rope width | 64 of 256 dims | all 72 |
//   | rope theta | 1e7 | 1e4 |
//   | rope axes | 3, interleaved by i % 3 | 2, contiguous blocks |
//   | attention | causal, GQA, KV cache | bidirectional, 16 uniform heads, packed |
//   | MLP | SwiGLU | fc1 / GELU / fc2 |
//   | activation | silu | gelu, and *two* different gelus |
//
// Every one of those runs to completion under the wrong reading and produces a
// fluent description of the wrong image, so each is checked against
// `tuili_model::qwen35_vision` — itself checked against a capture of the
// reference implementation on the real checkpoint — in tests/vision.rs, with the
// alternative reading computed and required to disagree. See
// notes/qwen3.5-vision.md for the measured deviation of each wrong reading.
//
// Dtypes. The residual stream is f32 everywhere and never narrowed. It reaches
// ~4184 by block 27 from ~8.6 at the input, and although that fits f16's 65504
// the *spacing* of f16 at 4184 is 4.0 — larger than the per-block update being
// added to it, so an f16 residual would quietly stop accumulating near the top
// of the tower. What is narrowed is only the GEMM operands, and the structural
// reason that is safe is that no GEMM in this tower ever reads the residual:
// every one of them (qkv, proj, fc1, fc2, and the merger's two) takes either a
// LayerNorm output or an attention output, both O(10). See
// `vision_layer_norm_f32`'s comment and the measurements in tests/vision.rs.

// ---------------------------------------------------------------- LayerNorm

// LayerNorm over a row of `d`: subtract the mean, divide by the standard
// deviation, scale, shift. One block a row.
//
// Not RMSNorm. The text tower's `rms_norm_f32` in ops.cu skips the centring and
// has no bias; on this checkpoint's block 0 that reading moves the output by
// 2.95 out of a peak of 5.09 (and RMSNorm-plus-bias by 0.319), so reusing it is
// not a precision compromise but a different function.
//
// Two passes — mean, then the variance of the centred values — rather than
// E[x^2] - E[x]^2. At the top of the tower a row's values are ~4000 and its
// variance ~2.5e3, so the one-pass form subtracts two numbers that agree to
// seven digits and keeps three.
//
// The reduction is a block tree, not a serial sum. That matters at those
// magnitudes: the host reference reduces in f64 precisely because a *sequential*
// f32 sum of 1152 values near 4000 carries ~7e-4 of relative error, which lands
// as ~1e-5 absolute on an output of order 0.02 and swamps the tolerance a layout
// test wants. A 256-thread tree sum over 1152 values has a depth of about 13
// instead of 1152 and comes in three orders below that; tests/vision.rs measures
// it against the f64 reference on the captured `last_hidden` rather than
// assuming it.
//
// Both outputs are written: the f32 one is what the tests compare and what a
// second consumer would read, the f16 one is the GEMM operand that always
// follows. Fusing the narrowing here saves a full pass over [N, 1152] twice a
// block, 54 times a tower.
extern "C" __global__ void vision_layer_norm_f32(float* __restrict__ out,
                                                 __half* __restrict__ out_h,
                                                 const float* __restrict__ x,
                                                 const float* __restrict__ w,
                                                 const float* __restrict__ b,
                                                 int d, float eps) {
    const size_t row = (size_t)blockIdx.x * d;
    const float* xr = x + row;

    float s = 0.0f;
    for (int i = threadIdx.x; i < d; i += blockDim.x) s += xr[i];
    const float mean = block_reduce_sum(s) / (float)d;

    float v = 0.0f;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        const float c = xr[i] - mean;
        v += c * c;
    }
    const float inv = rsqrtf(block_reduce_sum(v) / (float)d + eps);

    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        const float y = (xr[i] - mean) * inv * w[i] + b[i];
        out[row + i] = y;
        out_h[row + i] = __float2half(y);
    }
}

// ---------------------------------------------------------------- activations

// gelu_pytorch_tanh, plus the bias the preceding GEMM did not apply.
//
// x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 x^3))). This is what
// `vision_config.hidden_act` names and what all 27 block MLPs use. The merger
// uses the *exact* GELU below instead; config mentions only this one. The two
// agree to ~4.7e-4 absolute, which is why they are separate kernels rather than
// a runtime flag: a flag is a thing that can be passed wrong, and the failure it
// produces is "small and everywhere", the error class that gets blamed on
// quantization for a week.
//
// Using silu here instead — the text tower's activation — moves block 0's MLP by
// 0.19 out of a peak of 8.69.
// In place in the f32 buffer: the GEMM's output is read, the activation's output
// is written over it, and the f16 copy the next GEMM needs goes out beside it.
// One less [N, 4304] buffer, which at 4096 patches is 70 MB.
extern "C" __global__ void vision_gelu_tanh_f32(float* __restrict__ io,
                                                __half* __restrict__ out_h,
                                                const float* __restrict__ bias,
                                                int n_cols, long long n) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = io[i] + bias[i % n_cols];
    const float inner = 0.7978845608028654f * (v + 0.044715f * v * v * v);
    const float y = v * 0.5f * (1.0f + tanhf(inner));
    io[i] = y;
    out_h[i] = __float2half(y);
}

// The exact GELU: x * 0.5 * (1 + erf(x / sqrt(2))). `nn.GELU()` with no
// `approximate` argument, which is what the *merger* uses. 27 blocks on the tanh
// form, one merger on this one.
extern "C" __global__ void vision_gelu_erf_f32(float* __restrict__ io,
                                               __half* __restrict__ out_h,
                                               const float* __restrict__ bias,
                                               int n_cols, long long n) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = io[i] + bias[i % n_cols];
    const float y = v * 0.5f * (1.0f + erff(v * 0.70710678118654752f));
    io[i] = y;
    out_h[i] = __float2half(y);
}

// ---------------------------------------------------------------- vision RoPE

// The cos/sin tables: [n, head_dim] each, from [n, 2] (h, w) positions.
//
// Three things here that a text-side habit gets wrong, and all three run:
//
//  1. the frequency table has `rope_dim` = head_dim/2 = 36 slots and the
//     exponent is divided by 36, *not* by head_dim. "The first 18 dims of a
//     72-wide table" is a different table; that reading moves block 0's
//     attention output by 0.49 out of a peak of 1.87.
//  2. theta is 1e4. ops.cu's rope kernels are called with 1e7 on this same
//     checkpoint's text side; 1e7 here moves the output by 0.23.
//  3. the two axes are *contiguous blocks*: dims [0, 18) rotate with h and
//     [18, 36) with w, then both are copied into [36, 72) so that the
//     `rotate_half` pairing (i, i + 36) finds the same angle at both ends.
//     The text side's mRoPE interleaves its three axes by `i % 3`. Interleaving
//     h and w here moves the output by 0.51.
//
// The exponent and the angle are computed in double, unlike ops.cu's rope
// kernels which use `__powf`/`__sincosf`. Vision positions top out at the grid
// side (a few hundred), so this is not about range; it costs nothing at
// n * 36 threads and it removes precision from the list of things a
// disagreement with the f64 host reference could mean.
extern "C" __global__ void vision_rope_tables_f32(float* __restrict__ cos_out,
                                                  float* __restrict__ sin_out,
                                                  const int* __restrict__ pos_ids,
                                                  int n, int head_dim,
                                                  int rope_dim, float theta) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * rope_dim) return;
    const int p = idx / rope_dim;
    const int j = idx % rope_dim;
    const int per_axis = rope_dim / 2;
    const int axis = j / per_axis;   // 0 = h, 1 = w. Blocked, not interleaved.
    const int i = j % per_axis;

    const double inv = pow((double)theta, -(double)(2 * i) / (double)rope_dim);
    const double angle = (double)pos_ids[p * 2 + axis] * inv;
    const float c = (float)cos(angle);
    const float s = (float)sin(angle);

    const size_t base = (size_t)p * head_dim + j;
    cos_out[base] = c;
    sin_out[base] = s;
    // The duplication `cat((freqs, freqs))` performs, and the reason it is
    // needed: rotate_half pairs dim i with dim i + rope_dim, and both must carry
    // the same angle. The whole head rotates; there is no unrotated tail as
    // there is on the text side.
    cos_out[base + rope_dim] = c;
    sin_out[base + rope_dim] = s;
}

// ------------------------------------------------------------ qkv split + rope

// Split the fused projection into q, k, v and rotate q and k, in one pass.
//
// `qkv` is [n, 3 * heads * head_dim]; the outputs are [n, heads, head_dim].
//
// **The 3456 columns are three contiguous blocks of 1152**, `[all q | all k |
// all v]`: the reference is `reshape(seq, 3, heads, -1)`, with the 3 *before*
// the head axis. The text tower does the opposite — `view(.., heads, 2 *
// head_dim)`, so a head's query and its gate sit next to each other — and
// ops.cu's `split_qkv_f32` and `rope_qk_packed_f32` are written for that. A
// per-head interleaved reading of this tensor yields three correctly shaped
// tensors and moves block 0's attention output by 6.03 out of a peak of 1.87.
//
// Note that for head 0 component 0 the two readings name the same column, which
// is why tests/vision.rs probes heads 0, 1 and 15 rather than just the first.
//
// The rotation is `rotate_half`: (a, b) at (i, i + head_dim/2) goes to
// (a c - b s, b c + a s), with the *same* angle at both ends by construction of
// the table above. Rotating adjacent pairs (2i, 2i+1) instead moves block 0's
// attention output by 0.26.
//
// One thread a (patch, head, i) with i < head_dim/2: it owns both ends of one
// rotation pair, for q and for k, and copies the matching two components of v.
extern "C" __global__ void vision_qkv_rope_f32(float* __restrict__ q,
                                               float* __restrict__ k,
                                               float* __restrict__ v,
                                               const float* __restrict__ qkv,
                                               const float* __restrict__ cos_t,
                                               const float* __restrict__ sin_t,
                                               int n, int heads, int head_dim) {
    const int half = head_dim / 2;
    const int per = heads * half;
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)n * per) return;
    const int p = (int)(idx / per);
    const int r = (int)(idx % per);
    const int h = r / half;
    const int i = r % half;

    const int dim = heads * head_dim;
    const size_t src = (size_t)p * 3 * dim + (size_t)h * head_dim;
    const size_t dst = ((size_t)p * heads + h) * head_dim;
    const size_t tab = (size_t)p * head_dim;

    const float c0 = cos_t[tab + i], s0 = sin_t[tab + i];
    const float c1 = cos_t[tab + i + half], s1 = sin_t[tab + i + half];

    // q at offset 0, k at +dim, v at +2*dim: the three contiguous blocks.
    const float qa = qkv[src + i], qb = qkv[src + i + half];
    q[dst + i] = qa * c0 - qb * s0;
    q[dst + i + half] = qb * c1 + qa * s1;

    const float ka = qkv[src + dim + i], kb = qkv[src + dim + i + half];
    k[dst + i] = ka * c0 - kb * s0;
    k[dst + i + half] = kb * c1 + ka * s1;

    // v is not rotated.
    v[dst + i] = qkv[src + 2 * dim + i];
    v[dst + i + half] = qkv[src + 2 * dim + i + half];
}

// ------------------------------------------------------------------ attention

// Warps a block, queries a warp, keys a tile. head_dim <= 32 * VIS_ITER.
#define VIS_WARPS 8
#define VIS_QPW 2
#define VIS_BQ (VIS_WARPS * VIS_QPW)
#define VIS_BK 32
#define VIS_ITER 4

// Bidirectional attention inside one `cu_seqlens` segment, streaming.
//
// `q`, `k`, `v` are [n, heads, head_dim]; `out` is the same shape. One block
// serves `VIS_BQ` consecutive queries of one segment for one head, and streams
// the segment's keys in tiles with an online softmax, so nothing of size
// seq x seq is ever materialized. A 1024x1024 image is 4096 patches in one
// segment; its score matrix at 16 heads would be 1.0 GiB.
//
// Two properties, neither of which fails loudly if lost:
//
//   * **Not causal.** `Qwen3_5VisionAttention.is_causal` is False. Reusing
//     ops.cu's attention — where the mask is `j > positions[token]` — makes every
//     patch blind to everything below and right of it and moves block 0's output
//     by 2.08 out of a peak of 1.87, i.e. replaces it.
//   * **Segment-local, one segment a frame.** `cu_seqlens` comes from
//     `repeat_interleave(h * w, t)`, so a t-frame video is t independent blocks
//     of h*w patches and a packed batch of images never mixes them. This also
//     makes the cost linear in t rather than quadratic.
//
// Segments are ragged, so the block-to-work mapping is precomputed on the host
// rather than derived from blockIdx: `tile_a`/`tile_b` give this block's segment
// bounds and `tile_q0` its first query. That keeps a tile from straddling a
// boundary, which is the only way a kernel like this leaks across segments.
extern "C" __global__ void vision_attn_f32(float* __restrict__ out,
                                           const float* __restrict__ q,
                                           const float* __restrict__ k,
                                           const float* __restrict__ v,
                                           const int* __restrict__ tile_a,
                                           const int* __restrict__ tile_b,
                                           const int* __restrict__ tile_q0,
                                           int heads, int head_dim,
                                           float scale) {
    extern __shared__ float smem[];
    // Padded to head_dim + 1 so that lane j reading row j at a fixed column
    // (the score loop) walks a stride coprime with 32. head_dim 72 unpadded
    // would put four lanes in every bank.
    const int pad = head_dim + 1;
    float* ksh = smem;
    float* vsh = ksh + VIS_BK * pad;
    float* qsh = vsh + VIS_BK * pad;

    const int h = blockIdx.y;
    const int seg_a = tile_a[blockIdx.x];
    const int seg_b = tile_b[blockIdx.x];
    const int q0 = tile_q0[blockIdx.x];
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
    const int iters = (head_dim + WARP_SIZE - 1) / WARP_SIZE;

    for (int s = threadIdx.x; s < VIS_BQ * head_dim; s += blockDim.x) {
        const int u = s / head_dim, i = s % head_dim;
        const int t = q0 + u;
        qsh[u * pad + i] =
            (t < seg_b) ? q[((size_t)t * heads + h) * head_dim + i] : 0.0f;
    }

    float acc[VIS_QPW][VIS_ITER];
    float mrow[VIS_QPW], lrow[VIS_QPW];
#pragma unroll
    for (int u = 0; u < VIS_QPW; ++u) {
        mrow[u] = -INFINITY;
        lrow[u] = 0.0f;
#pragma unroll
        for (int r = 0; r < VIS_ITER; ++r) acc[u][r] = 0.0f;
    }

    for (int base = seg_a; base < seg_b; base += VIS_BK) {
        __syncthreads();
        for (int s = threadIdx.x; s < VIS_BK * head_dim; s += blockDim.x) {
            const int j = s / head_dim, i = s % head_dim;
            const int t = base + j;
            const bool live = t < seg_b;
            ksh[j * pad + i] =
                live ? k[((size_t)t * heads + h) * head_dim + i] : 0.0f;
            vsh[j * pad + i] =
                live ? v[((size_t)t * heads + h) * head_dim + i] : 0.0f;
        }
        __syncthreads();
        const int nk = min(VIS_BK, seg_b - base);

        for (int u = 0; u < VIS_QPW; ++u) {
            const int qt = q0 + warp * VIS_QPW + u;
            const int slot = warp * VIS_QPW + u;
            // Lane j scores key `base + j`. Both operands come from shared: the
            // q read is a broadcast, the k read is conflict-free by the padding.
            float sc = -INFINITY;
            if (lane < nk && qt < seg_b) {
                float dot = 0.0f;
                for (int i = 0; i < head_dim; ++i) {
                    dot += qsh[slot * pad + i] * ksh[lane * pad + i];
                }
                sc = dot * scale;
            }
            const float mnew = fmaxf(mrow[u], warp_reduce_max(sc));
            // Uniform across the warp: mnew is a warp reduction and mrow starts
            // uniform. -inf happens when this warp's query is past the segment;
            // exp(-inf - -inf) would be nan, so the whole update is skipped.
            if (mnew > -INFINITY) {
                const float p = (sc > -INFINITY) ? __expf(sc - mnew) : 0.0f;
                const float corr = __expf(mrow[u] - mnew);
                lrow[u] = lrow[u] * corr + warp_reduce_sum(p);
#pragma unroll
                for (int r = 0; r < VIS_ITER; ++r) acc[u][r] *= corr;
                for (int j = 0; j < nk; ++j) {
                    const float pj = __shfl_sync(FULL_MASK, p, j);
#pragma unroll
                    for (int r = 0; r < VIS_ITER; ++r) {
                        const int i = lane + r * WARP_SIZE;
                        if (r < iters && i < head_dim) {
                            acc[u][r] += pj * vsh[j * pad + i];
                        }
                    }
                }
                mrow[u] = mnew;
            }
        }
    }

    for (int u = 0; u < VIS_QPW; ++u) {
        const int qt = q0 + warp * VIS_QPW + u;
        if (qt >= seg_b) continue;
        const float inv = 1.0f / lrow[u];
        const size_t dst = ((size_t)qt * heads + h) * head_dim;
#pragma unroll
        for (int r = 0; r < VIS_ITER; ++r) {
            const int i = lane + r * WARP_SIZE;
            if (r < iters && i < head_dim) out[dst + i] = acc[u][r] * inv;
        }
    }
}

// --------------------------------------------------------------- preprocessing

// One [C, H, W] frame (or a pair of them) into [grid_h * grid_w, patch_dim],
// which is one row of `pixel_values` a patch.
//
// The two things this pins, both of which run either way:
//
// **1. Patches come out in spatial-merge-block order, not raster order.**
// `p = ((block_row * blocks_w + block_col) * merge + in_row) * merge + in_col`,
// so four *consecutive* patches are a 2x2 square. That is the entire reason the
// merger's `view(-1, 4 * hidden)` is a 2x2 pooling: the merger reorders nothing.
// Raster order (`p = row * grid_w + col`) makes the merger average four patches
// strung out horizontally while the position embeddings describe a different
// arrangement; on a 6x8 grid 36 of the 48 patches land somewhere else.
//
// **2. Inside a patch the 1536 components are (c, t, y, x) with decreasing
// stride.** `slot = ((c * temporal + t) * patch + y) * patch + x`. The three
// other orderings — (t, c, y, x), (c, t, x, y), (y, x, c, t) — fill every slot,
// type-check, and feed the patch embedding a transposed patch.
//
// `n_frames` is how many frames `frames` actually holds; temporal tap `t` reads
// frame `min(t, n_frames - 1)`. For a still image that is 1, so **both taps see
// the same pixels** — the processor `expand`s rather than zero-filling, and the
// Conv3d's two temporal taps therefore act as their sum. Filling only the first
// tap moves the patch embedding by 0.83 out of a peak of 3.15. For video it is
// 2, and an odd frame count has its last frame repeated by the caller passing
// the same offset twice.
extern "C" __global__ void vision_patchify_f32(float* __restrict__ out,
                                               __half* __restrict__ out_h,
                                               const float* __restrict__ frames,
                                               long long frame_stride,
                                               int n_frames, int height,
                                               int width, int channels,
                                               int patch, int temporal,
                                               int merge, int grid_h,
                                               int grid_w) {
    const int patch_dim = channels * temporal * patch * patch;
    const long long total = (long long)grid_h * grid_w * patch_dim;
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    const int p = (int)(idx / patch_dim);
    const int slot = (int)(idx % patch_dim);

    const int x = slot % patch;
    const int y = (slot / patch) % patch;
    const int t = (slot / (patch * patch)) % temporal;
    const int c = slot / (patch * patch * temporal);

    const int blocks_w = grid_w / merge;
    const int in_col = p % merge;
    const int in_row = (p / merge) % merge;
    const int block_col = (p / (merge * merge)) % blocks_w;
    const int block_row = p / (merge * merge * blocks_w);
    const int row = block_row * merge + in_row;
    const int col = block_col * merge + in_col;

    const int frame = min(t, n_frames - 1);
    const float val = frames[(long long)frame * frame_stride +
                             (long long)(c * height + row * patch + y) * width +
                             col * patch + x];
    out[idx] = val;
    out_h[idx] = __float2half(val);
}

// Add the interpolated position embedding into the patch embedding.
//
// `table` is the learned 48x48 grid flattened to [2304, hidden]; `idx` and `wts`
// are [n, taps] from the host's bilinear resampling. Four taps a patch, weights
// summing to one.
//
// The resampling rule itself lives on the host (`pos_embed_taps`) because it is
// integer geometry per patch, not per feature: `src = index * (side - 1) /
// (size - 1)`, i.e. `align_corners = True`, which the model's __init__ sets and
// the library helper's own default contradicts. The False variant runs and moves
// the position embeddings by 5.31 out of a peak of 6.60. The gather order is
// block order like everything else; raster order there moves them by 4.83.
//
// One block a patch, threads over `hidden`.
extern "C" __global__ void vision_add_pos_embed_f32(
    float* __restrict__ hidden, const float* __restrict__ table,
    const int* __restrict__ idx, const float* __restrict__ wts, int hidden_size,
    int taps) {
    const int p = blockIdx.x;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float acc = 0.0f;
        for (int j = 0; j < taps; ++j) {
            const float w = wts[p * taps + j];
            if (w == 0.0f) continue;
            acc += w * table[(size_t)idx[p * taps + j] * hidden_size + i];
        }
        hidden[(size_t)p * hidden_size + i] += acc;
    }
}

// ------------------------------------------------------------------- splicing

// Write merger output row `f` over the embedding of the token at `dst_row[f]`.
//
// The reference's `get_placeholder_mask` insists the placeholder count equals
// the number of feature rows and raises otherwise; the launcher does the same,
// because a mismatch means the grid the tower ran on is not the grid the prompt
// was built for. `dst_row` is built on the host by scanning `input_ids` for
// image_token_id / video_token_id — **248056 / 248057 for this checkpoint, not
// Qwen2-VL's 151655 / 151656**, which are ordinary Thai-fragment vocabulary
// entries in this 248320-word vocabulary and would make the sequence lengths
// still line up while the image features went nowhere.
extern "C" __global__ void vision_splice_f32(float* __restrict__ embeds,
                                             const float* __restrict__ features,
                                             const int* __restrict__ dst_row,
                                             int out_hidden, int n_features) {
    const int f = blockIdx.x;
    if (f >= n_features) return;
    const size_t dst = (size_t)dst_row[f] * out_hidden;
    const size_t src = (size_t)f * out_hidden;
    for (int i = threadIdx.x; i < out_hidden; i += blockDim.x) {
        embeds[dst + i] = features[src + i];
    }
}
