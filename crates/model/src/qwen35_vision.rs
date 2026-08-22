//! Host-side reference for the Qwen3.5 vision tower. See notes/qwen3.5-vision.md.
//!
//! Deliberately the slow, obvious version: plain `f32`, explicit shapes, one
//! loop per thing the reference implementation does. It exists to be *read* as
//! the spec and to be the thing CUDA kernels answer to.
//!
//! The vision tower is a denser field of layout traps than the text side,
//! because almost nothing about it matches the text side's conventions:
//!
//! | | text tower | vision tower |
//! |---|---|---|
//! | normalization | RMSNorm, no bias | LayerNorm, with bias |
//! | MLP | SwiGLU, three matrices | fc1 / GELU / fc2, two matrices |
//! | linear bias | none | on every projection |
//! | attention | causal, GQA, output gate | bidirectional, 16 uniform heads, no gate |
//! | q/k/v packing | q interleaves with its gate per head | `[all q \| all k \| all v]` |
//! | RoPE width | 64 of 256 dims (partial) | all 72 dims |
//! | RoPE theta | 1e7 | 1e4 |
//! | RoPE axes | 3, interleaved by `i % 3` | 2, in contiguous blocks |
//!
//! Every one of those differences is a place where carrying a text-side habit
//! into the vision tower runs to completion and produces a fluent caption of
//! the wrong image. Each is pinned against a capture of the reference
//! implementation on the real checkpoint (`tools/capture_qwen35_vision.py`).
//!
//! Names follow `transformers.models.qwen3_5.modeling_qwen3_5` and
//! `transformers.vision_utils`.

use std::f32::consts::PI;

// ------------------------------------------------------------------ dimensions

/// The 27B checkpoint's `vision_config`, spelled out.
///
/// `out_hidden_size` is 5120 here, matching the text tower's `hidden_size`. The
/// `Qwen3_5VisionConfig` class default is 3584 (the 9B), so a loader that falls
/// back to the class default builds a merger whose output does not fit the
/// language model — that one at least fails loudly.
#[derive(Clone, Copy, Debug)]
pub struct VisionDims {
    pub depth: usize,
    pub hidden: usize,
    pub heads: usize,
    pub intermediate: usize,
    pub out_hidden: usize,
    pub in_channels: usize,
    pub patch: usize,
    pub temporal_patch: usize,
    pub merge: usize,
    /// `num_position_embeddings`; the learned grid is `sqrt` of this per side.
    pub num_position_embeddings: usize,
    pub eps: f32,
    pub rope_theta: f32,
}

impl VisionDims {
    /// Qwen3.5-27B.
    pub const QWEN35_27B: Self = Self {
        depth: 27,
        hidden: 1152,
        heads: 16,
        intermediate: 4304,
        out_hidden: 5120,
        in_channels: 3,
        patch: 16,
        temporal_patch: 2,
        merge: 2,
        num_position_embeddings: 2304,
        eps: 1e-6,
        rope_theta: 10_000.0,
    };

    /// 1152 / 16 = 72. Not a power of two, which matters for a kernel that
    /// wants one warp per head: 72 lanes do not fit in 32 or 64.
    pub fn head_dim(&self) -> usize {
        self.hidden / self.heads
    }

    /// The rotary frequency table's width: `head_dim / 2` = 36, split 18 for
    /// the h axis and 18 for w.
    pub fn rope_dim(&self) -> usize {
        self.head_dim() / 2
    }

    /// 48. The learned position grid is 48x48, resampled per image.
    pub fn grid_per_side(&self) -> usize {
        (self.num_position_embeddings as f64).sqrt() as usize
    }

    /// 3 * 2 * 16 * 16 = 1536: the width of one row of `pixel_values`.
    pub fn patch_dim(&self) -> usize {
        self.in_channels * self.temporal_patch * self.patch * self.patch
    }

    /// 4: how many patches the merger folds into one language-model token.
    pub fn merge_unit(&self) -> usize {
        self.merge * self.merge
    }

    /// `patch_size * spatial_merge_size` = 32. The resize granularity, *not*
    /// `patch_size`: the grid has to be even in both axes so the merger's 2x2
    /// blocks are whole.
    pub fn resize_factor(&self) -> usize {
        self.patch * self.merge
    }
}

// ----------------------------------------------------------------- activations

/// `gelu_pytorch_tanh`: the tanh approximation. This is what
/// `vision_config.hidden_act` names and what all 27 block MLPs use.
pub fn gelu_tanh(x: f32) -> f32 {
    let inner = (2.0f32 / PI).sqrt() * (x + 0.044_715 * x * x * x);
    x * 0.5 * (1.0 + inner.tanh())
}

/// Exact GELU. The *merger* uses this one — `nn.GELU()` with no `approximate`
/// argument — while the blocks use the tanh form above.
///
/// The two agree to about 5e-4 absolute, so mixing them up is a small numeric
/// error rather than a layout catastrophe. It is listed here anyway because
/// "small and everywhere" is exactly the error class that gets blamed on
/// quantization for a week.
pub fn gelu_erf(x: f32) -> f32 {
    x * 0.5 * (1.0 + erf(x / 2.0f32.sqrt()))
}

/// Abramowitz & Stegun 7.1.26, good to ~1.5e-7 absolute — under f32's own
/// resolution at these magnitudes, so the choice of approximation does not
/// leak into the comparison against the reference.
fn erf(x: f32) -> f32 {
    let s = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let y = 1.0
        - (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736) * t
            + 0.254_829_592)
            * t
            * (-x * x).exp();
    s * y
}

/// LayerNorm over the last dimension: subtract the mean, divide by the standard
/// deviation, scale, shift.
///
/// Every normalization in the vision tower is this — the two per block, and the
/// merger's. Every normalization in the text tower is RMSNorm. Dropping the mean
/// subtraction and the bias runs and moves the block-0 output by 0.32 out of a
/// peak of 5.1; dropping only the mean subtraction moves it by 3.0.
pub fn layer_norm_rows(x: &[f32], w: &[f32], b: &[f32], row_len: usize, eps: f32) -> Vec<f32> {
    assert_eq!(w.len(), row_len);
    assert_eq!(b.len(), row_len);
    assert_eq!(x.len() % row_len, 0);
    let mut out = Vec::with_capacity(x.len());
    for row in x.chunks(row_len) {
        let n = row_len as f32;
        let mean: f32 = row.iter().sum::<f32>() / n;
        let var: f32 = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
        let inv = (var + eps).sqrt().recip();
        for ((v, g), c) in row.iter().zip(w).zip(b) {
            out.push(g * ((v - mean) * inv) + c);
        }
    }
    out
}

/// `y = x W^T + b`, with `w` row-major `[out_dim, in_dim]` — the safetensors
/// layout, no transpose needed.
///
/// Every linear in the vision tower has a bias. The text tower's have none
/// (`attention_bias: false`, and the MLPs are bias-free). Loading the vision
/// tower with the text tower's loader silently drops 12 bias tensors per block;
/// that is the same failure as the AWQ loader dropping Qwen's QKV biases, and it
/// reads as fluent nonsense.
pub fn linear(x: &[f32], w: &[f32], b: &[f32], rows: usize, in_dim: usize, out_dim: usize) -> Vec<f32> {
    assert_eq!(x.len(), rows * in_dim);
    assert_eq!(w.len(), out_dim * in_dim);
    assert_eq!(b.len(), out_dim);
    let mut out = vec![0.0f32; rows * out_dim];
    for t in 0..rows {
        let xt = &x[t * in_dim..(t + 1) * in_dim];
        for o in 0..out_dim {
            let wo = &w[o * in_dim..(o + 1) * in_dim];
            let mut acc = b[o];
            for (a, bb) in xt.iter().zip(wo) {
                acc += a * bb;
            }
            out[t * out_dim + o] = acc;
        }
    }
    out
}

// -------------------------------------------------------------- preprocessing

/// The dynamic-resolution rule, from `qwen2_vl.image_processing_qwen2_vl
/// .smart_resize`.
///
/// `factor` is `patch_size * spatial_merge_size` = 32. Rounding to `patch_size`
/// = 16 instead gives an odd grid, and then `h // merge` truncates: the tower
/// runs, one row and one column of patches are silently dropped from the
/// merger's view, and the position field no longer matches the pixels.
///
/// Returns `None` when the aspect ratio exceeds 200:1, which is where the
/// reference raises rather than resizing.
pub fn smart_resize(
    height: usize,
    width: usize,
    factor: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> Option<(usize, usize)> {
    let (h, w) = (height as f64, width as f64);
    if h.max(w) / h.min(w) > 200.0 {
        return None;
    }
    let f = factor as f64;
    // `round(x / factor) * factor`, with Python's round-half-to-even. The
    // difference from round-half-away only shows up on exact .5 ratios, which
    // for factor 32 means heights like 16, 48, 80 — common enough in thumbnails
    // to be worth matching rather than hoping about.
    let round_to = |v: f64| -> usize { (round_half_even(v / f) * f) as usize };
    let mut h_bar = round_to(h);
    let mut w_bar = round_to(w);
    if h_bar * w_bar > max_pixels {
        let beta = ((h * w) / max_pixels as f64).sqrt();
        h_bar = factor.max(((h / beta / f).floor() * f) as usize);
        w_bar = factor.max(((w / beta / f).floor() * f) as usize);
    } else if h_bar * w_bar < min_pixels {
        let beta = (min_pixels as f64 / (h * w)).sqrt();
        h_bar = ((h * beta / f).ceil() * f) as usize;
        w_bar = ((w * beta / f).ceil() * f) as usize;
    }
    Some((h_bar, w_bar))
}

fn round_half_even(v: f64) -> f64 {
    let r = v.round();
    if (v - v.trunc()).abs() == 0.5 && r % 2.0 != 0.0 {
        r - v.signum()
    } else {
        r
    }
}

/// Where component `(c, t, y, x)` of a patch sits in its `patch_dim`-wide row.
///
/// Stride order is `c > t > y > x`. The processor's `patchify` permutes to
/// `[.., merge, merge, channel, patch, patch]`, inserts the temporal axis
/// between `channel` and the two spatial axes, then flattens the last four.
/// A `(t, c, y, x)` or `(c, t, x, y)` reading feeds the Conv3d a transposed
/// patch; nothing complains.
pub fn patch_slot(c: usize, t: usize, y: usize, x: usize, dims: &VisionDims) -> usize {
    ((c * dims.temporal_patch + t) * dims.patch + y) * dims.patch + x
}

/// Which `(row, col)` of the patch grid patch index `p` is, within one frame.
///
/// Patches arrive in spatial-merge-block order:
/// `p = ((block_row * blocks_w + block_col) * merge + in_row) * merge + in_col`.
/// So four consecutive patches are a 2x2 spatial block — which is what makes the
/// merger's `view(-1, 4 * hidden)` a 2x2 pooling rather than a row-of-four
/// pooling. Raster order (`p = row * grid_w + col`) is the natural guess, it
/// runs, and it makes the merger average four patches strung out horizontally
/// while the position embeddings describe a different arrangement entirely.
pub fn patch_row_col(p: usize, grid_w: usize, merge: usize) -> (usize, usize) {
    let blocks_w = grid_w / merge;
    let in_col = p % merge;
    let in_row = (p / merge) % merge;
    let block_col = (p / (merge * merge)) % blocks_w;
    let block_row = p / (merge * merge * blocks_w);
    (block_row * merge + in_row, block_col * merge + in_col)
}

/// Flatten one `[C, H, W]` frame into `[grid_h * grid_w, patch_dim]`.
///
/// The temporal axis is filled by *repeating the same frame* `temporal_patch`
/// times for a still image — the processor does `expand`, not zero-fill. Both
/// temporal taps of the Conv3d therefore see the same pixels and act as their
/// sum. Zeroing the second tap runs and roughly halves the patch embedding.
/// (For video the two taps hold two consecutive frames, and a clip whose frame
/// count is odd has its last frame duplicated to make it even.)
pub fn patchify(
    frame: &[f32],
    height: usize,
    width: usize,
    dims: &VisionDims,
) -> (Vec<f32>, usize, usize) {
    let (p, m, tp, c_n) = (dims.patch, dims.merge, dims.temporal_patch, dims.in_channels);
    assert_eq!(frame.len(), c_n * height * width);
    let (gh, gw) = (height / p, width / p);
    let mut out = vec![0.0f32; gh * gw * dims.patch_dim()];
    for idx in 0..gh * gw {
        let (row, col) = patch_row_col(idx, gw, m);
        for c in 0..c_n {
            for t in 0..tp {
                for y in 0..p {
                    for x in 0..p {
                        let src = (c * height + row * p + y) * width + col * p + x;
                        out[idx * dims.patch_dim() + patch_slot(c, t, y, x, dims)] = frame[src];
                    }
                }
            }
        }
    }
    (out, gh, gw)
}

// ------------------------------------------------------------------- geometry

/// One `(t, h, w)` grid entry. `h` and `w` count 16-pixel patches, `t` counts
/// temporal patches — so a 4-frame clip has `t = 2`, not 4.
#[derive(Clone, Copy, Debug)]
pub struct Grid {
    pub t: usize,
    pub h: usize,
    pub w: usize,
}

impl Grid {
    /// Rows of `pixel_values` this entry contributes.
    pub fn patches(&self) -> usize {
        self.t * self.h * self.w
    }

    /// Language-model tokens this entry becomes after the merger.
    pub fn tokens(&self, merge: usize) -> usize {
        self.patches() / (merge * merge)
    }
}

/// Attention segment boundaries: **one segment per frame**, not per entry.
///
/// `get_vision_cu_seqlens` uses `repeat_interleave(h * w, t)`, so a `t`-frame
/// video is `t` independent attention blocks of `h * w` patches. Letting
/// attention span the whole entry runs and mixes frames that the model never
/// intended to see each other. It also means the vision attention cost is
/// linear in `t`, not quadratic — worth knowing before sizing a kernel.
pub fn cu_seqlens(grids: &[Grid]) -> Vec<usize> {
    let mut cu = vec![0usize];
    for g in grids {
        for _ in 0..g.t {
            cu.push(cu.last().unwrap() + g.h * g.w);
        }
    }
    cu
}

/// The `(h, w)` index of every patch, in the order the patches arrive.
///
/// Returns `[total_patches, 2]` flattened. Note what is *not* here: a temporal
/// index. Vision RoPE has two axes; the `(h, w)` pairs simply repeat `t` times
/// for a video, so two frames of a clip carry identical rotary phase and are
/// distinguished only by being in different attention segments.
pub fn vision_position_ids(grids: &[Grid], merge: usize) -> Vec<u32> {
    let mut out = Vec::new();
    for g in grids {
        let frame_len = g.h * g.w;
        let mut frame = Vec::with_capacity(frame_len * 2);
        for p in 0..frame_len {
            let (row, col) = patch_row_col(p, g.w, merge);
            frame.push(row as u32);
            frame.push(col as u32);
        }
        for _ in 0..g.t {
            out.extend_from_slice(&frame);
        }
    }
    out
}

/// Bilinear taps into the learned 48x48 position grid, per patch.
///
/// Returns `(indices, weights)`, each `[total_patches, 4]` flattened, in
/// (h0,w0), (h0,w1), (h1,w0), (h1,w1) order.
///
/// `align_corners = true`, which `Qwen3_5VisionModel::__init__` sets and the
/// library helper's own default contradicts. With it the source coordinate is
/// `index * (side - 1) / (size - 1)`; without it, `(index + 0.5) * side / size
/// - 0.5`. The false variant runs and moves the position embeddings by 5.3 out
/// of a peak of 6.6 — the position field is then simply a different function of
/// the image, which degrades spatial grounding while leaving fluency intact.
pub fn pos_embed_taps(grids: &[Grid], side: usize, merge: usize) -> (Vec<usize>, Vec<f32>) {
    let mut idx = Vec::new();
    let mut wts = Vec::new();
    // One axis: two taps and their weights for target `index` on an axis of
    // length `size`, resampling a `side`-long source.
    let axis = |index: usize, size: usize| -> ((usize, usize), (f32, f32)) {
        let src = index as f64 * (side as f64 - 1.0) / (size.saturating_sub(1)).max(1) as f64;
        let fl = src.floor();
        let t0 = (fl as isize).clamp(0, side as isize - 1) as usize;
        let t1 = (fl as isize + 1).clamp(0, side as isize - 1) as usize;
        let d = (src - fl).abs();
        (
            (t0, t1),
            (
                (1.0 - d).max(0.0) as f32,
                (1.0 - (src - fl - 1.0).abs()).max(0.0) as f32,
            ),
        )
    };
    for g in grids {
        for _ in 0..g.t {
            for p in 0..g.h * g.w {
                let (row, col) = patch_row_col(p, g.w, merge);
                let ((h0, h1), (a0, a1)) = axis(row, g.h);
                let ((w0, w1), (b0, b1)) = axis(col, g.w);
                idx.extend_from_slice(&[
                    h0 * side + w0,
                    h0 * side + w1,
                    h1 * side + w0,
                    h1 * side + w1,
                ]);
                wts.extend_from_slice(&[a0 * b0, a0 * b1, a1 * b0, a1 * b1]);
            }
        }
    }
    (idx, wts)
}

/// Gather the interpolated position embedding for each patch.
pub fn gather_pos_embed(
    table: &[f32],
    hidden: usize,
    idx: &[usize],
    wts: &[f32],
    taps: usize,
) -> Vec<f32> {
    assert_eq!(idx.len(), wts.len());
    let n = idx.len() / taps;
    let mut out = vec![0.0f32; n * hidden];
    for p in 0..n {
        for j in 0..taps {
            let (row, w) = (idx[p * taps + j], wts[p * taps + j]);
            if w == 0.0 {
                continue;
            }
            let src = &table[row * hidden..(row + 1) * hidden];
            for (acc, v) in out[p * hidden..(p + 1) * hidden].iter_mut().zip(src) {
                *acc += w * v;
            }
        }
    }
    out
}

// ---------------------------------------------------------------- patch embed

/// The patch embedding: a `Conv3d` with kernel equal to stride, which is a
/// per-patch GEMM and nothing more.
///
/// `w` is `proj.weight` flattened to `[hidden, patch_dim]` — the checkpoint
/// stores it as `[1152, 3, 2, 16, 16]` and the flatten is a no-op view, so a
/// loader can point straight at it. There is no sliding window, no padding, and
/// no overlap to reason about; treating this as a real convolution is wasted
/// work.
pub fn patch_embed(
    pixels: &[f32],
    w: &[f32],
    b: &[f32],
    n_patches: usize,
    dims: &VisionDims,
) -> Vec<f32> {
    linear(pixels, w, b, n_patches, dims.patch_dim(), dims.hidden)
}

// ----------------------------------------------------------------- vision RoPE

/// Cosine and sine tables for vision RoPE: `[n_patches, head_dim]` each.
///
/// `position_ids` is `[n_patches, 2]` — `(h, w)`.
///
/// Three things here that a text-side habit gets wrong:
///
/// 1. The frequency table has `head_dim / 2` = 36 slots and the exponent is
///    divided by **36**, not by `head_dim`. This is the same shape of error as
///    the text side's partial rope; here it moves the block-0 attention output
///    by 0.49 out of a peak of 1.87.
/// 2. `theta` is 1e4. The text side is 1e7. Using 1e7 moves the output by 0.23.
/// 3. The axis layout is **blocked**: dims `[0, 18)` rotate with the h
///    position, dims `[18, 36)` with w, and `cat((emb, emb))` copies both
///    blocks into `[36, 72)`. The text side's mRoPE for the very same
///    checkpoint interleaves its three axes by `i % 3`. Interleaving h and w
///    here moves the output by 0.51.
///
/// So the full-head layout is, per patch:
/// `[h*f0..h*f17, w*f0..w*f17, h*f0..h*f17, w*f0..w*f17]`, and `rotate_half`
/// pairs index `i` with `i + 36`, which is why the duplication is there.
pub fn vision_rope_tables(
    position_ids: &[u32],
    dims: &VisionDims,
) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(position_ids.len() % 2, 0);
    let n = position_ids.len() / 2;
    let head_dim = dims.head_dim();
    let rope_dim = dims.rope_dim(); // 36
    let per_axis = rope_dim / 2; // 18 frequencies per axis
    // f64 for the frequency and the angle. Vision positions top out at the grid
    // size (a few hundred at most), so unlike the text side's 262144-token
    // context this is not near f32's resolution limit — but it costs nothing
    // and removes precision from the list of things a mismatch could mean.
    let inv: Vec<f64> = (0..per_axis)
        .map(|i| (dims.rope_theta as f64).powf(-((2 * i) as f64 / rope_dim as f64)))
        .collect();
    let mut cos = vec![0.0f32; n * head_dim];
    let mut sin = vec![0.0f32; n * head_dim];
    for p in 0..n {
        for (axis, &pos) in position_ids[p * 2..p * 2 + 2].iter().enumerate() {
            for i in 0..per_axis {
                let angle = pos as f64 * inv[i];
                let (s, c) = (angle.sin() as f32, angle.cos() as f32);
                let j = axis * per_axis + i;
                cos[p * head_dim + j] = c;
                cos[p * head_dim + j + rope_dim] = c;
                sin[p * head_dim + j] = s;
                sin[p * head_dim + j + rope_dim] = s;
            }
        }
    }
    (cos, sin)
}

/// Apply vision RoPE in place to `[n_patches, heads, head_dim]`.
///
/// Pairing is `(i, i + head_dim/2)` — `rotate_half`, the non-interleaved
/// convention. Pairing adjacent dims `(2i, 2i+1)` instead runs and moves the
/// block-0 attention output by 0.26. The whole head rotates; there is no
/// unrotated tail as there is on the text side.
pub fn apply_vision_rope(
    x: &mut [f32],
    cos: &[f32],
    sin: &[f32],
    n_patches: usize,
    heads: usize,
    head_dim: usize,
) {
    assert_eq!(x.len(), n_patches * heads * head_dim);
    assert_eq!(cos.len(), n_patches * head_dim);
    let half = head_dim / 2;
    for p in 0..n_patches {
        for h in 0..heads {
            let base = (p * heads + h) * head_dim;
            for i in 0..half {
                let (a, b) = (x[base + i], x[base + i + half]);
                let (c0, s0) = (cos[p * head_dim + i], sin[p * head_dim + i]);
                let (c1, s1) = (cos[p * head_dim + i + half], sin[p * head_dim + i + half]);
                // rotate_half sends (a, b) to (-b, a), so the first half takes
                // -b*sin and the second +a*sin.
                x[base + i] = a * c0 - b * s0;
                x[base + i + half] = b * c1 + a * s1;
            }
        }
    }
}

// ------------------------------------------------------------------ attention

/// Split `qkv`'s output into q, k and v, each `[n_patches, heads, head_dim]`.
///
/// The reference is
/// `qkv(h).reshape(seq, 3, heads, -1).permute(1, 0, 2, 3).unbind(0)`: the 3
/// sits *before* the head axis, so the 3456 columns are three contiguous blocks
/// of 1152 — `[all q | all k | all v]`.
///
/// This is the opposite convention from the text tower, where `q_proj`'s output
/// is `view(.., heads, 2 * head_dim)` and each head's query and gate sit next to
/// each other. Reading this tensor per-head-interleaved gives three
/// correctly-shaped tensors and moves the block-0 attention output by 6.0 out of
/// a peak of 1.87 — that is, it produces something entirely unrelated, fluently.
pub fn split_qkv(
    qkv: &[f32],
    n_patches: usize,
    heads: usize,
    head_dim: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let dim = heads * head_dim;
    assert_eq!(qkv.len(), n_patches * 3 * dim);
    let mut parts = [
        vec![0.0f32; n_patches * dim],
        vec![0.0f32; n_patches * dim],
        vec![0.0f32; n_patches * dim],
    ];
    for p in 0..n_patches {
        for (s, part) in parts.iter_mut().enumerate() {
            let src = p * 3 * dim + s * dim;
            part[p * dim..(p + 1) * dim].copy_from_slice(&qkv[src..src + dim]);
        }
    }
    let [q, k, v] = parts;
    (q, k, v)
}

/// Bidirectional attention inside each `cu_seqlens` segment.
///
/// `q`, `k`, `v` are `[n_patches, heads, head_dim]`; the result is
/// `[n_patches, heads * head_dim]`.
///
/// Two properties, both easy to get wrong and neither of which fails loudly:
///
/// - **Not causal.** `Qwen3_5VisionAttention::is_causal` is `false`. A causal
///   mask moves the block-0 output by 2.08 out of a peak of 1.87, i.e. it
///   replaces it. Reusing the text tower's attention kernel is exactly how this
///   happens.
/// - **Segment-local.** Each frame of each image is its own segment. Attending
///   across segments in a packed batch mixes unrelated images.
pub fn segment_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    cu: &[usize],
    heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let n = *cu.last().unwrap();
    assert_eq!(q.len(), n * heads * head_dim);
    let scale = (head_dim as f32).sqrt().recip();
    let mut out = vec![0.0f32; n * heads * head_dim];
    let mut scores: Vec<f32> = Vec::new();
    for seg in cu.windows(2) {
        let (a, b) = (seg[0], seg[1]);
        for t in a..b {
            for h in 0..heads {
                let qh = &q[(t * heads + h) * head_dim..(t * heads + h + 1) * head_dim];
                scores.clear();
                let mut max = f32::NEG_INFINITY;
                for s in a..b {
                    let kh = &k[(s * heads + h) * head_dim..(s * heads + h + 1) * head_dim];
                    let dot: f32 = qh.iter().zip(kh).map(|(x, y)| x * y).sum();
                    let sc = dot * scale;
                    scores.push(sc);
                    max = max.max(sc);
                }
                let mut denom = 0.0f32;
                for sc in scores.iter_mut() {
                    *sc = (*sc - max).exp();
                    denom += *sc;
                }
                let o = &mut out[(t * heads + h) * head_dim..(t * heads + h + 1) * head_dim];
                for (i, &pw) in scores.iter().enumerate() {
                    let w = pw / denom;
                    let vh = &v[((a + i) * heads + h) * head_dim..((a + i) * heads + h + 1) * head_dim];
                    for (acc, &val) in o.iter_mut().zip(vh) {
                        *acc += w * val;
                    }
                }
            }
        }
    }
    out
}

// --------------------------------------------------------------------- blocks

/// Weights of one `Qwen3_5VisionBlock`, as they sit in the checkpoint.
pub struct BlockWeights<'a> {
    pub norm1_w: &'a [f32],
    pub norm1_b: &'a [f32],
    pub norm2_w: &'a [f32],
    pub norm2_b: &'a [f32],
    /// `[3 * hidden, hidden]`.
    pub qkv_w: &'a [f32],
    pub qkv_b: &'a [f32],
    /// `[hidden, hidden]`.
    pub proj_w: &'a [f32],
    pub proj_b: &'a [f32],
    /// `[intermediate, hidden]`.
    pub fc1_w: &'a [f32],
    pub fc1_b: &'a [f32],
    /// `[hidden, intermediate]`.
    pub fc2_w: &'a [f32],
    pub fc2_b: &'a [f32],
}

/// The block MLP: `fc2(gelu_tanh(fc1(x)))`.
///
/// Two matrices, not three. The text tower's MLP is SwiGLU — gate, up, down —
/// and `intermediate_size` there is the width of *two* of them. Here 4304 is one
/// width, it is not divisible into a gate/up pair in any natural way, and a
/// SwiGLU reading would have to invent a split.
pub fn vision_mlp(x: &[f32], b: &BlockWeights, rows: usize, dims: &VisionDims) -> Vec<f32> {
    let mut h = linear(x, b.fc1_w, b.fc1_b, rows, dims.hidden, dims.intermediate);
    for v in h.iter_mut() {
        *v = gelu_tanh(*v);
    }
    linear(&h, b.fc2_w, b.fc2_b, rows, dims.intermediate, dims.hidden)
}

/// One vision block, pre-norm with two residual adds:
///
/// ```text
/// h = h + proj(attn(norm1(h)))
/// h = h + fc2(gelu(fc1(norm2(h))))
/// ```
///
/// The residual carries the *unnormalized* stream, and it grows: by the last of
/// the 27 blocks the hidden state reaches ~4200 in magnitude with a row variance
/// around 2.5e3, against ~0.14 at the input. That is a 4-order-of-magnitude
/// growth along the residual path, and it is why the merger's LayerNorm is
/// load-bearing rather than cosmetic. A kernel that keeps this stream in f16
/// (max 65504) has about 15x of headroom left at the top of the tower; bf16 or
/// f32 accumulation is the safe choice.
pub fn vision_block(
    hidden: &mut [f32],
    w: &BlockWeights,
    cu: &[usize],
    n_patches: usize,
    cos: &[f32],
    sin: &[f32],
    dims: &VisionDims,
) {
    let (hd, heads, d) = (dims.head_dim(), dims.heads, dims.hidden);
    let normed = layer_norm_rows(hidden, w.norm1_w, w.norm1_b, d, dims.eps);
    let qkv = linear(&normed, w.qkv_w, w.qkv_b, n_patches, d, 3 * d);
    let (mut q, mut k, v) = split_qkv(&qkv, n_patches, heads, hd);
    apply_vision_rope(&mut q, cos, sin, n_patches, heads, hd);
    apply_vision_rope(&mut k, cos, sin, n_patches, heads, hd);
    let ctx = segment_attention(&q, &k, &v, cu, heads, hd);
    let attn = linear(&ctx, w.proj_w, w.proj_b, n_patches, d, d);
    for (h, a) in hidden.iter_mut().zip(&attn) {
        *h += a;
    }
    let normed = layer_norm_rows(hidden, w.norm2_w, w.norm2_b, d, dims.eps);
    let mlp = vision_mlp(&normed, w, n_patches, dims);
    for (h, m) in hidden.iter_mut().zip(&mlp) {
        *h += m;
    }
}

// --------------------------------------------------------------------- merger

/// Fold four normalized patches into one row of `4 * hidden`.
///
/// This is a plain reshape *because* patches arrive in 2x2-block order — see
/// `patch_row_col`. Nothing in the merger reorders anything, so if the patch
/// order coming out of preprocessing is raster, this silently pools a
/// horizontal run of four patches instead of a 2x2 square.
pub fn merger_shuffle(normed: &[f32], hidden: usize, merge_unit: usize) -> Vec<f32> {
    let wide = hidden * merge_unit;
    assert_eq!(normed.len() % wide, 0);
    let tokens = normed.len() / wide;
    let mut out = vec![0.0f32; normed.len()];
    for tok in 0..tokens {
        for u in 0..merge_unit {
            // Patch `tok * merge_unit + u` becomes slot `u` of token `tok`. The
            // indices are the identity, which is the whole point: the grouping
            // is free *given* that preprocessing emitted 2x2 blocks. Spelled out
            // so the assumption is visible rather than implicit in a reshape.
            let src = (tok * merge_unit + u) * hidden;
            let dst = tok * wide + u * hidden;
            out[dst..dst + hidden].copy_from_slice(&normed[src..src + hidden]);
        }
    }
    out
}

/// The patch merger: LayerNorm **per patch**, then group, then two linears with
/// exact GELU between them.
///
/// `use_postshuffle_norm` is `false` for this checkpoint, and the checkpoint
/// proves it: `merger.norm.weight` is `[1152]`, not `[4608]`. So the norm runs
/// over one patch's 1152 features *before* the 2x2 grouping. Normalizing the
/// grouped 4608 instead — tiling the gain four times to make the shapes work —
/// runs and moves the merger input by 9.99 out of a peak of 6.81.
///
/// The activation is `nn.GELU()`, the exact one, while the 27 blocks above use
/// the tanh approximation. `vision_config.hidden_act` names only the latter.
///
/// `hidden_in` is `[n_patches, hidden]` (the tower's last hidden state);
/// the result is `[n_patches / 4, out_hidden]`.
pub fn patch_merger(
    hidden_in: &[f32],
    norm_w: &[f32],
    norm_b: &[f32],
    fc1_w: &[f32],
    fc1_b: &[f32],
    fc2_w: &[f32],
    fc2_b: &[f32],
    n_patches: usize,
    dims: &VisionDims,
) -> Vec<f32> {
    let wide = dims.hidden * dims.merge_unit();
    let tokens = n_patches / dims.merge_unit();
    let normed = layer_norm_rows(hidden_in, norm_w, norm_b, dims.hidden, dims.eps);
    let grouped = merger_shuffle(&normed, dims.hidden, dims.merge_unit());
    let mut h = linear(&grouped, fc1_w, fc1_b, tokens, wide, wide);
    for v in h.iter_mut() {
        *v = gelu_erf(*v);
    }
    linear(&h, fc2_w, fc2_b, tokens, wide, dims.out_hidden)
}

// -------------------------------------------------- splicing into the text side

/// Token ids from the 27B `config.json`. **Not** 151655/151656 — those belong to
/// Qwen2-VL's 152k vocabulary and decode to Thai text fragments in this
/// checkpoint's 248320-entry vocabulary. Hard-coding the older ids replaces two
/// image placeholders with two arbitrary words; the sequence lengths still line
/// up, so nothing fails.
pub const IMAGE_TOKEN_ID: u32 = 248_056;
pub const VIDEO_TOKEN_ID: u32 = 248_057;
pub const VISION_START_TOKEN_ID: u32 = 248_053;
pub const VISION_END_TOKEN_ID: u32 = 248_054;

/// Replace each `image_token_id` / `video_token_id` embedding with the next
/// merger output row, in order.
///
/// `embeds` is `[seq, out_hidden]`, `features` is `[n_tokens, out_hidden]`. The
/// reference's `get_placeholder_mask` insists the counts match exactly, and so
/// does this: a mismatch means the grid the tower ran on is not the grid the
/// prompt was built for, which is a bug worth a panic rather than a shrug.
pub fn splice_image_features(
    embeds: &mut [f32],
    input_ids: &[u32],
    features: &[f32],
    out_hidden: usize,
) {
    assert_eq!(embeds.len(), input_ids.len() * out_hidden);
    let want = input_ids
        .iter()
        .filter(|&&t| t == IMAGE_TOKEN_ID || t == VIDEO_TOKEN_ID)
        .count();
    assert_eq!(
        features.len(),
        want * out_hidden,
        "{want} placeholder tokens but {} feature rows",
        features.len() / out_hidden
    );
    let mut next = 0;
    for (t, &id) in input_ids.iter().enumerate() {
        if id == IMAGE_TOKEN_ID || id == VIDEO_TOKEN_ID {
            embeds[t * out_hidden..(t + 1) * out_hidden]
                .copy_from_slice(&features[next * out_hidden..(next + 1) * out_hidden]);
            next += 1;
        }
    }
}

/// Which of T/H/W drives rotary frequency `i`, under `mrope_interleaved: true`.
///
/// The text config carries `mrope_section = [11, 11, 10]`, summing to 32 = the
/// number of partial-rope frequencies (`head_dim * 0.25 / 2`). With
/// interleaving the assignment is simply `i % 3` — verified by feeding
/// `apply_interleaved_mrope` a tensor whose value *is* the axis index, which
/// returns `[0,1,2,0,1,2,...,0,1]`. The section only encodes the resulting
/// counts: 11 slots land on T, 11 on H, 10 on W.
///
/// The trap: Qwen2-VL and Qwen2.5-VL use the *chunked* layout for the same
/// config field — `[0..11)` T, `[11..22)` H, `[22..32)` W. Both readings consume
/// the same three position rows and produce the same shapes. On a pure-text
/// prompt they are even identical, because `get_rope_index` sets T = H = W for
/// text tokens; the divergence appears only once an image is in the context,
/// which is the worst possible place for a bug to first show up.
pub fn interleaved_mrope_axis(i: usize, section: [usize; 3]) -> usize {
    // Mirrors the reference: H claims `i % 3 == 1` for `i < 3 * section[1]`,
    // W claims `i % 3 == 2` for `i < 3 * section[2]`, T keeps everything else.
    match i % 3 {
        1 if i < 3 * section[1] => 1,
        2 if i < 3 * section[2] => 2,
        _ => 0,
    }
}

/// 3-D text-side positions for a spliced sequence, mirroring
/// `Qwen3_5Model::get_rope_index`.
///
/// `token_types` is 0 for text, 1 for image, 2 for video, and must be run-length
/// contiguous per visual entry — that is what the reference groups on.
///
/// Returns `[3, seq]` flattened: T row, then H, then W.
///
/// The advance rule is the part to copy carefully. After a visual entry the
/// running scalar position advances by `max(llm_h, llm_w)` — the *larger spatial
/// extent*, not the token count and not `max(t, h, w)` as Qwen2-VL used. For a
/// 6x8 patch grid that is 12 image tokens but an advance of only 4, so the
/// sequence's maximum position ends up *below* its length; the reference reports
/// this as a negative `rope_delta`. An implementation that advances by the token
/// count runs and puts every post-image token at the wrong distance from
/// everything before the image.
pub fn llm_position_ids(token_types: &[u8], grids: &[Grid], merge: usize) -> Vec<u32> {
    let seq = token_types.len();
    let mut out = vec![0u32; 3 * seq];
    let mut pos = 0u32;
    let mut grid_iter = grids.iter();
    let mut i = 0;
    while i < seq {
        let kind = token_types[i];
        let mut j = i;
        while j < seq && token_types[j] == kind {
            j += 1;
        }
        if kind == 0 {
            for t in i..j {
                for axis in 0..3 {
                    out[axis * seq + t] = pos + (t - i) as u32;
                }
            }
            pos += (j - i) as u32;
        } else {
            let g = grid_iter.next().expect("a visual run with no grid entry");
            let (lt, lh, lw) = (g.t, g.h / merge, g.w / merge);
            assert_eq!(j - i, lt * lh * lw, "visual run length does not match its grid");
            let mut t = i;
            for ti in 0..lt {
                for hi in 0..lh {
                    for wi in 0..lw {
                        out[t] = pos + ti as u32;
                        out[seq + t] = pos + hi as u32;
                        out[2 * seq + t] = pos + wi as u32;
                        t += 1;
                    }
                }
            }
            pos += lh.max(lw) as u32;
        }
        i = j;
    }
    out
}
