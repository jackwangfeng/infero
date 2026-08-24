//! Launchers and the driver for the Qwen3.5 vision tower.
//!
//! In a file of its own, like `gdn.rs`, so this work and the text-side work do
//! not collide in `lib.rs`; the kernels are in `cu/vision.cu`.
//!
//! The one thing worth reading before using any of these: **this tower reverses
//! almost every convention the rest of the operator set is built around.** Its
//! normalization is LayerNorm and not RMSNorm, every linear carries a bias where
//! the text tower has none, its attention is bidirectional and its `qkv` is
//! `[all q | all k | all v]` where the text tower interleaves per head, its rope
//! turns all 72 dims with theta 1e4 over two contiguous axes where the text
//! tower turns 64 of 256 with theta 1e7 over three interleaved ones. Reaching
//! for `Kernels::rms_norm`, `Kernels::split_qkv`, `Kernels::rope_qk_packed` or
//! any of the `attn_*` family here compiles, runs, and produces a fluent caption
//! of the wrong image. `notes/qwen3.5-vision.md` gives the measured deviation of
//! each of those substitutions.
//!
//! What *is* reused from the text side: `add_bias`, `add_assign`, `to_f16`, and
//! `gemm_f16`. Those carry no vision-specific convention.
//!
//! ## The accumulation dtype
//!
//! The residual stream is f32 and is never narrowed. It grows four orders of
//! magnitude along the tower — peak 8.6 at the input, 4184 after 27 blocks — and
//! while 4184 sits comfortably inside f16's 65504, f16's *spacing* at 4184 is
//! 4.0, which is larger than a single block's contribution to the stream. An
//! f16 residual would therefore stop accumulating somewhere in the last third of
//! the tower without ever producing an inf. `tests/vision.rs`'
//! `an_f16_residual_stream_loses_the_block_update_at_the_top_of_the_tower`
//! measures that on the captured `last_hidden`.
//!
//! GEMM operands are f16, and the reason that is safe is structural rather than
//! empirical: **no GEMM in this tower reads the residual stream.** Every one of
//! the six — qkv, proj, fc1, fc2 and the merger's two — takes a LayerNorm output
//! or an attention output (a convex combination of one), all O(10). The one
//! place the residual's magnitude does reach a matrix is the merger, and it
//! passes through the merger's LayerNorm first, which is exactly why that norm
//! is load-bearing rather than cosmetic.

use anyhow::{Context, Result};
use tuili_gpu::{Buf, View, ViewMut, LaunchConfig, KernelArg};
use half::f16;

use crate::{ELEMENTWISE_BLOCK, Kernels, REDUCE_BLOCK, vision_src};

/// Queries one attention block serves. Must match `VIS_BQ` in `cu/vision.cu`
/// (`VIS_WARPS * VIS_QPW`): the host builds the block-to-segment mapping, so the
/// two have to agree.
pub const VISION_ATTN_BQ: usize = 32;
/// Threads in an attention block: `VIS_WARPS * 32`.
const VISION_ATTN_THREADS: u32 = 256;
/// Keys per streamed tile; `VIS_BK`.
const VISION_ATTN_BK: usize = 32;

/// The vision tower's shape. A local copy rather than a re-export of
/// `tuili_model::qwen35_vision::VisionDims`, because that crate is a
/// dev-dependency of this one (for the tests) and not a dependency of the
/// library.
#[derive(Clone, Copy, Debug)]
pub struct VisionShape {
    pub depth: usize,
    pub hidden: usize,
    pub heads: usize,
    pub intermediate: usize,
    pub out_hidden: usize,
    pub in_channels: usize,
    pub patch: usize,
    pub temporal_patch: usize,
    pub merge: usize,
    pub eps: f32,
    pub rope_theta: f32,
}

impl VisionShape {
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
        eps: 1e-6,
        rope_theta: 10_000.0,
    };

    /// 72 on this checkpoint. Not a power of two, which is why the attention
    /// kernel loops `(head_dim + 31) / 32` times over its accumulator instead of
    /// giving one lane one component.
    pub fn head_dim(&self) -> usize {
        self.hidden / self.heads
    }

    /// 36: the rotary frequency table's width, 18 for h and 18 for w.
    pub fn rope_dim(&self) -> usize {
        self.head_dim() / 2
    }

    /// 1536: the width of one row of `pixel_values`.
    pub fn patch_dim(&self) -> usize {
        self.in_channels * self.temporal_patch * self.patch * self.patch
    }

    /// 4: patches per language-model token.
    pub fn merge_unit(&self) -> usize {
        self.merge * self.merge
    }

    /// 4608: the merger's input width.
    pub fn merged(&self) -> usize {
        self.hidden * self.merge_unit()
    }
}

impl Kernels {
    /// LayerNorm over rows of `d`, writing the result in both f32 and f16.
    ///
    /// **Not RMSNorm.** `Kernels::rms_norm` skips the mean subtraction and has
    /// no bias; on block 0 of this checkpoint that reading moves the output by
    /// 2.95 out of a peak of 5.09.
    #[allow(clippy::too_many_arguments)]
    pub fn vision_layer_norm(
        &self,
        out: &mut ViewMut<'_, f32>,
        out_h: &mut ViewMut<'_, f16>,
        x: &View<'_, f32>,
        w: &View<'_, f32>,
        b: &View<'_, f32>,
        rows: usize,
        d: usize,
        eps: f32,
    ) -> Result<()> {
        debug_assert!(out.len() >= rows * d && out_h.len() >= rows * d);
        debug_assert!(x.len() >= rows * d);
        debug_assert!(w.len() >= d && b.len() >= d);
        let f = self
            .dev
            .kernels()
            .get("tuili_vision", vision_src(), "vision_layer_norm_f32")?;
        let cfg = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (REDUCE_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let dd = d as i32;
        let mut bl = self.dev.stream().launch_builder(&f);
        bl.arg(out).arg(out_h).arg(x).arg(w).arg(b).arg(&dd).arg(&eps);
        self.dev
            .profile()
            .time("vision_layer_norm", self.dev.stream(), || {
                unsafe { bl.launch(cfg) }.context("vision_layer_norm")?;
                Ok(())
            })?;
        Ok(())
    }

    /// Add a per-column bias and apply GELU **in place** in `io`, and write the
    /// f16 copy the following GEMM reads.
    ///
    /// `exact` picks the erf form (`nn.GELU()`, what the merger uses) over the
    /// tanh approximation (`gelu_pytorch_tanh`, what all 27 blocks use). The two
    /// agree to ~4.7e-4 absolute, so getting this wrong is a small error
    /// everywhere rather than a layout failure — which is the error class that
    /// gets blamed on quantization.
    #[allow(clippy::too_many_arguments)]
    pub fn vision_gelu(
        &self,
        io: &mut ViewMut<'_, f32>,
        out_h: &mut ViewMut<'_, f16>,
        bias: &View<'_, f32>,
        rows: usize,
        n_cols: usize,
        exact: bool,
    ) -> Result<()> {
        let n = rows * n_cols;
        debug_assert!(io.len() >= n && out_h.len() >= n);
        debug_assert!(bias.len() >= n_cols);
        let name = if exact {
            "vision_gelu_erf_f32"
        } else {
            "vision_gelu_tanh_f32"
        };
        let f = self.dev.kernels().get("tuili_vision", vision_src(), name)?;
        let cfg = LaunchConfig {
            grid_dim: ((n as u32).div_ceil(ELEMENTWISE_BLOCK).max(1), 1, 1),
            block_dim: (ELEMENTWISE_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let (c, count) = (n_cols as i32, n as i64);
        let mut bl = self.dev.stream().launch_builder(&f);
        bl.arg(io).arg(out_h).arg(bias).arg(&c).arg(&count);
        self.dev.profile().time(name, self.dev.stream(), || {
            unsafe { bl.launch(cfg) }.context(name)?;
            Ok(())
        })?;
        Ok(())
    }

    /// The vision rotary tables: `[n, head_dim]` cos and sin from `[n, 2]`
    /// `(h, w)` positions.
    ///
    /// `rope_dim` is `head_dim / 2` and the frequency exponent is divided by
    /// *that*, not by `head_dim`; `theta` is 1e4, not the text side's 1e7; and
    /// the two axes occupy contiguous blocks rather than interleaving. All three
    /// are the text side's habits, all three run.
    #[allow(clippy::too_many_arguments)]
    pub fn vision_rope_tables(
        &self,
        cos: &mut ViewMut<'_, f32>,
        sin: &mut ViewMut<'_, f32>,
        pos_ids: &View<'_, i32>,
        n: usize,
        head_dim: usize,
        rope_dim: usize,
        theta: f32,
    ) -> Result<()> {
        anyhow::ensure!(
            rope_dim * 2 == head_dim && rope_dim.is_multiple_of(2),
            "vision rope wants rope_dim = head_dim / 2 with two equal axes; got \
             rope_dim {rope_dim} for head_dim {head_dim}"
        );
        debug_assert!(cos.len() >= n * head_dim && sin.len() >= n * head_dim);
        debug_assert!(pos_ids.len() >= n * 2);
        let f = self
            .dev
            .kernels()
            .get("tuili_vision", vision_src(), "vision_rope_tables_f32")?;
        let total = (n * rope_dim) as u32;
        let cfg = LaunchConfig {
            grid_dim: (total.div_ceil(ELEMENTWISE_BLOCK).max(1), 1, 1),
            block_dim: (ELEMENTWISE_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let (nn, hd, rd) = (n as i32, head_dim as i32, rope_dim as i32);
        let mut bl = self.dev.stream().launch_builder(&f);
        bl.arg(cos)
            .arg(sin)
            .arg(pos_ids)
            .arg(&nn)
            .arg(&hd)
            .arg(&rd)
            .arg(&theta);
        self.dev
            .profile()
            .time("vision_rope_tables", self.dev.stream(), || {
                unsafe { bl.launch(cfg) }.context("vision_rope_tables")?;
                Ok(())
            })?;
        Ok(())
    }

    /// Split `[n, 3 * heads * head_dim]` into q, k, v and rotate q and k.
    ///
    /// The split is **three contiguous blocks**, `[all q | all k | all v]`.
    /// `Kernels::split_qkv` is for the text side's per-head interleaving and
    /// gives three correctly shaped tensors here holding unrelated values —
    /// block 0's attention output moves by 6.03 out of a peak of 1.87.
    #[allow(clippy::too_many_arguments)]
    pub fn vision_qkv_rope(
        &self,
        q: &mut ViewMut<'_, f32>,
        k: &mut ViewMut<'_, f32>,
        v: &mut ViewMut<'_, f32>,
        qkv: &View<'_, f32>,
        cos: &View<'_, f32>,
        sin: &View<'_, f32>,
        n: usize,
        heads: usize,
        head_dim: usize,
    ) -> Result<()> {
        anyhow::ensure!(
            head_dim.is_multiple_of(2),
            "rotate_half pairs i with i + head_dim/2, so head_dim must be even; \
             got {head_dim}"
        );
        let dim = heads * head_dim;
        debug_assert!(q.len() >= n * dim && k.len() >= n * dim && v.len() >= n * dim);
        debug_assert!(qkv.len() >= n * 3 * dim);
        debug_assert!(cos.len() >= n * head_dim && sin.len() >= n * head_dim);
        let f = self
            .dev
            .kernels()
            .get("tuili_vision", vision_src(), "vision_qkv_rope_f32")?;
        let total = (n * heads * head_dim / 2) as u32;
        let cfg = LaunchConfig {
            grid_dim: (total.div_ceil(ELEMENTWISE_BLOCK).max(1), 1, 1),
            block_dim: (ELEMENTWISE_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let (nn, h, hd) = (n as i32, heads as i32, head_dim as i32);
        let mut bl = self.dev.stream().launch_builder(&f);
        bl.arg(q)
            .arg(k)
            .arg(v)
            .arg(qkv)
            .arg(cos)
            .arg(sin)
            .arg(&nn)
            .arg(&h)
            .arg(&hd);
        self.dev
            .profile()
            .time("vision_qkv_rope", self.dev.stream(), || {
                unsafe { bl.launch(cfg) }.context("vision_qkv_rope")?;
                Ok(())
            })?;
        Ok(())
    }

    /// Bidirectional attention within each segment of `segs`.
    ///
    /// `q`, `k`, `v` and `out` are `[n, heads, head_dim]`. **Not causal, and not
    /// across segments.** The `attn_*` family in `ops.cu` is both causal and
    /// KV-cache shaped; a causal mask here moves block 0's output by 2.08 out of
    /// a peak of 1.87.
    #[allow(clippy::too_many_arguments)]
    pub fn vision_attn(
        &self,
        out: &mut ViewMut<'_, f32>,
        q: &View<'_, f32>,
        k: &View<'_, f32>,
        v: &View<'_, f32>,
        segs: &VisionSegments,
        heads: usize,
        head_dim: usize,
    ) -> Result<()> {
        anyhow::ensure!(
            head_dim <= 128,
            "the attention kernel keeps its output accumulator in four \
             registers a lane, so head_dim must be at most 128; got {head_dim}"
        );
        let dim = heads * head_dim;
        debug_assert!(out.len() >= segs.total * dim);
        debug_assert!(q.len() >= segs.total * dim);
        debug_assert!(k.len() >= segs.total * dim && v.len() >= segs.total * dim);
        let f = self
            .dev
            .kernels()
            .get("tuili_vision", vision_src(), "vision_attn_f32")?;
        // Two key tiles and one query tile, each row padded by one float.
        let shared = (2 * VISION_ATTN_BK + VISION_ATTN_BQ) * (head_dim + 1);
        let cfg = LaunchConfig {
            grid_dim: (segs.tiles as u32, heads as u32, 1),
            block_dim: (VISION_ATTN_THREADS, 1, 1),
            shared_mem_bytes: (shared * std::mem::size_of::<f32>()) as u32,
        };
        let (h, hd) = (heads as i32, head_dim as i32);
        let scale = (head_dim as f32).sqrt().recip();
        let (a, b, q0) = (
            segs.tile_a.as_view(),
            segs.tile_b.as_view(),
            segs.tile_q0.as_view(),
        );
        let mut bl = self.dev.stream().launch_builder(&f);
        bl.arg(out)
            .arg(q)
            .arg(k)
            .arg(v)
            .arg(&a)
            .arg(&b)
            .arg(&q0)
            .arg(&h)
            .arg(&hd)
            .arg(&scale);
        self.dev
            .profile()
            .time("vision_attn", self.dev.stream(), || {
                unsafe { bl.launch(cfg) }.context("vision_attn")?;
                Ok(())
            })?;
        Ok(())
    }

    /// One `[C, H, W]` frame (or a pair) into `[grid_h * grid_w, patch_dim]`.
    ///
    /// Patches come out in **spatial-merge-block order**, four consecutive
    /// patches to a 2x2 square, and each patch's components in `(c, t, y, x)`
    /// order. `frame_stride` is the element distance between frames in `frames`
    /// and `n_frames` how many it holds; temporal tap `t` reads frame
    /// `min(t, n_frames - 1)`, so a still image sees the same pixels in both
    /// taps, which is what the processor's `expand` does.
    #[allow(clippy::too_many_arguments)]
    pub fn vision_patchify(
        &self,
        out: &mut ViewMut<'_, f32>,
        out_h: &mut ViewMut<'_, f16>,
        frames: &View<'_, f32>,
        n_frames: usize,
        height: usize,
        width: usize,
        shape: &VisionShape,
    ) -> Result<()> {
        anyhow::ensure!(
            height.is_multiple_of(shape.patch * shape.merge)
                && width.is_multiple_of(shape.patch * shape.merge),
            "a {height}x{width} frame does not tile into whole {}x{} \
             spatial-merge blocks; smart_resize rounds to patch * merge = {}, \
             not to patch",
            shape.patch * shape.merge,
            shape.patch * shape.merge,
            shape.patch * shape.merge
        );
        let (gh, gw) = (height / shape.patch, width / shape.patch);
        let pd = shape.patch_dim();
        let total = gh * gw * pd;
        debug_assert!(out.len() >= total && out_h.len() >= total);
        let frame_elems = shape.in_channels * height * width;
        debug_assert!(frames.len() >= n_frames * frame_elems);

        let f = self
            .dev
            .kernels()
            .get("tuili_vision", vision_src(), "vision_patchify_f32")?;
        let cfg = LaunchConfig {
            grid_dim: (
                (total as u32).div_ceil(ELEMENTWISE_BLOCK).max(1),
                1,
                1,
            ),
            block_dim: (ELEMENTWISE_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let stride = frame_elems as i64;
        let nf = n_frames as i32;
        let (hh, ww, ch) = (height as i32, width as i32, shape.in_channels as i32);
        let (p, tp, m) = (
            shape.patch as i32,
            shape.temporal_patch as i32,
            shape.merge as i32,
        );
        let (g_h, g_w) = (gh as i32, gw as i32);
        let mut bl = self.dev.stream().launch_builder(&f);
        bl.arg(out)
            .arg(out_h)
            .arg(frames)
            .arg(&stride)
            .arg(&nf)
            .arg(&hh)
            .arg(&ww)
            .arg(&ch)
            .arg(&p)
            .arg(&tp)
            .arg(&m)
            .arg(&g_h)
            .arg(&g_w);
        self.dev
            .profile()
            .time("vision_patchify", self.dev.stream(), || {
                unsafe { bl.launch(cfg) }.context("vision_patchify")?;
                Ok(())
            })?;
        Ok(())
    }

    /// `hidden += bilinear(pos_embed_table)`, per patch.
    ///
    /// The taps and weights come from the host's `pos_embed_taps`, which uses
    /// `align_corners = true` — the value the model's `__init__` passes and the
    /// opposite of the library helper's default. The false variant runs and
    /// moves the position field by 5.31 out of a peak of 6.60.
    #[allow(clippy::too_many_arguments)]
    pub fn vision_add_pos_embed(
        &self,
        hidden: &mut ViewMut<'_, f32>,
        table: &View<'_, f32>,
        idx: &View<'_, i32>,
        wts: &View<'_, f32>,
        n: usize,
        hidden_size: usize,
        taps: usize,
    ) -> Result<()> {
        debug_assert!(hidden.len() >= n * hidden_size);
        debug_assert!(idx.len() >= n * taps && wts.len() >= n * taps);
        let f = self
            .dev
            .kernels()
            .get("tuili_vision", vision_src(), "vision_add_pos_embed_f32")?;
        let cfg = LaunchConfig {
            grid_dim: (n as u32, 1, 1),
            block_dim: (ELEMENTWISE_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let (hs, tp) = (hidden_size as i32, taps as i32);
        let mut bl = self.dev.stream().launch_builder(&f);
        bl.arg(hidden).arg(table).arg(idx).arg(wts).arg(&hs).arg(&tp);
        self.dev
            .profile()
            .time("vision_add_pos_embed", self.dev.stream(), || {
                unsafe { bl.launch(cfg) }.context("vision_add_pos_embed")?;
                Ok(())
            })?;
        Ok(())
    }

    /// Overwrite the embedding of each placeholder token with the next feature
    /// row. `dst_row[f]` is the token index feature row `f` belongs to; build it
    /// with [`splice_targets`].
    pub fn vision_splice(
        &self,
        embeds: &mut ViewMut<'_, f32>,
        features: &View<'_, f32>,
        dst_row: &View<'_, i32>,
        out_hidden: usize,
        n_features: usize,
    ) -> Result<()> {
        debug_assert!(features.len() >= n_features * out_hidden);
        debug_assert!(dst_row.len() >= n_features);
        if n_features == 0 {
            return Ok(());
        }
        let f = self
            .dev
            .kernels()
            .get("tuili_vision", vision_src(), "vision_splice_f32")?;
        let cfg = LaunchConfig {
            grid_dim: (n_features as u32, 1, 1),
            block_dim: (ELEMENTWISE_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let (oh, nf) = (out_hidden as i32, n_features as i32);
        let mut bl = self.dev.stream().launch_builder(&f);
        bl.arg(embeds).arg(features).arg(dst_row).arg(&oh).arg(&nf);
        self.dev
            .profile()
            .time("vision_splice", self.dev.stream(), || {
                unsafe { bl.launch(cfg) }.context("vision_splice")?;
                Ok(())
            })?;
        Ok(())
    }
}

/// Token ids from this checkpoint's `config.json`.
///
/// **Not 151655 / 151656.** Those are Qwen2-VL's, and in this 248320-word
/// vocabulary they decode to Thai text fragments: copying them over leaves the
/// sequence length right, the placeholder count wrong, and the image features
/// spliced into nothing.
pub const IMAGE_TOKEN_ID: u32 = 248_056;
pub const VIDEO_TOKEN_ID: u32 = 248_057;

/// Which token each merger output row replaces.
///
/// Errors rather than truncating when the counts disagree, mirroring
/// `get_placeholder_mask`: a mismatch means the grid the tower ran on is not the
/// grid the prompt was built for.
pub fn splice_targets(input_ids: &[u32], n_features: usize) -> Result<Vec<i32>> {
    let rows: Vec<i32> = input_ids
        .iter()
        .enumerate()
        .filter(|&(_, &t)| t == IMAGE_TOKEN_ID || t == VIDEO_TOKEN_ID)
        .map(|(i, _)| i as i32)
        .collect();
    anyhow::ensure!(
        rows.len() == n_features,
        "{} placeholder tokens but {n_features} feature rows",
        rows.len()
    );
    Ok(rows)
}

/// The attention block-to-segment mapping, on the device.
///
/// Segments are ragged — one per *frame*, of `h * w` patches each — so a fixed
/// `blockIdx -> query` rule would let a tile straddle a boundary and mix two
/// frames. This precomputes, for every attention block, which segment it lives
/// in and which query it starts at. Building it on the host costs one pass over
/// the segment list.
pub struct VisionSegments {
    tile_a: Buf<i32>,
    tile_b: Buf<i32>,
    tile_q0: Buf<i32>,
    pub tiles: usize,
    pub total: usize,
    pub segments: usize,
}

impl VisionSegments {
    /// `cu` is the cumulative segment boundary list, `[0, l0, l0+l1, ...]` —
    /// what `tuili_model::qwen35_vision::cu_seqlens` returns. One segment a
    /// frame, not a grid entry: a `t`-frame video is `t` segments.
    pub fn new(dev: &tuili_gpu::Device, cu: &[usize]) -> Result<Self> {
        anyhow::ensure!(
            cu.first() == Some(&0),
            "cu_seqlens must start at 0; got {:?}",
            cu.first()
        );
        let (mut a, mut b, mut q0) = (Vec::new(), Vec::new(), Vec::new());
        for w in cu.windows(2) {
            anyhow::ensure!(w[1] >= w[0], "cu_seqlens is not monotonic: {cu:?}");
            let len = w[1] - w[0];
            for tile in 0..len.div_ceil(VISION_ATTN_BQ) {
                a.push(w[0] as i32);
                b.push(w[1] as i32);
                q0.push((w[0] + tile * VISION_ATTN_BQ) as i32);
            }
        }
        let stream = dev.stream();
        Ok(Self {
            tiles: a.len(),
            total: *cu.last().unwrap(),
            segments: cu.len() - 1,
            tile_a: stream.clone_htod(&a)?,
            tile_b: stream.clone_htod(&b)?,
            tile_q0: stream.clone_htod(&q0)?,
        })
    }
}

/// Everything about one call's geometry that the device needs: the rotary
/// tables, the position-embedding taps, and the attention segmentation.
///
/// The host arrays come from `tuili_model::qwen35_vision` — `cu_seqlens`,
/// `vision_position_ids`, `pos_embed_taps` — rather than being recomputed here.
/// One copy of the block-order arithmetic, and it is the copy the capture tests
/// check.
pub struct VisionGeometry {
    pub segs: VisionSegments,
    pub cos: Buf<f32>,
    pub sin: Buf<f32>,
    pub interp_idx: Buf<i32>,
    pub interp_wts: Buf<f32>,
    pub taps: usize,
}

impl VisionGeometry {
    pub fn new(
        k: &Kernels,
        shape: &VisionShape,
        cu: &[usize],
        pos_ids: &[u32],
        interp_idx: &[usize],
        interp_wts: &[f32],
    ) -> Result<Self> {
        let segs = VisionSegments::new(k.device(), cu)?;
        let n = segs.total;
        anyhow::ensure!(
            pos_ids.len() == 2 * n,
            "{} position ids for {n} patches; vision rope has exactly two axes \
             (h, w) and no temporal one",
            pos_ids.len()
        );
        anyhow::ensure!(
            interp_idx.len() == interp_wts.len() && interp_idx.len().is_multiple_of(n),
            "the position-embedding taps do not divide into {n} patches"
        );
        let taps = interp_idx.len() / n;
        let stream = k.device().stream();
        let dpos: Vec<i32> = pos_ids.iter().map(|v| *v as i32).collect();
        let dpos = stream.clone_htod(&dpos)?;
        let didx: Vec<i32> = interp_idx.iter().map(|v| *v as i32).collect();
        let mut cos = stream.alloc_zeros::<f32>(n * shape.head_dim())?;
        let mut sin = stream.alloc_zeros::<f32>(n * shape.head_dim())?;
        k.vision_rope_tables(
            &mut cos.as_view_mut(),
            &mut sin.as_view_mut(),
            &dpos.as_view(),
            n,
            shape.head_dim(),
            shape.rope_dim(),
            shape.rope_theta,
        )?;
        Ok(Self {
            segs,
            cos,
            sin,
            interp_idx: stream.clone_htod(&didx)?,
            interp_wts: stream.clone_htod(interp_wts)?,
            taps,
        })
    }
}

/// One block's weights, on the device. Matrices in f16, norms and biases in f32.
///
/// Every one of the six biases is real. The text tower has none at all
/// (`attention_bias: false`, bias-free MLPs), so a loader written for it drops
/// twelve tensors a block here — the same failure as the AWQ loader dropping
/// Qwen's QKV bias, and it reads as fluent nonsense. Dropping just
/// `patch_embed.proj.bias` moves the patch embedding by 3.05 out of a peak of
/// 3.15.
pub struct VisionBlockWeights<'a> {
    pub norm1_w: View<'a, f32>,
    pub norm1_b: View<'a, f32>,
    pub norm2_w: View<'a, f32>,
    pub norm2_b: View<'a, f32>,
    /// `[3 * hidden, hidden]`, row-major: `[all q | all k | all v]` rows.
    pub qkv_w: View<'a, f16>,
    pub qkv_b: View<'a, f32>,
    /// `[hidden, hidden]`.
    pub proj_w: View<'a, f16>,
    pub proj_b: View<'a, f32>,
    /// `[intermediate, hidden]`.
    pub fc1_w: View<'a, f16>,
    pub fc1_b: View<'a, f32>,
    /// `[hidden, intermediate]`.
    pub fc2_w: View<'a, f16>,
    pub fc2_b: View<'a, f32>,
}

/// The whole tower's weights.
///
/// 333 tensors, 460.7M parameters, all BF16 in the checkpoint — the whole tower
/// is in `modules_to_not_convert`, so there is no block-dequantization path to
/// write here. There is also no deepstack merger: `deepstack_visual_indexes` is
/// empty and the tensors do not exist, whatever `modules_to_not_convert` claims.
pub struct VisionWeights<'a> {
    /// `[hidden, patch_dim]` — `proj.weight` flattened, which is a free view of
    /// the checkpoint's `[1152, 3, 2, 16, 16]`. The patch embedding is a GEMM,
    /// not a convolution: kernel equals stride and the input arrives pre-tiled.
    pub patch_embed_w: View<'a, f16>,
    pub patch_embed_b: View<'a, f32>,
    /// `[num_position_embeddings, hidden]`, the learned 48x48 grid.
    pub pos_embed: View<'a, f32>,
    pub blocks: Vec<VisionBlockWeights<'a>>,
    /// `[hidden]` — **not** `[4 * hidden]`. The merger normalizes each patch
    /// before it groups them, and the checkpoint settles it: a post-shuffle norm
    /// would make this 4608 wide.
    pub merger_norm_w: View<'a, f32>,
    pub merger_norm_b: View<'a, f32>,
    /// `[4 * hidden, 4 * hidden]`.
    pub merger_fc1_w: View<'a, f16>,
    pub merger_fc1_b: View<'a, f32>,
    /// `[out_hidden, 4 * hidden]`.
    pub merger_fc2_w: View<'a, f16>,
    pub merger_fc2_b: View<'a, f32>,
}

/// Activation buffers for one vision call, sized by the patch count.
///
/// About 85 KB a patch: 69 KB of f32 and 16 KB of f16. A 1024x1024 image is 4096
/// patches and 350 MB, which is why the caller should run one image (or one
/// frame) at a time on a small card rather than packing a whole conversation's
/// worth. Frames are independent attention segments, so splitting on a frame
/// boundary changes nothing about the result.
pub struct VisionScratch {
    max_patches: usize,
    pixels_h: Buf<f16>,
    hidden: Buf<f32>,
    normed: Buf<f32>,
    normed_h: Buf<f16>,
    qkv: Buf<f32>,
    qkv_split: Buf<f32>,
    ctx: Buf<f32>,
    ctx_h: Buf<f16>,
    wide: Buf<f32>,
    wide_h: Buf<f16>,
    sub: Buf<f32>,
    features: Buf<f32>,
}

impl VisionScratch {
    pub fn new(dev: &tuili_gpu::Device, shape: &VisionShape, max_patches: usize) -> Result<Self> {
        anyhow::ensure!(
            max_patches.is_multiple_of(shape.merge_unit()),
            "the merger folds {} patches into a token, so a call's patch count \
             must be a multiple of it; got {max_patches}",
            shape.merge_unit()
        );
        let s = dev.stream();
        let (n, d) = (max_patches, shape.hidden);
        // `wide` carries both the MLP's intermediate (4304 a patch) and the
        // merger's fc1 output (4608 a token = 1152 a patch), so one buffer at
        // the larger of the two serves both.
        let wide_n = shape.intermediate.max(shape.merged() / shape.merge_unit());
        Ok(Self {
            max_patches,
            pixels_h: s.alloc_zeros::<f16>(n * shape.patch_dim())?,
            hidden: s.alloc_zeros::<f32>(n * d)?,
            normed: s.alloc_zeros::<f32>(n * d)?,
            normed_h: s.alloc_zeros::<f16>(n * d)?,
            qkv: s.alloc_zeros::<f32>(n * 3 * d)?,
            qkv_split: s.alloc_zeros::<f32>(n * 3 * d)?,
            ctx: s.alloc_zeros::<f32>(n * d)?,
            ctx_h: s.alloc_zeros::<f16>(n * d)?,
            wide: s.alloc_zeros::<f32>(n * wide_n)?,
            wide_h: s.alloc_zeros::<f16>(n * wide_n)?,
            sub: s.alloc_zeros::<f32>(n * d)?,
            features: s.alloc_zeros::<f32>(n / shape.merge_unit() * shape.out_hidden)?,
        })
    }

    /// The tower's `last_hidden_state`, `[n_patches, hidden]` — the residual
    /// stream before the merger.
    pub fn last_hidden(&self) -> View<'_, f32> {
        self.hidden.as_view()
    }

    /// The merger's output, `[n_patches / 4, out_hidden]`. This is what the
    /// language model consumes — `pooler_output`, not `last_hidden_state`.
    pub fn features(&self) -> View<'_, f32> {
        self.features.as_view()
    }

    /// The f16 `pixel_values` the patch embedding reads, for a caller that
    /// produces patches some other way than [`Kernels::vision_patchify`].
    pub fn pixels_h_mut(&mut self) -> ViewMut<'_, f16> {
        self.pixels_h.as_view_mut()
    }
}

/// The tower: patch embedding, `depth` blocks, merger.
///
/// `pixel_values` must already be in `scratch.pixels_h` (see
/// [`Kernels::vision_patchify`], which writes it). On return
/// `scratch.features()` holds `[n_patches / 4, out_hidden]`.
pub fn vision_forward(
    k: &Kernels,
    shape: &VisionShape,
    w: &VisionWeights<'_>,
    geo: &VisionGeometry,
    scratch: &mut VisionScratch,
) -> Result<()> {
    let n = geo.segs.total;
    let (d, hd, heads) = (shape.hidden, shape.head_dim(), shape.heads);
    anyhow::ensure!(
        n <= scratch.max_patches,
        "{n} patches but the scratch was sized for {}",
        scratch.max_patches
    );
    anyhow::ensure!(
        w.blocks.len() == shape.depth,
        "{} block weight sets for a depth of {}",
        w.blocks.len(),
        shape.depth
    );
    anyhow::ensure!(
        n.is_multiple_of(shape.merge_unit()),
        "{n} patches do not group into whole 2x2 blocks; smart_resize rounds to \
         patch * merge for exactly this reason"
    );

    // ---- patch embedding: one GEMM a patch, plus bias --------------------
    //
    // `proj` is a Conv3d whose kernel equals its stride over an input already
    // cut into patches, so there is no window, no padding and no overlap. The
    // convolutional reading is wasted work, not a different answer.
    k.gemm_f16(
        &mut scratch.hidden.as_view_mut(),
        &scratch.pixels_h.as_view(),
        &w.patch_embed_w,
        n,
        shape.patch_dim(),
        d,
    )?;
    k.add_bias(
        &mut scratch.hidden.as_view_mut(),
        &w.patch_embed_b,
        d,
        n,
    )?;
    k.vision_add_pos_embed(
        &mut scratch.hidden.as_view_mut(),
        &w.pos_embed,
        &geo.interp_idx.as_view(),
        &geo.interp_wts.as_view(),
        n,
        d,
        geo.taps,
    )?;

    // ---- the blocks -----------------------------------------------------
    for b in &w.blocks {
        // h = h + proj(attn(norm1(h))), residual on the *unnormalized* stream.
        k.vision_layer_norm(
            &mut scratch.normed.as_view_mut(),
            &mut scratch.normed_h.as_view_mut(),
            &scratch.hidden.as_view(),
            &b.norm1_w,
            &b.norm1_b,
            n,
            d,
            shape.eps,
        )?;
        k.gemm_f16(
            &mut scratch.qkv.as_view_mut(),
            &scratch.normed_h.as_view(),
            &b.qkv_w,
            n,
            d,
            3 * d,
        )?;
        k.add_bias(&mut scratch.qkv.as_view_mut(), &b.qkv_b, 3 * d, n)?;
        {
            // q, k, v are three consecutive slabs of one buffer; the kernel
            // takes them as separate pointers, so hand it three views.
            let (mut q, mut rest) = scratch.qkv_split.split_at_mut(n * d);
            let (mut kk, mut vv) = rest.split_at_mut(n * d);
            k.vision_qkv_rope(
                &mut q,
                &mut kk,
                &mut vv,
                &scratch.qkv.as_view(),
                &geo.cos.as_view(),
                &geo.sin.as_view(),
                n,
                heads,
                hd,
            )?;
        }
        {
            let (q, rest) = scratch.qkv_split.split_at(n * d);
            let (kk, vv) = rest.split_at(n * d);
            k.vision_attn(
                &mut scratch.ctx.as_view_mut(),
                &q,
                &kk,
                &vv,
                &geo.segs,
                heads,
                hd,
            )?;
        }
        k.to_f16(
            &mut scratch.ctx_h.as_view_mut(),
            &scratch.ctx.as_view(),
            n * d,
        )?;
        k.gemm_f16(
            &mut scratch.sub.as_view_mut(),
            &scratch.ctx_h.as_view(),
            &b.proj_w,
            n,
            d,
            d,
        )?;
        k.add_bias(&mut scratch.sub.as_view_mut(), &b.proj_b, d, n)?;
        k.add_assign(
            &mut scratch.hidden.as_view_mut(),
            &scratch.sub.as_view(),
            n * d,
        )?;

        // h = h + fc2(gelu_tanh(fc1(norm2(h)))). Two matrices, not three: the
        // text tower's SwiGLU would have to invent a split of 4304.
        k.vision_layer_norm(
            &mut scratch.normed.as_view_mut(),
            &mut scratch.normed_h.as_view_mut(),
            &scratch.hidden.as_view(),
            &b.norm2_w,
            &b.norm2_b,
            n,
            d,
            shape.eps,
        )?;
        k.gemm_f16(
            &mut scratch.wide.as_view_mut(),
            &scratch.normed_h.as_view(),
            &b.fc1_w,
            n,
            d,
            shape.intermediate,
        )?;
        // The tanh approximation, which is what `hidden_act` names. The merger
        // below uses the exact one.
        k.vision_gelu(
            &mut scratch.wide.as_view_mut(),
            &mut scratch.wide_h.as_view_mut(),
            &b.fc1_b,
            n,
            shape.intermediate,
            false,
        )?;
        k.gemm_f16(
            &mut scratch.sub.as_view_mut(),
            &scratch.wide_h.as_view(),
            &b.fc2_w,
            n,
            shape.intermediate,
            d,
        )?;
        k.add_bias(&mut scratch.sub.as_view_mut(), &b.fc2_b, d, n)?;
        k.add_assign(
            &mut scratch.hidden.as_view_mut(),
            &scratch.sub.as_view(),
            n * d,
        )?;
    }

    // ---- merger ---------------------------------------------------------
    //
    // LayerNorm over each patch's `hidden`, *then* group four patches into one
    // 4608-wide row. The grouping is a pure reshape — the same buffer read with
    // a wider row — and it is a 2x2 pooling only because preprocessing emitted
    // patches in block order. Normalizing the grouped 4608 instead (tiling the
    // gain four times to make the shapes fit) runs and moves the merger input by
    // 9.99 out of a peak of 6.81.
    let tokens = n / shape.merge_unit();
    let wide = shape.merged();
    k.vision_layer_norm(
        &mut scratch.normed.as_view_mut(),
        &mut scratch.normed_h.as_view_mut(),
        &scratch.hidden.as_view(),
        &w.merger_norm_w,
        &w.merger_norm_b,
        n,
        d,
        shape.eps,
    )?;
    k.gemm_f16(
        &mut scratch.wide.as_view_mut(),
        &scratch.normed_h.as_view(),
        &w.merger_fc1_w,
        tokens,
        wide,
        wide,
    )?;
    // The exact GELU — `nn.GELU()` with no `approximate` argument — against the
    // tanh one the 27 blocks above use. `hidden_act` describes only the blocks.
    k.vision_gelu(
        &mut scratch.wide.as_view_mut(),
        &mut scratch.wide_h.as_view_mut(),
        &w.merger_fc1_b,
        tokens,
        wide,
        true,
    )?;
    k.gemm_f16(
        &mut scratch.features.as_view_mut(),
        &scratch.wide_h.as_view(),
        &w.merger_fc2_w,
        tokens,
        wide,
        shape.out_hidden,
    )?;
    k.add_bias(
        &mut scratch.features.as_view_mut(),
        &w.merger_fc2_b,
        shape.out_hidden,
        tokens,
    )?;
    Ok(())
}
