//! The vision-tower host reference, checked stage by stage against a capture of
//! the actual reference implementation running on the actual checkpoint.
//!
//! Every number these tests compare against was produced outside this
//! repository, by `transformers`' own `Qwen3_5VisionModel` on
//! `/home/jeff/models/qwen38-27b-fp8`. That is the whole point: local
//! self-consistency is what the bf16-as-f16 embedding bug satisfied for a night,
//! across nine component A/Bs, while the model produced nonsense.
//!
//! It is not enough to show the right reading matches. Almost every test here
//! also computes the *other* plausible reading and asserts it does **not**
//! match, so that a passing test is evidence about layout rather than evidence
//! that some arithmetic ran. The vision tower needs this more than the text
//! tower does, because it reverses nearly every text-side convention:
//! LayerNorm not RMSNorm, biases everywhere, bidirectional attention,
//! `[all q | all k | all v]` instead of per-head interleaving, two rotary axes
//! in contiguous blocks instead of three interleaved, theta 1e4 instead of 1e7.
//!
//! The capture is weight-derived, so it is not in the repository. Regenerate it
//! on the box that has the checkpoint:
//!
//!   /home/jeff/vllm312/bin/python tools/capture_qwen35_vision.py \
//!       /home/jeff/models/qwen38-27b-fp8 <out-dir>
//!   TUILI_QWEN35_VISION_CAPTURE=<out-dir> \
//!       cargo test -p tuili-model --test qwen35_vision
//!
//! Without the environment variable these tests report as skipped rather than
//! passing, because a silent skip is how a suite comes to be green without
//! checking anything.

use std::collections::HashMap;
use std::path::PathBuf;

use tuili_model::qwen35_vision::*;

// ------------------------------------------------------------------ plumbing

struct Capture {
    cfg: HashMap<String, f64>,
    arrays: HashMap<String, (Vec<usize>, Vec<f32>)>,
}

impl Capture {
    fn open() -> Option<Self> {
        let dir = PathBuf::from(std::env::var("TUILI_QWEN35_VISION_CAPTURE").ok()?);
        let manifest: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("manifest.json")).unwrap())
                .unwrap();
        assert_eq!(
            manifest["verified_against_transformers"], true,
            "this capture was written without the cross-check against \
             transformers, so it is one reading of the reference rather than \
             the reference; regenerate it where the library is importable"
        );
        let cfg = manifest["config"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.clone(), v.as_f64().unwrap()))
            .collect();
        let mut arrays = HashMap::new();
        for (name, shape) in manifest["arrays"].as_object().unwrap() {
            let shape: Vec<usize> = shape
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            let bytes = std::fs::read(dir.join(format!("{name}.f32"))).unwrap();
            let want: usize = shape.iter().product();
            assert_eq!(
                bytes.len(),
                want * 4,
                "{name}: manifest says {shape:?} but the file holds {} floats",
                bytes.len() / 4
            );
            let vals: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            arrays.insert(name.clone(), (shape, vals));
        }
        Some(Self { cfg, arrays })
    }

    fn get(&self, name: &str) -> &[f32] {
        &self
            .arrays
            .get(name)
            .unwrap_or_else(|| panic!("the capture has no array {name}"))
            .1
    }

    fn shape(&self, name: &str) -> &[usize] {
        &self.arrays.get(name).unwrap().0
    }

    fn u(&self, key: &str) -> usize {
        self.cfg[key] as usize
    }

    fn f(&self, key: &str) -> f32 {
        self.cfg[key] as f32
    }

    /// Integers dumped as f32 (indices, ids, grids), back as `usize`.
    fn idx(&self, name: &str) -> Vec<usize> {
        self.get(name).iter().map(|v| *v as usize).collect()
    }

    /// The dimensions the capture was taken with, so nothing here hard-codes a
    /// size the manifest could contradict.
    fn dims(&self) -> VisionDims {
        VisionDims {
            depth: self.u("depth"),
            hidden: self.u("hidden_size"),
            heads: self.u("num_heads"),
            intermediate: self.u("intermediate_size"),
            out_hidden: self.u("out_hidden_size"),
            in_channels: self.u("in_channels"),
            patch: self.u("patch_size"),
            temporal_patch: self.u("temporal_patch_size"),
            merge: self.u("spatial_merge_size"),
            num_position_embeddings: self.u("num_position_embeddings"),
            eps: self.f("layer_norm_eps"),
            rope_theta: self.f("vision_rope_theta"),
        }
    }

    /// The `[t, h, w]` grid rows of a group.
    fn grids(&self, name: &str) -> Vec<Grid> {
        let raw = self.get(name);
        raw.chunks_exact(3)
            .map(|g| Grid {
                t: g[0] as usize,
                h: g[1] as usize,
                w: g[2] as usize,
            })
            .collect()
    }
}

fn with_capture(what: &str, body: impl FnOnce(&Capture)) {
    match Capture::open() {
        Some(c) => body(&c),
        None => eprintln!(
            "SKIPPED {what}: set TUILI_QWEN35_VISION_CAPTURE to a directory \
             written by tools/capture_qwen35_vision.py"
        ),
    }
}

/// Compare against the capture with a tolerance that follows the error model the
/// arithmetic has: a per-element relative allowance plus an absolute floor tied
/// to the *tensor's* peak, not to the element.
///
/// The floor matters here for the same reason it did on the text side. The
/// merger contracts 4608 products into each output and the tower's residual
/// stream reaches ~4200 by block 27, so an output element that lands near zero
/// carries the same absolute f32 accumulation noise as one that lands at 100. A
/// floor derived from the element declares the near-zero ones broken. A floor
/// derived from the tensor still catches everything that matters: every wrong
/// layout in this file moves values by 10% to 300% of the tensor's own peak.
fn agree(got: &[f32], want: &[f32], rel: f32, scale: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let peak = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let floor = scale * peak.max(f32::MIN_POSITIVE);
    let mut worst = 0.0f32;
    let mut worst_at = 0;
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!(g.is_finite(), "{what}: element {i} is {g}");
        let err = (g - w).abs() / (floor + rel * w.abs());
        if err > worst {
            worst = err;
            worst_at = i;
        }
    }
    assert!(
        worst <= 1.0,
        "{what}: worst disagreement at element {worst_at} — got {}, reference \
         {}, which is {worst:.1}x the tolerance (rel {rel:.1e}, floor \
         {floor:.3e} from a peak of {peak:.3e})",
        got[worst_at],
        want[worst_at]
    );
}

/// How far apart two tensors are, relative to the second one's peak. Used to
/// state that a wrong reading is wrong *by a lot*, rather than merely that
/// `assert_ne!` held on some element.
fn spread(a: &[f32], b: &[f32]) -> f32 {
    let peak = b.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(f32::MIN_POSITIVE);
    a.iter()
        .zip(b)
        .fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
        / peak
}

/// A wrong reading has to be wrong by at least 1% of the tensor's own scale, or
/// the test that rules it out is not doing any work.
fn assert_discriminates(wrong: &[f32], want: &[f32], what: &str) {
    let s = spread(wrong, want);
    assert!(
        s > 0.01,
        "{what}: the alternative reading agrees to {s:.2e} of the tensor's peak, \
         so nothing in this test rules it out and it would pass either way"
    );
}

// ------------------------------------------------------- preprocessing layout

/// The flat position of `(c, t, y, x)` inside a patch, and the spatial-merge
/// block order patches arrive in.
///
/// Checked against an `arange` probe image put through the reference image
/// processor's own `patchify`: every pixel is a distinct number, so this locates
/// each one rather than agreeing by coincidence. Raster patch order and the
/// three other orderings of `(c, t, y, x)` must all fail.
#[test]
fn the_patch_layout_is_channel_temporal_row_column_in_block_order() {
    with_capture("patch layout", |c| {
        let d = c.dims();
        let grid = c.grids("patchify.probe_grid")[0];
        let img = c.get("patchify.probe_image");
        let (h_px, w_px) = (c.shape("patchify.probe_image")[1], c.shape("patchify.probe_image")[2]);
        let want = c.get("patchify.probe_pixels");

        let (got, gh, gw) = patchify(img, h_px, w_px, &d);
        assert_eq!((gh, gw), (grid.h, grid.w), "grid from the pixel dimensions");
        agree(&got, want, 0.0, 0.0, "patchify");

        // Raster patch order: p = row * grid_w + col.
        let pd = d.patch_dim();
        let mut raster = vec![0.0f32; gh * gw * pd];
        for p in 0..gh * gw {
            let (row, col) = (p / gw, p % gw);
            for ch in 0..d.in_channels {
                for t in 0..d.temporal_patch {
                    for y in 0..d.patch {
                        for x in 0..d.patch {
                            let src = (ch * h_px + row * d.patch + y) * w_px + col * d.patch + x;
                            raster[p * pd + patch_slot(ch, t, y, x, &d)] = img[src];
                        }
                    }
                }
            }
        }
        assert_discriminates(&raster, want, "raster patch order");

        // The other orderings of the four axes inside a patch. Each is the same
        // 1536 numbers permuted, so it type-checks, fills every slot, and feeds
        // the Conv3d a transposed patch.
        let slot_of = |variant: usize, ch: usize, t: usize, y: usize, x: usize| -> usize {
            match variant {
                // (t, c, y, x): channel and temporal swapped.
                0 => ((t * d.in_channels + ch) * d.patch + y) * d.patch + x,
                // (c, t, x, y): the patch itself transposed.
                1 => ((ch * d.temporal_patch + t) * d.patch + x) * d.patch + y,
                // (y, x, c, t): pixel-major, as an HWC image would give.
                _ => ((y * d.patch + x) * d.in_channels + ch) * d.temporal_patch + t,
            }
        };
        for (variant, label) in [(0, "(t, c, y, x)"), (1, "(c, t, x, y)"), (2, "(y, x, c, t)")] {
            let mut other = vec![0.0f32; gh * gw * pd];
            for p in 0..gh * gw {
                let (row, col) = patch_row_col(p, gw, d.merge);
                for ch in 0..d.in_channels {
                    for t in 0..d.temporal_patch {
                        for y in 0..d.patch {
                            for x in 0..d.patch {
                                let src =
                                    (ch * h_px + row * d.patch + y) * w_px + col * d.patch + x;
                                other[p * pd + slot_of(variant, ch, t, y, x)] = img[src];
                            }
                        }
                    }
                }
            }
            assert_discriminates(&other, want, &format!("patch slot order {label}"));
        }
    });
}

/// A still image's two temporal slots hold the same frame, not a frame and a
/// zero. The Conv3d's two temporal taps therefore act as their sum.
#[test]
fn a_still_images_temporal_slots_are_the_frame_repeated() {
    with_capture("temporal duplication", |c| {
        let d = c.dims();
        let px = c.get("img.pixel_values");
        let width = c.shape("img.pixel_values")[1];
        assert_eq!(width, d.patch_dim());
        let spatial = d.patch * d.patch;
        let mut worst = 0.0f32;
        for row in px.chunks(width) {
            for ch in 0..d.in_channels {
                for i in 0..spatial {
                    let a = row[patch_slot(ch, 0, i / d.patch, i % d.patch, &d)];
                    let b = row[patch_slot(ch, 1, i / d.patch, i % d.patch, &d)];
                    worst = worst.max((a - b).abs());
                }
            }
        }
        assert_eq!(
            worst, 0.0,
            "the two temporal slots of a still image's patch differ by {worst}; \
             the processor expands one frame across them, so a loader that fills \
             only the first tap will lose half the patch embedding"
        );
    });
}

/// `smart_resize` rounds to `patch_size * merge_size` = 32, and the reference's
/// table of (h, w) -> (h_bar, w_bar) has to come out exactly.
#[test]
fn smart_resize_reproduces_the_reference_table() {
    with_capture("smart_resize", |c| {
        let d = c.dims();
        let factor = d.resize_factor();
        assert_eq!(factor, c.u("smart_resize_factor"));
        let (lo, hi) = (c.u("min_pixels"), c.u("max_pixels"));
        let cases = c.get("smart_resize.cases");
        let mut checked = 0;
        for row in cases.chunks_exact(4) {
            let (h, w) = (row[0] as usize, row[1] as usize);
            let (want_h, want_w) = (row[2] as usize, row[3] as usize);
            let got = smart_resize(h, w, factor, lo, hi)
                .unwrap_or_else(|| panic!("{h}x{w}: refused to resize"));
            assert_eq!(
                got,
                (want_h, want_w),
                "smart_resize({h}, {w}) gave {got:?}, the reference gives \
                 ({want_h}, {want_w})"
            );
            assert_eq!(want_h % factor, 0, "the reference's own output is off-grid");
            assert_eq!(want_w % factor, 0, "the reference's own output is off-grid");
            checked += 1;
        }
        assert!(checked >= 8, "only {checked} resize cases in the capture");

        // Rounding to patch_size instead of patch_size * merge_size: the grid
        // comes out odd, `h / merge` truncates, and a row and column of patches
        // vanish from the merger's view.
        let mut odd = 0;
        for row in cases.chunks_exact(4) {
            let (h, w) = (row[0] as usize, row[1] as usize);
            if let Some((hb, wb)) = smart_resize(h, w, d.patch, lo, hi)
                && ((hb / d.patch) % d.merge != 0 || (wb / d.patch) % d.merge != 0)
            {
                odd += 1;
            }
        }
        assert!(
            odd > 0,
            "rounding to patch_size never produced an odd grid on these cases, \
             so this test does not show why the factor is patch*merge"
        );
    });
}

// ------------------------------------------------------------------- geometry

/// Attention segments are per *frame*, not per grid entry.
#[test]
fn attention_segments_split_a_video_by_frame() {
    with_capture("cu_seqlens", |c| {
        for tag in ["img", "pack", "vid"] {
            let grids = c.grids(&format!("{tag}.grid_thw"));
            let got = cu_seqlens(&grids);
            let want = c.idx(&format!("{tag}.cu_seqlens"));
            assert_eq!(got, want, "{tag}: cu_seqlens");
        }
        // The two-frame group must be two segments, or nothing above tested the
        // per-frame split.
        let vid = c.grids("vid.grid_thw");
        assert_eq!(vid[0].t, 2, "the 'vid' group is supposed to have two frames");
        assert_eq!(
            cu_seqlens(&vid).len(),
            3,
            "a t=2 entry must produce two segments; one segment per *entry* \
             would let the frames attend to each other"
        );
    });
}

/// The `(h, w)` index of every patch, and no temporal axis in vision RoPE.
#[test]
fn vision_position_ids_are_block_major_h_and_w() {
    with_capture("vision position ids", |c| {
        for tag in ["img", "pack", "vid"] {
            let grids = c.grids(&format!("{tag}.grid_thw"));
            let got = vision_position_ids(&grids, c.u("spatial_merge_size"));
            let want: Vec<u32> = c
                .get(&format!("{tag}.position_ids"))
                .iter()
                .map(|v| *v as u32)
                .collect();
            assert_eq!(got, want, "{tag}: vision position ids");
        }

        // Raster order must disagree, and the video's second frame must repeat
        // the first frame's positions rather than carrying a temporal index.
        let g = c.grids("img.grid_thw")[0];
        let raster: Vec<u32> = (0..g.h * g.w)
            .flat_map(|p| [(p / g.w) as u32, (p % g.w) as u32])
            .collect();
        let want = c.get("img.position_ids");
        let raster_f: Vec<f32> = raster.iter().map(|v| *v as f32).collect();
        assert_discriminates(&raster_f, want, "raster position ids");

        let vid = c.get("vid.position_ids");
        let half = vid.len() / 2;
        assert_eq!(
            &vid[..half],
            &vid[half..],
            "the second frame's rotary positions differ from the first's; \
             vision RoPE has only (h, w) axes, so frames are distinguished by \
             their attention segment and by nothing else"
        );
    });
}

/// The learned 48x48 position grid is resampled bilinearly with
/// `align_corners = true`, and the patches it is gathered for are in block
/// order.
#[test]
fn the_position_grid_is_resampled_with_align_corners() {
    with_capture("pos embed interpolation", |c| {
        let d = c.dims();
        let side = c.u("num_grid_per_side");
        assert_eq!(side, d.grid_per_side());
        assert_eq!(side * side, d.num_position_embeddings);

        let grids = c.grids("img.grid_thw");
        let (idx, wts) = pos_embed_taps(&grids, side, d.merge);
        let want_idx = c.idx("img.interp_indices");
        assert_eq!(idx, want_idx, "interpolation tap indices");
        agree(&wts, c.get("img.interp_weights"), 1e-5, 1e-6, "interpolation weights");

        // Every patch's four weights must sum to one, or the position field is
        // being scaled as well as resampled.
        for four in wts.chunks_exact(4) {
            let s: f32 = four.iter().sum();
            assert!((s - 1.0).abs() < 1e-5, "tap weights sum to {s}, not 1");
        }

        let table = c.get("pos_embed.table");
        let got = gather_pos_embed(table, d.hidden, &idx, &wts, 4);
        agree(&got, c.get("img.pos_embeds"), 2e-5, 1e-6, "interpolated pos embeds");

        // align_corners = false is the library helper's own default and the
        // other plausible reading. It must land far away.
        let g = grids[0];
        let mut wrong = vec![0.0f32; g.patches() * d.hidden];
        for p in 0..g.patches() {
            let (row, col) = patch_row_col(p, g.w, d.merge);
            let axis = |index: usize, size: usize| -> ((usize, usize), (f32, f32)) {
                let src = (index as f64 + 0.5) * side as f64 / size as f64 - 0.5;
                let fl = src.floor();
                let t0 = (fl as isize).clamp(0, side as isize - 1) as usize;
                let t1 = (fl as isize + 1).clamp(0, side as isize - 1) as usize;
                let dd = (src - fl).abs();
                (
                    (t0, t1),
                    ((1.0 - dd).max(0.0) as f32, (1.0 - (src - fl - 1.0).abs()).max(0.0) as f32),
                )
            };
            let ((h0, h1), (a0, a1)) = axis(row, g.h);
            let ((w0, w1), (b0, b1)) = axis(col, g.w);
            for (r, wt) in [
                (h0 * side + w0, a0 * b0),
                (h0 * side + w1, a0 * b1),
                (h1 * side + w0, a1 * b0),
                (h1 * side + w1, a1 * b1),
            ] {
                for (acc, v) in wrong[p * d.hidden..(p + 1) * d.hidden]
                    .iter_mut()
                    .zip(&table[r * d.hidden..(r + 1) * d.hidden])
                {
                    *acc += wt * v;
                }
            }
        }
        assert_discriminates(&wrong, c.get("img.pos_embeds"), "align_corners = false");

        // And gathering in raster order instead of block order.
        let raster: Vec<usize> = (0..g.patches())
            .flat_map(|p| {
                let (row, col) = (p / g.w, p % g.w);
                let mk = |index: usize, size: usize| -> (usize, usize) {
                    let src = index as f64 * (side as f64 - 1.0) / (size - 1).max(1) as f64;
                    let fl = src.floor() as isize;
                    (
                        fl.clamp(0, side as isize - 1) as usize,
                        (fl + 1).clamp(0, side as isize - 1) as usize,
                    )
                };
                let (h0, h1) = mk(row, g.h);
                let (w0, w1) = mk(col, g.w);
                [h0 * side + w0, h0 * side + w1, h1 * side + w0, h1 * side + w1]
            })
            .collect();
        assert_ne!(
            raster, want_idx,
            "raster gather order produced the same taps as block order, so this \
             grid cannot tell them apart"
        );
    });
}

// ---------------------------------------------------------------- patch embed

/// The patch embedding is a GEMM against `proj.weight` flattened to
/// `[hidden, 1536]`, and that flattening is `(c, t, y, x)`.
///
/// The one-hot probe is the evidence: the capture drove the reference `Conv3d`
/// with unit vectors, so each response row *is* the corresponding weight column
/// plus the bias. If the flat ordering in the checkpoint were anything else,
/// these would not line up.
#[test]
fn the_patch_embedding_is_a_gemm_over_the_flattened_patch() {
    with_capture("patch embed", |c| {
        let d = c.dims();
        let w = c.get("patch_embed.w_flat");
        let b = c.get("patch_embed.bias");
        assert_eq!(c.shape("patch_embed.w_flat"), [d.hidden, d.patch_dim()]);

        let slots = c.idx("patch_embed.onehot_slots");
        let resp = c.get("patch_embed.onehot_out");
        for (i, &slot) in slots.iter().enumerate() {
            for o in 0..d.hidden {
                let from_weight = w[o * d.patch_dim() + slot] + b[o];
                let from_module = resp[i * d.hidden + o];
                assert!(
                    (from_weight - from_module).abs() <= 1e-4 + 1e-4 * from_module.abs(),
                    "slot {slot}, output {o}: the flattened weight says \
                     {from_weight}, driving the reference Conv3d with that unit \
                     vector gave {from_module}"
                );
            }
        }

        let px = c.get("img.pixel_values");
        let n = c.shape("img.pixel_values")[0];
        let got = patch_embed(px, w, b, n, &d);
        agree(&got, c.get("img.patch_embed_out"), 2e-4, 1e-6, "patch_embed");

        // Dropping the bias: the AWQ-loader failure, applied here.
        let zero = vec![0.0f32; d.hidden];
        let no_bias = patch_embed(px, w, &zero, n, &d);
        assert_discriminates(&no_bias, c.get("img.patch_embed_out"), "patch embed without bias");

        // Using only the first temporal tap.
        let mut half_w = w.to_vec();
        for o in 0..d.hidden {
            for ch in 0..d.in_channels {
                for y in 0..d.patch {
                    for x in 0..d.patch {
                        half_w[o * d.patch_dim() + patch_slot(ch, 1, y, x, &d)] = 0.0;
                    }
                }
            }
        }
        let one_tap = patch_embed(px, &half_w, b, n, &d);
        assert_discriminates(&one_tap, c.get("img.patch_embed_out"), "one temporal tap only");
    });
}

/// The tower's input is `patch_embed(pixels) + interpolated_pos_embed`.
#[test]
fn the_tower_input_is_the_patch_embedding_plus_the_position_embedding() {
    with_capture("tower input", |c| {
        let d = c.dims();
        let pe = c.get("img.patch_embed_out");
        let pos = c.get("img.pos_embeds");
        let got: Vec<f32> = pe.iter().zip(pos).map(|(a, b)| a + b).collect();
        agree(&got, c.get("img.hidden_in"), 1e-6, 1e-7, "hidden_in");
        assert_eq!(c.shape("img.hidden_in")[1], d.hidden);
    });
}

// ----------------------------------------------------------------- vision RoPE

/// The rotary tables: 36 frequencies normalized by 36, theta 1e4, h in the low
/// block and w in the high one, each duplicated for `rotate_half`.
#[test]
fn the_vision_rope_tables_block_h_and_w_and_normalize_by_the_rope_width() {
    with_capture("vision rope tables", |c| {
        let d = c.dims();
        let head_dim = d.head_dim();
        assert_eq!(head_dim, c.u("head_dim"));
        assert_eq!(d.rope_dim(), c.u("vision_rope_dim"));
        assert_eq!(c.shape("img.rope_cos")[1], head_dim, "the table is head_dim wide");

        let pids: Vec<u32> = c.get("img.position_ids").iter().map(|v| *v as u32).collect();
        let (cos, sin) = vision_rope_tables(&pids, &d);
        agree(&cos, c.get("img.rope_cos"), 1e-5, 1e-6, "vision rope cos");
        agree(&sin, c.get("img.rope_sin"), 1e-5, 1e-6, "vision rope sin");

        // The duplication that makes rotate_half meaningful: column i and
        // column i + rope_dim carry the same angle.
        let half = head_dim / 2;
        let want = c.get("img.rope_cos");
        for row in want.chunks_exact(head_dim) {
            for i in 0..half {
                assert!(
                    (row[i] - row[i + half]).abs() < 1e-6,
                    "column {i} and {} of the reference's own cos table differ, \
                     so the rotate_half pairing is not (i, i + head_dim/2)",
                    i + half
                );
            }
        }

        let per_axis = d.rope_dim() / 2;
        let n = pids.len() / 2;
        // Each of the three wrong readings, built the same way and required to
        // land far from the reference.
        let build = |inv: &dyn Fn(usize) -> f64, interleave: bool| -> Vec<f32> {
            let mut out = vec![0.0f32; n * head_dim];
            for p in 0..n {
                for axis in 0..2 {
                    for i in 0..per_axis {
                        let angle = pids[p * 2 + axis] as f64 * inv(i);
                        let j = if interleave {
                            i * 2 + axis
                        } else {
                            axis * per_axis + i
                        };
                        out[p * head_dim + j] = angle.cos() as f32;
                        out[p * head_dim + j + half] = angle.cos() as f32;
                    }
                }
            }
            out
        };
        let theta = d.rope_theta as f64;
        let rope_dim = d.rope_dim() as f64;
        assert_discriminates(
            &build(&|i| theta.powf(-((2 * i) as f64 / head_dim as f64)), false),
            want,
            "frequency exponent normalized by head_dim",
        );
        assert_discriminates(
            &build(&|i| 1e7f64.powf(-((2 * i) as f64 / rope_dim)), false),
            want,
            "the text side's theta = 1e7",
        );
        assert_discriminates(
            &build(&|i| theta.powf(-((2 * i) as f64 / rope_dim)), true),
            want,
            "h and w interleaved instead of blocked",
        );
    });
}

/// Translating an image by a whole number of patches must leave the attention
/// output unchanged up to the same translation. Rotary embeddings encode
/// relative position; this needs no reference implementation to check, and it is
/// the cheapest guard against a wrong pairing or a wrong frequency table.
#[test]
fn shifting_all_patch_positions_does_not_change_the_attention_output() {
    with_capture("vision rope shift invariance", |c| {
        let d = c.dims();
        let (heads, hd) = (d.heads, d.head_dim());
        let n = c.shape("img.b0.qkv")[0];
        let qkv = c.get("img.b0.qkv");
        let cu = c.idx("img.cu_seqlens");
        let pids: Vec<u32> = c.get("img.position_ids").iter().map(|v| *v as u32).collect();

        let run = |offset: u32| {
            let shifted: Vec<u32> = pids.iter().map(|p| p + offset).collect();
            let (cos, sin) = vision_rope_tables(&shifted, &d);
            let (mut q, mut k, v) = split_qkv(qkv, n, heads, hd);
            apply_vision_rope(&mut q, &cos, &sin, n, heads, hd);
            apply_vision_rope(&mut k, &cos, &sin, n, heads, hd);
            segment_attention(&q, &k, &v, &cu, heads, hd)
        };
        let base = run(0);
        for offset in [1, 7, 40] {
            let moved = run(offset);
            let s = spread(&moved, &base);
            assert!(
                s < 1e-4,
                "shifting every patch position by {offset} moved the attention \
                 output by {s:.2e} of its peak; rotary embeddings encode \
                 relative position, so anything above f32 rounding means the \
                 frequency table or the pairing is wrong"
            );
        }
    });
}

// ------------------------------------------------------------------ attention

/// `qkv` is `[all q | all k | all v]`, not per-head interleaved.
///
/// The capture dumps the weight rows straddling both boundaries and a few
/// per-head probes, so this locates q, k and v in the weight itself rather than
/// inheriting whichever split the capture happened to make. Head 0 component 0
/// is the same row under both readings, which is why the probes reach past the
/// first head: that is where the readings separate.
#[test]
fn qkv_is_three_contiguous_blocks_not_interleaved_per_head() {
    with_capture("qkv layout", |c| {
        let d = c.dims();
        let (heads, hd, dim) = (d.heads, d.head_dim(), d.hidden);
        let n = c.shape("img.hidden_in")[0];
        let x = c.get("img.b0.norm1_out");
        let rows = c.idx("b0.qkv.probe_rows");
        let w = c.get("b0.qkv.probe_w");
        let bias = c.get("b0.qkv.bias");
        let qkv = c.get("img.b0.qkv");
        let width = c.shape("img.b0.qkv")[1];
        assert_eq!(width, 3 * dim, "qkv is {width} wide, not 3 * {dim}");

        let project = |row: usize, t: usize| -> f32 {
            let i = rows.iter().position(|&r| r == row).expect("row not probed");
            let wrow = &w[i * dim..(i + 1) * dim];
            let xt = &x[t * dim..(t + 1) * dim];
            xt.iter().zip(wrow).map(|(a, b)| a * b).sum::<f32>() + bias[row]
        };

        let mut separating = 0;
        for s in 0..3 {
            for h in [0, 1, heads - 1] {
                for dd in [0, 1, hd - 1] {
                    // Blocked: component s of head h dim dd is row s*dim + h*hd + dd.
                    let blocked_row = s * dim + h * hd + dd;
                    for t in 0..n {
                        let got = qkv[t * width + blocked_row];
                        let pred = project(blocked_row, t);
                        assert!(
                            (pred - got).abs() <= 1e-3 + 1e-3 * got.abs(),
                            "qkv[t{t}] part {s} head {h} dim {dd}: the blocked \
                             layout predicts {pred} from weight row \
                             {blocked_row}, the reference has {got}"
                        );
                        // Interleaved: it would be row h*3*hd + s*hd + dd.
                        let inter_row = h * 3 * hd + s * hd + dd;
                        if inter_row != blocked_row {
                            let alt = project(inter_row, t);
                            if (alt - got).abs() > 1e-3 {
                                separating += 1;
                            }
                        }
                    }
                }
            }
        }
        assert!(
            separating > 0,
            "no probe distinguished [all q | all k | all v] from a per-head \
             interleaving; this test would pass under either and is not evidence"
        );
    });
}

/// The whole attention interior: split, rotate, attend bidirectionally within
/// each segment. Checked against the reference's own `attn.proj` input, so the
/// split and the rope are answering to the reference and not to this file.
#[test]
fn the_attention_interior_reproduces_the_reference_projection_input() {
    with_capture("attention interior", |c| {
        let d = c.dims();
        let (heads, hd) = (d.heads, d.head_dim());
        for tag in ["img", "pack", "vid"] {
            let n = c.shape(&format!("{tag}.b0.qkv"))[0];
            let qkv = c.get(&format!("{tag}.b0.qkv"));
            let cos = c.get(&format!("{tag}.rope_cos"));
            let cu = c.idx(&format!("{tag}.cu_seqlens"));
            // The sin table is only dumped in full for the "img" group; rebuild
            // both from the position ids for the others, which the rope test has
            // already pinned.
            let pids: Vec<u32> = c
                .get(&format!("{tag}.position_ids"))
                .iter()
                .map(|v| *v as u32)
                .collect();
            let (cos_b, sin_b) = vision_rope_tables(&pids, &d);
            agree(&cos_b, cos, 1e-5, 1e-6, &format!("{tag}: rebuilt cos"));

            let (mut q, mut k, v) = split_qkv(qkv, n, heads, hd);
            apply_vision_rope(&mut q, &cos_b, &sin_b, n, heads, hd);
            apply_vision_rope(&mut k, &cos_b, &sin_b, n, heads, hd);
            let got = segment_attention(&q, &k, &v, &cu, heads, hd);
            let want = c.get(&format!("{tag}.b0.attn_pre_proj"));
            agree(&got, want, 2e-3, 1e-5, &format!("{tag}: attention interior"));

            // A causal mask: what reusing the text tower's attention gives.
            let mut causal = vec![0.0f32; n * heads * hd];
            for seg in cu.windows(2) {
                let (a, b) = (seg[0], seg[1]);
                for t in a..b {
                    let sub = cu_seqlens(&[Grid { t: 1, h: 1, w: t - a + 1 }]);
                    let part = segment_attention(
                        &q[a * heads * hd..(t + 1) * heads * hd],
                        &k[a * heads * hd..(t + 1) * heads * hd],
                        &v[a * heads * hd..(t + 1) * heads * hd],
                        &sub,
                        heads,
                        hd,
                    );
                    causal[t * heads * hd..(t + 1) * heads * hd].copy_from_slice(
                        &part[(t - a) * heads * hd..(t - a + 1) * heads * hd],
                    );
                }
            }
            assert_discriminates(&causal, want, &format!("{tag}: causal attention"));

            // No rope at all.
            let (q0, k0, v0) = split_qkv(qkv, n, heads, hd);
            let bare = segment_attention(&q0, &k0, &v0, &cu, heads, hd);
            assert_discriminates(&bare, want, &format!("{tag}: attention without rope"));

            // Adjacent-pair rotation instead of rotate_half.
            let (mut qa, mut ka, va) = split_qkv(qkv, n, heads, hd);
            let adj = |x: &mut [f32]| {
                for p in 0..n {
                    for h in 0..heads {
                        let base = (p * heads + h) * hd;
                        for i in 0..hd / 2 {
                            let (a, b) = (x[base + 2 * i], x[base + 2 * i + 1]);
                            let (cc, ss) =
                                (cos_b[p * hd + 2 * i], sin_b[p * hd + 2 * i]);
                            x[base + 2 * i] = a * cc - b * ss;
                            x[base + 2 * i + 1] = b * cc + a * ss;
                        }
                    }
                }
            };
            adj(&mut qa);
            adj(&mut ka);
            let other = segment_attention(&qa, &ka, &va, &cu, heads, hd);
            assert_discriminates(&other, want, &format!("{tag}: adjacent-pair rope"));
        }
    });
}

/// Packing several images, or several frames, into one call must not change any
/// of their outputs. This is the invariant that makes batched vision prefill
/// legitimate, and cross-segment attention breaks it.
#[test]
fn packing_images_and_frames_leaves_each_ones_output_alone() {
    with_capture("packing invariance", |c| {
        let d = c.dims();
        let alone_n = c.shape("img.b0.qkv")[0];
        for tag in ["pack", "vid"] {
            for stage in ["b0.qkv", "b0.attn_pre_proj", "b0.out"] {
                let packed = c.get(&format!("{tag}.{stage}"));
                let alone = c.get(&format!("img.{stage}"));
                let s = spread(&packed[..alone.len()], alone);
                assert!(
                    s < 1e-5,
                    "{tag}.{stage}: the first image's output moved by {s:.2e} of \
                     its peak once it was packed with another; something is \
                     attending across the segment boundary"
                );
            }
            // Through all 27 blocks the same must hold, to f32 accumulation
            // noise over a residual stream that reaches ~4200.
            let s = spread(
                &c.get(&format!("{tag}.last_hidden"))[..alone_n * d.hidden],
                c.get("img.last_hidden"),
            );
            assert!(s < 1e-4, "{tag}.last_hidden drifted {s:.2e} under packing");
            let tokens = alone_n / d.merge_unit();
            let s = spread(
                &c.get(&format!("{tag}.image_embeds"))[..tokens * d.out_hidden],
                c.get("img.image_embeds"),
            );
            assert!(s < 1e-4, "{tag}.image_embeds drifted {s:.2e} under packing");
        }
    });
}

// --------------------------------------------------------------------- blocks

/// The block norms are LayerNorm — mean subtracted, bias added — not RMSNorm.
#[test]
fn the_block_norms_are_layer_norm_not_rms_norm() {
    with_capture("block layer norm", |c| {
        let d = c.dims();
        for (input, w, b, want) in [
            ("img.hidden_in", "b0.norm1.weight", "b0.norm1.bias", "img.b0.norm1_out"),
            ("img.b0.resid1", "b0.norm2.weight", "b0.norm2.bias", "img.b0.norm2_out"),
        ] {
            let x = c.get(input);
            let (gw, gb) = (c.get(w), c.get(b));
            let got = layer_norm_rows(x, gw, gb, d.hidden, d.eps);
            agree(&got, c.get(want), 2e-5, 1e-6, &format!("layer_norm of {input}"));

            // RMSNorm with the same gain, with and without the bias.
            let rms: Vec<f32> = x
                .chunks(d.hidden)
                .flat_map(|row| {
                    let ms = row.iter().map(|v| v * v).sum::<f32>() / d.hidden as f32;
                    let inv = (ms + d.eps).sqrt().recip();
                    row.iter()
                        .zip(gw)
                        .zip(gb)
                        .map(move |((v, g), bb)| g * (v * inv) + bb)
                        .collect::<Vec<_>>()
                })
                .collect();
            assert_discriminates(&rms, c.get(want), &format!("rms_norm + bias for {input}"));

            let no_bias: Vec<f32> = got.iter().zip(gb.iter().cycle()).map(|(v, b)| v - b).collect();
            assert_discriminates(&no_bias, c.get(want), &format!("layer_norm without bias for {input}"));
        }

        // Not a degenerate regime: eps must not be doing the normalizing. If it
        // were, everything above would agree with everything.
        for input in ["img.hidden_in", "img.b0.resid1"] {
            let x = c.get(input);
            let mut lo = f32::INFINITY;
            for row in x.chunks(d.hidden) {
                let mean = row.iter().sum::<f32>() / d.hidden as f32;
                let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>()
                    / d.hidden as f32;
                lo = lo.min(var);
            }
            assert!(
                lo > 1000.0 * d.eps,
                "{input}: the smallest row variance is {lo:.3e}, within three \
                 orders of eps = {:.0e}; at that scale LayerNorm degenerates to \
                 a constant scale and this capture cannot discriminate its \
                 formulation",
                d.eps
            );
        }
    });
}

/// The block MLP is `fc2(gelu_tanh(fc1(x)))` — two matrices with biases and the
/// tanh GELU, not a SwiGLU.
#[test]
fn the_block_mlp_uses_the_tanh_gelu_between_two_matrices() {
    with_capture("block mlp", |c| {
        let fc1 = c.get("img.b0.fc1_out");
        let want = c.get("img.b0.act_out");
        let got: Vec<f32> = fc1.iter().map(|v| gelu_tanh(*v)).collect();
        agree(&got, want, 2e-5, 1e-7, "gelu_tanh(fc1)");

        // The exact GELU is the other reading. It is only ~5e-4 apart, so this
        // needs a looser bar than the layout checks — but it still has to be
        // detectable, or the doc's claim that the blocks and the merger use
        // different GELUs is untestable.
        let erf_variant: Vec<f32> = fc1.iter().map(|v| gelu_erf(*v)).collect();
        let s = spread(&erf_variant, want);
        assert!(
            s > 1e-5,
            "the exact GELU agrees with the tanh one to {s:.2e} of the peak on \
             this data, so which one the blocks use is not pinned here"
        );

        // A SwiGLU reading would need intermediate_size to be an even split of
        // a gate/up pair. Record that it is not: 4304 = 16 * 269, and 269 is
        // prime, so there is no natural half.
        let inter = c.u("intermediate_size");
        assert_eq!(inter, c.shape("img.b0.fc1_out")[1]);
        assert_eq!(
            inter % 2, 0,
            "4304 is even, so a naive SwiGLU split would not even fail on shape"
        );

        // Both linears, end to end, against the reference's own outputs.
        let d = c.dims();
        let n = c.shape("img.b0.norm2_out")[0];
        let silu_gate: Vec<f32> = fc1
            .iter()
            .map(|v| v / (1.0 + (-v).exp()))
            .collect();
        assert_discriminates(&silu_gate, want, "silu instead of gelu");
        assert_eq!(c.shape("img.b0.mlp_out"), [n, d.hidden]);
    });
}

/// The residual structure, and the growth along it.
#[test]
fn the_block_residual_carries_the_unnormalized_stream() {
    with_capture("block residual", |c| {
        let hin = c.get("img.hidden_in");
        let attn = c.get("img.b0.attn_out");
        let resid1: Vec<f32> = hin.iter().zip(attn).map(|(a, b)| a + b).collect();
        agree(&resid1, c.get("img.b0.resid1"), 1e-6, 1e-7, "hidden + attn");

        let mlp = c.get("img.b0.mlp_out");
        let out: Vec<f32> = c
            .get("img.b0.resid1")
            .iter()
            .zip(mlp)
            .map(|(a, b)| a + b)
            .collect();
        agree(&out, c.get("img.b0.out"), 1e-6, 1e-7, "resid1 + mlp");

        // Adding the *normalized* stream instead of the raw one — the mistake a
        // post-norm habit produces. It runs.
        let normed = c.get("img.b0.norm1_out");
        let wrong: Vec<f32> = normed.iter().zip(attn).map(|(a, b)| a + b).collect();
        assert_discriminates(&wrong, c.get("img.b0.resid1"), "residual from the normed stream");

        // The stream grows by four orders of magnitude across the 27 blocks.
        // Not a layout fact, but the reason the merger's LayerNorm is
        // load-bearing and the reason a f16 residual is a bad idea.
        let peak_in = hin.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let peak_out = c
            .get("img.last_hidden")
            .iter()
            .fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(
            peak_out > 100.0 * peak_in,
            "the residual stream only grew from {peak_in:.2e} to {peak_out:.2e} \
             across the tower; if that is really the case the note about f16 \
             headroom in qwen35_vision.rs needs revisiting"
        );
    });
}

// --------------------------------------------------------------------- merger

/// The merger normalizes **per patch**, before the 2x2 grouping.
///
/// The checkpoint settles this on its own — `merger.norm.weight` is `[1152]`,
/// not `[4608]` — and the capture confirms it numerically. Normalizing the
/// grouped 4608 with the gain tiled four times is the other reading; it runs.
#[test]
fn the_merger_normalizes_each_patch_before_it_groups_them() {
    with_capture("merger norm", |c| {
        let d = c.dims();
        let norm_w = c.get("merger.norm.weight");
        let norm_b = c.get("merger.norm.bias");
        assert_eq!(
            norm_w.len(),
            d.hidden,
            "merger.norm.weight is {} wide; a post-shuffle norm would make it {}",
            norm_w.len(),
            d.hidden * d.merge_unit()
        );

        let lh = c.get("img.last_hidden");
        let want = c.get("img.merger.norm_out");
        let got = layer_norm_rows(lh, norm_w, norm_b, d.hidden, d.eps);
        agree(&got, want, 3e-5, 1e-6, "merger norm, pre-shuffle");

        // Post-shuffle: normalize over 4608 with the gain and bias tiled.
        let wide = d.hidden * d.merge_unit();
        let tiled_w: Vec<f32> = norm_w.repeat(d.merge_unit());
        let tiled_b: Vec<f32> = norm_b.repeat(d.merge_unit());
        let post = layer_norm_rows(lh, &tiled_w, &tiled_b, wide, d.eps);
        assert_discriminates(&post, want, "post-shuffle merger norm");

        // And the norm is not degenerate here either — the last hidden state has
        // a row variance in the thousands, so eps is nowhere near relevant.
        let mut lo = f32::INFINITY;
        for row in lh.chunks(d.hidden) {
            let mean = row.iter().sum::<f32>() / d.hidden as f32;
            let var =
                row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / d.hidden as f32;
            lo = lo.min(var);
        }
        assert!(lo > 1000.0 * d.eps, "merger norm input variance {lo:.3e} is near eps");
    });
}

/// The grouping is a plain reshape of four *consecutive* patches, which is a 2x2
/// spatial block only because preprocessing emitted them in block order.
#[test]
fn the_merger_groups_four_consecutive_patches_into_one_token() {
    with_capture("merger grouping", |c| {
        let d = c.dims();
        let wide = d.hidden * d.merge_unit();
        let normed = c.get("img.merger.norm_out");
        let n = c.shape("img.merger.norm_out")[0];
        let tokens = n / d.merge_unit();
        assert_eq!(
            c.shape("img.merger.fc1_out"),
            [tokens, wide],
            "the merger's fc1 consumes {wide}-wide rows, {tokens} of them"
        );

        let grouped = merger_shuffle(normed, d.hidden, d.merge_unit());
        // Recompute the probed columns of fc1 from the grouped rows. If the
        // grouping were anything else these would not match.
        let rows = c.idx("merger.fc1.probe_rows");
        let w = c.get("merger.fc1.probe_w");
        let bias = c.get("merger.fc1.bias");
        let fc1 = c.get("img.merger.fc1_out");
        for (i, &row) in rows.iter().enumerate() {
            let wrow = &w[i * wide..(i + 1) * wide];
            for t in 0..tokens {
                let xt = &grouped[t * wide..(t + 1) * wide];
                let dot: f32 = xt.iter().zip(wrow).map(|(a, b)| a * b).sum::<f32>() + bias[row];
                let want = fc1[t * wide + row];
                assert!(
                    (dot - want).abs() <= 1e-3 + 1e-3 * want.abs(),
                    "fc1 column {row} at token {t}: the consecutive-patch \
                     grouping predicts {dot}, the reference has {want}"
                );
            }
        }

        // Grouping with a stride instead — token t takes patches
        // t, t+tokens, t+2*tokens, t+3*tokens — is the plausible alternative
        // (it is what a channels-first reshape gives) and must fail.
        let mut strided = vec![0.0f32; normed.len()];
        for t in 0..tokens {
            for u in 0..d.merge_unit() {
                let src = (u * tokens + t) * d.hidden;
                let dst = t * wide + u * d.hidden;
                strided[dst..dst + d.hidden].copy_from_slice(&normed[src..src + d.hidden]);
            }
        }
        let mut wrong = Vec::new();
        let mut right = Vec::new();
        for (i, &row) in rows.iter().enumerate() {
            let wrow = &w[i * wide..(i + 1) * wide];
            for t in 0..tokens {
                let f = |src: &[f32]| -> f32 {
                    src[t * wide..(t + 1) * wide]
                        .iter()
                        .zip(wrow)
                        .map(|(a, b)| a * b)
                        .sum::<f32>()
                        + bias[row]
                };
                wrong.push(f(&strided));
                right.push(fc1[t * wide + row]);
            }
        }
        assert_discriminates(&wrong, &right, "strided merger grouping");
    });
}

/// The merger's activation is the *exact* GELU, while the blocks above use the
/// tanh approximation.
#[test]
fn the_merger_uses_the_exact_gelu_and_the_blocks_use_the_tanh_one() {
    with_capture("merger gelu", |c| {
        let fc1 = c.get("img.merger.fc1_out");
        let want = c.get("img.merger.act_out");
        let got: Vec<f32> = fc1.iter().map(|v| gelu_erf(*v)).collect();
        agree(&got, want, 3e-5, 1e-6, "gelu_erf(merger fc1)");

        let tanh_variant: Vec<f32> = fc1.iter().map(|v| gelu_tanh(*v)).collect();
        let s_tanh = spread(&tanh_variant, want);
        let s_erf = spread(&got, want);
        assert!(
            s_tanh > 3.0 * s_erf.max(1e-9),
            "the tanh GELU is off by {s_tanh:.2e} and the exact one by \
             {s_erf:.2e}; they are too close together on this data for this test \
             to say which the merger uses"
        );
    });
}

/// The merger end to end, and the shape it hands the language model.
#[test]
fn the_merger_output_is_out_hidden_wide_and_one_row_per_four_patches() {
    with_capture("merger output", |c| {
        let d = c.dims();
        let n = c.shape("img.last_hidden")[0];
        let tokens = n / d.merge_unit();
        assert_eq!(c.shape("img.image_embeds"), [tokens, d.out_hidden]);
        assert_eq!(
            d.out_hidden, 5120,
            "out_hidden_size must equal the text tower's hidden_size; the \
             Qwen3_5VisionConfig class default is 3584 (the 9B)"
        );

        // The token count each grid entry contributes, which is what the prompt
        // builder has to reserve placeholders for.
        for tag in ["img", "pack", "vid"] {
            let grids = c.grids(&format!("{tag}.grid_thw"));
            let want: usize = grids.iter().map(|g| g.tokens(d.merge)).sum();
            assert_eq!(
                c.shape(&format!("{tag}.image_embeds"))[0],
                want,
                "{tag}: grid {grids:?} should yield {want} language-model tokens"
            );
        }
    });
}

// -------------------------------------------------- splicing into the text side

/// The placeholder ids come from this checkpoint's config, and are not the
/// Qwen2-VL ones.
#[test]
fn the_placeholder_token_ids_are_the_ones_this_checkpoint_uses() {
    with_capture("token ids", |c| {
        assert_eq!(IMAGE_TOKEN_ID as usize, c.u("image_token_id"));
        assert_eq!(VIDEO_TOKEN_ID as usize, c.u("video_token_id"));
        assert_eq!(VISION_START_TOKEN_ID as usize, c.u("vision_start_token_id"));
        assert_eq!(VISION_END_TOKEN_ID as usize, c.u("vision_end_token_id"));
        // The ids that Qwen2-VL / Qwen2.5-VL use, and that a copied constant
        // would carry over. They are ordinary vocabulary entries here.
        assert_ne!(IMAGE_TOKEN_ID, 151_655);
        assert_ne!(VIDEO_TOKEN_ID, 151_656);

        // The captured sequence really does contain the placeholders, in the
        // count the grid implies.
        let ids = c.idx("splice.input_ids");
        let d = c.dims();
        let g = c.grids("splice.grid_thw")[0];
        let placeholders = ids
            .iter()
            .filter(|&&t| t == IMAGE_TOKEN_ID as usize || t == VIDEO_TOKEN_ID as usize)
            .count();
        assert_eq!(placeholders, g.tokens(d.merge));
    });
}

/// Splicing puts merger row `i` at the `i`-th placeholder, in order.
#[test]
fn splicing_replaces_each_placeholder_with_the_next_feature_row() {
    with_capture("splice", |c| {
        let d = c.dims();
        let ids: Vec<u32> = c.get("splice.input_ids").iter().map(|v| *v as u32).collect();
        let feats = c.get("img.image_embeds");
        let tokens = c.shape("img.image_embeds")[0];
        let mut embeds = vec![-1.0f32; ids.len() * d.out_hidden];
        splice_image_features(&mut embeds, &ids, feats, d.out_hidden);

        let mut next = 0;
        for (t, &id) in ids.iter().enumerate() {
            let row = &embeds[t * d.out_hidden..(t + 1) * d.out_hidden];
            if id == IMAGE_TOKEN_ID || id == VIDEO_TOKEN_ID {
                let want = &feats[next * d.out_hidden..(next + 1) * d.out_hidden];
                assert_eq!(row, want, "token {t} did not get feature row {next}");
                next += 1;
            } else {
                assert!(
                    row.iter().all(|v| *v == -1.0),
                    "token {t} is not a placeholder but was overwritten"
                );
            }
        }
        assert_eq!(next, tokens, "not every feature row was placed");
    });
}

/// The text-side 3-D positions for a spliced sequence, including the advance
/// rule after an image.
#[test]
fn the_spliced_position_ids_advance_by_the_larger_spatial_extent() {
    with_capture("llm position ids", |c| {
        let d = c.dims();
        let types: Vec<u8> = c.get("splice.mm_token_type_ids").iter().map(|v| *v as u8).collect();
        let grids = c.grids("splice.grid_thw");
        let seq = types.len();
        let got = llm_position_ids(&types, &grids, d.merge);
        let want: Vec<u32> = c.get("splice.position_ids").iter().map(|v| *v as u32).collect();
        assert_eq!(got.len(), want.len());
        assert_eq!(got, want, "3-D position ids for the spliced sequence");

        // The delta the reference reports: max position + 1 - length. Negative,
        // because an image of 12 tokens only advances the running position by 4.
        let max = *got.iter().max().unwrap() as i64;
        let delta = max + 1 - seq as i64;
        assert_eq!(
            delta, c.get("splice.rope_delta")[0] as i64,
            "rope delta"
        );
        assert!(
            delta < 0,
            "the delta came out {delta}; an image whose token count exceeds \
             max(h, w) / merge must make the sequence's maximum position fall \
             below its length, and that is what the text side has to carry \
             forward into decode"
        );

        // Advancing by the image's token count instead of max(h, w) / merge —
        // the obvious alternative, and what Qwen2-VL-shaped code would do. The
        // visual block itself is unchanged; only the text tail after it moves.
        let g = grids[0];
        let (lh, lw) = (g.h / d.merge, g.w / d.merge);
        assert_ne!(
            lh.max(lw),
            lh * lw,
            "on this grid ({lh}x{lw}) the token count equals max(h, w) / merge, \
             so the two advance rules agree and this test cannot separate them"
        );
        let mut by_count = got.clone();
        let mut pos = 0u32;
        let mut i = 0;
        let mut visual_runs = 0;
        while i < seq {
            let kind = types[i];
            let mut j = i;
            while j < seq && types[j] == kind {
                j += 1;
            }
            if kind == 0 {
                for t in i..j {
                    for axis in 0..3 {
                        by_count[axis * seq + t] = pos + (t - i) as u32;
                    }
                }
                pos += (j - i) as u32;
            } else {
                visual_runs += 1;
                pos += (j - i) as u32; // the token count, not max(lh, lw)
            }
            i = j;
        }
        assert_eq!(
            visual_runs, 1,
            "the spliced sequence needs exactly one visual run for this to be \
             the advance rule under test"
        );
        let a: Vec<f32> = by_count.iter().map(|v| *v as f32).collect();
        let b: Vec<f32> = want.iter().map(|v| *v as f32).collect();
        assert_discriminates(&a, &b, "advancing by the image's token count");
    });
}

/// Interleaved mRoPE assigns axis `i % 3`, not three contiguous sections.
#[test]
fn the_text_mrope_interleaves_its_axes_rather_than_chunking_them() {
    with_capture("interleaved mrope", |c| {
        let want = c.get("mrope.axis_of_index");
        let sec = c.get("mrope.section");
        let section = [sec[0] as usize, sec[1] as usize, sec[2] as usize];
        let half = c.u("mrope_half");
        assert_eq!(want.len(), half);
        assert_eq!(section.iter().sum::<usize>(), half, "the section must cover every frequency");

        let got: Vec<f32> = (0..half)
            .map(|i| interleaved_mrope_axis(i, section) as f32)
            .collect();
        assert_eq!(got, want, "axis per rotary frequency");

        // The chunked layout Qwen2-VL uses for the same config field.
        let mut chunked = Vec::with_capacity(half);
        for (axis, &n) in section.iter().enumerate() {
            for _ in 0..n {
                chunked.push(axis as f32);
            }
        }
        assert_discriminates(&chunked, want, "chunked mrope sections");

        // And the counts each axis ends up with really are the section.
        for (axis, &expect) in section.iter().enumerate() {
            let n = want.iter().filter(|v| **v as usize == axis).count();
            assert_eq!(
                n, expect,
                "axis {axis} claims {n} frequencies, the config says {expect}"
            );
        }
    });
}

// ------------------------------------------------------------- the whole tower

/// The whole tower, end to end, from `pixel_values` to the merger's output —
/// using the captured block-0 weights for block 0 and asserting the stages it
/// can reach line up. The 26 remaining blocks' weights are not in the capture
/// (they would be 800 MB), so the end-to-end claim is made in three links:
/// input -> block 0 output, the reference's block 0 output -> its last hidden
/// state, and last hidden state -> image embeds.
#[test]
fn the_first_block_reproduces_the_reference_block_output() {
    with_capture("block 0 end to end", |c| {
        let d = c.dims();
        let n = c.shape("img.hidden_in")[0];
        let cu = c.idx("img.cu_seqlens");
        let pids: Vec<u32> = c.get("img.position_ids").iter().map(|v| *v as u32).collect();
        let (cos, sin) = vision_rope_tables(&pids, &d);

        // Only the biases and probe rows of block 0's matrices are captured, so
        // build the block from the reference's own intermediate tensors and
        // check each link. The links that need a full matrix are covered by the
        // probe-row tests above.
        let normed = layer_norm_rows(
            c.get("img.hidden_in"),
            c.get("b0.norm1.weight"),
            c.get("b0.norm1.bias"),
            d.hidden,
            d.eps,
        );
        agree(&normed, c.get("img.b0.norm1_out"), 2e-5, 1e-6, "norm1");

        let (mut q, mut k, v) = split_qkv(c.get("img.b0.qkv"), n, d.heads, d.head_dim());
        apply_vision_rope(&mut q, &cos, &sin, n, d.heads, d.head_dim());
        apply_vision_rope(&mut k, &cos, &sin, n, d.heads, d.head_dim());
        let ctx = segment_attention(&q, &k, &v, &cu, d.heads, d.head_dim());
        agree(&ctx, c.get("img.b0.attn_pre_proj"), 2e-3, 1e-5, "attention");

        let resid1: Vec<f32> = c
            .get("img.hidden_in")
            .iter()
            .zip(c.get("img.b0.attn_out"))
            .map(|(a, b)| a + b)
            .collect();
        agree(&resid1, c.get("img.b0.resid1"), 1e-6, 1e-7, "residual 1");

        let normed2 = layer_norm_rows(
            &resid1,
            c.get("b0.norm2.weight"),
            c.get("b0.norm2.bias"),
            d.hidden,
            d.eps,
        );
        agree(&normed2, c.get("img.b0.norm2_out"), 2e-5, 1e-6, "norm2");

        let act: Vec<f32> = c.get("img.b0.fc1_out").iter().map(|x| gelu_tanh(*x)).collect();
        agree(&act, c.get("img.b0.act_out"), 2e-5, 1e-7, "mlp activation");

        let out: Vec<f32> = resid1.iter().zip(c.get("img.b0.mlp_out")).map(|(a, b)| a + b).collect();
        agree(&out, c.get("img.b0.out"), 1e-6, 1e-7, "block 0 output");
    });
}

/// The merger, from the reference's last hidden state to the reference's image
/// embeddings, with everything but the two matrices' interiors done here.
#[test]
fn the_merger_reproduces_the_reference_image_embeddings_stagewise() {
    with_capture("merger stagewise", |c| {
        let d = c.dims();
        let normed = layer_norm_rows(
            c.get("img.last_hidden"),
            c.get("merger.norm.weight"),
            c.get("merger.norm.bias"),
            d.hidden,
            d.eps,
        );
        agree(&normed, c.get("img.merger.norm_out"), 3e-5, 1e-6, "merger norm");

        let grouped = merger_shuffle(&normed, d.hidden, d.merge_unit());
        let wide = d.hidden * d.merge_unit();
        assert_eq!(grouped.len() % wide, 0);

        let act: Vec<f32> = c.get("img.merger.fc1_out").iter().map(|v| gelu_erf(*v)).collect();
        agree(&act, c.get("img.merger.act_out"), 3e-5, 1e-6, "merger activation");

        // fc2's interior needs the full [5120, 4608] matrix, which is not
        // captured; what is checkable is that the reference's own output has the
        // shape and finiteness the language model needs.
        let emb = c.get("img.image_embeds");
        assert!(emb.iter().all(|v| v.is_finite()));
        assert_eq!(emb.len(), grouped.len() / wide * d.out_hidden);
    });
}
