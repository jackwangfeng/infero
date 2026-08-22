//! The Qwen3.5 vision-tower kernels against the host reference, and through it
//! against a capture of the reference implementation on the real checkpoint.
//!
//! The reference is `tuili_model::qwen35_vision`, reached through a
//! dev-dependency on the crate above this one — the same arrangement
//! `partial_rope.rs` uses, and for the same reason: a second copy of a layout
//! whose whole difficulty is which axis goes where will eventually disagree with
//! the first, and the copy that stops being checked against the capture is the
//! one the kernel is checked against.
//!
//! Dimensions come from the capture manifest, not from literals in this file.
//! (`VisionDims::QWEN35_27B` is validated against the checkpoint by the model
//! crate's own tests; the reduced shape used by the end-to-end tests below is
//! deliberately *not* the real one and says so.)
//!
//! Every test that pins a layout also computes the other plausible reading and
//! asserts it disagrees, by at least 1% of the tensor's own peak. Without that a
//! green test only says some arithmetic ran. This matters more here than
//! anywhere else in the operator set, because the vision tower reverses nearly
//! every text-side convention and each substitution runs to completion:
//!
//!   * `rms_norm` for LayerNorm — no centring, no bias;
//!   * `split_qkv` for the blocked `[all q | all k | all v]` split;
//!   * `rope_qk_*` with theta 1e7, an exponent over `head_dim`, or three
//!     interleaved axes;
//!   * any `attn_*` kernel, all of which are causal and KV-cache shaped;
//!   * silu for gelu, or the tanh gelu where the exact one belongs.
//!
//! `notes/qwen3.5-vision.md` records how far each of those lands. The numbers
//! this file prints are the same measurements taken through the kernels.
//!
//! Regenerate the capture on the box that has the checkpoint:
//!
//!   /home/jeff/vllm312/bin/python tools/capture_qwen35_vision.py \
//!       /home/jeff/models/qwen38-27b-fp8 <out-dir>
//!   TUILI_QWEN35_VISION_CAPTURE=<out-dir> \
//!       cargo test --release -p tuili-kernels --test vision
//!
//! Without the variable the capture-driven tests report SKIPPED rather than
//! passing, because a silent skip is how a suite comes to be green without
//! checking anything. The tests that need no capture run either way.

mod common;

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::Result;
use half::f16;
use tuili_kernels::vision::{VisionShape, VisionSegments};
use tuili_model::qwen35_vision as vref;
use tuili_model::qwen35_vision::VisionDims;

use common::*;

// ------------------------------------------------------------------ plumbing

/// A pinhole reader for the capture manifest.
///
/// Hand-rolled rather than `serde_json` so that this test target needs no new
/// dependency: the manifest is machine-written, flat, and pretty-printed, and
/// every lookup below fails loudly if the shape it expects is not there. It is
/// not a JSON parser and should not be reused as one.
fn json_token(src: &str, key: &str) -> Option<String> {
    let needle = format!("\"{key}\":");
    let at = src.find(&needle)? + needle.len();
    let rest = &src[at..];
    let end = rest
        .find(|c| c == ',' || c == '}')
        .unwrap_or(rest.len());
    Some(rest[..end].trim().to_string())
}

fn json_array(src: &str, key: &str) -> Option<Vec<usize>> {
    let needle = format!("\"{key}\":[");
    let at = src.find(&needle)? + needle.len();
    let rest = &src[at..];
    let end = rest.find(']')?;
    Some(
        rest[..end]
            .split(',')
            .map(|t| t.trim().parse::<usize>().expect("shape entry"))
            .collect(),
    )
}

struct Capture {
    cfg: HashMap<String, f64>,
    shapes: HashMap<String, Vec<usize>>,
    dir: PathBuf,
    arrays: std::sync::Mutex<HashMap<String, Vec<f32>>>,
}

impl Capture {
    fn open() -> Option<Self> {
        let dir = PathBuf::from(std::env::var("TUILI_QWEN35_VISION_CAPTURE").ok()?);
        // Whitespace-stripped so the scanner above sees `"key":value` with
        // nothing between them.
        let raw = std::fs::read_to_string(dir.join("manifest.json")).unwrap();
        let flat: String = raw.chars().filter(|c| !c.is_whitespace()).collect();
        assert_eq!(
            json_token(&flat, "verified_against_transformers").as_deref(),
            Some("true"),
            "this capture was written without the cross-check against \
             transformers, so it is one reading of the reference rather than the \
             reference"
        );
        // The config block, by name. Anything absent panics at the lookup.
        let mut cfg = HashMap::new();
        for key in [
            "depth",
            "hidden_size",
            "num_heads",
            "head_dim",
            "intermediate_size",
            "out_hidden_size",
            "in_channels",
            "patch_size",
            "temporal_patch_size",
            "spatial_merge_size",
            "num_position_embeddings",
            "num_grid_per_side",
            "layer_norm_eps",
            "vision_rope_theta",
            "vision_rope_dim",
            "image_token_id",
            "video_token_id",
        ] {
            let t = json_token(&flat, key)
                .unwrap_or_else(|| panic!("the manifest has no config key {key}"));
            cfg.insert(key.to_string(), t.parse::<f64>().expect("config number"));
        }
        // Array shapes: whatever files are on disk, keyed by the manifest.
        let mut shapes = HashMap::new();
        for entry in std::fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            let name = path.file_name().unwrap().to_string_lossy().to_string();
            if let Some(stem) = name.strip_suffix(".f32") {
                let shape = json_array(&flat, stem)
                    .unwrap_or_else(|| panic!("{stem}.f32 is on disk but not in the manifest"));
                let bytes = std::fs::metadata(&path).unwrap().len() as usize;
                assert_eq!(
                    bytes / 4,
                    shape.iter().product::<usize>(),
                    "{stem}: the manifest says {shape:?} but the file holds {} floats",
                    bytes / 4
                );
                shapes.insert(stem.to_string(), shape);
            }
        }
        Some(Self {
            cfg,
            shapes,
            dir,
            arrays: std::sync::Mutex::new(HashMap::new()),
        })
    }

    /// Arrays are read on demand and cached: `pos_embed.table` alone is 10 MB
    /// and most tests want three or four of the seventy-eight.
    fn get(&self, name: &str) -> Vec<f32> {
        let mut cache = self.arrays.lock().unwrap();
        if let Some(v) = cache.get(name) {
            return v.clone();
        }
        assert!(
            self.shapes.contains_key(name),
            "the capture has no array {name}"
        );
        let bytes = std::fs::read(self.dir.join(format!("{name}.f32"))).unwrap();
        let vals: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        cache.insert(name.to_string(), vals.clone());
        vals
    }

    fn shape(&self, name: &str) -> &[usize] {
        self.shapes
            .get(name)
            .unwrap_or_else(|| panic!("the capture has no array {name}"))
    }

    fn u(&self, key: &str) -> usize {
        self.cfg[key] as usize
    }

    fn f(&self, key: &str) -> f32 {
        self.cfg[key] as f32
    }

    fn idx(&self, name: &str) -> Vec<usize> {
        self.get(name).iter().map(|v| *v as usize).collect()
    }

    /// The host reference's dimensions, from the manifest — never from literals
    /// here, so that a manifest which contradicts this file fails rather than
    /// being quietly overridden.
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

    /// The same dimensions in the kernel crate's own shape struct, so the two
    /// cannot drift apart within a test.
    fn shape_gpu(&self) -> VisionShape {
        let d = self.dims();
        VisionShape {
            depth: d.depth,
            hidden: d.hidden,
            heads: d.heads,
            intermediate: d.intermediate,
            out_hidden: d.out_hidden,
            in_channels: d.in_channels,
            patch: d.patch,
            temporal_patch: d.temporal_patch,
            merge: d.merge,
            eps: d.eps,
            rope_theta: d.rope_theta,
        }
    }

    fn grids(&self, name: &str) -> Vec<vref::Grid> {
        self.get(name)
            .chunks_exact(3)
            .map(|g| vref::Grid {
                t: g[0] as usize,
                h: g[1] as usize,
                w: g[2] as usize,
            })
            .collect()
    }
}

fn with_capture(what: &str, body: impl FnOnce(&Capture) -> Result<()>) -> Result<()> {
    match Capture::open() {
        Some(c) => body(&c),
        None => {
            eprintln!(
                "SKIPPED {what}: set TUILI_QWEN35_VISION_CAPTURE to a directory \
                 written by tools/capture_qwen35_vision.py"
            );
            Ok(())
        }
    }
}

// ------------------------------------------------------------------- metrics

/// Peak-relative tolerance, the same error model the model crate's capture tests
/// use: a per-element relative allowance plus an absolute floor tied to the
/// *tensor's* peak rather than the element's own magnitude.
///
/// The floor is what makes this usable here. The merger contracts 4608 products
/// into every output and the residual stream reaches ~4200, so an output element
/// that happens to land near zero carries the same absolute accumulation noise
/// as one that lands at 100; a floor derived from the element declares the
/// near-zero ones broken. It still catches everything that matters, because
/// every wrong layout in this file moves values by 10% to 300% of the tensor's
/// own peak.
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
         {}, which is {worst:.2}x the tolerance (rel {rel:.1e}, floor {floor:.3e} \
         from a peak of {peak:.3e})",
        got[worst_at],
        want[worst_at]
    );
    eprintln!("{what}: within {worst:.2} of the tolerance (peak {peak:.3e})");
}

/// How far apart two tensors are, as a fraction of the second one's peak.
fn spread(a: &[f32], b: &[f32]) -> f32 {
    let peak = b
        .iter()
        .fold(0.0f32, |m, v| m.max(v.abs()))
        .max(f32::MIN_POSITIVE);
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
         so nothing here rules it out and the test would pass either way"
    );
    eprintln!("{what}: rejected, {s:.3} of the tensor's peak away");
}

/// Round-trip through f16, to model what a GEMM operand actually sees.
fn as_f16(x: &[f32]) -> Vec<f16> {
    x.iter().map(|v| f16::from_f32(*v)).collect()
}

// --------------------------------------------------------------- LayerNorm

/// The vision norm is LayerNorm — mean subtracted, bias added — and reusing the
/// text tower's `rms_norm` is not a precision compromise but a different
/// function.
///
/// Checked at both ends of the tower, because they are numerically different
/// problems: block 0's input has a row variance around 0.14 and values under 9,
/// the merger's input has a row variance in the thousands and values around
/// 4200. The second is where the host reference reduces in f64, so it is also
/// where a f32 kernel reduction has to be shown to be good enough.
#[test]
fn the_layer_norm_kernel_is_centred_with_a_bias_and_not_rms_norm() -> Result<()> {
    with_capture("vision layer norm", |c| {
        let k = kernels()?;
        let stream = k.device().stream().clone();
        let d = c.dims();

        for (input, gw, gb, want_name) in [
            ("img.hidden_in", "b0.norm1.weight", "b0.norm1.bias", "img.b0.norm1_out"),
            ("img.b0.resid1", "b0.norm2.weight", "b0.norm2.bias", "img.b0.norm2_out"),
            ("img.last_hidden", "merger.norm.weight", "merger.norm.bias", "img.merger.norm_out"),
        ] {
            let x = c.get(input);
            let rows = x.len() / d.hidden;
            let (w, b) = (c.get(gw), c.get(gb));
            let want = c.get(want_name);

            let dx = stream.clone_htod(&x)?;
            let dw = stream.clone_htod(&w)?;
            let db = stream.clone_htod(&b)?;
            let mut dout = stream.alloc_zeros::<f32>(rows * d.hidden)?;
            let mut dout_h = stream.alloc_zeros::<f16>(rows * d.hidden)?;
            k.vision_layer_norm(
                &mut dout.as_view_mut(),
                &mut dout_h.as_view_mut(),
                &dx.as_view(),
                &dw.as_view(),
                &db.as_view(),
                rows,
                d.hidden,
                d.eps,
            )?;
            let got = stream.clone_dtoh(&dout)?;
            let got_h = stream.clone_dtoh(&dout_h)?;
            k.device().synchronize()?;

            // Against the reference implementation's own output.
            agree(&got, &want, 3e-5, 1e-6, &format!("layer_norm of {input}"));
            // And against the host reference, which reduces in f64: this is the
            // measurement that says a block-tree f32 reduction is enough even
            // where a *sequential* f32 sum would carry ~7e-4 of relative error.
            let host = vref::layer_norm_rows(&x, &w, &b, d.hidden, d.eps);
            let vs_host = spread(&got, &host);
            eprintln!("{input}: kernel vs f64-reduced host reference {vs_host:.2e} of peak");
            assert!(
                vs_host < 1e-5,
                "the kernel's f32 reduction is {vs_host:.2e} of peak away from the \
                 f64 host reference on {input}; a tree reduction over {} values \
                 should be three orders better than that, so this is a bug and \
                 not the precision the reference's f64 was guarding against",
                d.hidden
            );

            // The fused f16 copy the following GEMM reads must be exactly the
            // narrowing of the f32 one, not a separately rounded computation.
            for (i, (&f, &h)) in got.iter().zip(&got_h).enumerate() {
                assert_eq!(
                    h,
                    f16::from_f32(f),
                    "{input}: the f16 copy at {i} is not the narrowing of the f32 one"
                );
            }

            // RMSNorm with the same gain and bias, which is what reaching for
            // `Kernels::rms_norm` gives once a bias add is bolted on.
            let mut drms = stream.alloc_zeros::<f32>(rows * d.hidden)?;
            k.rms_norm(
                &mut drms.as_view_mut(),
                &dx.as_view(),
                &dw.as_view(),
                rows,
                d.hidden,
                d.eps,
            )?;
            let mut rms = stream.clone_dtoh(&drms)?;
            k.device().synchronize()?;
            assert_discriminates(&rms, &want, &format!("rms_norm for {input}"));
            for (i, v) in rms.iter_mut().enumerate() {
                *v += b[i % d.hidden];
            }
            assert_discriminates(&rms, &want, &format!("rms_norm + bias for {input}"));

            // And the norm is not degenerate here: if eps were doing the
            // normalizing, every formulation would agree and none of the above
            // would mean anything.
            let mut lo = f32::INFINITY;
            for row in x.chunks(d.hidden) {
                let mean = row.iter().map(|v| *v as f64).sum::<f64>() / d.hidden as f64;
                let var = row
                    .iter()
                    .map(|v| (*v as f64 - mean) * (*v as f64 - mean))
                    .sum::<f64>()
                    / d.hidden as f64;
                lo = lo.min(var as f32);
            }
            assert!(
                lo > 1000.0 * d.eps,
                "{input}: smallest row variance {lo:.3e} is within three orders \
                 of eps {:.0e}; at that scale LayerNorm degenerates to a constant \
                 scale and this data cannot discriminate its formulation",
                d.eps
            );
        }
        Ok(())
    })
}

// -------------------------------------------------------------- vision RoPE

/// The rotary tables: 36 frequencies normalized by 36, theta 1e4, h in the low
/// block and w in the high one, each duplicated so `rotate_half` finds the same
/// angle at both ends of a pair.
///
/// All three of the alternatives here are text-side habits carried over, and all
/// three produce a correctly shaped table. The kernel computes its exponent and
/// angle in double precisely so that a disagreement with the f64 host reference
/// cannot be blamed on `__powf`, which is what the text-side rope kernels in
/// `ops.cu` use.
#[test]
fn the_rope_table_kernel_blocks_h_and_w_and_normalizes_by_the_rope_width() -> Result<()> {
    with_capture("vision rope tables", |c| {
        let k = kernels()?;
        let stream = k.device().stream().clone();
        let d = c.dims();
        let (hd, rope_dim) = (d.head_dim(), d.rope_dim());
        assert_eq!(hd, c.u("head_dim"), "head_dim from the manifest");
        assert_eq!(rope_dim, c.u("vision_rope_dim"));
        assert_eq!(c.shape("img.rope_cos")[1], hd, "the table is head_dim wide");

        let pids_f = c.get("img.position_ids");
        let n = pids_f.len() / 2;
        let pids: Vec<u32> = pids_f.iter().map(|v| *v as u32).collect();
        let dpids: Vec<i32> = pids_f.iter().map(|v| *v as i32).collect();
        let dpids = stream.clone_htod(&dpids)?;
        let mut dcos = stream.alloc_zeros::<f32>(n * hd)?;
        let mut dsin = stream.alloc_zeros::<f32>(n * hd)?;
        k.vision_rope_tables(
            &mut dcos.as_view_mut(),
            &mut dsin.as_view_mut(),
            &dpids.as_view(),
            n,
            hd,
            rope_dim,
            d.rope_theta,
        )?;
        let got_cos = stream.clone_dtoh(&dcos)?;
        let got_sin = stream.clone_dtoh(&dsin)?;
        k.device().synchronize()?;

        let want_cos = c.get("img.rope_cos");
        let want_sin = c.get("img.rope_sin");
        // 1e-5 against the capture, matching the model crate's own tolerance for
        // this comparison: the reference builds its table in f32 (position times
        // inverse frequency, then cos), so the residual here is the *capture's*
        // rounding and not the kernel's. Measured worst case 1.9e-7 absolute on
        // values of order 0.06, i.e. 3e-6 relative — which is f32 and nothing
        // else. The kernel-versus-host comparison below is the tight one, since
        // both of those compute in double.
        agree(&got_cos, &want_cos, 1e-5, 1e-6, "vision rope cos");
        agree(&got_sin, &want_sin, 1e-5, 1e-6, "vision rope sin");
        let (hc, hs) = vref::vision_rope_tables(&pids, &d);
        agree(&got_cos, &hc, 1e-6, 1e-7, "vision rope cos vs host");
        agree(&got_sin, &hs, 1e-6, 1e-7, "vision rope sin vs host");

        // The duplication: column i and column i + rope_dim carry the same
        // angle, which is what makes the (i, i + 36) pairing meaningful.
        for row in got_cos.chunks_exact(hd) {
            for i in 0..rope_dim {
                assert_eq!(
                    row[i], row[i + rope_dim],
                    "column {i} and {} of the kernel's cos table differ, so \
                     rotate_half's pairing has nothing to pair",
                    i + rope_dim
                );
            }
        }

        // The three wrong readings, built here and required to land far away.
        let per_axis = rope_dim / 2;
        let build = |inv: &dyn Fn(usize) -> f64, interleave: bool| -> Vec<f32> {
            let mut out = vec![0.0f32; n * hd];
            for p in 0..n {
                for axis in 0..2 {
                    for i in 0..per_axis {
                        let angle = pids[p * 2 + axis] as f64 * inv(i);
                        let j = if interleave { i * 2 + axis } else { axis * per_axis + i };
                        out[p * hd + j] = angle.cos() as f32;
                        out[p * hd + j + rope_dim] = angle.cos() as f32;
                    }
                }
            }
            out
        };
        let theta = d.rope_theta as f64;
        assert_discriminates(
            &build(&|i| theta.powf(-((2 * i) as f64 / hd as f64)), false),
            &want_cos,
            "the exponent normalized by head_dim rather than rope_dim",
        );
        assert_discriminates(
            &build(&|i| 1e7f64.powf(-((2 * i) as f64 / rope_dim as f64)), false),
            &want_cos,
            "the text side's theta = 1e7",
        );
        assert_discriminates(
            &build(&|i| theta.powf(-((2 * i) as f64 / rope_dim as f64)), true),
            &want_cos,
            "h and w interleaved rather than blocked",
        );
        Ok(())
    })
}

// -------------------------------------------------------- qkv split and rope

/// `qkv` is `[all q | all k | all v]` — three contiguous 1152-wide blocks — and
/// not the text tower's per-head interleaving.
///
/// The kernel is driven with an identity rotation (cos = 1, sin = 0) so that only
/// the split is under test, then with the real tables in the next test. The
/// interleaved reading gives three correctly shaped tensors; note that for head 0
/// component 0 the two readings name the same column, which is why the
/// discrimination below runs over the whole tensor rather than a first-head
/// probe.
#[test]
fn the_qkv_kernel_splits_three_contiguous_blocks_not_per_head_interleaved() -> Result<()> {
    with_capture("vision qkv split", |c| {
        let k = kernels()?;
        let stream = k.device().stream().clone();
        let d = c.dims();
        let (heads, hd, dim) = (d.heads, d.head_dim(), d.hidden);
        let qkv = c.get("img.b0.qkv");
        let n = c.shape("img.b0.qkv")[0];
        assert_eq!(c.shape("img.b0.qkv")[1], 3 * dim, "qkv is three blocks of {dim}");

        let dqkv = stream.clone_htod(&qkv)?;
        let dcos = stream.clone_htod(&vec![1.0f32; n * hd])?;
        let dsin = stream.clone_htod(&vec![0.0f32; n * hd])?;
        let mut dq = stream.alloc_zeros::<f32>(n * dim)?;
        let mut dk = stream.alloc_zeros::<f32>(n * dim)?;
        let mut dv = stream.alloc_zeros::<f32>(n * dim)?;
        k.vision_qkv_rope(
            &mut dq.as_view_mut(),
            &mut dk.as_view_mut(),
            &mut dv.as_view_mut(),
            &dqkv.as_view(),
            &dcos.as_view(),
            &dsin.as_view(),
            n,
            heads,
            hd,
        )?;
        let (gq, gk, gv) = (
            stream.clone_dtoh(&dq)?,
            stream.clone_dtoh(&dk)?,
            stream.clone_dtoh(&dv)?,
        );
        k.device().synchronize()?;

        // With an identity rotation the split must be bit-exact against the
        // host reference: it is a permuted copy and nothing else.
        let (wq, wk, wv) = vref::split_qkv(&qkv, n, heads, hd);
        assert_eq!(gq, wq, "q under an identity rotation is not a pure copy");
        assert_eq!(gk, wk, "k under an identity rotation is not a pure copy");
        assert_eq!(gv, wv, "v is not a pure copy — it must never be rotated");

        // The per-head interleaved reading: component s of head h at
        // `h * 3 * head_dim + s * head_dim + i` instead of `s * dim + h * hd + i`.
        let mut inter_q = vec![0.0f32; n * dim];
        for p in 0..n {
            for h in 0..heads {
                for i in 0..hd {
                    inter_q[(p * heads + h) * hd + i] =
                        qkv[p * 3 * dim + h * 3 * hd + i];
                }
            }
        }
        assert_discriminates(&inter_q, &wq, "per-head interleaved q");

        // The probe rows locate q, k and v in the *weight* rather than inheriting
        // whichever split the capture happened to make: reproduce the captured
        // qkv column from the captured weight row, at heads past the first, where
        // the two readings separate.
        let rows = c.idx("b0.qkv.probe_rows");
        let pw = c.get("b0.qkv.probe_w");
        let bias = c.get("b0.qkv.bias");
        let x = c.get("img.b0.norm1_out");
        let project = |row: usize, t: usize| -> f32 {
            let i = rows.iter().position(|&r| r == row).expect("row not probed");
            x[t * dim..(t + 1) * dim]
                .iter()
                .zip(&pw[i * dim..(i + 1) * dim])
                .map(|(a, b)| a * b)
                .sum::<f32>()
                + bias[row]
        };
        let mut separating = 0;
        for s in 0..3 {
            for h in [0usize, 1, heads - 1] {
                for dd in [0usize, 1, hd - 1] {
                    let blocked = s * dim + h * hd + dd;
                    let interleaved = h * 3 * hd + s * hd + dd;
                    for t in 0..n {
                        let want = qkv[t * 3 * dim + blocked];
                        let pred = project(blocked, t);
                        assert!(
                            (pred - want).abs() <= 1e-3 + 1e-3 * want.abs(),
                            "part {s} head {h} dim {dd} token {t}: the blocked \
                             layout predicts {pred} from weight row {blocked}, \
                             the reference has {want}"
                        );
                        if interleaved != blocked
                            && (project(interleaved, t) - want).abs() > 1e-3
                        {
                            separating += 1;
                        }
                    }
                }
            }
        }
        assert!(
            separating > 0,
            "no probe distinguished the blocked layout from a per-head \
             interleaving, so this test would pass under either"
        );
        eprintln!("qkv layout: {separating} probes separate blocked from interleaved");
        Ok(())
    })
}

/// The full split-and-rotate: `rotate_half` pairs (i, i + 36), v is untouched,
/// and rotating adjacent pairs instead lands far away.
#[test]
fn the_qkv_kernel_rotates_half_pairs_and_leaves_v_alone() -> Result<()> {
    with_capture("vision qkv rope", |c| {
        let k = kernels()?;
        let stream = k.device().stream().clone();
        let d = c.dims();
        let (heads, hd, dim) = (d.heads, d.head_dim(), d.hidden);
        let qkv = c.get("img.b0.qkv");
        let n = c.shape("img.b0.qkv")[0];
        let pids: Vec<u32> = c.get("img.position_ids").iter().map(|v| *v as u32).collect();
        let (cos, sin) = vref::vision_rope_tables(&pids, &d);

        let dqkv = stream.clone_htod(&qkv)?;
        let dcos = stream.clone_htod(&cos)?;
        let dsin = stream.clone_htod(&sin)?;
        let mut dq = stream.alloc_zeros::<f32>(n * dim)?;
        let mut dk = stream.alloc_zeros::<f32>(n * dim)?;
        let mut dv = stream.alloc_zeros::<f32>(n * dim)?;
        k.vision_qkv_rope(
            &mut dq.as_view_mut(),
            &mut dk.as_view_mut(),
            &mut dv.as_view_mut(),
            &dqkv.as_view(),
            &dcos.as_view(),
            &dsin.as_view(),
            n,
            heads,
            hd,
        )?;
        let (gq, gk, gv) = (
            stream.clone_dtoh(&dq)?,
            stream.clone_dtoh(&dk)?,
            stream.clone_dtoh(&dv)?,
        );
        k.device().synchronize()?;

        let (mut wq, mut wk, wv) = vref::split_qkv(&qkv, n, heads, hd);
        assert_eq!(gv, wv, "v moved; the rotation must not touch it");
        vref::apply_vision_rope(&mut wq, &cos, &sin, n, heads, hd);
        vref::apply_vision_rope(&mut wk, &cos, &sin, n, heads, hd);
        agree(&gq, &wq, 1e-6, 1e-7, "rotated q");
        agree(&gk, &wk, 1e-6, 1e-7, "rotated k");

        // Adjacent-pair rotation, the interleaved convention some models use.
        let (mut aq, _, _) = vref::split_qkv(&qkv, n, heads, hd);
        for p in 0..n {
            for h in 0..heads {
                let base = (p * heads + h) * hd;
                for i in 0..hd / 2 {
                    let (a, b) = (aq[base + 2 * i], aq[base + 2 * i + 1]);
                    let (cc, ss) = (cos[p * hd + 2 * i], sin[p * hd + 2 * i]);
                    aq[base + 2 * i] = a * cc - b * ss;
                    aq[base + 2 * i + 1] = b * cc + a * ss;
                }
            }
        }
        assert_discriminates(&aq, &wq, "adjacent-pair rotation");

        // No rotation at all — the failure mode where the tables are built and
        // then not applied.
        let (bare, _, _) = vref::split_qkv(&qkv, n, heads, hd);
        assert_discriminates(&bare, &wq, "q left unrotated");
        Ok(())
    })
}

// ------------------------------------------------------------------ attention

/// Run the attention kernel over `q`, `k`, `v` with the given segment
/// boundaries. Shared by the tests below.
#[allow(clippy::too_many_arguments)]
fn run_attn(
    k: &tuili_kernels::Kernels,
    q: &[f32],
    kk: &[f32],
    v: &[f32],
    cu: &[usize],
    heads: usize,
    hd: usize,
) -> Result<Vec<f32>> {
    let stream = k.device().stream().clone();
    let n = *cu.last().unwrap();
    let segs = VisionSegments::new(k.device(), cu)?;
    let dq = stream.clone_htod(&q.to_vec())?;
    let dk = stream.clone_htod(&kk.to_vec())?;
    let dv = stream.clone_htod(&v.to_vec())?;
    let mut dout = stream.alloc_zeros::<f32>(n * heads * hd)?;
    k.vision_attn(
        &mut dout.as_view_mut(),
        &dq.as_view(),
        &dk.as_view(),
        &dv.as_view(),
        &segs,
        heads,
        hd,
    )?;
    let out = stream.clone_dtoh(&dout)?;
    k.device().synchronize()?;
    Ok(out)
}

/// The attention interior reproduces the reference's own `attn.proj` input, for
/// a single image, a packed pair of images, and a two-frame video.
///
/// The reference's q/k/v split and rope application happen inside
/// `Qwen3_5VisionAttention.forward` where a hook cannot reach, so the capture
/// records both endpoints — the `attn.qkv` output and the `attn.proj` input — and
/// this rebuilds the interior between them. The answer is therefore the reference
/// implementation's, not this file's.
///
/// Two alternatives must fail: a causal mask, which is what reusing any of
/// `ops.cu`'s `attn_*` kernels gives, and one segment spanning the whole packed
/// batch, which is what dropping `cu_seqlens` gives.
#[test]
fn the_attention_kernel_is_bidirectional_within_each_segment() -> Result<()> {
    with_capture("vision attention", |c| {
        let k = kernels()?;
        let d = c.dims();
        let (heads, hd) = (d.heads, d.head_dim());
        for tag in ["img", "pack", "vid"] {
            let qkv = c.get(&format!("{tag}.b0.qkv"));
            let n = c.shape(&format!("{tag}.b0.qkv"))[0];
            let cu = c.idx(&format!("{tag}.cu_seqlens"));
            let pids: Vec<u32> = c
                .get(&format!("{tag}.position_ids"))
                .iter()
                .map(|v| *v as u32)
                .collect();
            let (cos, sin) = vref::vision_rope_tables(&pids, &d);
            let (mut q, mut kk, v) = vref::split_qkv(&qkv, n, heads, hd);
            vref::apply_vision_rope(&mut q, &cos, &sin, n, heads, hd);
            vref::apply_vision_rope(&mut kk, &cos, &sin, n, heads, hd);

            let want = c.get(&format!("{tag}.b0.attn_pre_proj"));
            let got = run_attn(&k, &q, &kk, &v, &cu, heads, hd)?;
            agree(&got, &want, 2e-3, 1e-5, &format!("{tag}: attention interior"));

            // And against the host reference, which is the tighter comparison:
            // both are f32, so this isolates the online-softmax rescaling from
            // the reference implementation's own bf16-weight arithmetic.
            let host = vref::segment_attention(&q, &kk, &v, &cu, heads, hd);
            let vs_host = spread(&got, &host);
            eprintln!("{tag}: kernel vs host segment_attention {vs_host:.2e} of peak");
            assert!(
                vs_host < 2e-6,
                "{tag}: the streaming kernel is {vs_host:.2e} of peak from the \
                 host's direct softmax; an online rescaling should agree to f32 \
                 rounding, so this is a bug"
            );

            // A causal mask, built with the host reference one prefix at a time.
            let mut causal = vec![0.0f32; n * heads * hd];
            for seg in cu.windows(2) {
                let (a, b) = (seg[0], seg[1]);
                for t in a..b {
                    let sub = vec![0usize, t - a + 1];
                    let part = vref::segment_attention(
                        &q[a * heads * hd..(t + 1) * heads * hd],
                        &kk[a * heads * hd..(t + 1) * heads * hd],
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
            assert_discriminates(&causal, &want, &format!("{tag}: causal attention"));

            // One segment spanning everything, which is what happens if
            // cu_seqlens is ignored. For the single image that is the same
            // computation, so it is only a discriminating check for the packed
            // and video groups — and it has to be, or nothing here shows the
            // kernel respects the boundary.
            if cu.len() > 2 {
                let whole = run_attn(&k, &q, &kk, &v, &[0, n], heads, hd)?;
                assert_discriminates(
                    &whole,
                    &want,
                    &format!("{tag}: attention across the segment boundary"),
                );
            }
        }
        Ok(())
    })
}

/// Packing several images, or several frames, into one call leaves each one's
/// output alone. This is the invariant that makes batched vision prefill
/// legitimate, and cross-segment attention breaks it.
///
/// Checked through the kernel rather than through the capture: the capture
/// already establishes that the *reference* has this property, so what is under
/// test here is that a ragged block-to-segment mapping does not leak.
#[test]
fn packing_images_and_frames_leaves_each_ones_kernel_output_alone() -> Result<()> {
    with_capture("vision attention packing", |c| {
        let k = kernels()?;
        let d = c.dims();
        let (heads, hd) = (d.heads, d.head_dim());

        let alone = {
            let qkv = c.get("img.b0.qkv");
            let n = c.shape("img.b0.qkv")[0];
            let pids: Vec<u32> = c.get("img.position_ids").iter().map(|v| *v as u32).collect();
            let (cos, sin) = vref::vision_rope_tables(&pids, &d);
            let (mut q, mut kk, v) = vref::split_qkv(&qkv, n, heads, hd);
            vref::apply_vision_rope(&mut q, &cos, &sin, n, heads, hd);
            vref::apply_vision_rope(&mut kk, &cos, &sin, n, heads, hd);
            run_attn(&k, &q, &kk, &v, &c.idx("img.cu_seqlens"), heads, hd)?
        };

        for tag in ["pack", "vid"] {
            let qkv = c.get(&format!("{tag}.b0.qkv"));
            let n = c.shape(&format!("{tag}.b0.qkv"))[0];
            let cu = c.idx(&format!("{tag}.cu_seqlens"));
            let pids: Vec<u32> = c
                .get(&format!("{tag}.position_ids"))
                .iter()
                .map(|v| *v as u32)
                .collect();
            let (cos, sin) = vref::vision_rope_tables(&pids, &d);
            let (mut q, mut kk, v) = vref::split_qkv(&qkv, n, heads, hd);
            vref::apply_vision_rope(&mut q, &cos, &sin, n, heads, hd);
            vref::apply_vision_rope(&mut kk, &cos, &sin, n, heads, hd);
            let packed = run_attn(&k, &q, &kk, &v, &cu, heads, hd)?;
            let s = spread(&packed[..alone.len()], &alone);
            assert!(
                s == 0.0,
                "{tag}: the first image's attention output moved by {s:.2e} of \
                 its peak once it was packed with another. The two runs feed the \
                 first image bit-identical q/k/v, so anything other than zero is \
                 a block reading past its segment"
            );
            eprintln!("{tag}: packing leaves the first image bit-identical");
        }
        Ok(())
    })
}

/// The attention kernel against the host reference on segment lengths chosen to
/// break tiling, with no capture involved.
///
/// The kernel serves 16 queries a block and streams keys 32 at a time, so the
/// lengths here are deliberately coprime with both: a segment shorter than one
/// query tile, one that ends mid-key-tile, and one long enough for the online
/// softmax to rescale many times.
#[test]
fn the_attention_kernel_handles_ragged_segments() -> Result<()> {
    let k = kernels()?;
    // head_dim 72 is the real one, and the awkward part: it is not a multiple of
    // the warp size, so the accumulator loop's tail is exercised here.
    const HEADS: usize = 3;
    const HD: usize = 72;
    let lens = [7usize, 16, 17, 33, 200];
    let mut cu = vec![0usize];
    for l in lens {
        cu.push(cu.last().unwrap() + l);
    }
    let n = *cu.last().unwrap();
    let q = pseudo_random(n * HEADS * HD, 0x9a1);
    let kk = pseudo_random(n * HEADS * HD, 0x9a2);
    let v = pseudo_random(n * HEADS * HD, 0x9a3);

    let got = run_attn(&k, &q, &kk, &v, &cu, HEADS, HD)?;
    let want = vref::segment_attention(&q, &kk, &v, &cu, HEADS, HD);
    let (abs, at) = max_abs_diff(&got, &want);
    eprintln!(
        "ragged segments {lens:?}: max abs diff from the host reference {abs:.2e}"
    );
    assert!(
        abs < 1e-6,
        "segment lengths {lens:?}: kernel and host differ by {abs} at element \
         {at} (got {}, want {})",
        got[at],
        want[at]
    );

    // Every output row must be a convex combination of its own segment's values,
    // so no row may exceed the range of the v rows it is allowed to see. A block
    // that read past its segment would usually still satisfy this, which is why
    // the comparison above is the real check — but a block that read *garbage*
    // would not, and this says so without a reference.
    for (seg, w) in cu.windows(2).enumerate() {
        let (a, b) = (w[0], w[1]);
        for h in 0..HEADS {
            for i in 0..HD {
                let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
                for t in a..b {
                    let x = v[(t * HEADS + h) * HD + i];
                    lo = lo.min(x);
                    hi = hi.max(x);
                }
                for t in a..b {
                    let o = got[(t * HEADS + h) * HD + i];
                    assert!(
                        o >= lo - 1e-6 && o <= hi + 1e-6,
                        "segment {seg} row {t} head {h} dim {i}: output {o} is \
                         outside [{lo}, {hi}], the range of the values that \
                         segment holds"
                    );
                }
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------- activations

/// Two GELUs in one tower: the 27 blocks use `gelu_pytorch_tanh`, the merger
/// uses the exact erf form, and `vision_config.hidden_act` names only the first.
///
/// They agree to about 5e-4 absolute, so this is not a layout test — it is the
/// "small and everywhere" error class that gets attributed to quantization. The
/// bar below is therefore relative: the right GELU has to be at least a few times
/// closer than the wrong one, or the claim that they differ is untestable on this
/// data.
#[test]
fn the_blocks_use_the_tanh_gelu_and_the_merger_the_exact_one() -> Result<()> {
    with_capture("vision gelu", |c| {
        let k = kernels()?;
        let stream = k.device().stream().clone();

        // (activation input, reference output, which GELU, the bias already
        // folded into the captured input)
        for (src, want_name, exact, label) in [
            ("img.b0.fc1_out", "img.b0.act_out", false, "block (tanh)"),
            ("img.merger.fc1_out", "img.merger.act_out", true, "merger (erf)"),
        ] {
            let x = c.get(src);
            let want = c.get(want_name);
            let cols = c.shape(src)[1];
            let rows = x.len() / cols;

            // The captured input is the linear's output, bias included, so drive
            // the kernel with a zero bias here; the bias path gets its own check
            // below.
            let zero = stream.clone_htod(&vec![0.0f32; cols])?;
            let run = |exact: bool| -> Result<Vec<f32>> {
                let mut dx = stream.clone_htod(&x)?;
                let mut dh = stream.alloc_zeros::<f16>(x.len())?;
                k.vision_gelu(
                    &mut dx.as_view_mut(),
                    &mut dh.as_view_mut(),
                    &zero.as_view(),
                    rows,
                    cols,
                    exact,
                )?;
                let out = stream.clone_dtoh(&dx)?;
                k.device().synchronize()?;
                Ok(out)
            };
            let got = run(exact)?;
            agree(&got, &want, 3e-5, 1e-6, &format!("{label} activation"));

            // The host reference's own version of the same activation. Its erf is
            // an Abramowitz & Stegun approximation good to ~1.5e-7 while the
            // kernel calls `erff`, so this comparison is about agreement of the
            // formula, not of the special function.
            let host: Vec<f32> = if exact {
                x.iter().map(|v| vref::gelu_erf(*v)).collect()
            } else {
                x.iter().map(|v| vref::gelu_tanh(*v)).collect()
            };
            let vs_host = spread(&got, &host);
            eprintln!("{label}: kernel vs host reference {vs_host:.2e} of peak");
            assert!(vs_host < 1e-6, "{label}: kernel and host differ by {vs_host:.2e}");

            // The other GELU, through the same kernel family.
            let other = run(!exact)?;
            let s_right = spread(&got, &want);
            let s_wrong = spread(&other, &want);
            eprintln!(
                "{label}: correct form is {s_right:.2e} of peak from the reference, \
                 the other one {s_wrong:.2e}"
            );
            assert!(
                s_wrong > 3.0 * s_right.max(1e-9),
                "{label}: the two GELUs are {s_wrong:.2e} and {s_right:.2e} from \
                 the reference — too close on this data for the test to say which \
                 one the reference used"
            );
        }

        // The bias the kernel folds in must be the same bias the preceding
        // linear would have added. Feed it the captured output minus the real
        // bias and require the captured activation back.
        let bias = c.get("b0.fc1.bias");
        let fc1 = c.get("img.b0.fc1_out");
        let cols = c.shape("img.b0.fc1_out")[1];
        let rows = fc1.len() / cols;
        assert_eq!(bias.len(), cols);
        let debiased: Vec<f32> = fc1
            .iter()
            .enumerate()
            .map(|(i, v)| v - bias[i % cols])
            .collect();
        let dbias = stream.clone_htod(&bias)?;
        let mut dx = stream.clone_htod(&debiased)?;
        let mut dh = stream.alloc_zeros::<f16>(fc1.len())?;
        k.vision_gelu(
            &mut dx.as_view_mut(),
            &mut dh.as_view_mut(),
            &dbias.as_view(),
            rows,
            cols,
            false,
        )?;
        let got = stream.clone_dtoh(&dx)?;
        k.device().synchronize()?;
        agree(&got, &c.get("img.b0.act_out"), 1e-4, 1e-6, "gelu with the fc1 bias");

        // silu, the text tower's activation, on the block's own input.
        let silu: Vec<f32> = fc1.iter().map(|v| v / (1.0 + (-v).exp())).collect();
        assert_discriminates(&silu, &c.get("img.b0.act_out"), "silu instead of gelu");
        Ok(())
    })
}

// --------------------------------------------------------------- preprocessing

/// The patchify kernel emits patches in spatial-merge-block order with `(c, t,
/// y, x)` components, and a still image's two temporal taps hold the same frame.
///
/// The probe image is an `arange` — every pixel a distinct number — put through
/// the reference image processor's own `patchify`, so this locates each pixel
/// rather than agreeing by coincidence. It is a pure gather, so the comparison is
/// exact rather than toleranced.
#[test]
fn the_patchify_kernel_emits_block_order_and_channel_temporal_row_column() -> Result<()> {
    with_capture("vision patchify", |c| {
        let k = kernels()?;
        let stream = k.device().stream().clone();
        let d = c.dims();
        let shape = c.shape_gpu();
        let img = c.get("patchify.probe_image");
        let (h_px, w_px) = (
            c.shape("patchify.probe_image")[1],
            c.shape("patchify.probe_image")[2],
        );
        let want = c.get("patchify.probe_pixels");
        let grid = c.grids("patchify.probe_grid")[0];
        let (gh, gw) = (h_px / d.patch, w_px / d.patch);
        assert_eq!((gh, gw), (grid.h, grid.w), "grid from the pixel dimensions");

        let dimg = stream.clone_htod(&img)?;
        let pd = d.patch_dim();
        let mut dout = stream.alloc_zeros::<f32>(gh * gw * pd)?;
        let mut dout_h = stream.alloc_zeros::<f16>(gh * gw * pd)?;
        k.vision_patchify(
            &mut dout.as_view_mut(),
            &mut dout_h.as_view_mut(),
            &dimg.as_view(),
            1,
            h_px,
            w_px,
            &shape,
        )?;
        let got = stream.clone_dtoh(&dout)?;
        let got_h = stream.clone_dtoh(&dout_h)?;
        k.device().synchronize()?;

        assert_eq!(got, want, "the kernel's patch layout is not the reference's");
        // And the host reference, bit for bit: both are gathers of the same
        // pixels, so anything but equality is a different layout.
        let (host, _, _) = vref::patchify(&img, h_px, w_px, &d);
        assert_eq!(got, host, "kernel and host patchify disagree");
        for (i, (&f, &h)) in got.iter().zip(&got_h).enumerate() {
            assert_eq!(h, f16::from_f32(f), "the f16 patch copy differs at {i}");
        }

        // A still image's two temporal slots must hold the same pixels: the
        // processor expands one frame across them, so both Conv3d taps see the
        // same input and act as their sum.
        let spatial = d.patch * d.patch;
        for row in got.chunks(pd) {
            for ch in 0..d.in_channels {
                for i in 0..spatial {
                    let a = row[vref::patch_slot(ch, 0, i / d.patch, i % d.patch, &d)];
                    let b = row[vref::patch_slot(ch, 1, i / d.patch, i % d.patch, &d)];
                    assert_eq!(a, b, "the two temporal slots differ");
                }
            }
        }

        // Raster patch order, and the three other orderings inside a patch. Each
        // fills every one of the 1536 slots and type-checks.
        let slot_of = |variant: usize, ch: usize, t: usize, y: usize, x: usize| -> usize {
            match variant {
                0 => ((t * d.in_channels + ch) * d.patch + y) * d.patch + x,
                1 => ((ch * d.temporal_patch + t) * d.patch + x) * d.patch + y,
                _ => ((y * d.patch + x) * d.in_channels + ch) * d.temporal_patch + t,
            }
        };
        for (variant, label) in [
            (usize::MAX, "raster patch order"),
            (0, "slot order (t, c, y, x)"),
            (1, "slot order (c, t, x, y)"),
            (2, "slot order (y, x, c, t)"),
        ] {
            let mut other = vec![0.0f32; gh * gw * pd];
            for p in 0..gh * gw {
                let (row, col) = if variant == usize::MAX {
                    (p / gw, p % gw)
                } else {
                    vref::patch_row_col(p, gw, d.merge)
                };
                for ch in 0..d.in_channels {
                    for t in 0..d.temporal_patch {
                        for y in 0..d.patch {
                            for x in 0..d.patch {
                                let src =
                                    (ch * h_px + row * d.patch + y) * w_px + col * d.patch + x;
                                let slot = if variant == usize::MAX {
                                    vref::patch_slot(ch, t, y, x, &d)
                                } else {
                                    slot_of(variant, ch, t, y, x)
                                };
                                other[p * pd + slot] = img[src];
                            }
                        }
                    }
                }
            }
            assert_discriminates(&other, &want, label);
        }
        Ok(())
    })
}

/// The learned 48x48 position grid is resampled with `align_corners = true` and
/// gathered in block order.
///
/// The same kernel is driven with all three tap sets, so what is under test is
/// the geometry rather than the gather: `align_corners = false` is the library
/// helper's own default, and raster gather order is the natural guess.
#[test]
fn the_position_grid_gather_uses_align_corners_and_block_order() -> Result<()> {
    with_capture("vision pos embed", |c| {
        let k = kernels()?;
        let stream = k.device().stream().clone();
        let d = c.dims();
        let side = c.u("num_grid_per_side");
        assert_eq!(side, d.grid_per_side());
        assert_eq!(side * side, d.num_position_embeddings);

        let grids = c.grids("img.grid_thw");
        let g = grids[0];
        let n = g.patches();
        let table = c.get("pos_embed.table");
        let dtable = stream.clone_htod(&table)?;
        let want = c.get("img.pos_embeds");

        let gather = |idx: &[usize], wts: &[f32]| -> Result<Vec<f32>> {
            let didx: Vec<i32> = idx.iter().map(|v| *v as i32).collect();
            let didx = stream.clone_htod(&didx)?;
            let dwts = stream.clone_htod(&wts.to_vec())?;
            // Into a zeroed buffer, so the kernel's `+=` returns the gather
            // itself — the tower calls it on top of the patch embedding.
            let mut dh = stream.alloc_zeros::<f32>(n * d.hidden)?;
            k.vision_add_pos_embed(
                &mut dh.as_view_mut(),
                &dtable.as_view(),
                &didx.as_view(),
                &dwts.as_view(),
                n,
                d.hidden,
                idx.len() / n,
            )?;
            let out = stream.clone_dtoh(&dh)?;
            k.device().synchronize()?;
            Ok(out)
        };

        let (idx, wts) = vref::pos_embed_taps(&grids, side, d.merge);
        assert_eq!(idx, c.idx("img.interp_indices"), "interpolation tap indices");
        agree(&wts, &c.get("img.interp_weights"), 1e-5, 1e-6, "tap weights");
        for four in wts.chunks_exact(4) {
            let s: f32 = four.iter().sum();
            assert!(
                (s - 1.0).abs() < 1e-5,
                "tap weights sum to {s}, not 1: the position field is being \
                 scaled as well as resampled"
            );
        }
        let got = gather(&idx, &wts)?;
        agree(&got, &want, 3e-5, 1e-6, "interpolated position embeddings");
        let host = vref::gather_pos_embed(&table, d.hidden, &idx, &wts, 4);
        agree(&got, &host, 1e-6, 1e-7, "position embeddings vs host");

        // align_corners = false: src = (i + 0.5) * side / size - 0.5.
        let axis_false = |index: usize, size: usize| -> ((usize, usize), (f32, f32)) {
            let src = (index as f64 + 0.5) * side as f64 / size as f64 - 0.5;
            let fl = src.floor();
            let t0 = (fl as isize).clamp(0, side as isize - 1) as usize;
            let t1 = (fl as isize + 1).clamp(0, side as isize - 1) as usize;
            let dd = (src - fl).abs();
            (
                (t0, t1),
                (
                    (1.0 - dd).max(0.0) as f32,
                    (1.0 - (src - fl - 1.0).abs()).max(0.0) as f32,
                ),
            )
        };
        let (mut fi, mut fw) = (Vec::new(), Vec::new());
        for p in 0..n {
            let (row, col) = vref::patch_row_col(p, g.w, d.merge);
            let ((h0, h1), (a0, a1)) = axis_false(row, g.h);
            let ((w0, w1), (b0, b1)) = axis_false(col, g.w);
            fi.extend_from_slice(&[
                h0 * side + w0,
                h0 * side + w1,
                h1 * side + w0,
                h1 * side + w1,
            ]);
            fw.extend_from_slice(&[a0 * b0, a0 * b1, a1 * b0, a1 * b1]);
        }
        assert_discriminates(&gather(&fi, &fw)?, &want, "align_corners = false");

        // Raster gather order with the correct interpolation rule.
        let (mut ri, mut rw) = (Vec::new(), Vec::new());
        let axis_true = |index: usize, size: usize| -> ((usize, usize), (f32, f32)) {
            let src = index as f64 * (side as f64 - 1.0) / (size.saturating_sub(1)).max(1) as f64;
            let fl = src.floor();
            let t0 = (fl as isize).clamp(0, side as isize - 1) as usize;
            let t1 = (fl as isize + 1).clamp(0, side as isize - 1) as usize;
            let dd = (src - fl).abs();
            (
                (t0, t1),
                (
                    (1.0 - dd).max(0.0) as f32,
                    (1.0 - (src - fl - 1.0).abs()).max(0.0) as f32,
                ),
            )
        };
        for p in 0..n {
            let (row, col) = (p / g.w, p % g.w);
            let ((h0, h1), (a0, a1)) = axis_true(row, g.h);
            let ((w0, w1), (b0, b1)) = axis_true(col, g.w);
            ri.extend_from_slice(&[
                h0 * side + w0,
                h0 * side + w1,
                h1 * side + w0,
                h1 * side + w1,
            ]);
            rw.extend_from_slice(&[a0 * b0, a0 * b1, a1 * b0, a1 * b1]);
        }
        assert_discriminates(&gather(&ri, &rw)?, &want, "raster gather order");
        Ok(())
    })
}

// ---------------------------------------------------------------- patch embed

/// The patch embedding is one GEMM against `proj.weight` flattened to
/// `[hidden, 1536]`, plus a bias — no window, no padding, no overlap.
///
/// This is also the first measurement of what f16 GEMM operands cost on *real*
/// weights and real activations, rather than on random data: the capture holds
/// the whole `[1152, 1536]` matrix.
#[test]
fn the_patch_embedding_is_a_gemm_and_carries_its_bias() -> Result<()> {
    with_capture("vision patch embed", |c| {
        let k = kernels()?;
        let stream = k.device().stream().clone();
        let d = c.dims();
        let w = c.get("patch_embed.w_flat");
        let b = c.get("patch_embed.bias");
        assert_eq!(c.shape("patch_embed.w_flat"), [d.hidden, d.patch_dim()]);

        let px = c.get("img.pixel_values");
        let n = c.shape("img.pixel_values")[0];
        let want = c.get("img.patch_embed_out");

        let dpx = stream.clone_htod(&as_f16(&px))?;
        let dw = stream.clone_htod(&as_f16(&w))?;
        let db = stream.clone_htod(&b)?;
        let mut dout = stream.alloc_zeros::<f32>(n * d.hidden)?;
        k.gemm_f16(
            &mut dout.as_view_mut(),
            &dpx.as_view(),
            &dw.as_view(),
            n,
            d.patch_dim(),
            d.hidden,
        )?;
        k.add_bias(&mut dout.as_view_mut(), &db.as_view(), d.hidden, n)?;
        let got = stream.clone_dtoh(&dout)?;
        k.device().synchronize()?;

        // The f16 operands are the whole tolerance story: 1536 products of
        // f16-rounded factors, accumulated in f32.
        agree(&got, &want, 3e-3, 3e-4, "patch embedding");
        let host = vref::patch_embed(&px, &w, &b, n, &d);
        let vs_host_f32 = spread(&got, &host);
        // The same arithmetic with the operands pre-rounded on the host, which is
        // what the kernel actually computes. If the kernel matches *this* much
        // more closely than it matches the f32 reference, the gap is narrowing
        // and not a bug.
        let px16: Vec<f32> = as_f16(&px).iter().map(|v| v.to_f32()).collect();
        let w16: Vec<f32> = as_f16(&w).iter().map(|v| v.to_f32()).collect();
        let host16 = vref::patch_embed(&px16, &w16, &b, n, &d);
        let vs_host_f16 = spread(&got, &host16);
        eprintln!(
            "patch embedding: kernel vs f32 host {vs_host_f32:.2e} of peak, vs \
             f16-operand host {vs_host_f16:.2e} — the difference is what \
             narrowing the operands costs"
        );
        assert!(
            vs_host_f16 < vs_host_f32,
            "the kernel is no closer to the f16-operand reference \
             ({vs_host_f16:.2e}) than to the f32 one ({vs_host_f32:.2e}), so the \
             disagreement is not explained by operand narrowing"
        );

        // Dropping the bias: the vision tower has one on every projection and
        // the text tower has none, so a text-side loader silently loses it.
        let zero = stream.clone_htod(&vec![0.0f32; d.hidden])?;
        let mut dnb = stream.alloc_zeros::<f32>(n * d.hidden)?;
        k.gemm_f16(
            &mut dnb.as_view_mut(),
            &dpx.as_view(),
            &dw.as_view(),
            n,
            d.patch_dim(),
            d.hidden,
        )?;
        k.add_bias(&mut dnb.as_view_mut(), &zero.as_view(), d.hidden, n)?;
        let no_bias = stream.clone_dtoh(&dnb)?;
        k.device().synchronize()?;
        assert_discriminates(&no_bias, &want, "patch embedding without its bias");

        // Only the first temporal tap, which is what zero-filling the second
        // slot instead of repeating the frame amounts to.
        let mut half_w = w.clone();
        for o in 0..d.hidden {
            for ch in 0..d.in_channels {
                for y in 0..d.patch {
                    for x in 0..d.patch {
                        half_w[o * d.patch_dim() + vref::patch_slot(ch, 1, y, x, &d)] = 0.0;
                    }
                }
            }
        }
        let dhw = stream.clone_htod(&as_f16(&half_w))?;
        let mut d1 = stream.alloc_zeros::<f32>(n * d.hidden)?;
        k.gemm_f16(
            &mut d1.as_view_mut(),
            &dpx.as_view(),
            &dhw.as_view(),
            n,
            d.patch_dim(),
            d.hidden,
        )?;
        k.add_bias(&mut d1.as_view_mut(), &db.as_view(), d.hidden, n)?;
        let one_tap = stream.clone_dtoh(&d1)?;
        k.device().synchronize()?;
        assert_discriminates(&one_tap, &want, "only the first temporal tap");

        // And the tower's input is this plus the position embedding.
        let (idx, wts) = vref::pos_embed_taps(&c.grids("img.grid_thw"), d.grid_per_side(), d.merge);
        let didx: Vec<i32> = idx.iter().map(|v| *v as i32).collect();
        let didx = stream.clone_htod(&didx)?;
        let dwts = stream.clone_htod(&wts)?;
        let dtable = stream.clone_htod(&c.get("pos_embed.table"))?;
        k.vision_add_pos_embed(
            &mut dout.as_view_mut(),
            &dtable.as_view(),
            &didx.as_view(),
            &dwts.as_view(),
            n,
            d.hidden,
            4,
        )?;
        let hidden_in = stream.clone_dtoh(&dout)?;
        k.device().synchronize()?;
        agree(&hidden_in, &c.get("img.hidden_in"), 3e-3, 3e-4, "tower input");
        Ok(())
    })
}

/// The qkv projection on the captured probe rows: real weights, real
/// activations, f16 operands.
///
/// 45 rows of the `[3456, 1152]` matrix are in the capture, chosen to straddle
/// both block boundaries and to reach heads past the first. Reproducing the
/// reference's own `attn.qkv` columns from them measures the f16 GEMM error where
/// it matters and confirms which row feeds which (part, head, component).
#[test]
fn the_qkv_projection_reproduces_the_reference_on_the_probe_rows() -> Result<()> {
    with_capture("vision qkv projection", |c| {
        let k = kernels()?;
        let stream = k.device().stream().clone();
        let d = c.dims();
        let dim = d.hidden;
        let rows = c.idx("b0.qkv.probe_rows");
        let pw = c.get("b0.qkv.probe_w");
        let bias = c.get("b0.qkv.bias");
        let x = c.get("img.b0.norm1_out");
        let n = c.shape("img.b0.norm1_out")[0];
        let qkv = c.get("img.b0.qkv");
        let m = rows.len();

        let dx = stream.clone_htod(&as_f16(&x))?;
        let dw = stream.clone_htod(&as_f16(&pw))?;
        let probe_bias: Vec<f32> = rows.iter().map(|&r| bias[r]).collect();
        let dbias = stream.clone_htod(&probe_bias)?;
        let mut dout = stream.alloc_zeros::<f32>(n * m)?;
        k.gemm_f16(&mut dout.as_view_mut(), &dx.as_view(), &dw.as_view(), n, dim, m)?;
        k.add_bias(&mut dout.as_view_mut(), &dbias.as_view(), m, n)?;
        let got = stream.clone_dtoh(&dout)?;
        k.device().synchronize()?;

        let mut want = Vec::with_capacity(n * m);
        for t in 0..n {
            for &r in &rows {
                want.push(qkv[t * 3 * dim + r]);
            }
        }
        agree(&got, &want, 3e-3, 3e-4, "qkv projection on the probe rows");
        eprintln!(
            "qkv projection: {:.2e} of peak from the reference with f16 operands \
             over k = {dim}",
            spread(&got, &want)
        );
        Ok(())
    })
}

// ------------------------------------------------------------------- merger

/// The merger normalizes each patch's 1152 features *before* it groups four
/// patches into a 4608-wide row, and the grouping is a plain reshape.
///
/// Both halves are checked at once by reproducing the reference's own `fc1`
/// columns from the captured probe rows: that only works if the norm ran per
/// patch and if reading the same buffer with a four-times-wider row is the
/// grouping. Two alternatives must fail — normalizing the grouped 4608 with the
/// gain tiled four times, and grouping by stride, which is what a channels-first
/// reshape gives.
#[test]
fn the_merger_normalizes_each_patch_before_grouping_them() -> Result<()> {
    with_capture("vision merger", |c| {
        let k = kernels()?;
        let stream = k.device().stream().clone();
        let d = c.dims();
        let wide = d.hidden * d.merge_unit();
        let norm_w = c.get("merger.norm.weight");
        let norm_b = c.get("merger.norm.bias");
        assert_eq!(
            norm_w.len(),
            d.hidden,
            "merger.norm.weight is {} wide; a post-shuffle norm would make it {wide}",
            norm_w.len()
        );

        let lh = c.get("img.last_hidden");
        let n = c.shape("img.last_hidden")[0];
        let tokens = n / d.merge_unit();
        let rows = c.idx("merger.fc1.probe_rows");
        let pw = c.get("merger.fc1.probe_w");
        let fc1 = c.get("img.merger.fc1_out");
        let fc1_bias = c.get("merger.fc1.bias");
        let m = rows.len();
        assert_eq!(c.shape("merger.fc1.probe_w"), [m, wide]);
        assert_eq!(c.shape("img.merger.fc1_out"), [tokens, wide]);

        let dlh = stream.clone_htod(&lh)?;
        let dnw = stream.clone_htod(&norm_w)?;
        let dnb = stream.clone_htod(&norm_b)?;
        let dpw = stream.clone_htod(&as_f16(&pw))?;
        let probe_bias: Vec<f32> = rows.iter().map(|&r| fc1_bias[r]).collect();
        let dbias = stream.clone_htod(&probe_bias)?;
        let mut want = Vec::with_capacity(tokens * m);
        for t in 0..tokens {
            for &r in &rows {
                want.push(fc1[t * wide + r]);
            }
        }

        // The real path: norm over 1152-wide rows, then the same buffer read as
        // `tokens` rows of 4608. The reshape is not a kernel — it is the GEMM's
        // row length changing, which is the whole content of the claim.
        let mut dnorm = stream.alloc_zeros::<f32>(n * d.hidden)?;
        let mut dnorm_h = stream.alloc_zeros::<f16>(n * d.hidden)?;
        k.vision_layer_norm(
            &mut dnorm.as_view_mut(),
            &mut dnorm_h.as_view_mut(),
            &dlh.as_view(),
            &dnw.as_view(),
            &dnb.as_view(),
            n,
            d.hidden,
            d.eps,
        )?;
        let mut dout = stream.alloc_zeros::<f32>(tokens * m)?;
        k.gemm_f16(
            &mut dout.as_view_mut(),
            &dnorm_h.as_view(),
            &dpw.as_view(),
            tokens,
            wide,
            m,
        )?;
        k.add_bias(&mut dout.as_view_mut(), &dbias.as_view(), m, tokens)?;
        let got = stream.clone_dtoh(&dout)?;
        let normed = stream.clone_dtoh(&dnorm)?;
        k.device().synchronize()?;
        agree(&normed, &c.get("img.merger.norm_out"), 3e-5, 1e-6, "merger norm");
        agree(&got, &want, 4e-3, 4e-4, "merger fc1 on the probe rows");

        // Post-shuffle norm: the same kernel over 4608-wide rows with the gain
        // and bias tiled four times to make the shapes fit.
        let tiled_w = norm_w.repeat(d.merge_unit());
        let tiled_b = norm_b.repeat(d.merge_unit());
        let dtw = stream.clone_htod(&tiled_w)?;
        let dtb = stream.clone_htod(&tiled_b)?;
        let mut dpost = stream.alloc_zeros::<f32>(n * d.hidden)?;
        let mut dpost_h = stream.alloc_zeros::<f16>(n * d.hidden)?;
        k.vision_layer_norm(
            &mut dpost.as_view_mut(),
            &mut dpost_h.as_view_mut(),
            &dlh.as_view(),
            &dtw.as_view(),
            &dtb.as_view(),
            tokens,
            wide,
            d.eps,
        )?;
        let mut dpo = stream.alloc_zeros::<f32>(tokens * m)?;
        k.gemm_f16(
            &mut dpo.as_view_mut(),
            &dpost_h.as_view(),
            &dpw.as_view(),
            tokens,
            wide,
            m,
        )?;
        k.add_bias(&mut dpo.as_view_mut(), &dbias.as_view(), m, tokens)?;
        let post = stream.clone_dtoh(&dpo)?;
        k.device().synchronize()?;
        assert_discriminates(&post, &want, "post-shuffle merger norm");
        assert_discriminates(
            &stream.clone_dtoh(&dpost)?,
            &c.get("img.merger.norm_out"),
            "post-shuffle merger norm output",
        );

        // Grouping by stride: token t takes patches t, t+tokens, t+2*tokens,
        // t+3*tokens, which is what a channels-first reshape gives.
        let mut strided = vec![0.0f32; normed.len()];
        for t in 0..tokens {
            for u in 0..d.merge_unit() {
                let src = (u * tokens + t) * d.hidden;
                let dst = t * wide + u * d.hidden;
                strided[dst..dst + d.hidden].copy_from_slice(&normed[src..src + d.hidden]);
            }
        }
        let dstr = stream.clone_htod(&as_f16(&strided))?;
        let mut dso = stream.alloc_zeros::<f32>(tokens * m)?;
        k.gemm_f16(
            &mut dso.as_view_mut(),
            &dstr.as_view(),
            &dpw.as_view(),
            tokens,
            wide,
            m,
        )?;
        k.add_bias(&mut dso.as_view_mut(), &dbias.as_view(), m, tokens)?;
        let strided_out = stream.clone_dtoh(&dso)?;
        k.device().synchronize()?;
        assert_discriminates(&strided_out, &want, "strided merger grouping");
        Ok(())
    })
}

// -------------------------------------------------- the accumulation dtype

/// Why the residual stream is f32 and not f16, measured on the captured tensors
/// rather than argued from the format's range.
///
/// The tempting reading is "peak 4184 against f16's 65504, so there is 15x of
/// headroom and f16 is fine". That is the wrong quantity. What matters is f16's
/// *spacing* at 4184 — 4.0 — against the size of the per-block update being added
/// to it. When the increment is smaller than half the spacing, `h + delta`
/// rounds back to `h` and the block has no effect at all: the tower silently
/// stops updating near its top, with no inf and no nan to notice.
///
/// This is not a kernel test; it is the evidence for a decision the kernels
/// encode, kept next to them so that a later "why not f16 here too" has an answer
/// with numbers.
#[test]
fn an_f16_residual_stream_loses_the_block_update_at_the_top_of_the_tower() -> Result<()> {
    with_capture("residual dtype", |c| {
        let d = c.dims();
        let top = c.get("img.last_hidden");
        let bottom = c.get("img.hidden_in");
        // A representative per-block contribution. Block 0's is the smallest of
        // the 27 relative to the top of the stream, which makes this the
        // *optimistic* case for f16.
        let update = c.get("img.b0.mlp_out");

        let peak = |v: &[f32]| v.iter().fold(0.0f32, |m, x| m.max(x.abs()));
        let (p_top, p_bot, p_up) = (peak(&top), peak(&bottom), peak(&update));
        assert!(
            p_top > 100.0 * p_bot,
            "the residual stream only grew from {p_bot:.2e} to {p_top:.2e}; if \
             that is really the case the f16 argument here needs revisiting"
        );

        // f16's spacing at the top of the stream, from the format itself.
        let spacing = {
            let h = f16::from_f32(p_top);
            let next = f16::from_bits(h.to_bits() + 1);
            (next.to_f32() - h.to_f32()).abs()
        };
        eprintln!(
            "residual: {p_bot:.3} at the input, {p_top:.1} after {} blocks; f16 \
             spacing there is {spacing:.3}, a typical block update is {p_up:.3}",
            d.depth
        );

        // Accumulate the update into the stream in each dtype and compare
        // against the f64 truth. The denominator is the *update's* norm, not the
        // stream's: the question is how much of what this block contributed
        // survives being added to what is already there, and against a stream of
        // 4184 any error looks small.
        let mut e32 = 0.0f64;
        let mut e16 = 0.0f64;
        let mut nu = 0.0f64;
        let mut vanished = 0usize;
        for (h, u) in top.iter().zip(&update) {
            let truth = *h as f64 + *u as f64;
            let g32 = (h + u) as f64;
            let h16 = f16::from_f32(*h);
            let g16 = f16::from_f32(h16.to_f32() + u);
            e32 += (g32 - truth) * (g32 - truth);
            e16 += (g16.to_f32() as f64 - truth) * (g16.to_f32() as f64 - truth);
            nu += (*u as f64) * (*u as f64);
            if g16 == h16 {
                vanished += 1;
            }
        }
        let (nu, e32, e16) = (nu.sqrt(), e32.sqrt(), e16.sqrt());
        let (f32_frac, f16_frac) = (e32 / nu, e16 / nu);
        let vanished_pct = 100.0 * vanished as f64 / top.len() as f64;
        eprintln!(
            "one accumulation at the top of the tower: f32 loses {f32_frac:.2e} \
             of the update's norm, f16 loses {f16_frac:.2e} — {:.0}x worse — and \
             {vanished_pct:.2}% of elements come back unchanged, their update \
             gone entirely",
            f16_frac / f32_frac
        );
        assert!(
            f16_frac > 100.0 * f32_frac && f16_frac > 1e-2,
            "f16 accumulation lost {f16_frac:.2e} of the update against f32's \
             {f32_frac:.2e}; if the two are really that close then the f32 \
             residual is costing bandwidth for nothing and this decision should \
             be revisited"
        );
        // And this is the *optimistic* pairing: block 0's contribution is one of
        // the smallest of the 27, so the error above is injected on every block
        // and compounds along the same path that grew four orders of magnitude.

        // And the counterpart: every GEMM operand in this tower is O(10), which
        // is the structural reason narrowing *those* is safe. Nothing that
        // reaches a matrix ever carries the residual's magnitude.
        for name in [
            "img.b0.norm1_out",
            "img.b0.norm2_out",
            "img.b0.attn_pre_proj",
            "img.b0.act_out",
            "img.merger.norm_out",
            "img.merger.act_out",
        ] {
            let p = peak(&c.get(name));
            assert!(
                p < 100.0,
                "{name} peaks at {p:.1}; it is a GEMM operand, so f16 operands \
                 are only safe while it stays orders below f16's 65504 and its \
                 spacing stays far below the values themselves"
            );
            eprintln!("  GEMM operand {name}: peak {p:.2}");
        }
        Ok(())
    })
}

// ------------------------------------------------------------------- splicing

/// Splicing puts merger row `i` at the `i`-th placeholder, in order, and refuses
/// when the counts disagree.
#[test]
fn splicing_replaces_each_placeholder_with_the_next_feature_row() -> Result<()> {
    with_capture("vision splice", |c| {
        let k = kernels()?;
        let stream = k.device().stream().clone();
        let d = c.dims();
        let ids: Vec<u32> = c.get("splice.input_ids").iter().map(|v| *v as u32).collect();
        let feats = c.get("img.image_embeds");
        let tokens = c.shape("img.image_embeds")[0];

        // The ids really are this checkpoint's, not Qwen2-VL's.
        assert_eq!(
            tuili_kernels::vision::IMAGE_TOKEN_ID as usize,
            c.u("image_token_id")
        );
        assert_eq!(
            tuili_kernels::vision::VIDEO_TOKEN_ID as usize,
            c.u("video_token_id")
        );

        let dst = tuili_kernels::vision::splice_targets(&ids, tokens)?;
        let ddst = stream.clone_htod(&dst)?;
        let dfeat = stream.clone_htod(&feats)?;
        // A sentinel everywhere, so an untouched row is visible.
        let mut dembeds = stream.clone_htod(&vec![-1.0f32; ids.len() * d.out_hidden])?;
        k.vision_splice(
            &mut dembeds.as_view_mut(),
            &dfeat.as_view(),
            &ddst.as_view(),
            d.out_hidden,
            tokens,
        )?;
        let got = stream.clone_dtoh(&dembeds)?;
        k.device().synchronize()?;

        let mut want = vec![-1.0f32; ids.len() * d.out_hidden];
        vref::splice_image_features(&mut want, &ids, &feats, d.out_hidden);
        assert_eq!(got, want, "the kernel's splice is not the reference's");

        let mut next = 0;
        for (t, &id) in ids.iter().enumerate() {
            let row = &got[t * d.out_hidden..(t + 1) * d.out_hidden];
            if id == vref::IMAGE_TOKEN_ID || id == vref::VIDEO_TOKEN_ID {
                assert_eq!(
                    row,
                    &feats[next * d.out_hidden..(next + 1) * d.out_hidden],
                    "token {t} did not get feature row {next}"
                );
                next += 1;
            } else {
                assert!(
                    row.iter().all(|v| *v == -1.0),
                    "token {t} is not a placeholder but was overwritten"
                );
            }
        }
        assert_eq!(next, tokens, "not every feature row was placed");

        // Qwen2-VL's ids find no placeholders in this vocabulary, so the count
        // check has to refuse rather than splice nothing. This is the failure the
        // reference's `get_placeholder_mask` exists to raise.
        let old_ids: Vec<u32> = ids
            .iter()
            .map(|&t| if t == vref::IMAGE_TOKEN_ID { 151_655 } else { t })
            .collect();
        assert!(
            tuili_kernels::vision::splice_targets(&old_ids, tokens).is_err(),
            "splice_targets accepted a sequence whose placeholders were written \
             with Qwen2-VL's 151655; the counts do not match and it must refuse, \
             or the image features go into the void quietly"
        );
        Ok(())
    })
}

// ------------------------------------------------------------- the whole tower

/// `y = x W^T + b` with the two operands narrowed to f16 first and the products
/// accumulated in f32 — what `gemm_f16` actually computes.
///
/// Only the arithmetic is re-done here; every layout decision in the comparison
/// chain below still comes from `tuili_model::qwen35_vision`.
fn linear16(x: &[f32], w: &[f32], b: &[f32], rows: usize, k: usize, n: usize) -> Vec<f32> {
    let xh = as_f16(x);
    let wh = as_f16(w);
    let mut out = vec![0.0f32; rows * n];
    for t in 0..rows {
        for o in 0..n {
            let mut acc = 0.0f32;
            for i in 0..k {
                acc += xh[t * k + i].to_f32() * wh[o * k + i].to_f32();
            }
            out[t * n + o] = acc + b[o];
        }
    }
    out
}

/// Random weights for the reduced shape, on the host.
struct HostWeights {
    patch_w: Vec<f32>,
    patch_b: Vec<f32>,
    table: Vec<f32>,
    blocks: Vec<[Vec<f32>; 12]>,
    m_norm_w: Vec<f32>,
    m_norm_b: Vec<f32>,
    m_fc1_w: Vec<f32>,
    m_fc1_b: Vec<f32>,
    m_fc2_w: Vec<f32>,
    m_fc2_b: Vec<f32>,
}

/// A deliberately *not* real shape: small enough that the host reference's
/// triple loops finish, with `head_dim` kept at the real and awkward 72 and
/// `merge` at 2, because those are the two numbers the kernels' indexing depends
/// on. The real 27x1152 tower is exercised by the timing test below.
fn reduced_dims() -> VisionDims {
    VisionDims {
        depth: 3,
        hidden: 144,
        heads: 2,
        intermediate: 200,
        out_hidden: 160,
        in_channels: 3,
        patch: 16,
        temporal_patch: 2,
        merge: 2,
        num_position_embeddings: 2304,
        eps: 1e-6,
        rope_theta: 10_000.0,
    }
}

fn reduced_shape(d: &VisionDims) -> VisionShape {
    VisionShape {
        depth: d.depth,
        hidden: d.hidden,
        heads: d.heads,
        intermediate: d.intermediate,
        out_hidden: d.out_hidden,
        in_channels: d.in_channels,
        patch: d.patch,
        temporal_patch: d.temporal_patch,
        merge: d.merge,
        eps: d.eps,
        rope_theta: d.rope_theta,
    }
}

fn host_weights(d: &VisionDims) -> HostWeights {
    // Scaled so the residual stream grows the way the real tower's does rather
    // than staying O(1): 1/sqrt(fan_in) keeps each sublayer's output O(1) and the
    // residual then accumulates across depth.
    let scaled = |n: usize, fan_in: usize, seed: u64| -> Vec<f32> {
        let s = (fan_in as f32).sqrt().recip();
        pseudo_random(n, seed).iter().map(|v| v * s).collect()
    };
    let (h, i, o) = (d.hidden, d.intermediate, d.out_hidden);
    let wide = h * d.merge_unit();
    HostWeights {
        patch_w: scaled(h * d.patch_dim(), d.patch_dim(), 0x100),
        patch_b: scaled(h, 16, 0x101),
        table: scaled(d.num_position_embeddings * h, 16, 0x102),
        blocks: (0..d.depth)
            .map(|b| {
                let s = 0x200 + b as u64 * 0x40;
                [
                    // The gains sit near 1, as trained LayerNorm gains do; a
                    // zero-mean gain would leave the residual stream flat and the
                    // f16 question invisible.
                    pseudo_random(h, s).iter().map(|v| 1.0 + 0.1 * v).collect(),
                    scaled(h, 64, s + 1),
                    pseudo_random(h, s + 2).iter().map(|v| 1.0 + 0.1 * v).collect(),
                    scaled(h, 64, s + 3),
                    scaled(3 * h * h, h, s + 4),
                    scaled(3 * h, 64, s + 5),
                    scaled(h * h, h, s + 6),
                    scaled(h, 64, s + 7),
                    scaled(i * h, h, s + 8),
                    scaled(i, 64, s + 9),
                    scaled(h * i, i, s + 10),
                    scaled(h, 64, s + 11),
                ]
            })
            .collect(),
        m_norm_w: pseudo_random(h, 0x300).iter().map(|v| 1.0 + 0.1 * v).collect(),
        m_norm_b: scaled(h, 64, 0x301),
        m_fc1_w: scaled(wide * wide, wide, 0x302),
        m_fc1_b: scaled(wide, 64, 0x303),
        m_fc2_w: scaled(o * wide, wide, 0x304),
        m_fc2_b: scaled(o, 64, 0x305),
    }
}

/// The whole tower on the device against the host reference: patchify, patch
/// embedding, position embedding, every block, merger.
///
/// The comparison target is the host reference with its GEMM operands narrowed to
/// f16, because that is what the kernels compute. The gap to the *unnarrowed*
/// reference is reported too — that number is the price of f16 operands, and
/// separating the two is what makes a disagreement mean "bug" rather than
/// "precision".
#[test]
fn the_whole_tower_matches_the_host_reference() -> Result<()> {
    let k = kernels()?;
    let stream = k.device().stream().clone();
    let d = reduced_dims();
    let shape = reduced_shape(&d);
    let hw = host_weights(&d);

    // A 64x96 image: a 4x6 patch grid, so 24 patches and 6 merged tokens, and
    // neither grid axis equals the other or divides the query tile.
    let (h_px, w_px) = (64usize, 96usize);
    let frame = pseudo_random(d.in_channels * h_px * w_px, 0x400);
    let (pixels, gh, gw) = vref::patchify(&frame, h_px, w_px, &d);
    assert_eq!((gh, gw), (4, 6));
    let grids = vec![vref::Grid { t: 1, h: gh, w: gw }];
    let n = grids[0].patches();
    let cu = vref::cu_seqlens(&grids);
    let pids = vref::vision_position_ids(&grids, d.merge);
    let (interp_idx, interp_wts) = vref::pos_embed_taps(&grids, d.grid_per_side(), d.merge);
    let (cos, sin) = vref::vision_rope_tables(&pids, &d);

    // ---- the host reference, twice: f32 throughout, and with f16 operands ----
    let run_host = |narrow: bool| -> (Vec<f32>, Vec<f32>) {
        let lin = |x: &[f32], w: &[f32], b: &[f32], rows, kk, nn| {
            if narrow {
                linear16(x, w, b, rows, kk, nn)
            } else {
                vref::linear(x, w, b, rows, kk, nn)
            }
        };
        let mut hidden = lin(&pixels, &hw.patch_w, &hw.patch_b, n, d.patch_dim(), d.hidden);
        let pos = vref::gather_pos_embed(&hw.table, d.hidden, &interp_idx, &interp_wts, 4);
        for (a, b) in hidden.iter_mut().zip(&pos) {
            *a += b;
        }
        for blk in &hw.blocks {
            let [n1w, n1b, n2w, n2b, qkv_w, qkv_b, proj_w, proj_b, fc1_w, fc1_b, fc2_w, fc2_b] =
                blk;
            let normed = vref::layer_norm_rows(&hidden, n1w, n1b, d.hidden, d.eps);
            let qkv = lin(&normed, qkv_w, qkv_b, n, d.hidden, 3 * d.hidden);
            let (mut q, mut kk, v) = vref::split_qkv(&qkv, n, d.heads, d.head_dim());
            vref::apply_vision_rope(&mut q, &cos, &sin, n, d.heads, d.head_dim());
            vref::apply_vision_rope(&mut kk, &cos, &sin, n, d.heads, d.head_dim());
            let ctx = vref::segment_attention(&q, &kk, &v, &cu, d.heads, d.head_dim());
            let attn = lin(&ctx, proj_w, proj_b, n, d.hidden, d.hidden);
            for (a, b) in hidden.iter_mut().zip(&attn) {
                *a += b;
            }
            let normed = vref::layer_norm_rows(&hidden, n2w, n2b, d.hidden, d.eps);
            let mut mid = lin(&normed, fc1_w, fc1_b, n, d.hidden, d.intermediate);
            for v in mid.iter_mut() {
                *v = vref::gelu_tanh(*v);
            }
            let out = lin(&mid, fc2_w, fc2_b, n, d.intermediate, d.hidden);
            for (a, b) in hidden.iter_mut().zip(&out) {
                *a += b;
            }
        }
        let wide = d.hidden * d.merge_unit();
        let tokens = n / d.merge_unit();
        let normed = vref::layer_norm_rows(&hidden, &hw.m_norm_w, &hw.m_norm_b, d.hidden, d.eps);
        let grouped = vref::merger_shuffle(&normed, d.hidden, d.merge_unit());
        let mut mid = lin(&grouped, &hw.m_fc1_w, &hw.m_fc1_b, tokens, wide, wide);
        for v in mid.iter_mut() {
            *v = vref::gelu_erf(*v);
        }
        let feats = lin(&mid, &hw.m_fc2_w, &hw.m_fc2_b, tokens, wide, d.out_hidden);
        (hidden, feats)
    };
    let (want_h32, want_f32_) = run_host(false);
    let (want_h16, want_f16_) = run_host(true);

    // The merger's grouping being the identity is the claim `merger_shuffle`
    // documents and the kernel path relies on: the device never reorders, it just
    // hands the GEMM a wider row. Check it here rather than trusting it.
    {
        let probe = pseudo_random(n * d.hidden, 0x401);
        assert_eq!(
            vref::merger_shuffle(&probe, d.hidden, d.merge_unit()),
            probe,
            "merger_shuffle is not the identity, so reading the same buffer with \
             a 4x wider row is not the grouping and the device path is wrong"
        );
    }

    // ---- the device -----------------------------------------------------
    let geo = tuili_kernels::vision::VisionGeometry::new(
        &k, &shape, &cu, &pids, &interp_idx, &interp_wts,
    )?;
    let mut scratch = tuili_kernels::vision::VisionScratch::new(k.device(), &shape, n)?;

    let dframe = stream.clone_htod(&frame)?;
    let mut dpix32 = stream.alloc_zeros::<f32>(n * d.patch_dim())?;
    {
        let mut pix_h = scratch.pixels_h_mut();
        k.vision_patchify(
            &mut dpix32.as_view_mut(),
            &mut pix_h,
            &dframe.as_view(),
            1,
            h_px,
            w_px,
            &shape,
        )?;
    }
    assert_eq!(
        stream.clone_dtoh(&dpix32)?,
        pixels,
        "the device patchify disagrees with the host one"
    );

    // Upload the weights, keeping the owning slices alive for the whole call.
    let dpatch_w = stream.clone_htod(&as_f16(&hw.patch_w))?;
    let dpatch_b = stream.clone_htod(&hw.patch_b)?;
    let dtable = stream.clone_htod(&hw.table)?;
    let dm_norm_w = stream.clone_htod(&hw.m_norm_w)?;
    let dm_norm_b = stream.clone_htod(&hw.m_norm_b)?;
    let dm_fc1_w = stream.clone_htod(&as_f16(&hw.m_fc1_w))?;
    let dm_fc1_b = stream.clone_htod(&hw.m_fc1_b)?;
    let dm_fc2_w = stream.clone_htod(&as_f16(&hw.m_fc2_w))?;
    let dm_fc2_b = stream.clone_htod(&hw.m_fc2_b)?;
    let mut norms = Vec::new();
    let mut mats = Vec::new();
    for blk in &hw.blocks {
        for j in [0usize, 1, 2, 3, 5, 7, 9, 11] {
            norms.push(stream.clone_htod(&blk[j])?);
        }
        for j in [4usize, 6, 8, 10] {
            mats.push(stream.clone_htod(&as_f16(&blk[j]))?);
        }
    }
    let blocks = (0..d.depth)
        .map(|b| tuili_kernels::vision::VisionBlockWeights {
            norm1_w: norms[b * 8].as_view(),
            norm1_b: norms[b * 8 + 1].as_view(),
            norm2_w: norms[b * 8 + 2].as_view(),
            norm2_b: norms[b * 8 + 3].as_view(),
            qkv_w: mats[b * 4].as_view(),
            qkv_b: norms[b * 8 + 4].as_view(),
            proj_w: mats[b * 4 + 1].as_view(),
            proj_b: norms[b * 8 + 5].as_view(),
            fc1_w: mats[b * 4 + 2].as_view(),
            fc1_b: norms[b * 8 + 6].as_view(),
            fc2_w: mats[b * 4 + 3].as_view(),
            fc2_b: norms[b * 8 + 7].as_view(),
        })
        .collect();
    let weights = tuili_kernels::vision::VisionWeights {
        patch_embed_w: dpatch_w.as_view(),
        patch_embed_b: dpatch_b.as_view(),
        pos_embed: dtable.as_view(),
        blocks,
        merger_norm_w: dm_norm_w.as_view(),
        merger_norm_b: dm_norm_b.as_view(),
        merger_fc1_w: dm_fc1_w.as_view(),
        merger_fc1_b: dm_fc1_b.as_view(),
        merger_fc2_w: dm_fc2_w.as_view(),
        merger_fc2_b: dm_fc2_b.as_view(),
    };

    tuili_kernels::vision::vision_forward(&k, &shape, &weights, &geo, &mut scratch)?;
    let got_h = stream.clone_dtoh(&scratch.last_hidden())?;
    let got_f = stream.clone_dtoh(&scratch.features())?;
    k.device().synchronize()?;

    let peak = |v: &[f32]| v.iter().fold(0.0f32, |m, x| m.max(x.abs()));
    eprintln!(
        "reduced tower: residual peak {:.2} at the input scale, {:.2} after {} \
         blocks",
        peak(&pixels),
        peak(&want_h32),
        d.depth
    );
    for (name, got, w16, w32) in [
        ("last_hidden", &got_h, &want_h16, &want_h32),
        ("image features", &got_f, &want_f16_, &want_f32_),
    ] {
        let s16 = spread(got, w16);
        let s32 = spread(got, w32);
        let narrowing = spread(w16, w32);
        eprintln!(
            "{name}: kernel vs f16-operand reference {s16:.2e} of peak, vs f32 \
             reference {s32:.2e}; narrowing alone accounts for {narrowing:.2e}"
        );
        // Two decades below the 1% floor at which every wrong layout in this
        // file lands, so the tolerance cannot be hiding one.
        assert!(
            s16 < 1e-3,
            "{name}: the kernel is {s16:.2e} of peak from the reference computing \
             the same narrowed arithmetic — within two decades of the 1e-2 at \
             which a wrong layout shows up, so this can no longer be called \
             precision"
        );
        assert!(
            s16 < s32,
            "{name}: the kernel is no closer to the narrowed reference than to \
             the f32 one, so the disagreement is not explained by f16 operands"
        );
        // The whole gap has to be attributable to narrowing. Two different
        // roundings of the same 3-block chain diverge and the divergence
        // compounds with depth, so the bar is the *same order* as the
        // reference's own f16-versus-f32 disagreement, not tighter than it: an
        // extra error source would push the kernel past that.
        assert!(
            s16 <= 2.0 * narrowing,
            "{name}: the kernel is {s16:.2e} from the narrowed reference while \
             narrowing itself only moves the answer {narrowing:.2e}, so something \
             other than operand precision is contributing"
        );
    }
    Ok(())
}

/// The real tower — 27 blocks, hidden 1152, head_dim 72 — at two realistic image
/// sizes, to confirm the launch geometry holds at full scale and to measure it.
///
/// Weights are zeros: the timing does not depend on their values and 460M random
/// f16 would cost more to generate on the host than the measurement costs to
/// take. Zero LayerNorm gains do not degenerate into nan — the variance floor is
/// `eps` and the gain multiplies to zero — so the finiteness check below is still
/// worth making.
///
/// What the numbers say: the tower is attention-bound, quadratically in the patch
/// count per frame, which is the shape of cost to expect from a segment that is
/// one whole frame. A 4096-patch frame is a 1024x1024 image; anything larger
/// should be checked against this scaling before being allowed through.
#[test]
fn the_real_shape_runs_and_is_worth_timing() -> Result<()> {
    let k = kernels()?;
    let stream = k.device().stream().clone();
    let shape = VisionShape::QWEN35_27B;
    let d = shape.hidden;
    let wide = shape.merged();

    // One allocation per distinct weight, shared by all 27 blocks: this measures
    // the kernels, not the loader, and 27 copies of the same zeros would cost
    // 900 MB for nothing.
    let qkv_w = stream.alloc_zeros::<f16>(3 * d * d)?;
    let proj_w = stream.alloc_zeros::<f16>(d * d)?;
    let fc1_w = stream.alloc_zeros::<f16>(shape.intermediate * d)?;
    let fc2_w = stream.alloc_zeros::<f16>(d * shape.intermediate)?;
    let norm = stream.alloc_zeros::<f32>(d)?;
    let qkv_b = stream.alloc_zeros::<f32>(3 * d)?;
    let fc1_b = stream.alloc_zeros::<f32>(shape.intermediate)?;
    let patch_w = stream.alloc_zeros::<f16>(d * shape.patch_dim())?;
    let table = stream.alloc_zeros::<f32>(2304 * d)?;
    let m_fc1_w = stream.alloc_zeros::<f16>(wide * wide)?;
    let m_fc1_b = stream.alloc_zeros::<f32>(wide)?;
    let m_fc2_w = stream.alloc_zeros::<f16>(shape.out_hidden * wide)?;
    let m_fc2_b = stream.alloc_zeros::<f32>(shape.out_hidden)?;

    let weights = tuili_kernels::vision::VisionWeights {
        patch_embed_w: patch_w.as_view(),
        patch_embed_b: norm.as_view(),
        pos_embed: table.as_view(),
        blocks: (0..shape.depth)
            .map(|_| tuili_kernels::vision::VisionBlockWeights {
                norm1_w: norm.as_view(),
                norm1_b: norm.as_view(),
                norm2_w: norm.as_view(),
                norm2_b: norm.as_view(),
                qkv_w: qkv_w.as_view(),
                qkv_b: qkv_b.as_view(),
                proj_w: proj_w.as_view(),
                proj_b: norm.as_view(),
                fc1_w: fc1_w.as_view(),
                fc1_b: fc1_b.as_view(),
                fc2_w: fc2_w.as_view(),
                fc2_b: norm.as_view(),
            })
            .collect(),
        merger_norm_w: norm.as_view(),
        merger_norm_b: norm.as_view(),
        merger_fc1_w: m_fc1_w.as_view(),
        merger_fc1_b: m_fc1_b.as_view(),
        merger_fc2_w: m_fc2_w.as_view(),
        merger_fc2_b: m_fc2_b.as_view(),
    };

    for (label, side) in [("512x512", 512usize), ("1024x1024", 1024)] {
        let grid = side / shape.patch;
        let grids = vec![vref::Grid { t: 1, h: grid, w: grid }];
        let n = grids[0].patches();
        let cu = vref::cu_seqlens(&grids);
        let host_dims = VisionDims::QWEN35_27B;
        let pids = vref::vision_position_ids(&grids, shape.merge);
        let (idx, wts) = vref::pos_embed_taps(&grids, host_dims.grid_per_side(), shape.merge);
        let geo = tuili_kernels::vision::VisionGeometry::new(&k, &shape, &cu, &pids, &idx, &wts)?;
        let mut scratch = tuili_kernels::vision::VisionScratch::new(k.device(), &shape, n)?;

        // Once to warm the caches and force every NVRTC compile, then timed.
        tuili_kernels::vision::vision_forward(&k, &shape, &weights, &geo, &mut scratch)?;
        k.device().synchronize()?;
        let started = std::time::Instant::now();
        tuili_kernels::vision::vision_forward(&k, &shape, &weights, &geo, &mut scratch)?;
        k.device().synchronize()?;
        let ms = started.elapsed().as_secs_f64() * 1e3;

        let feats = stream.clone_dtoh(&scratch.features())?;
        k.device().synchronize()?;
        assert!(
            feats.iter().all(|v| v.is_finite()),
            "{label}: the tower produced a non-finite feature at the real shape"
        );
        assert_eq!(feats.len(), n / shape.merge_unit() * shape.out_hidden);
        eprintln!(
            "{label}: {n} patches -> {} tokens, {} blocks in {ms:.1} ms \
             ({:.1} us a block, {:.0} patches/ms)",
            n / shape.merge_unit(),
            shape.depth,
            ms * 1e3 / shape.depth as f64,
            n as f64 / ms
        );
        // Guessing at which kernel is slow has a poor record in this project, so
        // ask. `TUILI_PROFILE` serializes the stream and times each launch.
        if k.device().profile().enabled() {
            eprintln!("{label}:\n{}", k.device().profile().report());
            k.device().profile().reset();
        }
    }
    Ok(())
}

/// The kernel crate's own `VisionShape::QWEN35_27B` against the checkpoint.
///
/// This is the constant a loader would actually reach for, and a hard-coded
/// constant that nothing checks is how `VisionDims::QWEN35_27B` went unvalidated
/// while every test around it took its dimensions from the manifest. Both
/// constants are checked here, against each other and against the capture's
/// config — which is the checkpoint's `vision_config`.
#[test]
fn the_hard_coded_shape_matches_the_checkpoint() -> Result<()> {
    with_capture("vision shape constant", |c| {
        let s = VisionShape::QWEN35_27B;
        let d = VisionDims::QWEN35_27B;
        // Against the checkpoint.
        assert_eq!(s.depth, c.u("depth"));
        assert_eq!(s.hidden, c.u("hidden_size"));
        assert_eq!(s.heads, c.u("num_heads"));
        assert_eq!(s.intermediate, c.u("intermediate_size"));
        assert_eq!(s.out_hidden, c.u("out_hidden_size"), "the class default is 3584, the 9B's");
        assert_eq!(s.in_channels, c.u("in_channels"));
        assert_eq!(s.patch, c.u("patch_size"));
        assert_eq!(s.temporal_patch, c.u("temporal_patch_size"));
        assert_eq!(s.merge, c.u("spatial_merge_size"));
        assert_eq!(s.eps, c.f("layer_norm_eps"));
        assert_eq!(s.rope_theta, c.f("vision_rope_theta"), "1e4, not the text side's 1e7");
        // Derived quantities the kernels index with.
        assert_eq!(s.head_dim(), c.u("head_dim"));
        assert_eq!(s.rope_dim(), c.u("vision_rope_dim"));
        assert_eq!(s.patch_dim(), 1536);
        assert_eq!(s.merged(), s.hidden * s.merge * s.merge);
        // And against the model crate's copy, field by field, so the two cannot
        // drift: this crate cannot depend on that one at build time, so nothing
        // but a test keeps them equal.
        assert_eq!(
            (s.depth, s.hidden, s.heads, s.intermediate, s.out_hidden, s.in_channels),
            (d.depth, d.hidden, d.heads, d.intermediate, d.out_hidden, d.in_channels),
            "VisionShape and VisionDims disagree"
        );
        assert_eq!(
            (s.patch, s.temporal_patch, s.merge, s.eps, s.rope_theta),
            (d.patch, d.temporal_patch, d.merge, d.eps, d.rope_theta),
            "VisionShape and VisionDims disagree"
        );
        assert_eq!(s.head_dim(), d.head_dim());
        assert_eq!(s.patch_dim(), d.patch_dim());
        Ok(())
    })
}
