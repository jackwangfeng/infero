//! The image side of preprocessing, against Pillow.
//!
//! Everything downstream of a resized frame is pinned by `qwen35_vision.rs`
//! against a capture of the reference implementation. The resize itself is not,
//! because the capture holds the processor's *output* and not its input — so this
//! file checks it against the same library the reference processor calls.
//! `Qwen2VLImageProcessor` resizes through `transformers.image_transforms.resize`,
//! which converts to a PIL image, calls `Image.resize(..., BICUBIC)`, and converts
//! back; so Pillow is the authority, not a bicubic formula.
//!
//! The fixtures in `fixtures/pil_bicubic*.u8` were written by Pillow 12.3.0 on
//! three shapes — a downscale (the common path for this model, whose `max_pixels`
//! is 4096x4096), an upscale, and a same-aspect reduction. They are 8-bit in and
//! 8-bit out, which is what the reference pipeline actually moves around.
//!
//! The module under test is not yet declared in `lib.rs`, so it is included by
//! path. It has no `crate::` references for exactly this reason; once the
//! `pub mod` line lands this include can become a normal `use`.
#[path = "../src/qwen35_vision_image.rs"]
mod image;

use std::path::PathBuf;

use tuili_model::qwen35_vision::{self as vref, VisionDims};

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

struct Case {
    name: String,
    src_h: usize,
    src_w: usize,
    dst_h: usize,
    dst_w: usize,
    channels: usize,
}

/// The fixture manifest, read with the same pinhole scanner style the kernel
/// tests use rather than pulling in a JSON dependency for four integers a case.
fn cases() -> Vec<Case> {
    let raw = std::fs::read_to_string(fixtures().join("pil_bicubic.json")).unwrap();
    let flat: String = raw.chars().filter(|c| !c.is_whitespace()).collect();
    let num = |seg: &str, key: &str| -> usize {
        let needle = format!("\"{key}\":");
        let at = seg.find(&needle).unwrap() + needle.len();
        let rest = &seg[at..];
        let end = rest.find([',', '}']).unwrap_or(rest.len());
        rest[..end].parse().unwrap()
    };
    flat.split("{\"name\":\"")
        .skip(1)
        .map(|seg| {
            let name = seg[..seg.find('"').unwrap()].to_string();
            Case {
                name,
                src_h: num(seg, "src_h"),
                src_w: num(seg, "src_w"),
                dst_h: num(seg, "dst_h"),
                dst_w: num(seg, "dst_w"),
                channels: num(seg, "channels"),
            }
        })
        .collect()
}

/// The resampler reproduces Pillow byte for byte, or says how far off it is.
///
/// Bit-exactness is the goal and not a nicety: Pillow accumulates in fixed point
/// with an 8-bit intermediate between its two passes, and any implementation that
/// differs differs in *every patch of every image*, which is the error class that
/// gets attributed to the model.
#[test]
fn the_resampler_reproduces_pillows_bicubic() {
    let cases = cases();
    assert!(cases.len() >= 3, "only {} resize fixtures", cases.len());
    for c in &cases {
        let src = std::fs::read(fixtures().join(format!("{}.src.u8", c.name))).unwrap();
        let want = std::fs::read(fixtures().join(format!("{}.dst.u8", c.name))).unwrap();
        assert_eq!(src.len(), c.src_h * c.src_w * c.channels);
        assert_eq!(want.len(), c.dst_h * c.dst_w * c.channels);

        let got = image::pil_resize_u8(
            &src, c.src_h, c.src_w, c.channels, c.dst_h, c.dst_w,
        );
        assert_eq!(got.len(), want.len());
        let mut worst = 0i32;
        let mut differing = 0usize;
        for (&g, &w) in got.iter().zip(&want) {
            let d = (g as i32 - w as i32).abs();
            if d != 0 {
                differing += 1;
            }
            worst = worst.max(d);
        }
        eprintln!(
            "{}: {} of {} bytes differ from Pillow, worst by {worst} LSB",
            c.name,
            differing,
            want.len()
        );
        assert!(
            worst <= 1,
            "{}: {differing} bytes differ from Pillow by up to {worst} LSB — more \
             than the half-LSB the fixed-point accumulation can account for, so \
             the filter, the window, or the antialiasing rule is wrong",
            c.name
        );
        assert!(
            differing * 100 < want.len(),
            "{}: {differing} of {} bytes differ; even at 1 LSB that is too many \
             to be rounding",
            c.name,
            want.len()
        );
    }
}

/// The three plausible wrong resamplers, and how far each lands.
///
/// Without this the test above only says "some bicubic ran". Each of these runs,
/// produces a plausible image, and shifts every patch.
#[test]
fn the_wrong_bicubic_variants_are_visibly_different() {
    for c in &cases() {
        let src = std::fs::read(fixtures().join(format!("{}.src.u8", c.name))).unwrap();
        let want = std::fs::read(fixtures().join(format!("{}.dst.u8", c.name))).unwrap();
        let scale = c.src_h as f64 / c.dst_h as f64;

        // 1. No antialiasing: the plain 4-tap kernel regardless of scale.
        let plain = resample_variant(&src, c, -0.5, false);
        // 2. OpenCV's a = -0.75, with antialiasing.
        let opencv = resample_variant(&src, c, -0.75, true);
        // 3. Full precision between the passes (no 8-bit intermediate).
        let full = resample_full_precision(&src, c);

        for (label, got, must_differ) in [
            ("no antialiasing", &plain, scale > 1.5),
            ("a = -0.75", &opencv, true),
            ("no 8-bit intermediate", &full, false),
        ] {
            let worst = got
                .iter()
                .zip(&want)
                .map(|(&g, &w)| (g as i32 - w as i32).abs())
                .max()
                .unwrap();
            let differing = got
                .iter()
                .zip(&want)
                .filter(|&(&g, &w)| g != w)
                .count();
            eprintln!(
                "{} [{label}]: worst {worst} LSB, {differing} of {} bytes differ",
                c.name,
                want.len()
            );
            if must_differ {
                assert!(
                    worst > 1,
                    "{} [{label}]: agrees with Pillow to {worst} LSB, so this \
                     fixture cannot rule it out",
                    c.name
                );
            }
        }
        // Upscaling is the case where antialiasing does not apply — filterscale
        // is clamped to 1 — so record that rather than pretending the check
        // discriminates there.
        if scale <= 1.0 {
            assert_eq!(
                plain,
                image::pil_resize_u8(&src, c.src_h, c.src_w, c.channels, c.dst_h, c.dst_w),
                "{}: this is an upscale, so the antialiasing widening must be \
                 inactive and the two must agree exactly",
                c.name
            );
        }
    }
}

/// A variant resampler: `a` selects the kernel constant, `antialias` whether the
/// filter widens when downscaling.
fn resample_variant(src: &[u8], c: &Case, a: f64, antialias: bool) -> Vec<u8> {
    let kernel = move |x: f64| -> f64 {
        let x = x.abs();
        if x < 1.0 {
            ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
        } else if x < 2.0 {
            (((x * a) - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a
        } else {
            0.0
        }
    };
    let coeffs = |in_size: usize, out_size: usize| -> Vec<(usize, Vec<f64>)> {
        let scale = in_size as f64 / out_size as f64;
        let fs = if antialias { scale.max(1.0) } else { 1.0 };
        let support = 2.0 * fs;
        (0..out_size)
            .map(|xx| {
                let center = (xx as f64 + 0.5) * scale;
                let lo = ((center - support + 0.5).max(0.0)) as usize;
                let hi = (((center + support + 0.5) as isize).max(0) as usize).min(in_size);
                let mut k: Vec<f64> = (lo..hi)
                    .map(|x| kernel((x as f64 - center + 0.5) / fs))
                    .collect();
                let ww: f64 = k.iter().sum();
                if ww != 0.0 {
                    for v in k.iter_mut() {
                        *v /= ww;
                    }
                }
                (lo, k)
            })
            .collect()
    };
    let clip = |v: f64| -> u8 {
        let r = (v + 0.5).floor();
        r.clamp(0.0, 255.0) as u8
    };
    let hk = coeffs(c.src_w, c.dst_w);
    let mut mid = vec![0u8; c.src_h * c.dst_w * c.channels];
    for y in 0..c.src_h {
        for (xx, (lo, k)) in hk.iter().enumerate() {
            for ch in 0..c.channels {
                let mut acc = 0.0;
                for (i, w) in k.iter().enumerate() {
                    acc += w * src[((y * c.src_w) + lo + i) * c.channels + ch] as f64;
                }
                mid[((y * c.dst_w) + xx) * c.channels + ch] = clip(acc);
            }
        }
    }
    let vk = coeffs(c.src_h, c.dst_h);
    let mut out = vec![0u8; c.dst_h * c.dst_w * c.channels];
    for (yy, (lo, k)) in vk.iter().enumerate() {
        for x in 0..c.dst_w {
            for ch in 0..c.channels {
                let mut acc = 0.0;
                for (i, w) in k.iter().enumerate() {
                    acc += w * mid[(((lo + i) * c.dst_w) + x) * c.channels + ch] as f64;
                }
                out[((yy * c.dst_w) + x) * c.channels + ch] = clip(acc);
            }
        }
    }
    out
}

/// The same two passes with an f64 intermediate — more accurate than Pillow, and
/// therefore not Pillow.
fn resample_full_precision(src: &[u8], c: &Case) -> Vec<u8> {
    let coeffs = |in_size: usize, out_size: usize| -> Vec<(usize, Vec<f64>)> {
        let scale = in_size as f64 / out_size as f64;
        let fs = scale.max(1.0);
        let support = 2.0 * fs;
        (0..out_size)
            .map(|xx| {
                let center = (xx as f64 + 0.5) * scale;
                let lo = ((center - support + 0.5).max(0.0)) as usize;
                let hi = (((center + support + 0.5) as isize).max(0) as usize).min(in_size);
                let mut k: Vec<f64> = (lo..hi)
                    .map(|x| image::bicubic((x as f64 - center + 0.5) / fs))
                    .collect();
                let ww: f64 = k.iter().sum();
                if ww != 0.0 {
                    for v in k.iter_mut() {
                        *v /= ww;
                    }
                }
                (lo, k)
            })
            .collect()
    };
    let hk = coeffs(c.src_w, c.dst_w);
    let mut mid = vec![0.0f64; c.src_h * c.dst_w * c.channels];
    for y in 0..c.src_h {
        for (xx, (lo, k)) in hk.iter().enumerate() {
            for ch in 0..c.channels {
                let mut acc = 0.0;
                for (i, w) in k.iter().enumerate() {
                    acc += w * src[((y * c.src_w) + lo + i) * c.channels + ch] as f64;
                }
                mid[((y * c.dst_w) + xx) * c.channels + ch] = acc;
            }
        }
    }
    let vk = coeffs(c.src_h, c.dst_h);
    let mut out = vec![0u8; c.dst_h * c.dst_w * c.channels];
    for (yy, (lo, k)) in vk.iter().enumerate() {
        for x in 0..c.dst_w {
            for ch in 0..c.channels {
                let mut acc = 0.0;
                for (i, w) in k.iter().enumerate() {
                    acc += w * mid[(((lo + i) * c.dst_w) + x) * c.channels + ch];
                }
                out[((yy * c.dst_w) + x) * c.channels + ch] =
                    (acc + 0.5).floor().clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}

/// Normalization is `2 * (x / 255) - 1`, on `[-1, 1]`, and the layout it writes
/// is the planar one `patchify` reads.
#[test]
fn normalization_uses_the_configs_half_and_not_clips_constants() {
    let (h, w, ch) = (4usize, 6usize, 3usize);
    let src: Vec<u8> = (0..h * w * ch).map(|i| (i * 7 % 256) as u8).collect();
    let got = image::normalize_planar(&src, h, w, ch, &image::IMAGE_MEAN, &image::IMAGE_STD);
    assert_eq!(got.len(), ch * h * w);

    for c in 0..ch {
        for y in 0..h {
            for x in 0..w {
                let raw = src[(y * w + x) * ch + c] as f32;
                let want = 2.0 * (raw / 255.0) - 1.0;
                let g = got[(c * h + y) * w + x];
                assert!(
                    (g - want).abs() < 1e-6,
                    "channel {c} at ({y}, {x}): got {g}, want {want} — the \
                     interleaved-to-planar transpose or the constants are wrong"
                );
            }
        }
    }
    assert!(got.iter().all(|v| (-1.0..=1.0).contains(v)), "outside [-1, 1]");

    // CLIP's constants are `Qwen2VLImageProcessor`'s *class* default and this
    // config overrides them. Using them runs and offsets every patch.
    const CLIP_MEAN: [f32; 3] = [0.481_454_66, 0.457_827_5, 0.408_210_73];
    const CLIP_STD: [f32; 3] = [0.268_629_54, 0.261_302_6, 0.275_777_1];
    let clip = image::normalize_planar(&src, h, w, ch, &CLIP_MEAN, &CLIP_STD);
    let worst = got
        .iter()
        .zip(&clip)
        .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
    assert!(
        worst > 0.05,
        "CLIP's normalization agrees with this config's to {worst}, so nothing \
         here shows which was used"
    );
    eprintln!("CLIP's constants move every pixel by up to {worst:.3} of the [-1, 1] range");
}

/// The whole preprocessing path, from an 8-bit image to the tower's input rows:
/// `smart_resize`, resize, normalize, patchify. Sizes and the grid have to line
/// up end to end, and the resize target has to be on the `patch * merge` grid.
#[test]
fn the_preprocessing_path_produces_a_whole_patch_grid() {
    let d = VisionDims::QWEN35_27B;
    let factor = d.resize_factor();
    assert_eq!(factor, 32, "patch * merge");
    let (min_px, max_px) = (65536usize, 16777216usize);

    for (h, w) in [(300usize, 400usize), (37, 1200), (1080, 1920), (64, 64)] {
        let (th, tw) = vref::smart_resize(h, w, factor, min_px, max_px)
            .unwrap_or_else(|| panic!("{h}x{w} refused"));
        let src: Vec<u8> = (0..h * w * 3).map(|i| (i * 31 % 251) as u8).collect();
        let frame = image::prepare_frame(&src, h, w, 3, th, tw, d.patch, d.merge);

        assert_eq!((frame.height, frame.width), (th, tw));
        assert_eq!(frame.planar.len(), 3 * th * tw);
        // The grid must be even in both axes, which is the entire reason
        // smart_resize rounds to patch * merge rather than patch.
        assert_eq!(frame.grid_h % d.merge, 0, "{h}x{w}: odd patch grid height");
        assert_eq!(frame.grid_w % d.merge, 0, "{h}x{w}: odd patch grid width");

        let (pixels, gh, gw) = vref::patchify(&frame.planar, th, tw, &d);
        assert_eq!((gh, gw), (frame.grid_h, frame.grid_w));
        assert_eq!(pixels.len(), gh * gw * d.patch_dim());
        let tokens = gh * gw / d.merge_unit();
        assert!(
            (64..=16384).contains(&tokens),
            "{h}x{w} -> {gh}x{gw} patches -> {tokens} tokens, outside the \
             min_pixels/max_pixels budget of 64..16384"
        );
        eprintln!(
            "{h}x{w} -> resize {th}x{tw} -> {gh}x{gw} patches -> {tokens} tokens"
        );

        // Rounding to `patch` instead would sometimes give an odd grid; record
        // when it does, so the reason for the factor is visible here too.
        if let Some((bh, bw)) = vref::smart_resize(h, w, d.patch, min_px, max_px) {
            let odd = !(bh / d.patch).is_multiple_of(d.merge)
                || !(bw / d.patch).is_multiple_of(d.merge);
            if odd {
                eprintln!(
                    "  (rounding to patch={} would have given {}x{} — an odd grid)",
                    d.patch,
                    bh / d.patch,
                    bw / d.patch
                );
            }
        }
    }
}
