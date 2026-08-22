//! Turning an actual image into `pixel_values` for the Qwen3.5 vision tower.
//!
//! `qwen35_vision` covers everything downstream of a resized frame: the grid
//! arithmetic, `smart_resize`, `patchify`, the position taps. What is here is the
//! step before that, which the capture cannot pin because it does not contain the
//! source image — only the processor's output. So this module is checked against
//! Pillow directly, in `tests/qwen35_vision_image.rs`, from a fixture generated
//! by the same Pillow the reference processor calls.
//!
//! Why it needs to be this careful. `Qwen2VLImageProcessor` resizes with
//! `PIL.Image.BICUBIC` — `transformers.image_transforms.resize` converts the
//! array to a PIL image, calls `Image.resize`, and converts back — so the
//! resampling is Pillow's, including its two-pass structure and its **8-bit
//! intermediate**. A "reasonable" bicubic gets every patch slightly wrong, which
//! is the same failure class as using CLIP's normalization constants or the wrong
//! GELU: it runs, it reads as fluent, and it gets blamed on the model.
//!
//! Three specific things a from-scratch bicubic gets wrong, in rising order of
//! how much they cost:
//!
//! 1. **No antialiasing.** Pillow widens the filter when downscaling
//!    (`filterscale = max(1, in/out)`), so a 4096 -> 1024 resize averages ~8
//!    source pixels per axis rather than sampling 4. A plain 4-tap bicubic
//!    aliases, and images are usually downscaled here, not up.
//! 2. **One pass in floating point.** Pillow resizes horizontally into an 8-bit
//!    buffer, then vertically out of it. Keeping full precision between the two
//!    passes is *more* accurate and still not what the reference computed.
//! 3. **`a = -0.75`.** OpenCV's bicubic constant. Pillow uses `a = -0.5`.
//!
//! This module is deliberately free of `crate::` references so that the test can
//! reach it before it is wired into `lib.rs`; see the report accompanying this
//! work for the one line `lib.rs` needs.

/// `image_mean` / `image_std` from `preprocessor_config.json`.
///
/// **0.5, not CLIP's.** `Qwen2VLImageProcessor`'s *class default* is
/// `OPENAI_CLIP_MEAN`/`STD` and this checkpoint's config overrides it, so a
/// loader that trusts the class default normalizes every patch with the wrong
/// offsets. The result is `2 * (x / 255) - 1`, on `[-1, 1]`.
pub const IMAGE_MEAN: [f32; 3] = [0.5, 0.5, 0.5];
pub const IMAGE_STD: [f32; 3] = [0.5, 0.5, 0.5];

/// `rescale_factor`.
pub const RESCALE: f32 = 1.0 / 255.0;

/// Pillow's bicubic kernel, with `a = -0.5`.
///
/// OpenCV and several tutorials use `a = -0.75`, which is a visibly different
/// filter — sharper, with more overshoot. Pillow's `support` is 2.0, so the
/// kernel is zero past `|x| >= 2`.
pub fn bicubic(x: f64) -> f64 {
    const A: f64 = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x * A) - 5.0 * A) * x + 8.0 * A) * x - 4.0 * A
    } else {
        0.0
    }
}

/// One output position's source window and weights, following Pillow's
/// `precompute_coeffs`.
///
/// The two details that matter: `filterscale` never drops below 1 (so upscaling
/// uses the plain kernel and downscaling widens it), and the window bounds are
/// computed with `+ 0.5` truncation rather than rounding, which shifts the window
/// by one for about half of all positions.
fn coeffs(in_size: usize, out_size: usize) -> Vec<(usize, Vec<f64>)> {
    let scale = in_size as f64 / out_size as f64;
    let filterscale = scale.max(1.0);
    let support = 2.0 * filterscale;
    let inv = 1.0 / filterscale;
    (0..out_size)
        .map(|xx| {
            let center = (xx as f64 + 0.5) * scale;
            let xmin = ((center - support + 0.5).max(0.0)) as usize;
            let xmax = (((center + support + 0.5) as isize).max(0) as usize).min(in_size);
            let mut k: Vec<f64> = (xmin..xmax)
                .map(|x| bicubic((x as f64 - center + 0.5) * inv))
                .collect();
            let ww: f64 = k.iter().sum();
            if ww != 0.0 {
                for v in k.iter_mut() {
                    *v /= ww;
                }
            }
            (xmin, k)
        })
        .collect()
}

/// Round-and-clamp to 8 bits, which is what Pillow's `clip8` does between its
/// two passes and at the end.
fn clip8(v: f64) -> u8 {
    // Pillow accumulates in fixed point with a half-LSB bias and then shifts,
    // which is round-half-up on non-negative values. `+ 0.5` then truncate is
    // the same thing for the range that survives the clamp.
    let r = (v + 0.5).floor();
    if r <= 0.0 {
        0
    } else if r >= 255.0 {
        255
    } else {
        r as u8
    }
}

/// Resize an interleaved `[H, W, C]` 8-bit image the way Pillow does: a
/// horizontal pass into an 8-bit intermediate, then a vertical pass.
///
/// Reproducing the intermediate quantization is not pedantry. It is a
/// half-LSB-per-pass difference that shows up in every single patch, and the
/// point of matching the reference here is that a later disagreement somewhere
/// else means something.
pub fn pil_resize_u8(
    src: &[u8],
    src_h: usize,
    src_w: usize,
    channels: usize,
    dst_h: usize,
    dst_w: usize,
) -> Vec<u8> {
    assert_eq!(src.len(), src_h * src_w * channels);
    // Horizontal: [src_h, src_w, C] -> [src_h, dst_w, C].
    let hk = coeffs(src_w, dst_w);
    let mut mid = vec![0u8; src_h * dst_w * channels];
    for y in 0..src_h {
        for (xx, (xmin, k)) in hk.iter().enumerate() {
            for c in 0..channels {
                let mut acc = 0.0f64;
                for (i, w) in k.iter().enumerate() {
                    acc += w * src[((y * src_w) + xmin + i) * channels + c] as f64;
                }
                mid[((y * dst_w) + xx) * channels + c] = clip8(acc);
            }
        }
    }
    // Vertical: [src_h, dst_w, C] -> [dst_h, dst_w, C].
    let vk = coeffs(src_h, dst_h);
    let mut out = vec![0u8; dst_h * dst_w * channels];
    for (yy, (ymin, k)) in vk.iter().enumerate() {
        for x in 0..dst_w {
            for c in 0..channels {
                let mut acc = 0.0f64;
                for (i, w) in k.iter().enumerate() {
                    acc += w * mid[(((ymin + i) * dst_w) + x) * channels + c] as f64;
                }
                out[((yy * dst_w) + x) * channels + c] = clip8(acc);
            }
        }
    }
    out
}

/// Rescale by 1/255 and normalize, turning interleaved `[H, W, C]` 8-bit into
/// planar `[C, H, W]` f32 — the layout `patchify` reads.
pub fn normalize_planar(
    src: &[u8],
    height: usize,
    width: usize,
    channels: usize,
    mean: &[f32],
    std: &[f32],
) -> Vec<f32> {
    assert_eq!(src.len(), height * width * channels);
    assert!(mean.len() >= channels && std.len() >= channels);
    let mut out = vec![0.0f32; channels * height * width];
    for c in 0..channels {
        for y in 0..height {
            for x in 0..width {
                let v = src[(y * width + x) * channels + c] as f32 * RESCALE;
                out[(c * height + y) * width + x] = (v - mean[c]) / std[c];
            }
        }
    }
    out
}

/// A frame ready for `patchify`, plus the patch grid it implies.
#[derive(Clone, Debug)]
pub struct PreparedFrame {
    /// Planar `[C, H, W]`, normalized to `[-1, 1]`.
    pub planar: Vec<f32>,
    pub height: usize,
    pub width: usize,
    /// Patch counts, `height / patch` and `width / patch`. Both are even,
    /// because the resize target was rounded to `patch * merge`.
    pub grid_h: usize,
    pub grid_w: usize,
}

/// Resize, rescale, normalize, and report the grid.
///
/// `target_h` / `target_w` come from `qwen35_vision::smart_resize`, which is
/// where the `patch * merge` rounding lives — this function refuses a target that
/// is not on that grid rather than silently truncating a row and a column out of
/// the merger's view.
#[allow(clippy::too_many_arguments)]
pub fn prepare_frame(
    src: &[u8],
    src_h: usize,
    src_w: usize,
    channels: usize,
    target_h: usize,
    target_w: usize,
    patch: usize,
    merge: usize,
) -> PreparedFrame {
    let factor = patch * merge;
    assert!(
        target_h % factor == 0 && target_w % factor == 0,
        "{target_h}x{target_w} is not a multiple of patch * merge = {factor}; \
         smart_resize rounds to that and not to patch, because an odd grid makes \
         `grid / merge` truncate and the position field then describes pixels the \
         merger never sees"
    );
    let resized = pil_resize_u8(src, src_h, src_w, channels, target_h, target_w);
    let planar = normalize_planar(
        &resized,
        target_h,
        target_w,
        channels,
        &IMAGE_MEAN,
        &IMAGE_STD,
    );
    PreparedFrame {
        planar,
        height: target_h,
        width: target_w,
        grid_h: target_h / patch,
        grid_w: target_w / patch,
    }
}
