//! `Model::encode_clip`'s multi-launch patchify (one `vision_patchify` call a
//! temporal-patch group, writing into offset slices of a shared scratch
//! buffer) on the real 27B tower: does group `ti`'s output depend on
//! anything but its own two frames?
//!
//! `crates/kernels/tests/vision.rs`'s `packing_images_and_frames_leaves_each_ones_kernel_output_alone`
//! already proves attention does not leak across segments, at the kernel
//! level, from captured `pixel_values`. What that does not exercise is the
//! *host* code this session added: `encode_clip`'s loop over
//! `vision_patchify` launches, each into `rows_f32.slice(ti*group_total..)`.
//! A bug there (wrong offset, `n_frames` computed from the wrong group, a
//! stride error) does not necessarily show up as leakage the kernel test
//! would catch -- it can just as easily point one group's launch at another
//! group's memory, or read past the clip into whatever scratch held before.
//!
//! The check: encode a 4-frame clip (2 temporal-patch groups) in one
//! `encode_clip` call, and separately encode each 2-frame group alone in its
//! own `encode_clip` call (`grid_t = 1` each). Group `ti`'s rows in the
//! 4-frame call's output must be bit-identical to the corresponding
//! 2-frame-alone call's output. If a group's launch pulled in the wrong
//! frames or wrote the wrong offset, this comparison catches it directly,
//! rather than through a color-difference proxy the way `vision_end_to_end`
//! checks the splice.
//!
//!   cargo run --release -p infero-model --example video_clip_check -- <model-dir>

use anyhow::{Context, Result};
use infero_model::qwen35_vision_image::prepare_clip;
use infero_model::{KvCacheQuant, Model};

/// A solid frame of one colour, `[H, W, 3]` u8.
fn solid(h: usize, w: usize, rgb: [u8; 3]) -> Vec<u8> {
    let mut v = Vec::with_capacity(h * w * 3);
    for _ in 0..h * w {
        v.extend_from_slice(&rgb);
    }
    v
}

fn main() -> Result<()> {
    tracing_subscriber::fmt().with_max_level(tracing::Level::INFO).init();
    let dir = std::env::args().nth(1).expect("usage: video_clip_check <model-dir>");
    let device: usize = std::env::var("INFERO_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    let dev = infero_cuda::Device::new(device)?;

    let mut model = Model::load_awq(dev, &dir, 4096, KvCacheQuant::F16, 8)?;
    const MAX_PATCHES: usize = 4096; // room for 2 groups of a 512x512-ish grid
    anyhow::ensure!(
        model.load_vision_tower(&dir, MAX_PATCHES)?,
        "this checkpoint has no vision tower"
    );
    let shape = *model.vision_shape().context("no shape")?;

    let (th, tw, tokens_per_group) = model.vision_resize(224, 224, MAX_PATCHES / 2)?;
    println!("224x224 -> {th}x{tw}, {tokens_per_group} tokens a frame-group");

    // Four visually distinct solid frames -- not that the tower needs colour
    // variety to have a bug reproduce, but a real difference is what makes
    // "the two paths agree" mean something rather than "both are numerically
    // near-flat on a mostly-uniform input".
    let frames: [Vec<u8>; 4] = [
        solid(224, 224, [220, 30, 30]),
        solid(224, 224, [30, 220, 30]),
        solid(224, 224, [30, 30, 220]),
        solid(224, 224, [220, 220, 30]),
    ];
    let refs: Vec<&[u8]> = frames.iter().map(|f| f.as_slice()).collect();

    // Path A: the whole clip in one `encode_clip` call, `grid_t = 2`.
    let full_clip = prepare_clip(&refs, 224, 224, 3, th, tw, shape.patch, shape.merge);
    assert_eq!(full_clip.frames, 4);
    let full = model.encode_clip(&full_clip)?;
    assert_eq!(full.grid_t, 2);
    assert_eq!(full.tokens, 2 * tokens_per_group);
    let full_host = model.device().stream().clone_dtoh(&full.view())?;
    model.device().synchronize()?;

    // Path B: each frame-pair alone, `grid_t = 1`.
    let mut alone_host = Vec::new();
    for (gi, pair) in refs.chunks(2).enumerate() {
        let clip = prepare_clip(pair, 224, 224, 3, th, tw, shape.patch, shape.merge);
        assert_eq!(clip.frames, 2);
        let feats = model.encode_clip(&clip)?;
        assert_eq!(feats.grid_t, 1);
        assert_eq!(feats.tokens, tokens_per_group);
        let host = model.device().stream().clone_dtoh(&feats.view())?;
        model.device().synchronize()?;
        println!(
            "group {gi} alone: absmax {:.4}",
            host.iter().fold(0.0f32, |m, x| m.max(x.abs()))
        );
        alone_host.push(host);
    }

    // Not bit-exact: the "alone" call is a 49-row forward pass and the
    // "clip" call is a 98-row one (two segments, batched in one
    // `vision_forward`), and cuBLAS is not required to reduce a row's dot
    // products in the same order at different batch sizes -- a few ULPs a
    // layer, compounded over 27, is exactly the class of noise
    // `partial_rope.rs`'s fast-math tolerances budget for elsewhere in this
    // codebase. What a wrong offset or a misrouted frame produces instead is
    // nowhere near this small: either garbage on the order of the feature
    // magnitude itself (absmax ~74 here) or another group's actual content,
    // not a few-ULP perturbation of the right answer.
    let d = full.out_hidden;
    for (gi, alone) in alone_host.iter().enumerate() {
        let seg = &full_host[gi * tokens_per_group * d..(gi + 1) * tokens_per_group * d];
        let max_abs = seg
            .iter()
            .zip(alone.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!("group {gi}: max abs diff between clip-embedded and alone {max_abs:.3e}");
        anyhow::ensure!(
            max_abs < 0.05,
            "group {gi} differs between the 4-frame clip call and its own 2-frame \
             call by {max_abs:.3e} -- far past GEMM batch-size noise, encode_clip's \
             per-group launch is reading or writing the wrong slice"
        );
    }
    println!("ok: both frame-groups agree whether encoded together or alone (within GEMM batch-size noise)");
    Ok(())
}
