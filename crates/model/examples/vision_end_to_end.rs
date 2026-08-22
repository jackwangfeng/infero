//! Vision, end to end on the real 27B: same prompt, two different images, and
//! the answers have to differ.
//!
//! Every stage of the tower is already pinned numerically —
//! `crates/kernels/tests/vision.rs` against a capture of the reference, and
//! `tests/qwen35_vision*.rs` on the host side. What none of that can show is
//! whether the features reach the language model, because a splice that writes
//! nowhere leaves fluent text behind: the model simply answers about the words
//! around the placeholders. So the check here is a *difference*, which is the
//! only thing that cannot be faked by a splice that did nothing.
//!
//!   cargo run --release -p tuili-model --example vision_end_to_end -- <model-dir>

use anyhow::{Context, Result};
use tuili_model::{BatchItem, KvCacheQuant, Model};

/// A solid frame of one colour, `[H, W, 3]` u8 — the crudest possible image
/// that still has content, and enough for a difference test.
fn solid(h: usize, w: usize, rgb: [u8; 3]) -> Vec<u8> {
    let mut v = Vec::with_capacity(h * w * 3);
    for _ in 0..h * w {
        v.extend_from_slice(&rgb);
    }
    v
}

/// Half one colour, half another, split down the middle — so the two test
/// images differ in layout as well as in mean, which a tower that ignored
/// position would still pass on colour alone.
fn split(h: usize, w: usize, left: [u8; 3], right: [u8; 3]) -> Vec<u8> {
    let mut v = Vec::with_capacity(h * w * 3);
    for _ in 0..h {
        for x in 0..w {
            v.extend_from_slice(if x < w / 2 { &left } else { &right });
        }
    }
    v
}

fn main() -> Result<()> {
    tracing_subscriber::fmt().with_max_level(tracing::Level::INFO).init();
    let dir = std::env::args().nth(1).expect("usage: vision_end_to_end <model-dir>");
    let device: usize = std::env::var("TUILI_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    let dev = tuili_cuda::Device::new(device)?;
    let tok = tuili_tokenizer::Tokenizer::from_hf_dir(&dir)?;

    let mut model = Model::load_awq(dev, &dir, 4096, KvCacheQuant::F16, 8)?;
    // 1024 patches is a 512x512 image at patch 16, which is plenty for a solid
    // frame and keeps the scratch at 87 MB.
    const MAX_PATCHES: usize = 1024;
    anyhow::ensure!(
        model.load_vision_tower(&dir, MAX_PATCHES)?,
        "this checkpoint has no vision tower"
    );
    let shape = *model.vision_shape().context("no shape")?;
    let (img_tok, _vid_tok) = model.vision_tokens().context("no vision tokens")?;

    // 224x224 in, resized by the processor's own rule.
    let (th, tw, tokens) = model.vision_resize(224, 224, MAX_PATCHES)?;
    println!("224x224 -> {th}x{tw}, {tokens} language-model tokens");

    let images: [(&str, Vec<u8>); 2] = [
        ("solid red", solid(224, 224, [220, 30, 30])),
        ("blue left, yellow right", split(224, 224, [30, 60, 220], [230, 220, 40])),
    ];

    let mut answers = Vec::new();
    let mut feature_sums = Vec::new();
    for (label, pixels) in &images {
        let frame = tuili_model::qwen35_vision_image::prepare_frame(
            pixels, 224, 224, 3, th, tw, shape.patch, shape.merge,
        );
        let feats = model.encode_image(&frame)?;
        anyhow::ensure!(
            feats.tokens == tokens,
            "{label}: the tower produced {} tokens where the resize said {tokens}",
            feats.tokens
        );
        // A cheap fingerprint, so that "the two answers differ" can be
        // attributed to the features differing rather than to sampling.
        let host = model.device().stream().clone_dtoh(&feats.view())?;
        model.device().synchronize()?;
        let sum: f64 = host.iter().map(|x| *x as f64).sum();
        let absmax = host.iter().fold(0.0f32, |m, x| m.max(x.abs()));
        println!("{label}: features sum {sum:.3} absmax {absmax:.4}");
        anyhow::ensure!(absmax.is_finite() && absmax > 0.0, "{label}: dead features");
        feature_sums.push(sum);

        // `<|vision_start|>` + one placeholder a token + `<|vision_end|>`, inside
        // an ordinary user turn. The placeholder count is what `vision_targets`
        // matches against, so it has to be exactly `tokens` long.
        let mut ids = tok.encode("<|im_start|>user\n", Some(false), false);
        ids.push(248_053);
        ids.extend(std::iter::repeat_n(img_tok, tokens));
        ids.push(248_054);
        ids.extend(tok.encode(
            "这张图是什么颜色？只回答颜色。<|im_end|>\n<|im_start|>assistant\n",
            Some(false),
            false,
        ));

        let mut pool = model.new_pool(4096, 1)?;
        let seq = pool.alloc().context("no kv slot")?;
        let mut out: Vec<u32> = Vec::new();
        // Prefill with the features attached, then decode greedily. One pass for
        // the prompt: the splice's row indices are per item, and a chunked
        // prefill would need the caller to cut the feature rows the same way.
        {
            let item = BatchItem {
                seq,
                tokens: &ids,
                wants_logits: true,
                vision: Some(&feats),
            };
            model.forward_batch_device(std::slice::from_ref(&item), &mut pool)?;
        }
        let mut next = argmax(model.logits_host()?);
        for _ in 0..24 {
            out.push(next);
            let item = BatchItem::new(seq, std::slice::from_ref(&next));
            model.forward_batch_device(std::slice::from_ref(&item), &mut pool)?;
            next = argmax(model.logits_host()?);
        }
        let text = tok.decode(&out, false);
        println!("{label}: {text:?}");
        answers.push(text);
    }

    anyhow::ensure!(
        (feature_sums[0] - feature_sums[1]).abs() > 1e-3,
        "the two images produced the same features, so the tower is not reading \
         the pixels: {:?}",
        feature_sums
    );
    anyhow::ensure!(
        answers[0] != answers[1],
        "greedy decoding gave the same answer for a red image and a blue/yellow \
         one, so the features are not reaching the language model — which is \
         exactly what a splice writing to the wrong rows looks like:\n  {:?}",
        answers[0]
    );
    println!("\nthe answers differ, so the features reach the language model");
    Ok(())
}

fn argmax(v: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, x) in v.iter().enumerate() {
        if *x > v[best] {
            best = i;
        }
    }
    best as u32
}
