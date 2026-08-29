//! Cross-step chunked prefill for a vision placeholder run, on the real 27B.
//!
//! `plan()`/`admit()` no longer require one image's or video's whole
//! placeholder run to land in a single `BatchItem` -- a chunk boundary can
//! now fall in the middle of it, and `BatchItem::vision_row_offset` tells
//! `forward_batch_device` which of the clip's rows that chunk's tokens
//! correspond to (`vision.rs`'s `notes/video-encoding-optimizations.md`,
//! item 1). What this checks: encoding the SAME prompt -- text, then an
//! image's placeholder run, then more text -- in one unchunked pass versus
//! two passes whose split point falls strictly inside the placeholder run
//! must agree, because splicing the same feature rows into the same token
//! positions is not supposed to depend on how many calls it took to get
//! there.
//!
//!   cargo run --release -p infero-model --example chunked_vision_prefill_check -- <model-dir>

use anyhow::{Context, Result};
use infero_model::{BatchItem, KvCacheQuant, Model};

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
    let dir = std::env::args().nth(1).expect("usage: chunked_vision_prefill_check <model-dir>");
    let device: usize = std::env::var("INFERO_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    let dev = infero_cuda::Device::new(device)?;
    let tok = infero_tokenizer::Tokenizer::from_hf_dir(&dir)?;

    let mut model = Model::load_awq(dev, &dir, 4096, KvCacheQuant::F16, 8)?;
    const MAX_PATCHES: usize = 1024;
    anyhow::ensure!(
        model.load_vision_tower(&dir, MAX_PATCHES)?,
        "this checkpoint has no vision tower"
    );
    let shape = *model.vision_shape().context("no shape")?;
    let (img_tok, _vid_tok) = model.vision_tokens().context("no vision tokens")?;

    let (th, tw, tokens) = model.vision_resize(224, 224, MAX_PATCHES)?;
    println!("224x224 -> {th}x{tw}, {tokens} language-model tokens");
    anyhow::ensure!(tokens >= 6, "need at least 6 placeholder tokens to split meaningfully, got {tokens}");

    let pixels = solid(224, 224, [200, 60, 20]);
    let frame = infero_model::qwen35_vision_image::prepare_frame(
        &pixels, 224, 224, 3, th, tw, shape.patch, shape.merge,
    );
    let feats = model.encode_image(&frame)?;
    anyhow::ensure!(feats.tokens == tokens, "resize said {tokens}, tower gave {}", feats.tokens);

    // Text, then `<|vision_start|>` + placeholder run + `<|vision_end|>`,
    // then more text -- so the run neither opens nor closes the prompt, and
    // a chunk boundary landing inside it has real tokens on both sides.
    let prefix = tok.encode("<|im_start|>user\nDescribe this: ", Some(false), false);
    let suffix = tok.encode(" What colour is it?<|im_end|>\n<|im_start|>assistant\n", Some(false), false);
    let vision_at = prefix.len() + 1; // +1 for the vision_start token below
    let mut prompt = prefix.clone();
    prompt.push(248_053); // <|vision_start|>
    prompt.extend(std::iter::repeat_n(img_tok, tokens));
    prompt.push(248_054); // <|vision_end|>
    prompt.extend(suffix);
    anyhow::ensure!(prompt[vision_at] == img_tok, "vision_at points at {} not the first placeholder", prompt[vision_at]);

    // Path A: the whole prompt, one `BatchItem`, `vision_row_offset = 0` --
    // today's only path before this change.
    let mut pool_a = model.new_pool(4096, 1)?;
    let seq_a = pool_a.alloc().context("no kv slot")?;
    let item_a = BatchItem {
        seq: seq_a,
        tokens: &prompt,
        wants_logits: true,
        vision: Some(&feats),
        vision_row_offset: 0,
        mrope: None,
        mrope_delta: 0,
    };
    model.forward_batch_device(std::slice::from_ref(&item_a), &mut pool_a)?;
    let logits_a = model.logits_host()?.to_vec();

    // Path B: split strictly inside the placeholder run -- a third of the
    // way through it, so both chunks carry a real, partial slice of `feats`.
    let offset_in_run = tokens / 3;
    anyhow::ensure!(offset_in_run > 0 && offset_in_run < tokens, "split degenerate: {offset_in_run}/{tokens}");
    let split = vision_at + offset_in_run;
    println!("splitting at token {split} of {} -- {offset_in_run}/{tokens} into the placeholder run", prompt.len());

    let mut pool_b = model.new_pool(4096, 1)?;
    let seq_b = pool_b.alloc().context("no kv slot")?;
    let item_b1 = BatchItem {
        seq: seq_b,
        tokens: &prompt[..split],
        wants_logits: false,
        vision: Some(&feats),
        vision_row_offset: 0,
        mrope: None,
        mrope_delta: 0,
    };
    model.forward_batch_device(std::slice::from_ref(&item_b1), &mut pool_b)?;
    let item_b2 = BatchItem {
        seq: seq_b,
        tokens: &prompt[split..],
        wants_logits: true,
        vision: Some(&feats),
        vision_row_offset: offset_in_run,
        mrope: None,
        mrope_delta: 0,
    };
    model.forward_batch_device(std::slice::from_ref(&item_b2), &mut pool_b)?;
    let logits_b = model.logits_host()?.to_vec();

    anyhow::ensure!(logits_a.len() == logits_b.len(), "vocab size mismatch");
    let argmax = |v: &[f32]| v.iter().enumerate().fold((0usize, f32::MIN), |(bi, bv), (i, &v)| if v > bv { (i, v) } else { (bi, bv) }).0;
    let (a_top, b_top) = (argmax(&logits_a), argmax(&logits_b));
    let worst = logits_a.iter().zip(&logits_b).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    println!("argmax: unchunked {a_top}, chunked (split inside the placeholder run) {b_top}; worst logit diff {worst:.4e}");

    // Same tolerance `forward.rs`'s `chunked_prefill_is_seamless` uses for
    // plain-text chunking: GEMM/attention batch-size noise between a
    // 1-chunk and a 2-chunk pass, not the mismatch a wrong `vision_row_offset`
    // would produce (which reads or writes another part of the clip, or
    // writes past its end, and shows up orders of magnitude larger than this).
    println!(
        "vision case: worst logit diff {worst:.4e}, argmax {}",
        if a_top == b_top { "agrees" } else { "DIFFERS" }
    );

    // Control: the same split, same real 27B, same depth -- but plain text,
    // no vision at all. If this shows the same order-of-magnitude diff as
    // the vision case above, that is this model's own chunked-prefill GEMM
    // noise at 64 layers deep, not a `vision_row_offset` bug -- the smaller
    // GGUF `forward.rs`'s `chunked_prefill_is_seamless` uses never exercises
    // that depth, so its 0.05 tolerance was never proven to transfer here.
    let control_prompt = tok.encode(
        "<|im_start|>user\nDescribe this: a solid orange-brown square, roughly \
         224 by 224 pixels, no other detail. What colour is it?<|im_end|>\n\
         <|im_start|>assistant\n",
        Some(false),
        false,
    );
    anyhow::ensure!(control_prompt.len() > 10, "control prompt too short to split meaningfully");
    let control_split = control_prompt.len() / 2;

    let mut pool_c1 = model.new_pool(4096, 1)?;
    let seq_c1 = pool_c1.alloc().context("no kv slot")?;
    let item_c1 = BatchItem::new(seq_c1, &control_prompt);
    model.forward_batch_device(std::slice::from_ref(&item_c1), &mut pool_c1)?;
    let logits_c1 = model.logits_host()?.to_vec();

    let mut pool_c2 = model.new_pool(4096, 1)?;
    let seq_c2 = pool_c2.alloc().context("no kv slot")?;
    let item_c2a = BatchItem::without_logits(seq_c2, &control_prompt[..control_split]);
    model.forward_batch_device(std::slice::from_ref(&item_c2a), &mut pool_c2)?;
    let item_c2b = BatchItem::new(seq_c2, &control_prompt[control_split..]);
    model.forward_batch_device(std::slice::from_ref(&item_c2b), &mut pool_c2)?;
    let logits_c2 = model.logits_host()?.to_vec();

    let control_worst = logits_c1.iter().zip(&logits_c2).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    let (c1_top, c2_top) = (argmax(&logits_c1), argmax(&logits_c2));
    println!(
        "text-only control: worst logit diff {control_worst:.4e}, argmax {}",
        if c1_top == c2_top { "agrees" } else { "DIFFERS" }
    );

    anyhow::ensure!(a_top == b_top, "chunked split changed the argmax token ({a_top} vs {b_top})");
    anyhow::ensure!(
        worst < control_worst * 5.0 + 0.05,
        "vision case diverges {worst:.4e}, far past the text-only control's {control_worst:.4e} -- \
         vision_row_offset is not slicing the right rows"
    );
    println!("ok: the vision case's divergence is in the same regime as the text-only control at this depth");
    Ok(())
}
