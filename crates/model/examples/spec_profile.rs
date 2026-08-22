//! Where a speculative round's time goes, measured rather than inferred.
//!
//! End to end the 27B does 30.2 tok/s plain and 38.5 with speculation at k=1,
//! a 1.27x against a mean acceptance of 1.72. The gap between those two numbers
//! is the whole question, and dividing wall-clock throughput by acceptance only
//! says a round costs more than a step — not which part.
//!
//! So this times the three things a round is made of, separately, with a
//! synchronise around each:
//!
//! * a plain decode step, one row — the thing speculation has to beat;
//! * a verification pass at `k + 1` rows, which is the same weights read once
//!   and should therefore cost the same, and does not;
//! * one draft step, whose memory bound is the head's 810 MiB plus the
//!   vocabulary projection.
//!
//! Run it again with `TUILI_PROFILE=1` for the per-kernel breakdown inside those
//! numbers. That serializes the stream and disables graph capture, so the totals
//! move — it answers "which kernel" and not "how long".
//!
//!   cargo run --release -p tuili-model --example spec_profile -- <model-dir>

use anyhow::{Context, Result};
use tuili_model::{BatchItem, KvCacheQuant, Model};

const REPS: usize = 12;

fn main() -> Result<()> {
    tracing_subscriber::fmt().with_max_level(tracing::Level::WARN).init();
    let dir = std::env::args().nth(1).expect("usage: spec_profile <model-dir>");
    let device: usize = std::env::var("TUILI_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    let k: usize = std::env::var("K").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    let dev = tuili_cuda::Device::new(device)?;
    let tok = tuili_tokenizer::Tokenizer::from_hf_dir(&dir)?;
    // `k + 1` logit rows for the verification pass, and the drafter wants at
    // least that many rows of its own.
    let mut model = Model::load_awq(dev, &dir, 8192, KvCacheQuant::F16, (k + 1).max(8))?;
    anyhow::ensure!(model.load_mtp_head(&dir, 64)?, "this checkpoint has no MTP head");

    let prompt = tok.encode(
        "<|im_start|>user\n用一句话说明什么是投机解码<|im_end|>\n<|im_start|>assistant\n",
        Some(false),
        false,
    );
    let mut pool = model.new_pool(8192, 2)?;
    model.enable_speculation(k, &pool)?;
    let seq = pool.alloc().context("no kv slot")?;

    // Prefill, then a few plain steps so the caches are warm and the graphs are
    // captured before anything is timed.
    let item = BatchItem::new(seq, &prompt);
    model.forward_batch_device(std::slice::from_ref(&item), &mut pool)?;
    let mut pending = argmax(model.logits_host()?);
    for _ in 0..4 {
        let it = BatchItem::new(seq, std::slice::from_ref(&pending));
        model.forward_batch_device(std::slice::from_ref(&it), &mut pool)?;
        pending = argmax(model.logits_host()?);
    }

    // ---- a plain decode step, one row ------------------------------------
    let base_len = pool.len(seq);
    let plain = time(
        REPS,
        |m: &mut Model| {
            let it = BatchItem::new(seq, std::slice::from_ref(&pending));
            m.forward_batch_device(std::slice::from_ref(&it), &mut pool)?;
            // Back to where it started, so a hundred reps do not walk the
            // sequence to its context limit and change the kv length the
            // attention reads — which would make the later reps a different
            // measurement from the earlier ones.
            pool.truncate(seq, base_len);
            Ok(())
        },
        &mut model,
    )?;

    // ---- a pass at `k + 1` rows -------------------------------------------
    //
    // Not the verification path — just the forward pass at that width, so the
    // number is about the kernels and not about the acceptance rule. The rows
    // are the pending token repeated, which is wrong text and the right shape.
    let rows: Vec<u32> = std::iter::repeat_n(pending, k + 1).collect();
    let tail = [k + 1];
    let wide = time(
        REPS,
        |m: &mut Model| {
        let it = BatchItem::new(seq, &rows);
        m.forward_batch_rows(std::slice::from_ref(&it), &mut pool, &tail)?;
        pool.truncate(seq, base_len);
        Ok(())
    }, &mut model);

    println!("\n  plain decode, 1 row       {plain:7.2} ms");
    match wide {
        Ok(t) => println!(
            "  forward at {} rows        {t:7.2} ms   {:+.1}% a row",
            k + 1,
            100.0 * (t - plain) / plain / k as f64
        ),
        Err(e) => println!("  forward at {} rows        skipped: {e:#}", k + 1),
    }
    println!(
        "\n  A pass that reads the same weights should cost the same whatever the\n  \
         row count. The batched FP8 mat-vec charges for rows instead, which is\n  \
         what caps speculation's payoff at 1.27x on an acceptance of 1.72."
    );
    let report = model.device().profile().report();
    if !report.starts_with("no kernels") {
        println!("\n{report}");
    }
    Ok(())
}

fn time(
    reps: usize,
    mut f: impl FnMut(&mut Model) -> Result<()>,
    model: &mut Model,
) -> Result<f64> {
    // One untimed rep so a first-call graph capture is not in the average.
    f(model)?;
    model.device().synchronize()?;
    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        f(model)?;
    }
    model.device().synchronize()?;
    Ok(t0.elapsed().as_secs_f64() * 1000.0 / reps as f64)
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
