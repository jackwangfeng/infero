//! What does a decode step cost as the batch grows?
//!
//! Continuous batching only pays if a step with `n` sequences costs far less
//! than `n` steps with one. This measures that directly, with the prompts
//! already in the pool so nothing but decode is timed.
//!
//!     cargo run --release -p infero-model --example batch_bench -- model.gguf

use std::time::Instant;

use anyhow::{Context, Result};
use infero_model::{BatchItem, KvCacheQuant, Model};

fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter("warn").init();

    let path = std::env::args()
        .nth(1)
        .context("usage: batch_bench <model.gguf> [context]")?;
    let ctx: usize = std::env::args()
        .nth(2)
        .and_then(|v| v.parse().ok())
        .unwrap_or(512);

    // The KV cache is 24% of a batch-32 step's memory traffic at this context
    // length, so how it is stored is part of what this measures.
    let kv = match std::env::var("INFERO_KV").as_deref() {
        Ok("tq2") => KvCacheQuant::Tq2,
        Ok("tq4") => KvCacheQuant::Tq4,
        Ok("tq8") => KvCacheQuant::Tq8,
        _ => KvCacheQuant::F16,
    };

    let dev = infero_cuda::Device::new(0)?;
    // A directory is an AWQ checkpoint, a file is a GGUF.
    let mut model = if std::path::Path::new(&path).is_dir() {
        Model::load_awq(dev, &path, 2048, kv, 128)?
    } else {
        let gguf = infero_gguf::Gguf::open(&path)?;
        Model::load_quantized(dev, &gguf, 2048, kv)?
    };

    println!(
        "\ndecode steps with {ctx} tokens of history per sequence\n\n\
         {:>6} {:>10} {:>12} {:>10}",
        "batch", "ms/step", "tokens/s", "speedup"
    );

    let mut baseline = 0.0f64;
    // `INFERO_BATCHES=8,32,64` overrides. Continuous batching means a server
    // under load runs wider than any fixed number here, and the tensor-core
    // GEMM is only defined up to `MMQ_MAX_TOKENS`.
    let batches: Vec<usize> = match std::env::var("INFERO_BATCHES") {
        Ok(v) => v.split(',').filter_map(|s| s.trim().parse().ok()).collect(),
        Err(_) => vec![1, 2, 4, 8, 16, 32],
    };
    for batch in batches {
        let mut pool = model.new_pool(batch * (ctx + 64), batch.max(1))?;
        let seqs: Vec<_> = (0..batch).map(|_| pool.alloc().unwrap()).collect();

        // Fill each sequence's history.
        let filler: Vec<u32> = (0..ctx).map(|i| (1000 + i % 5000) as u32).collect();
        for &seq in &seqs {
            for chunk in filler.chunks(infero_model::MAX_BATCH_TOKENS) {
                model.forward_batch(&[BatchItem::without_logits(seq, chunk)], &mut pool)?;
            }
        }

        let tok = [42u32];
        let step = |m: &mut Model, p: &mut _| -> Result<()> {
            let items: Vec<BatchItem<'_>> = seqs.iter().map(|&s| BatchItem::new(s, &tok)).collect();
            m.forward_batch(&items, p)?;
            Ok(())
        };

        for _ in 0..3 {
            step(&mut model, &mut pool)?;
        }
        // Warm-up ran before this, so the table only covers the timed steps.
        model.device().profile().reset();
        let reps = 20;
        let t = Instant::now();
        for _ in 0..reps {
            step(&mut model, &mut pool)?;
        }
        let ms = t.elapsed().as_secs_f64() * 1e3 / reps as f64;
        let tps = batch as f64 / (ms / 1e3);
        if batch == 1 {
            baseline = tps;
        }
        println!("{batch:>6} {ms:>10.2} {tps:>12.1} {:>9.2}x", tps / baseline);
        if model.device().profile().enabled() {
            println!("\n{}", model.device().profile().report());
        }
    }
    Ok(())
}
