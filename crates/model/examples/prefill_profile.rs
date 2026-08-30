//! One-shot per-kernel breakdown of a single long prefill, for AWQ/safetensors
//! checkpoints that `generate` (GGUF-only) can't load. Always run under
//! `INFERO_PROFILE=1` — that's the whole point, and it disables graph capture,
//! so the total won't match a normal serve.
//!
//!   INFERO_PROFILE=1 cargo run --release -p infero-model --example prefill_profile -- <model-dir> [n_tokens]

use anyhow::{Context, Result};
use infero_model::{BatchItem, KvCacheQuant, Model};

fn main() -> Result<()> {
    tracing_subscriber::fmt().with_max_level(tracing::Level::WARN).init();
    let mut args = std::env::args().skip(1);
    let dir = args.next().context("usage: prefill_profile <model-dir> [n_tokens]")?;
    let n_tokens: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(30500);

    let dev = infero_cuda::Device::new(0)?;
    let max_seq = (n_tokens + 128).next_power_of_two().max(8192);
    let mut model = Model::load_awq(dev, &dir, max_seq, KvCacheQuant::F16, 32)?;

    // Token id 100 for every position: content doesn't matter, only the shape
    // does, and this skips paying for a real tokenizer + a multi-MB prompt string.
    let prompt: Vec<u32> = vec![100u32; n_tokens];
    let mut pool = model.new_pool(max_seq, 1)?;
    let seq = pool.alloc().context("no kv slot")?;

    // One real forward call can't exceed `batch_tokens()` -- chunk the same
    // way the server's scheduler does for a long prompt.
    let budget = model.batch_tokens();
    println!("chunking {n_tokens} tokens at {budget} a pass");
    let t0 = std::time::Instant::now();
    for chunk in prompt.chunks(budget) {
        let item = BatchItem::new(seq, chunk);
        model.forward_batch_device(std::slice::from_ref(&item), &mut pool)?;
    }
    // `forward_batch_device` does not block on the GPU (the real server's own
    // sampling call is what waits, deliberately -- see the note in
    // `Scheduler::step`), so the loop above returns once every kernel is
    // *launched*, not once the last one has *run*. Without this, the reported
    // time silently drops however much of the GPU's queue is still draining
    // when the CPU reaches here -- measured on a 30552-token prefill as
    // missing the better part of a second.
    model.device().synchronize()?;
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("prefill {n_tokens} tokens: {ms:.1} ms ({:.1} tok/s)", n_tokens as f64 / (ms / 1000.0));

    if model.device().profile().enabled() {
        println!("{}", model.device().profile().report());
    }
    Ok(())
}
