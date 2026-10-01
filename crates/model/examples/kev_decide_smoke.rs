//! Real end-to-end smoke test for Kev-style decision scoring: loads a
//! real (offline-merged) Kev-0.8B checkpoint, scores one real yes/no
//! question against one real piece of text, and prints the probabilities.
//!
//!   cargo run --release -p infero-model --example kev_decide_smoke -- \
//!     <merged-checkpoint-dir> <kev_head.safetensors-dir>
//!
//! Both directories are usually the same one -- `merge_kev_lora.py`'s own
//! output directory carries the merged backbone and `kev_head.safetensors`/
//! `kev_head.json` side by side.

use anyhow::{Context, Result};
use infero_model::kev::{self, PointerHead, QuestionInput, ScoreQuestion, SpecialTokens};
use infero_model::{BatchItem, BatchItemKind, KvCacheQuant, Model};

fn main() -> Result<()> {
    tracing_subscriber::fmt().with_max_level(tracing::Level::WARN).init();
    let mut args = std::env::args().skip(1);
    let model_dir = args.next().context("usage: kev_decide_smoke <checkpoint-dir> <head-dir>")?;
    let head_dir = args.next().unwrap_or_else(|| model_dir.clone());

    // Real ids, resolved once offline against this exact checkpoint's real
    // tokenizer.json (`<|box_start|>`/`<|box_end|>`/`<|fim_prefix|>`/
    // `<|fim_middle|>`/`<|fim_suffix|>` -- `kev/model.py`'s own `SPECIAL`
    // list, in that order) -- not re-derived here, since this crate does
    // not depend on a tokenizer (see `kev.rs`'s module doc).
    let special = SpecialTokens {
        state: 248060,
        q: 248061,
        opt: 248049,
        opt_close: 248050,
        decide: 248062,
    };

    let head_file = infero_safetensors::File::open(format!("{head_dir}/kev_head.safetensors"))?;
    let head_meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(format!("{head_dir}/kev_head.json"))?)?;
    let d_pointer = head_meta["head_dim"].as_u64().context("head_dim")? as usize;
    let temperature = head_meta["temperature"].as_f64().context("temperature")? as f32;
    let get = |name: &str| -> Result<Vec<f32>> {
        head_file.get(name).with_context(|| format!("missing {name}"))?.to_f32()
    };
    let q_weight = get("q.weight")?;
    let q_bias = get("q.bias")?;
    let k_weight = get("k.weight")?;
    let k_bias = get("k.bias")?;
    println!(
        "loaded head: q.weight {} floats, d_pointer={d_pointer}, temperature={temperature}",
        q_weight.len()
    );

    let dev = infero_cuda::Device::new(0)?;
    let max_seq = 512usize;
    let mut model = Model::load_awq(dev, &model_dir, max_seq, KvCacheQuant::F16, 4)?;
    let d_model = model.config().d_model;
    let head = PointerHead::new(q_weight, q_bias, k_weight, k_bias, d_model, d_pointer, temperature)?;

    // Already-tokenized against this exact checkpoint's tokenizer (printed
    // by a one-off Python `AutoTokenizer` call, not re-derived here -- see
    // this file's own doc comment).
    let state: Vec<u32> = vec![
        1919, 1918, 14009, 1238, 799, 1834, 13, 561, 1870, 4131, 369, 16958, 321, 424, 11136, 1330, 5381, 3200, 13, 353,
        1318, 264, 20325, 6849, 13,
    ]; // "This product broke after one day. ... I want a refund immediately."
    let instr: Vec<u32> = vec![3742, 411, 3286, 6572, 30]; // "Is this review positive?"
    let no: Vec<u32> = vec![2083];
    let yes: Vec<u32> = vec![9405];

    let questions = vec![QuestionInput {
        instr,
        options: vec![no, yes],
    }];
    let enc = kev::encode(&special, &state, &questions, 4096, 4096)
        .map_err(|e| anyhow::anyhow!("{e}"))
        .context("encoding the smoke-test record")?;

    let mut pool = model.new_pool(max_seq, 8)?;
    let state_seq = pool.alloc().context("no kv slot for the state")?;
    let item = BatchItem::without_logits(state_seq, &enc.state_ids, BatchItemKind::Prefill);
    model.forward_batch_device(std::slice::from_ref(&item), &mut pool)?;

    let score_questions: Vec<ScoreQuestion<'_>> = enc.branches.iter().map(|b| ScoreQuestion { branch: b }).collect();
    let probs = kev::decide_batch(&mut model, &mut pool, state_seq, &head, &score_questions)?;
    pool.free(state_seq);

    println!("P(no)  = {:.4}", probs[0][0]);
    println!("P(yes) = {:.4}", probs[0][1]);
    anyhow::ensure!((probs[0][0] + probs[0][1] - 1.0).abs() < 1e-4, "probabilities must sum to 1");
    println!(
        "{}",
        if probs[0][1] > probs[0][0] { "verdict: yes" } else { "verdict: no" }
    );
    Ok(())
}
