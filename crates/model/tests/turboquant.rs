//! TurboQuant KV cache, end to end against the dense f16 cache.
//!
//! The kernel tests establish that the quantizer reproduces the paper's
//! distortion and unbiasedness guarantees. What those guarantees translate to
//! for a *particular* model is a separate question, and this file measures it
//! rather than assuming the paper's Llama-3.1-8B numbers carry over to a
//! 0.5B model with 64-wide heads.

use std::path::PathBuf;

use anyhow::Result;
use infero_cuda::Device;
use infero_gguf::Gguf;
use infero_model::{KvCacheQuant, Model, Sampler, SamplingParams};
use infero_tokenizer::Tokenizer;

/// Enough prompts, and varied enough, that a two-point difference between
/// settings is not just which token happened to be near a tie.
const PROMPTS: &[&str] = &[
    "The capital of France is",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
    "人工智能是",
    "The three primary colors are red,",
    "Water boils at a temperature of",
    "The largest planet in the solar system is",
    "import numpy as np\narr = np.zeros((3, 4))\nprint(arr.",
    "She opened the door and found",
    "In 1969, humans first landed on the",
    "The chemical symbol for gold is",
    "杭州是浙江省的",
    "To reverse a list in Python you can write list[",
    "The mitochondria is the powerhouse of the",
    "Once upon a time, in a village at the edge of the",
    "SELECT name, age FROM users WHERE age >",
    "The derivative of x squared with respect to x is",
];

fn model_path() -> Option<PathBuf> {
    let p = std::env::var("INFERO_TEST_GGUF")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../models/qwen2.5-0.5b-instruct-q8_0.gguf")
        });
    if !p.exists() {
        eprintln!("skipping: {} not downloaded", p.display());
        return None;
    }
    Some(p)
}

fn load(quant: KvCacheQuant) -> Result<Option<(Model, Tokenizer)>> {
    let Some(path) = model_path() else {
        return Ok(None);
    };
    let gguf = Gguf::open(&path)?;
    let tok = Tokenizer::from_gguf(&gguf)?;
    let model = Model::load_quantized(Device::new(0)?, &gguf, 1024, quant)?;
    Ok(Some((model, tok)))
}

/// One test in this file at a time; see the note in `batching.rs`. A decode
/// step captures a CUDA graph, and capture dies if another thread allocates on
/// the same context, so the default test harness fails these where
/// `--test-threads=1` does not.
static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn gpu_lock() -> std::sync::MutexGuard<'static, ()> {
    GPU.lock().unwrap_or_else(|e| e.into_inner())
}

macro_rules! setup {
    ($q:expr) => {{
        let _gpu = gpu_lock();
        match load($q)? {
            Some(v) => (v.0, v.1, _gpu),
            None => return Ok(()),
        }
    }};
}

fn argmax(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .fold((0u32, f32::NEG_INFINITY), |b, (i, &x)| {
            if x > b.1 { (i as u32, x) } else { b }
        })
        .0
}

#[allow(dead_code)]
fn top_k(v: &[f32], k: usize) -> Vec<u32> {
    let mut idx: Vec<u32> = (0..v.len() as u32).collect();
    idx.sort_unstable_by(|&a, &b| v[b as usize].total_cmp(&v[a as usize]));
    idx.truncate(k);
    idx
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
    let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    dot / (na * nb + 1e-30)
}

fn softmax(logits: &[f32]) -> Vec<f64> {
    let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f64> = logits.iter().map(|v| ((v - m) as f64).exp()).collect();
    let s: f64 = e.iter().sum();
    e.into_iter().map(|v| v / s).collect()
}

/// `KL(reference || candidate)` in nats.
///
/// The metric that actually says "did the model's prediction change": cosine
/// between raw logit vectors is dominated by their shared bulk and stays high
/// even when the top of the distribution has been rearranged.
fn kl_divergence(reference: &[f32], candidate: &[f32]) -> f64 {
    let p = softmax(reference);
    let q = softmax(candidate);
    p.iter()
        .zip(&q)
        .filter(|(pi, _)| **pi > 1e-12)
        .map(|(pi, qi)| pi * (pi / qi.max(1e-30)).ln())
        .sum()
}

/// Logits for the final position of each prompt.
fn logits_for(quant: KvCacheQuant) -> Result<Option<Vec<Vec<f32>>>> {
    let (mut model, tok) = match load(quant)? {
        Some(v) => v,
        None => return Ok(None),
    };
    let mut out = Vec::new();
    for p in PROMPTS {
        let ids = tok.encode(p, Some(false), false);
        let mut session = model.new_session()?;
        out.push(model.forward(&ids, &mut session)?.to_vec());
    }
    Ok(Some(out))
}

/// How closely each setting tracks the dense cache, all in one table.
#[test]
fn quality_against_the_dense_cache() -> Result<()> {
    let Some(reference) = logits_for(KvCacheQuant::F16)? else {
        return Ok(());
    };

    eprintln!(
        "\n  {:<10} {:>6}  {:>9}  {:>9}  {:>9}",
        "setting", "bits", "argmax", "KL(nats)", "cosine"
    );

    let mut results = Vec::new();
    for quant in [
        KvCacheQuant::Tq8,
        // Isolate the two sides. Both stay quantized so the comparison is
        // between bit allocations rather than between code paths: whichever
        // of these two holds up is the side where bits are cheap.
        KvCacheQuant::new(8, 2, false)?,
        KvCacheQuant::new(2, 8, false)?,
        // The allocations that isolation suggests: keep keys wide, let values
        // go cheap.
        KvCacheQuant::new(8, 4, false)?,
        KvCacheQuant::new(4, 2, false)?,
        KvCacheQuant::Tq4,
        KvCacheQuant::Tq4Mse,
        KvCacheQuant::new(4, 2, true)?,
        KvCacheQuant::Tq2,
        KvCacheQuant::Tq2Mse,
    ] {
        let got = logits_for(quant)?.expect("model is present");

        let mut agree = 0usize;
        let mut cos_sum = 0.0f64;
        let mut kl_sum = 0.0f64;
        for (g, r) in got.iter().zip(&reference) {
            if argmax(g) == argmax(r) {
                agree += 1;
            }
            cos_sum += cosine(g, r);
            kl_sum += kl_divergence(r, g);
        }
        let n = got.len();
        let kl = kl_sum / n as f64;
        eprintln!(
            "  {:<10} {:>5.2}  {:>5}/{:<3}  {:>9.4}  {:>9.5}",
            quant.name(),
            quant.bits_per_channel(64),
            agree,
            n,
            kl,
            cos_sum / n as f64,
        );
        results.push((quant, agree, kl));
    }

    let get = |q: KvCacheQuant| results.iter().find(|(k, _, _)| *k == q).unwrap();

    // 8 bits is the sanity floor, not a free lunch. Theorem 1 puts the
    // per-vector error at sqrt(D_mse) ~ 0.6% there, against roughly 0.05% for
    // f16, so a tenth of a nat after 24 layers is the expected cost rather
    // than a symptom. What would indicate broken plumbing is 8 bits failing to
    // be dramatically better than 4.
    let (_, agree8, kl8) = *get(KvCacheQuant::Tq8);
    let (_, _, kl4m) = *get(KvCacheQuant::Tq4Mse);
    assert!(
        agree8 >= PROMPTS.len() - 2,
        "tq8 changed {} of {} predictions",
        PROMPTS.len() - agree8,
        PROMPTS.len()
    );
    assert!(kl8 < 0.3, "tq8 KL {kl8:.4} nats");
    assert!(
        kl8 * 5.0 < kl4m,
        "8 bits should be far better than 4: KL {kl8:.4} vs {kl4m:.4}"
    );

    // The finding this file records: keys and values are not interchangeable.
    // At equal total width, spending the bits on keys wins by a wide margin,
    // because a key's error is amplified through the softmax while a value's
    // is averaged away.
    let (_, _, kl_kv) = *get(KvCacheQuant::new(8, 2, false)?);
    let (_, _, kl_vk) = *get(KvCacheQuant::new(2, 8, false)?);
    assert!(
        kl_kv < kl_vk / 2.0,
        "expected keys to matter far more than values: k8v2 KL {kl_kv:.4} vs k2v8 KL {kl_vk:.4}"
    );

    // The QJL stage is deliberately *not* asserted either way. Measured on
    // this model it helps at 4-bit keys and hurts at 2-bit keys, and the
    // kernel-level picture explains why the sign is not obvious: it removes a
    // multiplicative bias, which a softmax barely notices, and pays for that
    // in variance, which a softmax does notice. Reported, not claimed.
    let (_, _, kl4) = *get(KvCacheQuant::Tq4);
    eprintln!(
        "\n  qjl at 4-bit keys: KL {kl4:.4} with, {kl4m:.4} without \
         (+{:.2} bits/channel)",
        KvCacheQuant::Tq4.bits_per_channel(64) - KvCacheQuant::Tq4Mse.bits_per_channel(64)
    );
    Ok(())
}

/// A quantized cache must still be a *cache*: reading a position back has to
/// give the same answer whether it was written during prefill or one token at
/// a time.
#[test]
fn incremental_and_batched_writes_agree() -> Result<()> {
    let (mut model, tok, _gpu) = setup!(KvCacheQuant::new(8, 4, false)?);
    let ids = tok.encode(PROMPTS[0], Some(false), false);

    let mut session = model.new_session()?;
    let batched: Vec<f32> = model.forward(&ids, &mut session)?.to_vec();

    let mut session = model.new_session()?;
    let mut incremental = Vec::new();
    for &t in &ids {
        incremental = model.forward(&[t], &mut session)?.to_vec();
    }

    // Three sources of divergence stack here, none of them a bug: prefill goes
    // through cuBLAS in f16 while a single token takes the integer mat-vec with
    // an 8-bit activation, and the KV cache is quantized on top. The prediction
    // has to survive; the logit vectors do not have to coincide.
    assert_eq!(argmax(&batched), argmax(&incremental));
    let cos = cosine(&batched, &incremental);
    eprintln!("  batched vs incremental logit cosine: {cos:.6}");
    assert!(cos > 0.97, "cosine {cos:.6}");
    Ok(())
}

#[test]
fn a_quantized_cache_is_much_smaller() -> Result<()> {
    let Some(path) = model_path() else {
        return Ok(());
    };
    let _gpu = gpu_lock();
    let gguf = Gguf::open(&path)?;
    let dev = Device::new(0)?;

    let dense = Model::load_quantized(dev.clone(), &gguf, 4096, KvCacheQuant::F16)?
        .new_session()?
        .bytes();
    let tq4 = Model::load_quantized(dev.clone(), &gguf, 4096, KvCacheQuant::Tq4)?
        .new_session()?
        .bytes();
    let tq2 = Model::load_quantized(dev, &gguf, 4096, KvCacheQuant::Tq2)?
        .new_session()?
        .bytes();

    eprintln!(
        "  4096 positions: f16 {:.1} MiB, tq4 {:.1} MiB ({:.2}x), tq2 {:.1} MiB ({:.2}x)",
        dense as f64 / (1 << 20) as f64,
        tq4 as f64 / (1 << 20) as f64,
        dense as f64 / tq4 as f64,
        tq2 as f64 / (1 << 20) as f64,
        dense as f64 / tq2 as f64,
    );

    // The nominal ratios are 16/5 and 16/3; the per-vector norms eat into both,
    // more so on a 64-wide head than on the 128-wide heads the paper uses.
    assert!(dense as f64 / tq4 as f64 > 3.0, "tq4 saved too little");
    assert!(dense as f64 / tq2 as f64 > 5.0, "tq2 saved too little");
    Ok(())
}

/// Greedy generation from a quantized cache has to stay on the rails for long
/// enough to matter — a cache that degrades as it fills would pass a
/// single-step logit check and still be useless.
#[test]
fn generation_stays_coherent_over_a_long_run() -> Result<()> {
    // k8v4: the allocation the sweep above picks out, not the paper's
    // symmetric one.
    let (mut model, tok, _gpu) = setup!(KvCacheQuant::new(8, 4, false)?);
    let prompt = tok.encode(
        "Count from one to twenty: one, two, three,",
        Some(false),
        false,
    );

    let mut session = model.new_session()?;
    let mut sampler = Sampler::new(SamplingParams::greedy());
    let mut generated = Vec::new();
    let mut logits: Vec<f32> = model.forward(&prompt, &mut session)?.to_vec();

    for _ in 0..120 {
        let next = sampler.sample(&logits, &generated);
        if tok.is_eog(next) {
            break;
        }
        generated.push(next);
        logits = model.forward(&[next], &mut session)?.to_vec();
    }

    let text = tok.decode(&generated, true);
    eprintln!("  {:?}", text.chars().take(120).collect::<String>());
    // Not asserting on the exact continuation — only that 120 steps of
    // quantized attention still produce the counting the prompt sets up.
    let hits = ["four", "five", "six", "seven"]
        .iter()
        .filter(|w| text.contains(*w))
        .count();
    assert!(hits >= 3, "generation lost the thread: {text:?}");
    Ok(())
}
