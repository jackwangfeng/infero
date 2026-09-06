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
use infero_model::{BatchItem, BatchItemKind, KvCacheQuant, Model, Sampler, SamplingParams};
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
        out.push(model.forward(&ids, BatchItemKind::Prefill, &mut session)?.to_vec());
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
    let batched: Vec<f32> = model.forward(&ids, BatchItemKind::Prefill, &mut session)?.to_vec();

    let mut session = model.new_session()?;
    let mut incremental = Vec::new();
    for &t in &ids {
        incremental = model.forward(&[t], BatchItemKind::Prefill, &mut session)?.to_vec();
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
    let mut logits: Vec<f32> = model.forward(&prompt, BatchItemKind::Prefill, &mut session)?.to_vec();

    for _ in 0..120 {
        let next = sampler.sample(&logits, &generated);
        if tok.is_eog(next) {
            break;
        }
        generated.push(next);
        logits = model.forward(&[next], BatchItemKind::Decode, &mut session)?.to_vec();
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

/// A long TurboQuant prefill run must produce the same answer whether or not
/// the dequantize-and-dispatch path serves it: `INFERO_TQ_PREFILL_DEQUANT=0`
/// keeps every run on `tq_attn_decode`, the default routes runs wider than
/// `TQ_DEQUANT_THRESHOLD` through `tq_dequant_kv` + `attn_prefill_ws4`, and
/// the two are the same attention computed by different kernels.
///
/// `tq4-mse`, not `tq4`, and that is the point rather than a convenience: the
/// long path is gated off for any quantization whose score estimator carries
/// the QJL sign-sketch term, because a per-element dequantized key cannot
/// carry that term. See `the_qjl_estimator_keeps_long_runs_off_the_dequant_path`
/// below and the gate's own comment in `forward_batch_rows`. `tq4-mse` is
/// `tq4` with exactly that stage switched off, so it is both a real supported
/// setting and the widest-coverage one the long path can serve today.
///
/// Requires `INFERO_ATTN_MMA=1` in the environment *before the test binary
/// starts* (`Kernels::prefill_attention` caches it in a `OnceLock`): without
/// it no tile kernel is eligible, the dispatch keeps every run on
/// `tq_attn_decode`, and this test would pass without exercising anything.
#[test]
fn long_prefill_dequant_dispatch_agrees_with_tq_attn_decode() -> Result<()> {
    let _gpu = gpu_lock();
    if std::env::var("INFERO_ATTN_MMA").as_deref() != Ok("1") {
        eprintln!("skipping: needs INFERO_ATTN_MMA=1 set before the test binary starts");
        return Ok(());
    }
    let Some((logits_off, logits_on)) = both_paths(KvCacheQuant::Tq4Mse)? else {
        return Ok(());
    };

    let cos = cosine(&logits_off, &logits_on);
    eprintln!("  dequant-dispatch vs tq_attn_decode logit cosine: {cos:.6}");
    assert_eq!(
        argmax(&logits_off),
        argmax(&logits_on),
        "the two kernel paths disagree on the next token"
    );
    // Not bit-identical, and never will be: `attn_prefill_ws4` stages K/V as
    // f16 and accumulates PV in f16 fragments, where `tq_attn_decode` reads
    // the codebook in f32 throughout. Twenty-four layers of that is the whole
    // budget below.
    assert!(
        cos > 0.99,
        "cosine {cos:.6} -- two kernel routes to the same attention should \
         differ only by their accumulation precision"
    );
    // ...and it must not be *exactly* 1.0 either, or the long path never ran
    // and this test proved nothing. That is precisely how it read before the
    // dispatch was wired: identical to the last bit, because both settings
    // took the same kernel.
    assert!(
        cos < 0.99999,
        "cosine {cos:.6} is indistinguishable from 1.0 -- the two settings \
         took the same kernel, so this test is not exercising the dequant \
         dispatch at all (is TQ_DEQUANT_THRESHOLD above this prompt's length, \
         or did the eligibility gate turn the long path off?)"
    );
    Ok(())
}

/// The QJL-enabled settings — `tq4` among them, and `tq4` is what this whole
/// path was built for — must keep taking `tq_attn_decode` for long runs too,
/// bit for bit, until `tq_dequant_kv` can carry the QJL term.
///
/// `tq_attn_decode` scores a key as `scale·⟨q_rot, cb[code]⟩ +
/// qjl_scale·(√(π/2)/d)·γ·⟨q_qjl, s⟩`; a dequantized key materializes only the
/// first term. Measured on this model at this prompt length, dropping the
/// second is not a rounding difference — the two estimators sit at cosine
/// ~0.29 from each other and choose different next tokens, and the QJL one is
/// much the closer to an f16 cache (0.675 against 0.330). So the long path is
/// gated off for them rather than being allowed to quietly serve a materially
/// worse answer. See the gate's own comment in `forward_batch_rows`, which
/// also derives the fix (`⟨Q·q, s⟩ = ⟨q, Qᵀ·s⟩`, so the term folds into a
/// per-element key after all — kernel work, not dispatch work).
#[test]
fn the_qjl_estimator_keeps_long_runs_off_the_dequant_path() -> Result<()> {
    let _gpu = gpu_lock();
    if std::env::var("INFERO_ATTN_MMA").as_deref() != Ok("1") {
        eprintln!("skipping: needs INFERO_ATTN_MMA=1 set before the test binary starts");
        return Ok(());
    }
    let Some((logits_off, logits_on)) = both_paths(KvCacheQuant::Tq4)? else {
        return Ok(());
    };
    assert_eq!(
        logits_off, logits_on,
        "a QJL-enabled cache took a different path with the flag on; the long \
         run must stay on tq_attn_decode until the dequantizer carries the QJL term"
    );
    Ok(())
}

/// One long prefill, run twice over a freshly loaded model: once with
/// `INFERO_TQ_PREFILL_DEQUANT=0` and once at the default. Returns
/// `(logits_off, logits_on)`, or `None` when the test model isn't present.
///
/// The flag is read once at load (`Model::from_parts`), so it has to be set
/// around `load_quantized` rather than around `forward`.
fn both_paths(quant: KvCacheQuant) -> Result<Option<(Vec<f32>, Vec<f32>)>> {
    let Some(path) = model_path() else {
        return Ok(None);
    };
    let gguf = Gguf::open(&path)?;
    let tok = Tokenizer::from_gguf(&gguf)?;

    // Long enough to clear even a generously large `TQ_DEQUANT_THRESHOLD`.
    let prompt = "The quick brown fox jumps over the lazy dog. ".repeat(40);
    let ids = tok.encode(&prompt, Some(false), false);
    assert!(
        ids.len() > 256,
        "prompt too short to exercise the long-run path: {} tokens",
        ids.len()
    );

    let off = {
        unsafe { std::env::set_var("INFERO_TQ_PREFILL_DEQUANT", "0") };
        let mut model = Model::load_quantized(Device::new(0)?, &gguf, 1024, quant)?;
        unsafe { std::env::remove_var("INFERO_TQ_PREFILL_DEQUANT") };
        assert!(!model.tq_prefill_dequant(), "=0 should switch the path off");
        assert!(
            model.batch_tokens() >= ids.len(),
            "this prompt would be split across {} passes; the test wants one long item",
            ids.len().div_ceil(model.batch_tokens())
        );
        let mut session = model.new_session()?;
        model
            .forward(&ids, BatchItemKind::Prefill, &mut session)?
            .to_vec()
    };

    let on = {
        let mut model = Model::load_quantized(Device::new(0)?, &gguf, 1024, quant)?;
        assert!(model.tq_prefill_dequant(), "dequant-dispatch should default on");
        let mut session = model.new_session()?;
        model
            .forward(&ids, BatchItemKind::Prefill, &mut session)?
            .to_vec()
    };
    Ok(Some((off, on)))
}

/// Load with `INFERO_TQ_PREFILL_DEQUANT` forced for the duration of the load,
/// which is when `Model::from_parts` resolves it. Removed again immediately:
/// nothing reads it after load, and leaving it set would leak into the next
/// test.
fn load_tq(quant: KvCacheQuant, dequant: bool) -> Result<Option<(Model, Tokenizer)>> {
    let Some(path) = model_path() else {
        return Ok(None);
    };
    let gguf = Gguf::open(&path)?;
    let tok = Tokenizer::from_gguf(&gguf)?;
    if !dequant {
        unsafe { std::env::set_var("INFERO_TQ_PREFILL_DEQUANT", "0") };
    }
    let model = Model::load_quantized(Device::new(0)?, &gguf, 1024, quant);
    unsafe { std::env::remove_var("INFERO_TQ_PREFILL_DEQUANT") };
    let model = model?;
    assert_eq!(
        model.tq_prefill_dequant(),
        dequant,
        "INFERO_TQ_PREFILL_DEQUANT did not resolve as expected -- the rest of \
         this test would be measuring nothing"
    );
    Ok(Some((model, tok)))
}

/// A real-text prompt of exactly `n` tokens, chosen by `seed` out of three
/// unrelated passages.
///
/// Real text, deliberately, not pseudo-random token ids: at four-bit keys on a
/// 0.5B model, random-token logits are near-degenerate, and every comparison
/// below then measures the quantizer's noise floor rather than the dispatch.
/// Measured, on the very comparison `long_prefill_dequant_dispatch_agrees_with_
/// tq_attn_decode` makes: 0.996 on real text, 0.894 on token soup of the same
/// length.
fn prompt_tokens(tok: &Tokenizer, seed: usize, n: usize) -> Vec<u32> {
    const PASSAGES: [&str; 3] = [
        "The quick brown fox jumps over the lazy dog. ",
        "In a distant valley the river turned slowly toward the sea. ",
        "Every morning she walked to the market and bought fresh bread. ",
    ];
    let ids = tok.encode(&PASSAGES[seed % 3].repeat(200), Some(false), false);
    assert!(ids.len() >= n, "passage {seed} is too short for {n} tokens");
    ids[..n].to_vec()
}

/// The case the single-shot test above cannot reach: a long prefill chunk that
/// *continues* a sequence, so its `kv_len` (the whole cached span) is much
/// larger than its own token count.
///
/// This is where the long path's two most delicate numbers live, and both fail
/// silently rather than loudly if they are wrong:
///
///   - `kv_len` must be `KvPool::len(seq)` — the sequence's post-extension
///     length, history included — not this chunk's own width, or the chunk
///     attends only the keys it brought with it and forgets everything before.
///   - the synthetic `BatchLayout` handed to `attn_prefill_ws4` maps compact
///     scratch row `i` to logical position `i`, so its `slot_table` is the
///     identity — but its `positions` must stay this chunk's *absolute*
///     positions, because that is what the tile kernel's causal mask compares
///     each key index against (`abs_key <= row_last`, where `row_last` is read
///     out of `positions`). Making `positions` the identity too is an easy and
///     natural-looking mistake, since every other array in that layout is.
///
/// Both halves are MEASURED discriminators here, not assumed ones. Replacing
/// `positions` with the identity moves this comparison from 0.982 to 0.670
/// while leaving the single-chunk test above at its unchanged 0.980 — which is
/// exactly why this test exists as well as that one.
#[test]
fn a_continuation_chunk_attends_its_own_history_through_the_dequant_path() -> Result<()> {
    let _gpu = gpu_lock();
    if std::env::var("INFERO_ATTN_MMA").as_deref() != Ok("1") {
        eprintln!("skipping: needs INFERO_ATTN_MMA=1 set before the test binary starts");
        return Ok(());
    }
    // Two chunks, each comfortably over `TQ_DEQUANT_THRESHOLD`, so the *second*
    // one is a long run whose kv extent (400) is twice its own width (200).
    const CHUNK: usize = 200;
    let mut out = Vec::new();
    for dequant in [false, true] {
        let Some((mut model, tok)) = load_tq(KvCacheQuant::Tq4Mse, dequant)? else {
            return Ok(());
        };
        let vocab = model.config().vocab_size;
        let prompt = prompt_tokens(&tok, 1, 2 * CHUNK);
        let mut pool = model.new_pool(1024, 2)?;
        let seq = pool.alloc().expect("fresh pool had no rows");
        let mut logits = Vec::new();
        for chunk in prompt.chunks(CHUNK) {
            let item = BatchItem::new(seq, chunk, BatchItemKind::Prefill);
            logits =
                model.forward_batch(std::slice::from_ref(&item), &mut pool)?[..vocab].to_vec();
        }
        assert_eq!(pool.len(seq), 2 * CHUNK);
        out.push(logits);
    }

    let cos = cosine(&out[0], &out[1]);
    eprintln!("  continuation chunk, dequant off vs on: cosine {cos:.6}");
    assert_eq!(
        argmax(&out[0]),
        argmax(&out[1]),
        "a continuation chunk predicted differently through the dequant path"
    );
    // Looser than the single-shot test's 0.99 because two long runs stack their
    // f16-vs-f32 accumulation gap here, and the second one carries the first's
    // divergence in its cache as well. Measured: 0.982.
    assert!(cos > 0.96, "cosine {cos:.6}");
    assert!(
        cos < 0.99999,
        "cosine {cos:.6} is indistinguishable from 1.0 -- the long path never ran"
    );
    Ok(())
}

/// An interleaved batch: a decode item, then the long prefill under test, then
/// a short prefill chunk. That is deliberately not the easy "all short, then
/// all long" shape — it splits the short/decode items into two separate
/// contiguous runs on either side of the long one, which is exactly the
/// partition a naive two-flat-lists split gets wrong (it would hand the
/// trailing short item the leading run's base offset).
///
/// Every item's logits are compared against *the same batch* run with
/// `INFERO_TQ_PREFILL_DEQUANT=0`, item by item, rather than against a solo run.
/// That holds the batch shape fixed and varies only the dispatch, so a wrong
/// `run.base` shows up as one item reading another's query rows. Comparing
/// against a *solo* run would not work: the TurboQuant path's batch isolation
/// is only ~0.977 on this model *before* this change, so such a comparison has
/// no headroom left to detect anything.
///
/// On the tolerance. The measured numbers here are 0.983 / 0.970 / 0.997, and
/// they are not the long path's doing — only item 1 is long. Splitting a batch
/// into per-run calls changes each call's `n_tokens`, and `tq_attn_decode`
/// derives its split-K chunk count from that (`decode_chunks`), so the same
/// token's keys get summed in a different order. That is the same pre-existing
/// sensitivity that makes a solo run and a batched run differ by ~0.977 on this
/// path already. 0.95 is set below it and well above what a real bug does:
/// giving every short run the *first* run's base — precisely the mistake a
/// two-flat-lists partition makes on this interleaved shape — drops item 0 to
/// 0.694, which this catches with room to spare.
#[test]
fn an_interleaved_batch_dispatches_every_item_to_its_own_rows() -> Result<()> {
    let _gpu = gpu_lock();
    if std::env::var("INFERO_ATTN_MMA").as_deref() != Ok("1") {
        eprintln!("skipping: needs INFERO_ATTN_MMA=1 set before the test binary starts");
        return Ok(());
    }
    let mut runs: Vec<Vec<Vec<f32>>> = Vec::new();
    for dequant in [false, true] {
        let Some((mut model, tok)) = load_tq(KvCacheQuant::Tq4Mse, dequant)? else {
            return Ok(());
        };
        let vocab = model.config().vocab_size;
        let long = prompt_tokens(&tok, 0, 200);
        let short = prompt_tokens(&tok, 1, 40);
        let other = prompt_tokens(&tok, 2, 300);

        let mut pool = model.new_pool(1024, 4)?;
        let decoder = pool.alloc().expect("fresh pool had no rows");
        let target = pool.alloc().expect("fresh pool had no second row");
        let tail = pool.alloc().expect("fresh pool had no third row");
        let prime = BatchItem::without_logits(decoder, &other, BatchItemKind::Prefill);
        model.forward_batch_device(std::slice::from_ref(&prime), &mut pool)?;
        assert_eq!(pool.len(decoder), other.len());
        let next = [other[0]];
        let items = [
            BatchItem::new(decoder, &next, BatchItemKind::Decode),
            BatchItem::new(target, &long, BatchItemKind::Prefill),
            BatchItem::new(tail, &short, BatchItemKind::Prefill),
        ];
        let out = model.forward_batch(&items, &mut pool)?;
        // `logit_rows` is built in item order, one row per item here.
        runs.push((0..items.len()).map(|i| out[i * vocab..(i + 1) * vocab].to_vec()).collect());
    }

    let mut any_moved = false;
    for (i, (off, on)) in runs[0].iter().zip(&runs[1]).enumerate() {
        let cos = cosine(off, on);
        eprintln!("  item {i}: dequant off vs on cosine {cos:.9}");
        assert_eq!(
            argmax(off),
            argmax(on),
            "item {i} predicted differently once the batch took the dequant dispatch"
        );
        assert!(cos > 0.95, "item {i} cosine {cos:.9}");
        any_moved |= cos < 0.99999;
    }
    // Item 1 is the only long one, so it is the only one that may move at all;
    // if nothing moved, the long path never ran and this proved nothing.
    assert!(
        any_moved,
        "no item's logits moved -- the dequant dispatch never took the long path"
    );
    Ok(())
}
