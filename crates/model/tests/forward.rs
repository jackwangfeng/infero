//! The forward pass against Hugging Face's logits.
//!
//! Fluent output is not proof: a wrong RoPE base, a swapped gate/up projection
//! or an off-by-one causal mask all still produce readable English. These tests
//! compare the actual distribution against `transformers` running the same
//! checkpoint in f32, using fixtures from `scripts/make_logits_fixtures.py`.

use std::path::PathBuf;

use anyhow::Result;
use serde::Deserialize;
use tuili_cuda::Device;
use tuili_gguf::Gguf;
use tuili_model::{Model, Sampler, SamplingParams};
use tuili_tokenizer::Tokenizer;

#[derive(Deserialize)]
struct Fixtures {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    prompt: String,
    ids: Vec<u32>,
    top_ids: Vec<u32>,
    argmax: u32,
    argmax_piece: String,
    mean: f32,
    std: f32,
}

fn workspace() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// F16 by default: comparing a quantized build against f32 reference logits
/// would measure the quantizer, not the engine.
fn model_path() -> Option<PathBuf> {
    let p = std::env::var("TUILI_TEST_GGUF")
        .map(PathBuf::from)
        .unwrap_or_else(|_| workspace().join("models/qwen2.5-0.5b-instruct-fp16.gguf"));
    if !p.exists() {
        eprintln!("skipping: {} not downloaded", p.display());
        return None;
    }
    Some(p)
}

fn fixtures() -> Fixtures {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/qwen2.5-0.5b-instruct-logits.json");
    let raw = std::fs::read_to_string(&path).expect("reading logit fixtures");
    serde_json::from_str(&raw).expect("parsing logit fixtures")
}

fn load() -> Result<Option<(Model, Tokenizer)>> {
    let Some(path) = model_path() else {
        return Ok(None);
    };
    let gguf = Gguf::open(&path)?;
    let tokenizer = Tokenizer::from_gguf(&gguf)?;
    let model = Model::load(Device::new(0)?, &gguf, 2048)?;
    Ok(Some((model, tokenizer)))
}

/// One test in this file at a time; see the note in `batching.rs`. A decode
/// step captures a CUDA graph, and capture dies if another thread allocates on
/// the same context.
static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn gpu_lock() -> std::sync::MutexGuard<'static, ()> {
    GPU.lock().unwrap_or_else(|e| e.into_inner())
}

macro_rules! setup {
    () => {
        match load()? {
            Some(v) => v,
            None => return Ok(()),
        }
    };
}

#[test]
fn logits_match_huggingface() -> Result<()> {
    let _gpu = gpu_lock();
    let (mut model, _tok) = setup!();
    let fx = fixtures();

    // The mean logit is deliberately not asserted: our lm_head accumulates f32
    // over f16 weights where the reference is f32 throughout, which shifts the
    // whole vector by a small constant. Softmax is invariant to a constant, so
    // argmax, the top-10 set and the spread are what actually have to hold.
    let vocab_size = model.config().vocab_size;
    let mut failures = Vec::new();
    for case in &fx.cases {
        let mut session = model.new_session()?;
        let logits: Vec<f32> = model.forward(&case.ids, &mut session)?.to_vec();
        assert_eq!(logits.len(), vocab_size);

        let mean = logits.iter().sum::<f32>() / logits.len() as f32;
        let var =
            logits.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / (logits.len() - 1) as f32;
        let std = var.sqrt();
        let ours = top_k(&logits, 10);
        let theirs: std::collections::HashSet<u32> =
            case.top_ids.iter().take(10).copied().collect();
        let overlap = ours.iter().filter(|t| theirs.contains(t)).count();

        eprintln!(
            "  {:<46} argmax {:>6} (want {:>6})  mean {:>8.4} (want {:>8.4})  std {:>7.4} (want {:>7.4})  top10 {overlap}/10",
            format!("{:?}", &case.prompt[..case.prompt.len().min(40)]),
            argmax(&logits),
            case.argmax,
            mean,
            case.mean,
            std,
            case.std,
        );

        if argmax(&logits) != case.argmax {
            failures.push(format!(
                "{:?}: predicted {}, reference {} ({:?})",
                case.prompt,
                argmax(&logits),
                case.argmax,
                case.argmax_piece
            ));
        }
        // Top-10 as a set. Exact ordering deep in the tail is at the mercy of
        // f16 weights and fast-math intrinsics; membership is not.
        if overlap < 9 {
            failures.push(format!(
                "{:?}: only {overlap}/10 of the top tokens agree",
                case.prompt
            ));
        }
        // The spread catches a global scale error that argmax would hide.
        if (std - case.std).abs() / case.std > 0.02 {
            failures.push(format!(
                "{:?}: logit std {std} vs reference {}",
                case.prompt, case.std
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    Ok(())
}

/// Feeding a prompt one token at a time must land in the same state as feeding
/// it all at once — the property the KV cache exists to provide.
#[test]
fn incremental_decode_equals_batch_prefill() -> Result<()> {
    let _gpu = gpu_lock();
    let (mut model, _tok) = setup!();
    let fx = fixtures();
    let ids = &fx.cases[0].ids;

    let mut session = model.new_session()?;
    let batched: Vec<f32> = model.forward(ids, &mut session)?.to_vec();

    let mut session = model.new_session()?;
    let mut incremental = Vec::new();
    for &t in ids {
        incremental = model.forward(&[t], &mut session)?.to_vec();
    }

    assert_eq!(argmax(&batched), argmax(&incremental));
    let worst = batched
        .iter()
        .zip(&incremental)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    // The two paths take different kernels — cuBLAS for the batch, the
    // mat-vec for single tokens — so they agree to f16 precision, not exactly.
    assert!(worst < 0.05, "largest logit difference {worst}");
    Ok(())
}

/// Prompts longer than one prefill chunk must give the same answer as the
/// same prompt processed in a single chunk would.
#[test]
fn chunked_prefill_is_seamless() -> Result<()> {
    let _gpu = gpu_lock();
    let (mut model, tok) = setup!();

    let long = "The quick brown fox jumps over the lazy dog. ".repeat(40);
    let ids = tok.encode(&long, Some(false), false);
    assert!(
        ids.len() > tuili_model::PREFILL_CHUNK,
        "test needs a prompt longer than one chunk, got {}",
        ids.len()
    );

    let mut session = model.new_session()?;
    let all_at_once: Vec<f32> = model.forward(&ids, &mut session)?.to_vec();
    assert_eq!(session.len(), ids.len());

    // Same tokens, but handed over in two calls so the split lands elsewhere.
    let mut session = model.new_session()?;
    let split = ids.len() / 3;
    model.forward(&ids[..split], &mut session)?;
    let in_pieces: Vec<f32> = model.forward(&ids[split..], &mut session)?.to_vec();

    assert_eq!(argmax(&all_at_once), argmax(&in_pieces));
    let worst = all_at_once
        .iter()
        .zip(&in_pieces)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(worst < 0.05, "largest logit difference {worst}");
    Ok(())
}

#[test]
fn greedy_generation_is_deterministic() -> Result<()> {
    let _gpu = gpu_lock();
    let (mut model, tok) = setup!();
    let prompt = tok.encode("The capital of France is", Some(false), false);

    let mut run = || -> Result<Vec<u32>> {
        let mut session = model.new_session()?;
        let mut sampler = Sampler::new(SamplingParams::greedy());
        let mut out = Vec::new();
        let mut logits: Vec<f32> = model.forward(&prompt, &mut session)?.to_vec();
        for _ in 0..8 {
            let next = sampler.sample(&logits, &out);
            out.push(next);
            logits = model.forward(&[next], &mut session)?.to_vec();
        }
        Ok(out)
    };

    let a = run()?;
    let b = run()?;
    assert_eq!(a, b, "greedy decoding drifted between runs");
    assert!(
        tok.decode(&a, true).contains("Paris"),
        "expected Paris, got {:?}",
        tok.decode(&a, true)
    );
    Ok(())
}

#[test]
fn context_overflow_is_an_error_not_a_crash() -> Result<()> {
    let _gpu = gpu_lock();
    let Some(path) = model_path() else {
        return Ok(());
    };
    let gguf = Gguf::open(&path)?;
    let mut model = Model::load(Device::new(0)?, &gguf, 64)?;
    let mut session = model.new_session()?;

    let tokens: Vec<u32> = vec![100; 65];
    let err = model
        .forward(&tokens, &mut session)
        .unwrap_err()
        .to_string();
    assert!(err.contains("context overflow"), "{err}");
    // The model must still be usable afterwards.
    assert!(model.forward(&[100, 200], &mut session).is_ok());
    Ok(())
}

fn argmax(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .fold((0u32, f32::NEG_INFINITY), |best, (i, &x)| {
            if x > best.1 { (i as u32, x) } else { best }
        })
        .0
}

fn top_k(v: &[f32], k: usize) -> Vec<u32> {
    let mut idx: Vec<u32> = (0..v.len() as u32).collect();
    idx.sort_unstable_by(|&a, &b| v[b as usize].total_cmp(&v[a as usize]));
    idx.truncate(k);
    idx
}

/// The integer mat-vec against the float one, end to end.
///
/// Per matmul the two agree to a cosine of 0.999994, but a decode step chains
/// seven of them across every layer, so the question worth asking is what the
/// logits look like after that. `TUILI_NO_MMVQ` forces the float path with
/// everything else held fixed, which makes this a clean A/B.
#[test]
fn integer_decode_agrees_with_float_decode() -> Result<()> {
    let _gpu = gpu_lock();
    let Some(path) = model_path_quantized() else {
        return Ok(());
    };
    let gguf = Gguf::open(&path)?;
    let tok = Tokenizer::from_gguf(&gguf)?;
    let ids = tok.encode("The capital of France is", Some(false), false);

    let run = |float_path: bool| -> Result<Vec<f32>> {
        // Safety: single-threaded here, and read once at load.
        unsafe {
            if float_path {
                std::env::set_var("TUILI_NO_MMVQ", "1");
            } else {
                std::env::remove_var("TUILI_NO_MMVQ");
            }
        }
        let mut model = Model::load(Device::new(0)?, &gguf, 512)?;
        let mut session = model.new_session()?;
        model.forward(&ids, &mut session)?;
        // A decode step, which is the path under test.
        Ok(model.forward(&[12095], &mut session)?.to_vec())
    };

    let float = run(true)?;
    let integer = run(false)?;

    assert_eq!(argmax(&float), argmax(&integer), "the prediction changed");
    let dot: f64 = float
        .iter()
        .zip(&integer)
        .map(|(a, b)| *a as f64 * *b as f64)
        .sum();
    let na: f64 = float
        .iter()
        .map(|v| (*v as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    let nb: f64 = integer
        .iter()
        .map(|v| (*v as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    let cos = dot / (na * nb);
    eprintln!("  float vs integer decode: cosine {cos:.6}");
    assert!(cos > 0.999, "cosine {cos:.6}");
    Ok(())
}

/// A quantized build, so the integer path is actually reachable.
fn model_path_quantized() -> Option<PathBuf> {
    let p = workspace().join("models/qwen2.5-0.5b-instruct-q8_0.gguf");
    p.exists().then_some(p)
}
