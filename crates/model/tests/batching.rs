//! Batched forward passes against the same sequences run alone.
//!
//! Batching is a scheduling decision, not a numerical one: a sequence must not
//! be able to tell who else was in its batch. These tests hold it to that.

use std::path::PathBuf;

use anyhow::Result;
use infero_model::{BatchItem, KvCacheQuant, Model, Sampler, SamplingParams};
use infero_tokenizer::Tokenizer;

const PROMPTS: &[&str] = &[
    "The capital of France is",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
    "人工智能是",
    "The three primary colors are red,",
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

fn load() -> Result<Option<(Model, Tokenizer)>> {
    let Some(path) = model_path() else {
        return Ok(None);
    };
    let gguf = infero_gguf::Gguf::open(&path)?;
    let tok = Tokenizer::from_gguf(&gguf)?;
    let model = Model::load_quantized(infero_cuda::Device::new(0)?, &gguf, 512, KvCacheQuant::F16)?;
    Ok(Some((model, tok)))
}

/// One test in this file at a time.
///
/// Each builds its own `Model`, and a decode step captures a CUDA graph.
/// Capture is invalidated by an allocation from any other thread on the same
/// context, so two of these running concurrently fail with
/// `CUDA_ERROR_STREAM_CAPTURE_INVALIDATED` — reliably under the default test
/// harness, never under `--test-threads=1`. Serializing here keeps `cargo test`
/// green without asking everyone to remember the flag, which is the difference
/// between a suite people trust and one they learn to ignore.
static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn gpu_lock() -> std::sync::MutexGuard<'static, ()> {
    GPU.lock().unwrap_or_else(|e| e.into_inner())
}

macro_rules! setup {
    () => {{
        // Held for the rest of the test: the guard's lifetime is the body's.
        let _gpu = gpu_lock();
        match load()? {
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

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
    let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    dot / (na * nb + 1e-30)
}

/// Prefill several prompts at once and compare each against a solo run.
#[test]
fn batched_prefill_matches_solo_prefill() -> Result<()> {
    let (mut model, tok, _gpu) = setup!();
    let vocab = model.config().vocab_size;
    let ids: Vec<Vec<u32>> = PROMPTS
        .iter()
        .map(|p| tok.encode(p, Some(false), false))
        .collect();

    let mut solo = Vec::new();
    for prompt in &ids {
        let mut session = model.new_session()?;
        solo.push(model.forward(prompt, &mut session)?.to_vec());
    }

    let mut pool = model.new_pool(2048, 8)?;
    let seqs: Vec<_> = (0..ids.len()).map(|_| pool.alloc().unwrap()).collect();
    let items: Vec<BatchItem<'_>> = seqs
        .iter()
        .zip(&ids)
        .map(|(&seq, p)| BatchItem::new(seq, p))
        .collect();
    let batched = model.forward_batch(&items, &mut pool)?.to_vec();

    for (i, want) in solo.iter().enumerate() {
        let got = &batched[i * vocab..(i + 1) * vocab];
        let cos = cosine(got, want);
        eprintln!(
            "  prompt {i}: argmax {} vs {}, cosine {cos:.6}",
            argmax(got),
            argmax(want)
        );
        assert_eq!(
            argmax(got),
            argmax(want),
            "prompt {i} predicted differently inside a batch"
        );
        // Not exact, and for a reason worth naming: a solo prefill wants logits
        // for one row and so takes the integer vocab projection, while a batch
        // of four takes the float one. The 8-bit activation is the whole gap.
        assert!(cos > 0.9995, "prompt {i}: cosine {cos:.6}");
    }
    Ok(())
}

/// Sequences at different lengths decoding together, against the same
/// sequences run alone.
///
/// This does not demand equality, and the reason is a measured trade-off rather
/// than a defect. A one-token step takes the integer mat-vec and a many-token
/// step takes the tensor-core GEMM, because at one token the mat-vec is 1.9x
/// faster on Q4_K and 3.2x faster on Q6_K — unifying them would cost that much
/// on single-request latency, which is the case this engine exists to serve.
/// The two kernels sum over `k` in different orders, so greedy decoding can
/// eventually pick the other side of a near-tie.
///
/// What is asserted instead: the logits stay within float noise of each other
/// at every step, and any divergence happens late. A real bug — a mis-indexed
/// KV slot, a scale on the wrong group — destroys the cosine immediately, which
/// is what this is here to catch. See `tensor_core_gemm_gives_the_same_answer_
/// at_any_batch_size` for the invariance that does hold exactly, and
/// `a_batch_does_not_leak_between_its_members` for the property a server needs.
#[test]
fn batched_decode_tracks_solo_decode() -> Result<()> {
    let (mut model, tok, _gpu) = setup!();
    let vocab = model.config().vocab_size;
    let ids: Vec<Vec<u32>> = PROMPTS
        .iter()
        .map(|p| tok.encode(p, Some(false), false))
        .collect();

    const STEPS: usize = 8;
    let mut solo = Vec::new();
    for prompt in &ids {
        let mut session = model.new_session()?;
        let mut sampler = Sampler::new(SamplingParams::greedy());
        let mut out = Vec::new();
        let mut logits: Vec<f32> = model.forward(prompt, &mut session)?.to_vec();
        for _ in 0..STEPS {
            let next = sampler.sample(&logits, &out);
            out.push(next);
            logits = model.forward(&[next], &mut session)?.to_vec();
        }
        solo.push(out);
    }

    let mut pool = model.new_pool(2048, 8)?;
    let seqs: Vec<_> = (0..ids.len()).map(|_| pool.alloc().unwrap()).collect();
    let items: Vec<BatchItem<'_>> = seqs
        .iter()
        .zip(&ids)
        .map(|(&seq, p)| BatchItem::new(seq, p))
        .collect();
    let logits = model.forward_batch(&items, &mut pool)?.to_vec();

    let mut samplers: Vec<_> = (0..ids.len())
        .map(|_| Sampler::new(SamplingParams::greedy()))
        .collect();
    let mut generated: Vec<Vec<u32>> = vec![Vec::new(); ids.len()];
    let mut next: Vec<u32> = (0..ids.len())
        .map(|i| samplers[i].sample(&logits[i * vocab..(i + 1) * vocab], &generated[i]))
        .collect();

    for _ in 0..STEPS {
        for (i, t) in next.iter().enumerate() {
            generated[i].push(*t);
        }
        let step: Vec<BatchItem<'_>> = seqs
            .iter()
            .zip(&next)
            .map(|(&seq, t)| BatchItem::new(seq, std::slice::from_ref(t)))
            .collect();
        let logits = model.forward_batch(&step, &mut pool)?.to_vec();
        for i in 0..ids.len() {
            next[i] = samplers[i].sample(&logits[i * vocab..(i + 1) * vocab], &generated[i]);
        }
    }

    for (i, want) in solo.iter().enumerate() {
        let agree = generated[i]
            .iter()
            .zip(want)
            .take_while(|(a, b)| a == b)
            .count();
        eprintln!(
            "  seq {i}: first {agree}/{STEPS} agree\n         batched {:?}\n         solo    {:?}",
            tok.decode(&generated[i], true),
            tok.decode(want, true)
        );
        assert!(
            agree >= 5,
            "sequence {i} diverged at step {agree}, too early to be a near-tie"
        );
    }
    Ok(())
}

/// The property a server actually needs: a request's logits must not depend on
/// which *other* requests share its batch.
///
/// Unlike the test above this is exact. The batch width is held fixed so the
/// same kernels run in both cases, and only the batchmates change — which is
/// what a scheduler varies from step to step. A KV slot indexed by the wrong
/// column, or attention reading past a sequence's own history, shows up here as
/// a hard failure rather than a drifting cosine.
#[test]
fn a_batch_does_not_leak_between_its_members() -> Result<()> {
    let (mut model, tok, _gpu) = setup!();
    let vocab = model.config().vocab_size;
    let ids: Vec<Vec<u32>> = PROMPTS
        .iter()
        .map(|p| tok.encode(p, Some(false), false))
        .collect();

    // The subject is PROMPTS[0]; its two runs differ only in the company it
    // keeps. Both batches are three wide.
    let logits_for = |model: &mut Model, others: [usize; 2]| -> Result<Vec<f32>> {
        let mut pool = model.new_pool(2048, 8)?;
        let subject = pool.alloc().unwrap();
        let mut items = vec![BatchItem::new(subject, &ids[0])];
        let extra: Vec<_> = others.iter().map(|_| pool.alloc().unwrap()).collect();
        for (slot, &o) in extra.iter().zip(&others) {
            items.push(BatchItem::new(*slot, &ids[o]));
        }
        let out = model.forward_batch(&items, &mut pool)?.to_vec();
        Ok(out[..vocab].to_vec())
    };

    let a = logits_for(&mut model, [1, 2])?;
    let b = logits_for(&mut model, [3, 1])?;
    let differing = a
        .iter()
        .zip(&b)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count();
    eprintln!(
        "  {:?}: {differing} of {vocab} logits changed with the batchmates",
        tok.decode(&ids[0], true)
    );
    assert_eq!(
        differing, 0,
        "the subject's logits depend on who shares its batch"
    );
    Ok(())
}

/// A sequence joining a batch that is already mid-flight must not be affected
/// by the ones already running, nor disturb them.
#[test]
fn a_sequence_can_join_a_running_batch() -> Result<()> {
    let (mut model, tok, _gpu) = setup!();
    let vocab = model.config().vocab_size;
    let long = tok.encode(PROMPTS[1], Some(false), false);
    let joiner = tok.encode(PROMPTS[0], Some(false), false);

    let mut solo_session = model.new_session()?;
    let solo_joiner: Vec<f32> = model.forward(&joiner, &mut solo_session)?.to_vec();

    let mut pool = model.new_pool(2048, 8)?;
    let a = pool.alloc().unwrap();
    let items = [BatchItem::new(a, &long)];
    model.forward_batch(&items, &mut pool)?;
    // A few decode steps for A before B arrives.
    let mut tok_a = 100u32;
    for _ in 0..3 {
        let items = [BatchItem::new(a, std::slice::from_ref(&tok_a))];
        let logits = model.forward_batch(&items, &mut pool)?;
        tok_a = argmax(&logits[..vocab]);
    }

    // B joins mid-flight, in the same batch as A's next decode step.
    let b = pool.alloc().unwrap();
    let items = [
        BatchItem::new(a, std::slice::from_ref(&tok_a)),
        BatchItem::new(b, &joiner),
    ];
    let logits = model.forward_batch(&items, &mut pool)?.to_vec();
    let got_joiner = &logits[vocab..2 * vocab];

    let cos = cosine(got_joiner, &solo_joiner);
    eprintln!(
        "  joiner: argmax {} vs {}, cosine {cos:.6}",
        argmax(got_joiner),
        argmax(&solo_joiner)
    );
    assert_eq!(argmax(got_joiner), argmax(&solo_joiner));
    assert!(cos > 0.9999, "cosine {cos:.6}");
    Ok(())
}

/// Freed slots must be reusable without leaking the previous tenant's history.
#[test]
fn a_reused_sequence_row_starts_clean() -> Result<()> {
    let (mut model, tok, _gpu) = setup!();
    let vocab = model.config().vocab_size;
    let ids = tok.encode(PROMPTS[0], Some(false), false);

    // Small pool, so the second sequence is forced onto the first's slots.
    let mut pool = model.new_pool(ids.len() + 4, 2)?;
    let a = pool.alloc().unwrap();
    let first = model
        .forward_batch(&[BatchItem::new(a, &ids)], &mut pool)?
        .to_vec();
    pool.free(a);
    assert_eq!(pool.free_slots(), ids.len() + 4);

    let b = pool.alloc().unwrap();
    let second = model
        .forward_batch(&[BatchItem::new(b, &ids)], &mut pool)?
        .to_vec();

    assert_eq!(argmax(&first[..vocab]), argmax(&second[..vocab]));
    let cos = cosine(&first[..vocab], &second[..vocab]);
    assert!(
        cos > 0.9999,
        "recycled slots changed the answer: cosine {cos:.6}"
    );
    Ok(())
}
