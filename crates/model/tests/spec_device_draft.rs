//! `Model::draft_with_head_device_batch`: the device-resident draft loop's
//! own orchestration, on a real (if small) model and MTP head.
//!
//! `Kernels::gumbel_sample_rows`'s own statistical correctness (empirical
//! draws match the closed-form softmax) is already established by
//! `gpu_sampling.rs`'s `gumbel_sample_matches_the_softmax_it_draws_from` --
//! this file does not re-derive that. What it checks is the orchestration
//! this function adds around that kernel: determinism end to end (not just
//! the kernel's own), and that a real multi-step, multi-sequence round
//! actually runs, engages the batched vocab-projection path, and produces
//! structurally sound `Drafted` rows -- the same real small-model setup
//! `spec.rs`'s `batched_draft_matches_the_sequential_calls_it_replaces`
//! already uses and validates against.

use anyhow::{Context, Result};
use infero_model::mtp::{BatchDraftItem, HeadDims, MtpHead};
use infero_model::spec::DraftFeed;
use infero_model::weights::{AttnWeights, DenseFfn, Layer, Matrix, MtpWeights};
use infero_model::{BatchItem, BatchItemKind, KvCacheQuant, KvPool, Model, SeqId};
use infero_tokenizer::Tokenizer;

const PROMPT: &str = "The quick brown fox jumps over the lazy dog. \
Speculative decoding drafts several tokens ahead and verifies them in one pass.";

fn gpu_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn gguf_path(name: &str) -> Option<std::path::PathBuf> {
    let p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models").join(name);
    if !p.exists() {
        eprintln!("skipping: {} not downloaded", p.display());
        return None;
    }
    Some(p)
}

fn load(name: &str, max_logit_rows: usize) -> Result<Option<(Model, Tokenizer)>> {
    let Some(path) = gguf_path(name) else {
        return Ok(None);
    };
    let gguf = infero_gguf::Gguf::open(&path)?;
    let tok = Tokenizer::from_gguf(&gguf)?;
    let model = Model::load_full(
        infero_cuda::Device::new(0)?,
        &gguf,
        512,
        KvCacheQuant::F16,
        usize::MAX,
        max_logit_rows,
    )?;
    Ok(Some((model, tok)))
}

/// Mirrors `spec.rs`'s own `synthetic_head_branched` (a real, minimal,
/// correctly-dimensioned `MtpHead` with synthetic weights, no dependence on
/// the checkpoint shipping a real head) -- duplicated rather than shared
/// because integration test binaries cannot import each other's helpers.
fn synthetic_head_branched(
    dev: &infero_cuda::Device,
    cfg: &infero_model::Config,
    max_rows: usize,
    max_seq: usize,
    branches: usize,
) -> Result<MtpHead> {
    let dims = HeadDims::from_config(cfg);
    let (d, d_attn, d_kv) = (dims.d_model, dims.d_attn(), dims.d_kv());
    let seed = std::cell::Cell::new(0x9e37_79b9u32);
    let next = move || {
        seed.set(seed.get().wrapping_mul(1_664_525).wrapping_add(1_013_904_223));
        0.02 * (((seed.get() >> 8) as f32 / 8_388_608.0) - 1.0)
    };
    let m = |k: usize, n: usize| -> Result<Matrix> {
        let v: Vec<half::f16> = (0..k * n).map(|_| half::f16::from_f32(next())).collect();
        Matrix::upload_f16(dev, &v, k, n)
    };
    let vec1 = |n: usize| -> Result<infero_model::weights::Vector> {
        let v: Vec<f32> = (0..n).map(|_| 1.0 + next()).collect();
        Ok(dev.stream().clone_htod(&v)?)
    };
    let w = MtpWeights {
        fc: m(2 * d, d)?,
        pre_fc_norm_embedding: vec1(d)?,
        pre_fc_norm_hidden: vec1(d)?,
        norm: vec1(d)?,
        layer: Layer {
            attn_norm: vec1(d)?,
            attn: Some(AttnWeights {
                wq: m(d, 2 * d_attn)?,
                wk: m(d, d_kv)?,
                wv: m(d, d_kv)?,
                wo: m(d_attn, d)?,
                bq: None,
                bk: None,
                bv: None,
                bo: None,
                q_norm: Some(vec1(dims.d_head)?),
                k_norm: Some(vec1(dims.d_head)?),
                w_qkv: None,
                w_kv: None,
                output_gate: true,
            }),
            gdn: None,
            ffn_norm: vec1(d)?,
            dense: Some(DenseFfn {
                w_gate: m(d, dims.d_ff)?,
                w_up: m(d, dims.d_ff)?,
                w_down: m(dims.d_ff, d)?,
                w_gate_up: None,
            }),
            moe: None,
            blob: None,
        },
        device_bytes: 0,
    };
    let kern = infero_kernels::Kernels::new(dev.clone());
    MtpHead::new(dev, &kern, w, dims, max_rows, max_seq, branches)
}

fn prime(model: &mut Model, pool: &mut KvPool, seq: SeqId, prompt: &[u32]) -> Result<u32> {
    let item = BatchItem::new(seq, prompt, BatchItemKind::Prefill);
    model.forward_batch_device(std::slice::from_ref(&item), pool)?;
    let logits = model.logits_host()?;
    let mut best = 0usize;
    for (i, x) in logits.iter().enumerate() {
        if *x > logits[best] {
            best = i;
        }
    }
    Ok(best as u32)
}

fn sp(seed: u64) -> infero_model::SamplingParams {
    infero_model::SamplingParams {
        temperature: 0.8,
        top_p: 0.95,
        top_k: 32,
        seed: Some(seed),
        ..Default::default()
    }
}

#[test]
fn a_real_round_engages_and_is_structurally_sound() -> Result<()> {
    let _gpu = gpu_lock();
    const K: usize = 3;
    let Some((mut model, tok)) = load("qwen2.5-0.5b-instruct-q8_0.gguf", K + 1)? else {
        return Ok(());
    };
    let prompt = tok.encode(PROMPT, Some(false), false);
    let cfg = model.config().clone();
    let max_tokens = 2 * prompt.len().max(K + 1);
    let mut head = synthetic_head_branched(model.device(), &cfg, max_tokens, 4 * model.max_seq(), 4)?;
    head.fork(0, model.max_seq())?;
    model.install_mtp_head(head)?;

    let mut pool = model.new_pool(512, 4)?;
    model.enable_speculation(K, &pool)?;

    let seq0 = pool.alloc().context("no kv slot")?;
    let pending0 = prime(&mut model, &mut pool, seq0, &prompt)?;
    let seq1 = pool.alloc().context("no kv slot")?;
    let pending1 = prime(&mut model, &mut pool, seq1, &prompt)?;

    let feed0 = DraftFeed::after_prefill(&prompt, pending0);
    let feed1 = DraftFeed::after_prefill(&prompt, pending1);
    let items = [
        BatchDraftItem { branch: 0, feed: &feed0, history: &prompt },
        BatchDraftItem { branch: 1, feed: &feed1, history: &prompt },
    ];
    let mut s0 = infero_model::Sampler::new(sp(111));
    let mut s1 = infero_model::Sampler::new(sp(222));
    let mut samplers: Vec<&mut infero_model::Sampler> = vec![&mut s0, &mut s1];
    let drafts = model
        .draft_with_head_device_batch(K, &items, &mut samplers)?
        .context("logits_rows_batch_device declined at this shape -- unexpected for a real Q8_0 GGUF head")?;

    assert_eq!(drafts.len(), 2, "one drafted sequence a item");
    for (i, seq_draft) in drafts.iter().enumerate() {
        assert_eq!(seq_draft.len(), K, "sequence {i}: drafted a different number of tokens than k={K}");
        for (step, d) in seq_draft.iter().enumerate() {
            assert!(
                d.q.iter().any(|(t, w)| *t == d.token && *w > 0.0),
                "sequence {i} step {step}: drafted {}, which carries no weight in its own q",
                d.token
            );
            let total: f32 = d.q.iter().map(|(_, w)| w).sum();
            assert!(
                (total - 1.0).abs() < 1e-3,
                "sequence {i} step {step}: q sums to {total}, not 1.0"
            );
        }
    }
    Ok(())
}

#[test]
fn repeated_rounds_with_the_same_seed_are_bit_identical() -> Result<()> {
    let _gpu = gpu_lock();
    const K: usize = 2;
    let Some((mut model, tok)) = load("qwen2.5-0.5b-instruct-q8_0.gguf", K + 1)? else {
        return Ok(());
    };
    let prompt = tok.encode(PROMPT, Some(false), false);
    let cfg = model.config().clone();
    let max_tokens = 2 * prompt.len().max(K + 1);
    let mut head = synthetic_head_branched(model.device(), &cfg, max_tokens, 4 * model.max_seq(), 2)?;
    head.fork(0, model.max_seq())?;
    model.install_mtp_head(head)?;

    let mut pool = model.new_pool(512, 2)?;
    model.enable_speculation(K, &pool)?;

    // Reprime from scratch each rep on a fresh sequence, so both reps really
    // start from the identical real hidden state -- a shared mutable
    // `mtp_hidden`/KV cache would make "repeat the call" not actually repeat
    // the same inputs.
    let mut run = || -> Result<Vec<Vec<infero_model::mtp::Drafted>>> {
        let seq = pool.alloc().context("no kv slot")?;
        let pending = prime(&mut model, &mut pool, seq, &prompt)?;
        let feed = DraftFeed::after_prefill(&prompt, pending);
        let items = [BatchDraftItem { branch: 0, feed: &feed, history: &prompt }];
        let mut s = infero_model::Sampler::new(sp(4242));
        let mut samplers: Vec<&mut infero_model::Sampler> = vec![&mut s];
        let drafts = model
            .draft_with_head_device_batch(K, &items, &mut samplers)?
            .context("declined at this shape")?;
        pool.free(seq);
        Ok(drafts)
    };
    let a = run()?;
    let b = run()?;
    assert_eq!(a.len(), b.len());
    for (sa, sb) in a.iter().zip(&b) {
        assert_eq!(sa.len(), sb.len());
        for (da, db) in sa.iter().zip(sb) {
            assert_eq!(da.token, db.token, "same seed must draft the same token");
            assert_eq!(da.q, db.q, "same seed must reproduce the same q exactly");
        }
    }
    Ok(())
}
