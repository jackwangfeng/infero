//! End-to-end proof that turning M-RoPE on does not change a text-only
//! request's output by even one bit.
//!
//! `crates/kernels/tests/mrope.rs` already proves the kernel arithmetic is
//! bit-identical when a token's three axes are equal -- that is the rope
//! *kernel's* claim. This is the claim one level up: does `forward_batch_rows`
//! actually feed it `[T,H,W] = [p,p,p]` for a plain text request, end to end,
//! through the real `BatchItem`/`Acts`/`Weights::mrope_axis` plumbing, on both
//! a single prefill and a run of decode steps (where `mrope_delta` rather than
//! a precomputed array is what drives the position). Two models, identical
//! weights, differing only in `cfg.mrope_section` being `Some`/`None`, must
//! answer a shared prompt with bit-identical logits at every step -- not
//! "close", since this claims to be the *same* arithmetic, not an
//! approximation of it.

use anyhow::Result;
use half::f16;
use infero_cuda::Device;
use infero_model::qwen35_vision::interleaved_mrope_axis;
use infero_model::weights::{AttnWeights, DenseFfn, Layer, Matrix, Weights};
use infero_model::{BatchItem, Config, KvCacheQuant, KvPool, Model, SeqId};

const PROMPT: &[u32] = &[3, 17, 41, 5, 200, 61, 7, 12, 99];
const DECODE_STEPS: usize = 6;

static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());
fn gpu_lock() -> std::sync::MutexGuard<'static, ()> {
    GPU.lock().unwrap_or_else(|e| e.into_inner())
}

/// A tiny Qwen3.5-shaped, attention-only config (no GDN blocks -- the point
/// here is M-RoPE, not the recurrence, and `tests/spec_gdn.rs` already covers
/// that combination separately). `head_dim: 16` at `partial_rotary_factor:
/// 0.5` rotates 8 of 16, so `rotary_dim / 2 = 4` -- small enough to hand-pick
/// a real, non-degenerate `mrope_section` (`[2, 1, 1]`) that still exercises
/// all three axes.
fn config(mrope: bool) -> Result<Config> {
    let mut rope_parameters = serde_json::json!({
        "rope_theta": 500_000.0,
        "partial_rotary_factor": 0.5,
    });
    if mrope {
        rope_parameters["mrope_interleaved"] = serde_json::json!(true);
        rope_parameters["mrope_section"] = serde_json::json!([2, 1, 1]);
    }
    let json = serde_json::json!({
        "model_type": "qwen3_5",
        "tie_word_embeddings": false,
        "text_config": {
            "num_hidden_layers": 3,
            "hidden_size": 64,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "intermediate_size": 128,
            "vocab_size": 256,
            "max_position_embeddings": 512,
            "rms_norm_eps": 1e-6,
            "rope_parameters": rope_parameters,
        },
    });
    Config::from_hf(&json, "synthetic-mrope-check")
}

/// A deterministic fill, in a `Cell` so the same sequence of values can be
/// drawn twice -- once per model -- and land on identical weights.
struct Rng(std::cell::Cell<u32>);
impl Rng {
    fn next(&self) -> f32 {
        self.0
            .set(self.0.get().wrapping_mul(1_664_525).wrapping_add(1_013_904_223));
        ((self.0.get() >> 8) as f32 / 8_388_608.0) - 1.0
    }
}

/// Build a pure-attention model at `cfg`'s shape with arbitrary weights, and
/// with `Weights::mrope_axis` set from `interleaved_mrope_axis` when `cfg` has
/// a section, all zeros otherwise -- exactly what `weights::mrope_axis_table`
/// (private to that module) computes for a real loader, reproduced here since
/// this builder does not go through a loader.
///
/// Called with a *fresh* `Rng` seeded identically for both models under test,
/// so this is the one place their weights could diverge and does not.
fn synthetic_model(dev: &Device, cfg: &Config, rng: &Rng) -> Result<Model> {
    let (d, d_ff, vocab) = (cfg.d_model, cfg.d_ff, cfg.vocab_size);
    let (da, dkv) = (cfg.d_attn(), cfg.d_kv());
    let m = |k: usize, n: usize, scale: f32| -> Result<Matrix> {
        let v: Vec<f16> = (0..k * n).map(|_| f16::from_f32(scale * rng.next())).collect();
        Matrix::upload_f16(dev, &v, k, n)
    };
    let vec_at = |n: usize, centre: f32, spread: f32| -> Result<infero_model::weights::Vector> {
        let v: Vec<f32> = (0..n).map(|_| centre + spread * rng.next()).collect();
        Ok(dev.stream().clone_htod(&v)?)
    };

    let mut layers = Vec::with_capacity(cfg.n_layers);
    for _ in 0..cfg.n_layers {
        let attn = AttnWeights {
            wq: m(d, da, 0.2)?,
            wk: m(d, dkv, 0.2)?,
            wv: m(d, dkv, 0.2)?,
            wo: m(da, d, 0.2)?,
            bq: None,
            bk: None,
            bv: None,
            bo: None,
            q_norm: Some(vec_at(cfg.d_head, 1.0, 0.1)?),
            k_norm: Some(vec_at(cfg.d_head, 1.0, 0.1)?),
            w_qkv: None,
            w_kv: None,
            output_gate: false,
        };
        layers.push(Layer {
            attn_norm: vec_at(d, 1.0, 0.1)?,
            attn: Some(attn),
            gdn: None,
            ffn_norm: vec_at(d, 1.0, 0.1)?,
            dense: Some(DenseFfn {
                w_gate: m(d, d_ff, 0.2)?,
                w_up: m(d, d_ff, 0.2)?,
                w_down: m(d_ff, d, 0.2)?,
                w_gate_up: None,
            }),
            moe: None,
            blob: None,
        });
    }

    let half = (cfg.rotary_dim / 2).max(1);
    let mrope_axis: Vec<i32> = match cfg.mrope_section {
        Some(section) => (0..half).map(|i| interleaved_mrope_axis(i, section) as i32).collect(),
        None => vec![0i32; half],
    };

    let w = Weights {
        token_embd: m(d, vocab, 0.5)?,
        layers,
        output_norm: vec_at(d, 1.0, 0.1)?,
        output: Some(m(d, vocab, 0.5)?),
        output_split: None,
        rope_freqs: dev.stream().clone_htod(&vec![1.0f32; cfg.rotary_dim / 2])?,
        mrope_axis: dev.stream().clone_htod(&mrope_axis)?,
        device_bytes: 0,
        host_bytes: 0,
        max_blob_bytes: 0,
    };
    Model::from_weights(dev.clone(), cfg.clone(), w, 512, KvCacheQuant::F16, 8)
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

/// Prefill, then decode `steps` tokens greedily, returning every step's full
/// logits row (not just the argmax) -- the bit-exactness claim is about the
/// arithmetic, and an argmax comparison could pass on two logits vectors that
/// differ well outside rounding noise but still agree on which entry is
/// largest.
fn run(model: &mut Model, pool: &mut KvPool, seq: SeqId) -> Result<Vec<Vec<f32>>> {
    let mut logits = Vec::with_capacity(1 + DECODE_STEPS);
    let item = BatchItem::new(seq, PROMPT);
    model.forward_batch_device(std::slice::from_ref(&item), pool)?;
    let mut row = model.logits_host()?.to_vec();
    let mut next = argmax(&row);
    logits.push(row);
    for _ in 0..DECODE_STEPS {
        let item = BatchItem::new(seq, std::slice::from_ref(&next));
        model.forward_batch_device(std::slice::from_ref(&item), pool)?;
        row = model.logits_host()?.to_vec();
        next = argmax(&row);
        logits.push(row);
    }
    Ok(logits)
}

/// The test. Everything before this point is scaffolding shared with
/// `tests/spec_gdn.rs`'s pattern, adapted to build two configs from one RNG
/// stream rather than one.
#[test]
fn mrope_on_is_bit_identical_to_mrope_off_for_plain_text() -> Result<()> {
    let _gpu = gpu_lock();
    let dev = Device::new(0)?;

    let cfg_off = config(false)?;
    let cfg_on = config(true)?;
    assert_eq!(cfg_off.rotary_dim, cfg_on.rotary_dim, "the two configs must share a shape");
    assert!(cfg_off.mrope_section.is_none());
    assert_eq!(cfg_on.mrope_section, Some([2, 1, 1]));

    // Same seed for both: the weight streams are byte-identical, so any
    // difference in the two runs' output can only come from the M-RoPE
    // plumbing (`pos_stride`, `mrope_axis`, `Acts::mrope_positions`), not
    // from the models secretly being different.
    let rng_off = Rng(std::cell::Cell::new(0xC0FF_EE01));
    let rng_on = Rng(std::cell::Cell::new(0xC0FF_EE01));
    let mut model_off = synthetic_model(&dev, &cfg_off, &rng_off)?;
    let mut model_on = synthetic_model(&dev, &cfg_on, &rng_on)?;

    let mut pool_off = model_off.new_pool(512, 1)?;
    let seq_off = pool_off.alloc().unwrap();
    let logits_off = run(&mut model_off, &mut pool_off, seq_off)?;

    let mut pool_on = model_on.new_pool(512, 1)?;
    let seq_on = pool_on.alloc().unwrap();
    let logits_on = run(&mut model_on, &mut pool_on, seq_on)?;

    assert_eq!(logits_off.len(), logits_on.len());
    for (step, (a, b)) in logits_off.iter().zip(&logits_on).enumerate() {
        assert_eq!(
            a, b,
            "step {step}: mrope-on and mrope-off logits are not bit-identical \
             for a text-only request"
        );
    }
    Ok(())
}
