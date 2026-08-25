//! Speculative decoding on a model that carries recurrent state.
//!
//! `tests/spec.rs` establishes that the tokens do not change, on Qwen2.5 — a pure
//! attention model, where a rejected draft is undone by moving a length counter.
//! `tests/gdn_rollback.rs` establishes that the journal restores a GatedDeltaNet
//! layer's state and window, at the level of four kernel launches. Neither of
//! them runs the two together, and between them sits the part most likely to be
//! wrong: the hooks in `Model::linear_attention` that decide when to save the
//! window, when to run on the working copy of the state, and what to journal.
//!
//! The only checkpoint with those blocks is 51 GiB and does not fit on the card
//! this is developed on, so the model here is synthetic: Qwen3.5's block pattern
//! at 1/80th of its width, three GatedDeltaNet blocks and one gated
//! full-attention block, with arbitrary weights. Arbitrary weights make the text
//! meaningless and the test no weaker — what is being compared is speculative
//! decoding against ordinary decoding on the *same* model, and the acceptance
//! rule promises those are identical whatever the model is.
//!
//! The second half is the one that makes this evidence. Running the same loop
//! with the journal not installed leaves the rejected candidates folded into the
//! recurrent state, and the output has to *diverge*. Without that, a passing test
//! here would be consistent with the recurrence ignoring its own state.

use anyhow::Result;
use half::f16;
use tuili_cuda::Device;
use tuili_model::weights::{AttnWeights, DenseFfn, GdnWeights, Layer, Matrix, Weights};
use tuili_model::{BatchItem, Config, KvCacheQuant, KvPool, Model, SeqId};

/// Long enough for a polluted recurrent state to show up in the argmax.
const STEPS: usize = 24;
const PROMPT: &[u32] = &[3, 17, 41, 5, 200, 61, 7];

static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn gpu_lock() -> std::sync::MutexGuard<'static, ()> {
    GPU.lock().unwrap_or_else(|e| e.into_inner())
}

/// Qwen3.5's shape, small. `full_attention_interval` is 4, so blocks 0, 1 and 2
/// mix with a recurrence and block 3 is gated full attention — the same pattern
/// the 27B repeats sixteen times.
fn config() -> Result<Config> {
    let json = serde_json::json!({
        "model_type": "qwen3_5",
        "tie_word_embeddings": false,
        "text_config": {
            "num_hidden_layers": 4,
            "hidden_size": 64,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "intermediate_size": 128,
            "vocab_size": 256,
            "max_position_embeddings": 512,
            "rms_norm_eps": 1e-6,
            "attn_output_gate": true,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_key_head_dim": 16,
            "linear_value_head_dim": 16,
            "linear_conv_kernel_dim": 4,
            "rope_parameters": {"rope_theta": 10000000.0, "partial_rotary_factor": 0.25},
        },
    });
    Config::from_hf(&json, "synthetic-qwen3.5")
}

/// A deterministic fill, in a `Cell` so that two closures can share it.
struct Rng(std::cell::Cell<u32>);

impl Rng {
    fn next(&self) -> f32 {
        self.0
            .set(self.0.get().wrapping_mul(1_664_525).wrapping_add(1_013_904_223));
        ((self.0.get() >> 8) as f32 / 8_388_608.0) - 1.0
    }
}

/// A model at that shape with arbitrary weights.
///
/// The scales are chosen so that nothing overflows f16 through four blocks and so
/// that the recurrence neither saturates nor decays to nothing: `A_log` near -2
/// gives `g` around -0.15 and a per-token decay near 0.86, so a token's
/// contribution is still visible several tokens later. A state that decayed
/// instantly would make the rollback untestable — the second half of the test
/// below would fail, which is the point of having it.
fn synthetic_model(dev: &Device, cfg: &Config) -> Result<Model> {
    let la = cfg.linear_attn.expect("the config names linear dimensions");
    let (d, d_ff, vocab) = (cfg.d_model, cfg.d_ff, cfg.vocab_size);
    let (da, dkv) = (cfg.d_attn(), cfg.d_kv());
    let rng = Rng(std::cell::Cell::new(0x1234_5678));
    let m = |k: usize, n: usize, scale: f32| -> Result<Matrix> {
        let v: Vec<f16> = (0..k * n)
            .map(|_| f16::from_f32(scale * rng.next()))
            .collect();
        Matrix::upload_f16(dev, &v, k, n)
    };
    let vec_at = |n: usize, centre: f32, spread: f32| -> Result<tuili_model::weights::Vector> {
        let v: Vec<f32> = (0..n).map(|_| centre + spread * rng.next()).collect();
        Ok(dev.stream().clone_htod(&v)?)
    };

    let mut layers = Vec::with_capacity(cfg.n_layers);
    for i in 0..cfg.n_layers {
        let linear = (i + 1) % 4 != 0;
        let (attn, gdn) = if linear {
            (
                None,
                Some(GdnWeights {
                    in_proj_qkv: m(d, la.conv_channels(), 0.2)?,
                    in_proj_z: m(d, la.value_dim(), 0.2)?,
                    in_proj_a: m(d, la.value_heads, 0.2)?,
                    in_proj_b: m(d, la.value_heads, 0.2)?,
                    // The test builds its weights by hand and exercises the
                    // unstacked path; `stacked2`/`stacked_fp8_2` only fire in
                    // the real loader.
                    in_proj_ba: None,
                    in_proj_qz: None,
                    conv1d: vec_at(la.conv_channels() * la.conv_kernel, 0.3, 0.3)?,
                    // The decay: `g = -exp(A_log) * softplus(a + dt_bias)`.
                    a_log: vec_at(la.value_heads, -2.0, 0.2)?,
                    dt_bias: vec_at(la.value_heads, 0.0, 0.2)?,
                    // `Qwen3_5RMSNormGated`'s gain, the one norm in this
                    // architecture that is one-initialized rather than a delta.
                    norm: vec_at(la.value_head_dim, 1.0, 0.1)?,
                    out_proj: m(la.value_dim(), d, 0.2)?,
                }),
            )
        } else {
            (
                Some(AttnWeights {
                    // Gated: a query and its gate interleaved per head.
                    wq: m(d, 2 * da, 0.2)?,
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
                    output_gate: true,
                }),
                None,
            )
        };
        layers.push(Layer {
            attn_norm: vec_at(d, 1.0, 0.1)?,
            attn,
            gdn,
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

    let w = Weights {
        token_embd: m(d, vocab, 0.5)?,
        layers,
        output_norm: vec_at(d, 1.0, 0.1)?,
        output: Some(m(d, vocab, 0.5)?),
        output_split: None,
        rope_freqs: dev.stream().clone_htod(&vec![1.0f32; cfg.rotary_dim / 2])?,
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

fn prime(model: &mut Model, pool: &mut KvPool, seq: SeqId, prompt: &[u32]) -> Result<u32> {
    let item = BatchItem::new(seq, prompt);
    model.forward_batch_device(std::slice::from_ref(&item), pool)?;
    Ok(argmax(model.logits_host()?))
}

fn plain_greedy(model: &mut Model, steps: usize) -> Result<Vec<u32>> {
    let mut pool = model.new_pool(512, 1)?;
    let seq = pool.alloc().unwrap();
    let mut out = vec![prime(model, &mut pool, seq, PROMPT)?];
    for _ in 0..steps {
        let tok = *out.last().unwrap();
        let item = BatchItem::new(seq, std::slice::from_ref(&tok));
        model.forward_batch_device(std::slice::from_ref(&item), &mut pool)?;
        out.push(argmax(model.logits_host()?));
    }
    Ok(out)
}

/// How a drafter behaves, so that the acceptance count is chosen rather than
/// observed.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Drafter {
    /// Proposes what the model will choose: every draft accepted, bonus token
    /// emitted, and the journal replays the whole pass.
    Oracle,
    /// Proposes one right and one wrong: a partial acceptance, which is the case
    /// where the replay has to stop in the middle of the journal.
    Half,
    /// Proposes nothing the model would choose: the deepest rollback.
    Garbage,
}

fn speculate(
    model: &mut Model,
    k: usize,
    steps: usize,
    drafter: Drafter,
    rollback: bool,
    truth: &[u32],
) -> Result<(Vec<u32>, usize)> {
    let mut pool = model.new_pool(512, 1)?;
    if rollback {
        model.enable_speculation(k, &pool)?;
    } else {
        // The control: no journal, so `linear_attention` never arms, the
        // recurrence advances the persistent state over all `k + 1` candidates,
        // and nothing puts the rejected ones back.
        model.disable_speculation();
    }
    let seq = pool.alloc().unwrap();
    let mut pending = prime(model, &mut pool, seq, PROMPT)?;
    let mut out = vec![pending];
    let mut accepted = 0usize;
    while out.len() <= steps {
        // The drafter reads the unspeculated run, which is how a chosen
        // acceptance pattern is produced on a model whose text means nothing.
        let at = out.len() - 1;
        let proposal: Vec<u32> = (0..k)
            .map(|j| {
                let right = truth.get(at + 1 + j).copied().unwrap_or(0);
                match drafter {
                    Drafter::Oracle => right,
                    Drafter::Half if j == 0 => right,
                    _ => right.wrapping_add(37) % 256,
                }
            })
            .collect();
        let outcome = model.verify_draft(seq, &mut pool, pending, &proposal)?;
        accepted += outcome.accepted;
        pending = *outcome.tokens.last().unwrap();
        out.extend(outcome.tokens);
    }
    Ok((out, accepted))
}

/// A prefill chunk the same width as a verification pass does not steal its
/// captured graph.
///
/// This is the one hazard the rollback adds to CUDA graph capture, and it is
/// invisible from either side. A graph records the copies that stage a layer's
/// recurrent state and journal its inputs, and the capture key is
/// `(pool, tokens, kv bucket)` — under which a `k + 1`-token prefill chunk and a
/// `k + 1`-candidate verification pass are the same shape. Replaying the wrong
/// one is wrong in opposite directions: an ordinary pass would advance a working
/// copy and drop its state update on the floor, and a verification pass would
/// advance the persistent state and then have the journal replayed on top of it.
/// Neither fails, both produce fluent output.
///
/// So the prompt here is fed in chunks of exactly `k + 1`, which is enough to get
/// that shape warmed *and* captured before a single draft is verified.
#[test]
fn a_prefill_chunk_the_width_of_a_verification_pass_keeps_its_own_graph() -> Result<()> {
    let _gpu = gpu_lock();
    let Ok(dev) = Device::new(0) else {
        eprintln!("SKIPPED: no CUDA device");
        return Ok(());
    };
    let cfg = config()?;
    let mut model = synthetic_model(&dev, &cfg)?;
    let k = 2;
    let chunk = k + 1;
    // Nine tokens in threes: the first chunk warms the shape, the second is
    // captured, the third replays it — all before speculation starts.
    let prompt: Vec<u32> = (0..9u32).map(|i| 3 + i * 11).collect();

    let chunked_prefill = |model: &mut Model, pool: &mut KvPool, seq: SeqId| -> Result<u32> {
        let n = prompt.len();
        for (i, part) in prompt.chunks(chunk).enumerate() {
            let last = (i + 1) * chunk >= n;
            let item = if last {
                BatchItem::new(seq, part)
            } else {
                BatchItem::without_logits(seq, part)
            };
            model.forward_batch_device(std::slice::from_ref(&item), pool)?;
        }
        Ok(argmax(model.logits_host()?))
    };

    // Plain decoding, with the same chunked prefill.
    let mut plain = {
        let mut pool = model.new_pool(512, 1)?;
        let seq = pool.alloc().unwrap();
        let mut out = vec![chunked_prefill(&mut model, &mut pool, seq)?];
        for _ in 0..STEPS {
            let tok = *out.last().unwrap();
            let item = BatchItem::new(seq, std::slice::from_ref(&tok));
            model.forward_batch_device(std::slice::from_ref(&item), &mut pool)?;
            out.push(argmax(model.logits_host()?));
        }
        out
    };

    let mut pool = model.new_pool(512, 1)?;
    model.enable_speculation(k, &pool)?;
    let seq = pool.alloc().unwrap();
    let mut pending = chunked_prefill(&mut model, &mut pool, seq)?;
    let mut out = vec![pending];
    while out.len() <= STEPS {
        let at = out.len() - 1;
        // Half right, half wrong: partial acceptance every step, so the journal
        // is replayed to a middle row rather than to one end.
        let proposal: Vec<u32> = (0..k)
            .map(|j| {
                let right = plain.get(at + 1 + j).copied().unwrap_or(0);
                if j == 0 { right } else { right.wrapping_add(37) % 256 }
            })
            .collect();
        let outcome = model.verify_draft(seq, &mut pool, pending, &proposal)?;
        pending = *outcome.tokens.last().unwrap();
        out.extend(outcome.tokens);
    }
    plain.truncate(STEPS + 1);
    assert_eq!(
        &out[..=STEPS],
        &plain[..],
        "a verification pass and a prefill chunk of the same width shared a \
         captured graph"
    );
    Ok(())
}

/// Speculative decoding on a model with recurrent state emits exactly what
/// ordinary decoding emits — and does so *because* of the rollback.
#[test]
fn speculation_over_gateddeltanet_blocks_changes_nothing_but_the_speed() -> Result<()> {
    let _gpu = gpu_lock();
    let Ok(dev) = Device::new(0) else {
        eprintln!("SKIPPED: no CUDA device");
        return Ok(());
    };
    let cfg = config()?;
    assert!(cfg.linear_attn.is_some(), "the fixture has no linear blocks");
    let mut model = synthetic_model(&dev, &cfg)?;
    let plain = plain_greedy(&mut model, STEPS)?;
    assert_eq!(plain.len(), STEPS + 1);
    // A model whose argmax never moves would make every comparison below vacuous.
    let distinct: std::collections::BTreeSet<u32> = plain.iter().copied().collect();
    assert!(
        distinct.len() > 2,
        "the synthetic model emits {} distinct tokens in {STEPS} steps; a \
         degenerate model cannot show that a polluted state changes anything",
        distinct.len()
    );

    for k in [1usize, 2] {
        for drafter in [Drafter::Oracle, Drafter::Half, Drafter::Garbage] {
            if drafter == Drafter::Half && k == 1 {
                continue; // no partial case exists at k = 1
            }
            let (got, accepted) = speculate(&mut model, k, STEPS, drafter, true, &plain)?;
            assert_eq!(
                &got[..=STEPS],
                &plain[..],
                "k = {k}, {drafter:?}: speculation changed the output on a model \
                 with recurrent state"
            );
            let copies = model
                .gdn_rollback()
                .expect("the journal should be installed")
                .state_copies();
            assert!(
                copies > 0,
                "no layer's state was staged, so the verification pass ran \
                 straight into the persistent state and this run proves nothing"
            );
            eprintln!(
                "k = {k}, {drafter:?}: output identical, {accepted} drafts \
                 accepted, {copies} layer states staged"
            );
            match drafter {
                Drafter::Oracle => assert!(accepted > 0, "the oracle accepted nothing"),
                Drafter::Garbage => assert_eq!(accepted, 0, "garbage was accepted"),
                Drafter::Half => assert!(accepted > 0, "the half drafter accepted nothing"),
            }
        }
    }

    // And the control: the same loop with nothing putting the rejected
    // candidates back. The recurrent state then carries tokens the sequence does
    // not contain, and the output diverges — which is what makes the runs above
    // evidence about the rollback rather than about the acceptance rule.
    let (unrolled, _) = speculate(&mut model, 2, STEPS, Drafter::Garbage, false, &plain)?;
    let first_diff = unrolled
        .iter()
        .zip(&plain)
        .position(|(a, b)| a != b);
    eprintln!(
        "without the journal the output diverges at token {:?} of {STEPS}",
        first_diff
    );
    assert!(
        first_diff.is_some(),
        "with the rollback disabled, {STEPS} steps of speculation still produced \
         the unspeculated tokens. Either the recurrence is not sensitive to the \
         rejected candidates at these shapes, or the journal is not what is \
         making the runs above pass — and in both cases this test is not \
         evidence for the rollback."
    );
    Ok(())
}
