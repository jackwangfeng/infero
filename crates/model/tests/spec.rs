//! Speculative decoding on a real model: it must change the speed and nothing
//! else.
//!
//! The one test that matters here is the first one. Speculative decoding is a
//! pure throughput optimization — the greedy acceptance rule is constructed so
//! that a step emits a prefix of the target model's own argmax sequence — so any
//! difference in the tokens is a bug, and it is a bug of the worst kind, because
//! a drafter that is merely *bad* produces perfectly fluent output at a lower
//! speedup. Nothing in a sample of generated text distinguishes "the drafter is
//! weak" from "the rollback is wrong". Only this comparison does.
//!
//! Which model does not matter for that, and it is deliberately not Qwen3.5:
//! this is the machinery, not the head, and the machinery is exercised harder by
//! a drafter whose acceptance pattern the test chooses than by a real one. So
//! Qwen2.5-0.5B carries these, with four adversarial drafters that force every
//! acceptance count — all-accept (which exercises the bonus token), all-reject
//! (which exercises the deepest rollback), accept-then-reject, and noise. The MTP
//! head's own numbers are in `tests/qwen35_mtp_device.rs`, where the real weights
//! are.
//!
//! The last test measures a mean acceptance length end to end with a real
//! drafter — a quantized copy of the target model — because a number that came
//! out of the whole loop is worth more than one derived from a per-token
//! agreement rate.

use std::path::PathBuf;

use anyhow::Result;
use tuili_model::mtp::{HeadDims, MtpHead};
use tuili_model::spec::DraftFeed;
use tuili_model::weights::{AttnWeights, Layer, Matrix, MtpWeights};
use tuili_model::{BatchItem, KvCacheQuant, KvPool, Model, SeqId};
use tuili_tokenizer::Tokenizer;

const PROMPT: &str = "The capital of France is Paris, and the capital of Japan is";

/// Long enough that a wrong rollback has somewhere to drift, short enough to run
/// in a couple of seconds.
const STEPS: usize = 24;

fn gguf_path(name: &str) -> Option<PathBuf> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models").join(name);
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
    let gguf = tuili_gguf::Gguf::open(&path)?;
    let tok = Tokenizer::from_gguf(&gguf)?;
    let model = Model::load_full(
        tuili_cuda::Device::new(0)?,
        &gguf,
        512,
        KvCacheQuant::F16,
        usize::MAX,
        max_logit_rows,
    )?;
    Ok(Some((model, tok)))
}

/// One test at a time: each builds its own `Model`, and a decode step captures a
/// CUDA graph, which any other thread's allocation on the same context
/// invalidates. Same reasoning as `tests/batching.rs`.
static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn gpu_lock() -> std::sync::MutexGuard<'static, ()> {
    GPU.lock().unwrap_or_else(|e| e.into_inner())
}

/// The first index of the largest value.
///
/// Both sides of the comparison have to break ties the same way or the test
/// measures the tie-break rather than the rollback; `spec::verify_draft` uses
/// `mtp::argmax`, which is this rule.
fn argmax(v: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, x) in v.iter().enumerate() {
        if *x > v[best] {
            best = i;
        }
    }
    best as u32
}

/// Prefill and return the first generated token.
fn prime(model: &mut Model, pool: &mut KvPool, seq: SeqId, prompt: &[u32]) -> Result<u32> {
    let item = BatchItem::new(seq, prompt);
    model.forward_batch_device(std::slice::from_ref(&item), pool)?;
    Ok(argmax(model.logits_host()?))
}

/// Ordinary greedy decoding, one token per forward pass.
fn plain_greedy(model: &mut Model, prompt: &[u32], steps: usize) -> Result<Vec<u32>> {
    let mut pool = model.new_pool(512, 1)?;
    let seq = pool.alloc().unwrap();
    let mut out = vec![prime(model, &mut pool, seq, prompt)?];
    for _ in 0..steps {
        let tok = *out.last().unwrap();
        let item = BatchItem::new(seq, std::slice::from_ref(&tok));
        model.forward_batch_device(std::slice::from_ref(&item), &mut pool)?;
        out.push(argmax(model.logits_host()?));
    }
    Ok(out)
}

/// What a drafter proposes, given the step index and the token the previous step
/// settled on.
type Draft<'a> = Box<dyn FnMut(usize, u32) -> Vec<u32> + 'a>;

/// Speculative greedy decoding: draft `k`, verify `k + 1`, accept a prefix.
///
/// Returns the tokens generated and the acceptance length of every step.
fn speculative(
    model: &mut Model,
    prompt: &[u32],
    want: usize,
    k: usize,
    mut draft: Draft<'_>,
) -> Result<(Vec<u32>, Vec<usize>)> {
    let mut pool = model.new_pool(512, 1)?;
    model.enable_speculation(k, &pool)?;
    let seq = pool.alloc().unwrap();
    let mut pending = prime(model, &mut pool, seq, prompt)?;
    let mut out = vec![pending];
    let mut lengths = Vec::new();
    let mut step = 0usize;
    while out.len() <= want {
        let proposal = draft(step, pending);
        assert_eq!(proposal.len(), k, "the drafter changed its mind about k");
        let outcome = model.verify_draft(seq, &mut pool, pending, &proposal)?;
        assert!(
            !outcome.tokens.is_empty(),
            "a verification step emitted nothing, which would livelock"
        );
        assert!(
            outcome.tokens.len() <= k + 1,
            "a step emitted {} tokens from a {k}-token draft",
            outcome.tokens.len()
        );
        assert_eq!(outcome.accepted + 1, outcome.tokens.len());
        lengths.push(outcome.tokens.len());
        // The last token is the next step's pending one, and is deliberately not
        // in the cache: the sequence holds `accepted + 1` new entries.
        pending = *outcome.tokens.last().unwrap();
        out.extend(outcome.tokens);
        step += 1;
        assert!(step < 4 * want + 8, "no progress");
    }
    Ok((out, lengths))
}

/// The tokens speculative decoding emits are the tokens plain decoding emits.
///
/// Four drafters, chosen so that every acceptance count is forced rather than
/// hoped for:
///
/// * `oracle` proposes what the model will choose, so every step accepts all `k`
///   and emits the bonus token — the only path where `logits[k]` is read at all;
/// * `garbage` proposes a token the model will not choose, so every step accepts
///   nothing and rolls back the whole draft;
/// * `half` proposes one right and one wrong, the partial case;
/// * `noise` proposes pseudo-random ids, which mixes all three.
///
/// The oracle gets its answers from the plain run, which is the same thing a
/// perfect drafter would produce, and is the only drafter here that can reach
/// full acceptance reliably on a 0.5B model.
#[test]
fn greedy_speculation_emits_exactly_what_plain_decoding_emits() -> Result<()> {
    let _gpu = gpu_lock();
    let Some((mut model, tok)) = load("qwen2.5-0.5b-instruct-q8_0.gguf", 8)? else {
        return Ok(());
    };
    let prompt = tok.encode(PROMPT, Some(false), false);
    let plain = plain_greedy(&mut model, &prompt, STEPS)?;
    assert_eq!(plain.len(), STEPS + 1);
    eprintln!(
        "plain greedy: {:?}",
        tok.decode(&plain, false)
    );

    for k in [1usize, 2, 3] {
        // The oracle reads the plain run, which is what it is allowed to know:
        // its job is to exercise the all-accepted path, not to be realistic.
        let oracle_tokens = plain.clone();
        let drafters: Vec<(&str, Draft<'_>)> = vec![
            (
                "oracle",
                Box::new(move |_step, pending| {
                    // Whatever the plain run produced after `pending`, found by
                    // its first occurrence — the sequences agree by induction, so
                    // this is well defined for every step the test reaches.
                    let at = oracle_tokens
                        .iter()
                        .position(|t| *t == pending)
                        .unwrap_or(oracle_tokens.len() - 1);
                    (0..k)
                        .map(|j| {
                            oracle_tokens
                                .get(at + 1 + j)
                                .copied()
                                .unwrap_or(pending)
                        })
                        .collect()
                }),
            ),
            (
                "garbage",
                Box::new(move |_step, pending| vec![pending.wrapping_add(7) % 150_000; k]),
            ),
            (
                "half",
                {
                    let plain = plain.clone();
                    Box::new(move |_step, pending| {
                        let at = plain.iter().position(|t| *t == pending).unwrap_or(0);
                        (0..k)
                            .map(|j| {
                                if j == 0 {
                                    plain.get(at + 1).copied().unwrap_or(pending)
                                } else {
                                    (pending as usize * 31 + j) as u32 % 150_000
                                }
                            })
                            .collect()
                    })
                },
            ),
            (
                "noise",
                Box::new(move |step, pending| {
                    (0..k)
                        .map(|j| ((step * 7919 + j * 104_729 + pending as usize) % 150_000) as u32)
                        .collect()
                }),
            ),
        ];

        for (name, draft) in drafters {
            let (got, lengths) = speculative(&mut model, &prompt, STEPS, k, draft)?;
            let mean = lengths.iter().sum::<usize>() as f64 / lengths.len() as f64;
            eprintln!(
                "k = {k}, {name}: {} steps, mean acceptance length {mean:.2}",
                lengths.len()
            );
            assert_eq!(
                &got[..=STEPS],
                &plain[..],
                "k = {k}, drafter {name}: speculative decoding produced \
                 different tokens from plain greedy decoding. Speculation is \
                 supposed to be a speedup and nothing else.\n  plain: {:?}\n  \
                 spec:  {:?}",
                tok.decode(&plain, false),
                tok.decode(&got[..=STEPS], false),
            );
            // And the drafters really did behave differently, or three of these
            // four runs proved the same thing.
            match name {
                "oracle" => assert!(
                    mean > 1.0,
                    "the oracle drafter accepted nothing; it is not exercising \
                     the bonus-token path"
                ),
                "garbage" => assert_eq!(
                    mean, 1.0,
                    "the garbage drafter had a draft accepted, so the \
                     all-rejected rollback is not being exercised"
                ),
                _ => {}
            }
        }
    }
    Ok(())
}

/// A rejected draft leaves the sequence exactly as long as one ordinary step
/// would have, and returns the slots it borrowed.
///
/// The KV half of "leaves no trace". The recurrent-state half needs a model with
/// linear-attention blocks and lives in `tests/gdn_rollback.rs`, against the
/// kernels; this is the part that is about the pool.
#[test]
fn a_rejected_draft_returns_its_kv_slots_and_leaves_the_length_alone() -> Result<()> {
    let _gpu = gpu_lock();
    let Some((mut model, tok)) = load("qwen2.5-0.5b-instruct-q8_0.gguf", 8)? else {
        return Ok(());
    };
    let prompt = tok.encode(PROMPT, Some(false), false);
    let k = 3;
    let mut pool = model.new_pool(512, 1)?;
    model.enable_speculation(k, &pool)?;
    let seq = pool.alloc().unwrap();
    let pending = prime(&mut model, &mut pool, seq, prompt.as_slice())?;

    let len_before = pool.len(seq);
    let free_before = pool.free_slots();
    let table_before = pool.read_slot_table(model.device(), seq)?;
    assert_eq!(len_before, prompt.len());

    // A draft that cannot be accepted: three copies of a token the model would
    // not choose after `pending`.
    let doomed = vec![pending.wrapping_add(11) % 150_000; k];
    let outcome = model.verify_draft(seq, &mut pool, pending, &doomed)?;
    assert_eq!(outcome.accepted, 0, "the doomed draft was accepted");
    assert_eq!(outcome.tokens.len(), 1);

    assert_eq!(
        pool.len(seq),
        len_before + 1,
        "after a fully rejected draft the sequence should be one token longer — \
         the token the previous step had settled on — and not {}",
        pool.len(seq)
    );
    assert_eq!(
        pool.free_slots(),
        free_before - 1,
        "a rejected draft kept {} slots it should have returned",
        (free_before - 1) - pool.free_slots()
    );
    // The prompt's own mapping is untouched: a rollback that renumbered the
    // surviving prefix would still generate fluent text.
    let table_after = pool.read_slot_table(model.device(), seq)?;
    assert_eq!(
        &table_after[..len_before],
        &table_before[..len_before],
        "the surviving prefix's slot mapping changed"
    );

    // And the next step continues from the accepted token, not from the draft:
    // its argmax is what plain decoding gives after `pending`.
    let next = {
        let tok_in = *outcome.tokens.last().unwrap();
        let item = BatchItem::new(seq, std::slice::from_ref(&tok_in));
        model.forward_batch_device(std::slice::from_ref(&item), &mut pool)?;
        argmax(model.logits_host()?)
    };
    let plain = plain_greedy(&mut model, &prompt, 2)?;
    assert_eq!(
        (outcome.tokens[0], next),
        (plain[1], plain[2]),
        "decoding after a rejected draft diverged from plain decoding"
    );
    Ok(())
}

/// The MTP head, driven through the whole loop, on a model that loads.
///
/// The 27B cannot be loaded end to end yet — its vision tower is unimplemented —
/// so the head's *numbers* are checked against the real weights in
/// `tests/qwen35_mtp_device.rs`, against a capture of the reference
/// implementation. What that cannot check is the plumbing between the target
/// model and the drafter: that the hidden states the head reads are the ones the
/// pass just produced, that the shifted ids and positions line up, that the
/// drafter's cache tracks the accepted prefix, and that the acceptance rule sees
/// the drafts the head actually proposed.
///
/// So this builds a head at Qwen2.5-0.5B's shapes with arbitrary weights and
/// runs it. Its drafts are worthless — that is the point of the assertion at the
/// end, that they are mostly *rejected* — and the tokens that come out must
/// still be the ones plain greedy decoding produces. A drafter proposing noise is
/// the hardest case for the rollback and the easiest to get an accidental pass
/// from, which is why the acceptance count is asserted as well as the output.
#[test]
fn the_mtp_head_drives_the_loop_without_changing_the_output() -> Result<()> {
    let _gpu = gpu_lock();
    let Some((mut model, tok)) = load("qwen2.5-0.5b-instruct-q8_0.gguf", 8)? else {
        return Ok(());
    };
    let prompt = tok.encode(PROMPT, Some(false), false);
    let plain = plain_greedy(&mut model, &prompt, STEPS)?;

    let k = 2;
    let cfg = model.config().clone();
    let head = synthetic_head(model.device(), &cfg, prompt.len().max(k + 1), model.max_seq())?;
    model.install_mtp_head(head)?;
    assert!(model.has_mtp_head());

    let mut pool = model.new_pool(512, 1)?;
    model.enable_speculation(k, &pool)?;
    let seq = pool.alloc().unwrap();
    let mut pending = prime(&mut model, &mut pool, seq, &prompt)?;
    let mut feed = DraftFeed::after_prefill(&prompt, pending);
    let mut out = vec![pending];
    let mut accepted_total = 0usize;
    let mut drafted_total = 0usize;
    while out.len() <= STEPS {
        let draft = model.draft_with_head(k, &feed)?;
        assert_eq!(draft.len(), k, "the head proposed the wrong number of tokens");
        for t in &draft {
            assert!(
                (*t as usize) < cfg.vocab_size,
                "the head drafted token {t}, past the vocabulary"
            );
        }
        let outcome = model.verify_draft(seq, &mut pool, pending, &draft)?;
        accepted_total += outcome.accepted;
        drafted_total += outcome.drafted;
        pending = *outcome.tokens.last().unwrap();
        out.extend(outcome.tokens.iter().copied());
        feed = outcome.feed;
        assert_eq!(feed.positions.len(), feed.rows.len());
        assert_eq!(
            feed.positions.last().copied().unwrap() + 1,
            pool.len(seq),
            "the feed's last row should be the sequence's last cached token"
        );
    }
    assert_eq!(
        &out[..=STEPS],
        &plain[..],
        "the MTP head changed the output.\n  plain: {:?}\n  spec:  {:?}",
        tok.decode(&plain, false),
        tok.decode(&out[..=STEPS], false),
    );
    eprintln!(
        "a random-weight head at 0.5B shapes: {accepted_total} of {drafted_total} \
         drafts accepted, and the output is unchanged"
    );
    assert!(
        accepted_total * 4 < drafted_total,
        "a head with arbitrary weights had {accepted_total} of {drafted_total} \
         drafts accepted; either the model is degenerate or the drafts are not \
         reaching the acceptance rule"
    );
    Ok(())
}

/// An MTP head at a given model's shapes with arbitrary small weights.
///
/// Small on purpose: the head is twenty-seven f16 operations deep and the point
/// is to exercise the plumbing, not to survive an overflow. The values are
/// deterministic so a failure is reproducible.
fn synthetic_head_branched(
    dev: &tuili_cuda::Device,
    cfg: &tuili_model::Config,
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
    let vec1 = |n: usize| -> Result<tuili_model::weights::Vector> {
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
                output_gate: true,
            }),
            gdn: None,
            ffn_norm: vec1(d)?,
            w_gate: m(d, dims.d_ff)?,
            w_up: m(d, dims.d_ff)?,
            w_down: m(dims.d_ff, d)?,
            w_gate_up: None,
            blob: None,
        },
        device_bytes: 0,
    };
    MtpHead::new(dev, w, dims, max_rows, max_seq, branches)
}

/// The same, one branch, which is what the linear draft wants.
fn synthetic_head(
    dev: &tuili_cuda::Device,
    cfg: &tuili_model::Config,
    max_rows: usize,
    max_seq: usize,
) -> Result<MtpHead> {
    synthetic_head_branched(dev, cfg, max_rows, max_seq, 1)
}

/// A mean acceptance length, measured end to end with a drafter that is a real
/// model.
///
/// Reported rather than asserted against the notes' 1.9: that figure is for the
/// Qwen3.5 MTP head on the 27B, and this pair is a quantized 0.5B drafting for an
/// f16 copy of itself, which is a different — and much easier — problem. What the
/// number here establishes is that the loop works on two real models and that
/// acceptance is not accidentally pinned at 1. The head's own acceptance rate is
/// measured on the real weights in `tests/qwen35_mtp_device.rs`.
#[test]
fn a_real_drafter_reaches_a_useful_acceptance_length() -> Result<()> {
    let _gpu = gpu_lock();
    let Some((mut target, tok)) = load("qwen2.5-0.5b-instruct-fp16.gguf", 8)? else {
        return Ok(());
    };
    let Some((mut drafter, _)) = load("qwen2.5-0.5b-instruct-q4_k_m.gguf", 8)? else {
        return Ok(());
    };
    let prompt = tok.encode(PROMPT, Some(false), false);
    let k = 2;

    let mut pool = target.new_pool(512, 1)?;
    target.enable_speculation(k, &pool)?;
    let seq = pool.alloc().unwrap();
    let mut dpool = drafter.new_pool(512, 1)?;
    let dseq = dpool.alloc().unwrap();

    let mut pending = prime(&mut target, &mut pool, seq, &prompt)?;
    // The drafter sees the prompt too; from here on it is fed the tokens the
    // target confirmed, which is what keeps its cache in step.
    prime(&mut drafter, &mut dpool, dseq, &prompt)?;

    let mut lengths = Vec::new();
    let mut confirmed: Vec<u32> = vec![pending];
    while lengths.len() < 32 {
        // The drafter's cache holds the prompt and every confirmed token except
        // `pending`; feeding `pending` gives its first proposal, and each
        // proposal is fed back for the next.
        let mut proposal = Vec::with_capacity(k);
        let mut fed = pending;
        for _ in 0..k {
            let item = BatchItem::new(dseq, std::slice::from_ref(&fed));
            drafter.forward_batch_device(std::slice::from_ref(&item), &mut dpool)?;
            fed = argmax(drafter.logits_host()?);
            proposal.push(fed);
        }
        let outcome = target.verify_draft(seq, &mut pool, pending, &proposal)?;
        lengths.push(outcome.tokens.len());
        // Roll the drafter back to the accepted prefix: it consumed `pending`
        // plus `k` proposals, and only `pending` plus the accepted ones survive.
        // One behind the target, which is what makes this arithmetic and not a
        // guess: the drafter has been fed every confirmed token up to and
        // including `pending`.
        dpool.truncate(dseq, dpool.len(dseq) - (k - outcome.accepted));
        confirmed.extend(outcome.tokens.iter().copied());
        pending = *outcome.tokens.last().unwrap();
    }

    let mean = lengths.iter().sum::<usize>() as f64 / lengths.len() as f64;
    let accepted: usize = lengths.iter().map(|l| l - 1).sum();
    eprintln!(
        "q4 drafting for f16, k = {k}: mean acceptance length {mean:.2} over {} \
         steps ({accepted} of {} drafts accepted); text: {:?}",
        lengths.len(),
        lengths.len() * k,
        tok.decode(&confirmed, false)
    );
    assert!(
        mean > 1.0,
        "a quantized copy of the target model drafted {mean:.2} tokens a step, \
         which means nothing was ever accepted — the drafter and the target \
         disagree on every token, so either the rollback or the drafter's cache \
         is wrong"
    );
    assert!(
        mean <= k as f64 + 1.0,
        "a step cannot emit more than k + 1 tokens"
    );
    Ok(())
}

/// A gap in the drafter's history is refused, by name, rather than papered over.
///
/// The scheduler leans on this. The drafter's cache only advances on the rounds
/// that run, so any step that skips speculation — a second sequence arriving is
/// the ordinary one — leaves it behind, and it cannot catch up: `mtp_hidden`
/// holds the rows of the *last* pass, so the hidden states for the gap are gone.
/// `Scheduler::speculative_step` therefore compares the feed's first position
/// against `MtpHead::cached` and stops speculating for that sequence.
///
/// That check is only worth having because this one exists underneath it. If
/// priming across a gap were allowed, the drafter would attend over slots nobody
/// wrote — zeros, or another request's keys — and the scheduler's guard would be
/// dead code that nothing would notice the loss of. Two concurrent requests to
/// the 27B reached exactly here, as a 500 carrying this message.
///
/// It is an `Err`, not a panic: it is a reachable condition about state, not a
/// broken invariant, and the server turns it into a skip.
#[test]
fn priming_the_drafter_across_a_gap_is_an_error() -> Result<()> {
    let _gpu = gpu_lock();
    let Some((mut model, tok)) = load("qwen2.5-0.5b-instruct-q8_0.gguf", 8)? else {
        return Ok(());
    };
    let prompt = tok.encode(PROMPT, Some(false), false);
    let cfg = model.config().clone();
    const K: usize = 2;
    let head = synthetic_head(model.device(), &cfg, prompt.len().max(K + 1), model.max_seq())?;
    model.install_mtp_head(head)?;

    let mut pool = model.new_pool(512, 1)?;
    model.enable_speculation(K, &pool)?;
    let seq = pool.alloc().unwrap();
    let pending = prime(&mut model, &mut pool, seq, &prompt)?;

    // Contiguous first, so that the failure below is the gap and not the setup.
    let feed = DraftFeed::after_prefill(&prompt, pending);
    model.draft_with_head(K, &feed)?;
    let reached = model.mtp_head().expect("installed above").cached();
    // The prompt's rows, plus one slot for each draft after the first: a
    // `k`-token draft re-enters the head `k - 1` times through
    // `step_from_own_output`, and each of those writes a key. This is the drafter
    // running ahead of the target, which is the whole point of it, and the reason
    // the next round starts with `truncate`.
    assert_eq!(
        reached,
        prompt.len() + (K - 1),
        "priming over the prompt plus {} self-steps",
        K - 1
    );

    // Now ask for a row well past where the cache reaches, which is what a
    // sequence that decoded without speculation for a while looks like.
    let hole = reached + 5;
    let gapped = DraftFeed {
        rows: 0..1,
        positions: vec![hole],
        shifted: vec![pending],
    };
    let err = model
        .draft_with_head(K, &gapped)
        .expect_err("a gap in the drafter's history has to be refused");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("hole") && msg.contains(&hole.to_string()),
        "the refusal should name the gap and where it is, got: {msg}"
    );
    // And the refusal left the cache where it was, so a caller that skips this
    // round and comes back contiguous is still in business.
    assert_eq!(
        model.mtp_head().expect("installed above").cached(),
        reached,
        "a refused prime must not move the drafter's cache"
    );
    Ok(())
}

/// A tree of width one has to draft exactly what the linear path drafts.
///
/// This is the reduction that makes the tree machinery trustworthy: the fork, the
/// per-level `step_tree`, the per-node windows and the candidate loop all sit on
/// the path a width-one tree takes, so if any of them is wrong the tokens diverge
/// from the code that has been shipping. Same seed both ways, and the head
/// consumes one uniform a candidate either way, so the streams line up token for
/// token.
#[test]
fn a_tree_of_width_one_drafts_what_the_linear_path_drafts() -> Result<()> {
    let _gpu = gpu_lock();
    const K: usize = 3;
    let Some((mut model, tok)) = load("qwen2.5-0.5b-instruct-q8_0.gguf", K + 1)? else {
        return Ok(());
    };
    let _ = &tok;
    let prompt = tok.encode(PROMPT, Some(false), false);
    let cfg = model.config().clone();
    // One lane: a width-one tree never forks, which is what makes it the linear
    // draft.
    let head = synthetic_head_branched(model.device(), &cfg, prompt.len().max(K + 1), model.max_seq(), 1)?;
    model.install_mtp_head(head)?;

    let mut pool = model.new_pool(512, 1)?;
    model.enable_speculation(K, &pool)?;
    let seq = pool.alloc().unwrap();
    let pending = prime(&mut model, &mut pool, seq, &prompt)?;
    let feed = DraftFeed::after_prefill(&prompt, pending);

    let sp = tuili_model::SamplingParams {
        temperature: 0.8,
        top_p: 0.95,
        top_k: 32,
        seed: Some(4242),
        ..Default::default()
    };
    let linear = {
        let mut s = tuili_model::Sampler::new(sp.clone());
        model.draft_with_head_sampled(K, &feed, &mut s, &prompt)?
    };
    let tree = {
        let mut s = tuili_model::Sampler::new(sp.clone());
        model.draft_tree(1, K, &feed, &mut s, &prompt)?
    };

    assert_eq!(
        tree.nodes.len(),
        K,
        "a width-one tree of depth {K} should have {K} nodes"
    );
    let tree_tokens: Vec<u32> = tree.nodes.iter().map(|n| n.token).collect();
    let linear_tokens: Vec<u32> = linear.iter().map(|d| d.token).collect();
    assert_eq!(
        tree_tokens, linear_tokens,
        "a width-one tree drafted {tree_tokens:?} where the linear path drafted \
         {linear_tokens:?}"
    );
    // And the distributions have to match too, since the acceptance rule
    // composes with them.
    for (i, (n, l)) in tree.nodes.iter().zip(&linear).enumerate() {
        let q = &tree.qs[n.q_of];
        assert_eq!(
            q.len(),
            l.q.len(),
            "node {i}'s distribution has {} entries, the linear draft's {}",
            q.len(),
            l.q.len()
        );
        for (a, b) in q.iter().zip(&l.q) {
            assert_eq!(a.0, b.0, "node {i}: token order differs");
            assert!(
                (a.1 - b.1).abs() < 1e-6,
                "node {i}: probability {} against {}",
                a.1,
                b.1
            );
        }
    }
    Ok(())
}

/// A two-wide tree's shape: the node count, the lanes, and which nodes share a
/// distribution.
///
/// Siblings must share a `q` — they are i.i.d. draws from it, which is what the
/// acceptance rule is stated for — and cousins must not, since their parents saw
/// different tokens. Every leaf needs a lane to itself, or two paths would write
/// the same slot in the drafter's cache.
#[test]
fn a_two_wide_tree_is_well_formed() -> Result<()> {
    let _gpu = gpu_lock();
    const B: usize = 2;
    const D: usize = 3;
    let Some((mut model, tok)) = load("qwen2.5-0.5b-instruct-q8_0.gguf", 4)? else {
        return Ok(());
    };
    let prompt = tok.encode(PROMPT, Some(false), false);
    let cfg = model.config().clone();
    let widest = B.pow(D as u32 - 1);
    let head = synthetic_head_branched(
        model.device(),
        &cfg,
        prompt.len().max(widest * B),
        model.max_seq(),
        widest,
    )?;
    model.install_mtp_head(head)?;

    let mut pool = model.new_pool(512, 1)?;
    model.enable_speculation(D, &pool)?;
    let seq = pool.alloc().unwrap();
    let pending = prime(&mut model, &mut pool, seq, &prompt)?;
    let feed = DraftFeed::after_prefill(&prompt, pending);
    let mut s = tuili_model::Sampler::new(tuili_model::SamplingParams {
        temperature: 0.9,
        top_p: 0.95,
        top_k: 32,
        seed: Some(7),
        ..Default::default()
    });
    let tree = model.draft_tree(B, D, &feed, &mut s, &prompt)?;

    let want: usize = (1..=D).map(|l| B.pow(l as u32)).sum();
    assert_eq!(tree.nodes.len(), want, "a {B}-by-{D} tree should have {want} nodes");
    // One distribution a parent, and a parent is any node with children: the
    // root plus every level but the last. So `1 + B + ... + B^(D-1)`, which is
    // seven at two by three — not three, which is what this line said until the
    // test disagreed with the code and the code turned out to be right.
    let internal: usize = (0..D).map(|l| B.pow(l as u32)).sum();
    assert_eq!(
        tree.qs.len(),
        internal,
        "a {B}-by-{D} tree should carry {internal} distributions"
    );

    for (i, n) in tree.nodes.iter().enumerate() {
        assert!((n.token as usize) < cfg.vocab_size, "node {i} drafted {}", n.token);
        assert!(n.q_of < tree.qs.len(), "node {i} points at distribution {}", n.q_of);
        if let Some(p) = n.parent {
            assert!(p < i, "node {i}'s parent {p} comes after it");
        }
    }
    // Siblings share, cousins do not.
    for a in 0..tree.nodes.len() {
        for b in 0..tree.nodes.len() {
            if a == b {
                continue;
            }
            let same_parent = tree.nodes[a].parent == tree.nodes[b].parent;
            let same_q = tree.nodes[a].q_of == tree.nodes[b].q_of;
            assert_eq!(
                same_parent, same_q,
                "nodes {a} and {b}: parents {:?}/{:?} but distributions {}/{}",
                tree.nodes[a].parent, tree.nodes[b].parent,
                tree.nodes[a].q_of, tree.nodes[b].q_of
            );
        }
    }
    // Every deepest-level node on its own lane, and every path the right length.
    let leaves: Vec<usize> = (0..tree.nodes.len())
        .filter(|i| tree.path(*i).len() == D)
        .collect();
    assert_eq!(leaves.len(), B.pow(D as u32), "wrong number of leaves");
    let mut lanes: Vec<usize> = leaves.iter().map(|i| tree.nodes[*i].lane).collect();
    lanes.sort_unstable();
    lanes.dedup();
    assert_eq!(
        lanes.len(),
        leaves.len(),
        "two leaves share a lane, so two paths write the same cache slot"
    );
    Ok(())
}
