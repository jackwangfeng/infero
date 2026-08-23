//! Stochastic speculation preserves the target's distribution.
//!
//! This is the property that makes speculative decoding legitimate rather than
//! merely fast. If it fails, nothing crashes and the text does not obviously
//! degrade — the server just stops sampling from the distribution the request
//! asked for, at whatever temperature and top-p the caller chose. So it is
//! checked two ways: exactly, on the acceptance rule in isolation, and
//! statistically, on the composition.

use std::collections::HashMap;

use tuili_model::{Sampler, SamplingParams};

fn params(temperature: f32, top_p: f32, top_k: usize, rep: f32) -> SamplingParams {
    SamplingParams {
        temperature,
        top_p,
        top_k,
        repetition_penalty: rep,
        ..Default::default()
    }
}

/// A small distribution over `n` tokens from a fixed seed.
fn logits(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 23) as f32 - 1.0) * 4.0
        })
        .collect()
}

/// The composition of "draft from q, accept with min(1, p/q), else draw from
/// (p-q)+" is a draw from p. Checked by simulation on a small vocabulary,
/// against the sampler's own distribution as the truth.
///
/// The residual is drawn by the engine's own `SpecOutcome`-side function, not by
/// a copy of it here. A test that reimplements the rule and then agrees with
/// itself is exactly the failure this project has been bitten by; the only part
/// this file reimplements is the *composition* — draft, accept, recover — which
/// is what is under test.
#[test]
fn the_composition_of_draft_and_acceptance_is_a_draw_from_the_target() {
    const VOCAB: usize = 24;
    const ROUNDS: usize = 400_000;

    for (t, top_p, top_k) in [(1.0f32, 1.0f32, usize::MAX), (0.7, 0.95, 12), (1.4, 0.8, 8)] {
        let target_logits = logits(VOCAB, 0xaaa);
        // A deliberately mismatched drafter, so acceptance is far from certain
        // and the residual path is exercised heavily.
        let draft_logits = logits(VOCAB, 0xbbb);

        let sp = params(t, top_p, top_k, 1.0);
        let mut s = Sampler::new(sp.clone());
        let (tdist, ttotal) = s.distribution(&target_logits, &[]);
        let target: Vec<(u32, f64)> = tdist
            .iter()
            .map(|(tok, w)| (*tok, *w as f64 / ttotal))
            .collect();
        let mut s2 = Sampler::new(sp.clone());
        let (ddist, dtotal) = s2.distribution(&draft_logits, &[]);
        let draft: Vec<(u32, f64)> = ddist
            .iter()
            .map(|(tok, w)| (*tok, *w as f64 / dtotal))
            .collect();

        let p_of = |d: &[(u32, f64)], tok: u32| {
            d.iter().find(|(x, _)| *x == tok).map(|(_, p)| *p).unwrap_or(0.0)
        };

        // A simple deterministic generator, so a failure is reproducible.
        let mut st = 0x2545F4914F6CDD1Du64;
        let mut next = || {
            st ^= st << 13;
            st ^= st >> 7;
            st ^= st << 17;
            (st >> 11) as f64 / (1u64 << 53) as f64
        };

        let mut counts: HashMap<u32, usize> = HashMap::new();
        for _ in 0..ROUNDS {
            // Draft one token from q.
            let mut r = next();
            let mut drafted = draft.last().unwrap().0;
            for (tok, p) in &draft {
                r -= p;
                if r <= 0.0 {
                    drafted = *tok;
                    break;
                }
            }
            let q = p_of(&draft, drafted);
            let p = p_of(&target, drafted);
            let emitted = if q > 0.0 && p / q >= next() {
                drafted
            } else {
                // The engine's own residual draw. `q` is subtracted at every
                // token there, not only the drafted one; subtracting it only
                // there treats a sampled draft as a point mass and the
                // composition stops reproducing p — measured at 0.0745 off on
                // one token, about a hundred standard errors.
                let tdist: Vec<(u32, f32)> =
                    target.iter().map(|(t, p)| (*t, *p as f32)).collect();
                let qdist: Vec<(u32, f32)> =
                    draft.iter().map(|(t, p)| (*t, *p as f32)).collect();
                tuili_model::Model::draw_residual(&tdist, 1.0, &qdist, next())
            };
            *counts.entry(emitted).or_default() += 1;
        }

        // Compare against the target's own probabilities. With 400k rounds the
        // standard error on a probability p is sqrt(p(1-p)/N) < 0.0008, so 0.006
        // is well over seven sigma for the largest bin and far looser for the
        // rest — tight enough that a wrong rule shows, loose enough not to be
        // flaky.
        let mut worst = 0.0f64;
        let mut worst_tok = 0u32;
        for (tok, p) in &target {
            let seen = *counts.get(tok).unwrap_or(&0) as f64 / ROUNDS as f64;
            if (seen - p).abs() > worst {
                worst = (seen - p).abs();
                worst_tok = *tok;
            }
        }
        assert!(
            worst < 0.006,
            "t={t} top_p={top_p} top_k={top_k}: token {worst_tok} came out at a \
             rate {worst:.4} away from the target's probability; the \
             composition is not sampling from the target"
        );

        // And the mismatched drafter must actually be mismatched, or the test
        // would pass with a broken residual path.
        let overlap: f64 = target
            .iter()
            .map(|(tok, p)| p.min(p_of(&draft, *tok)))
            .sum();
        assert!(
            overlap < 0.9,
            "the drafter agrees with the target {overlap:.2} of the time, so \
             this test barely exercises rejection"
        );
    }
}

/// A draft the target's truncation excludes must always be rejected, or
/// speculation would smuggle in tokens the request's top-k or top-p ruled out.
#[test]
fn a_draft_outside_the_targets_support_cannot_be_accepted() {
    const VOCAB: usize = 512;
    let l = logits(VOCAB, 0x777);
    let mut s = Sampler::new(params(0.7, 0.5, 4, 1.0));
    let (dist, _) = s.distribution(&l, &[]);
    let support: Vec<u32> = dist.iter().map(|(t, _)| *t).collect();
    assert!(support.len() <= 4, "top_k should bound this to four tokens");

    // Any token outside that support has probability zero under the request's
    // transformation, so `p_target / p_draft` is zero and no draw accepts it.
    let outside = (0..VOCAB as u32).find(|t| !support.contains(t)).unwrap();
    let p_target = dist
        .iter()
        .find(|(t, _)| *t == outside)
        .map(|(_, w)| *w)
        .unwrap_or(0.0);
    assert_eq!(
        p_target, 0.0,
        "token {outside} is outside the support and must carry no weight"
    );
}

/// Does the multi-candidate rule still emit exactly the target's distribution?
///
/// This is the correctness question for a tree draft. The tempting rule — run
/// several branches and keep whichever accepted — conditions on the outcome and
/// biases what comes out, however natural it looks. The rule that does not is the
/// one-candidate rule with the residual carried forward: reject `x_i` and the
/// target becomes `norm((p - q)+)`, and the next candidate is tested against
/// that.
///
/// Held to the same standard as the single-candidate test above: Monte Carlo
/// against the target's own probabilities, with a deliberately mismatched drafter
/// so rejection and the residual path carry most of the traffic. `B = 1` is
/// included because it has to reduce to the rule the engine already ships.
#[test]
fn the_multi_candidate_composition_is_a_draw_from_the_target() {
    const VOCAB: usize = 64;
    const ROUNDS: usize = 400_000;

    for b in [1usize, 2, 4] {
        for (t, top_p, top_k) in [(1.0f32, 1.0f32, usize::MAX), (0.7, 0.95, 12)] {
            let target_logits = logits(VOCAB, 0xccc);
            let draft_logits = logits(VOCAB, 0xddd);
            let sp = params(t, top_p, top_k, 1.0);

            let mut s = Sampler::new(sp.clone());
            let (tdist, ttotal) = s.distribution(&target_logits, &[]);
            let target: Vec<(u32, f64)> = tdist
                .iter()
                .map(|(tok, w)| (*tok, *w as f64 / ttotal))
                .collect();
            let mut s2 = Sampler::new(sp.clone());
            let (ddist, dtotal) = s2.distribution(&draft_logits, &[]);
            let draft: Vec<(u32, f64)> = ddist
                .iter()
                .map(|(tok, w)| (*tok, *w as f64 / dtotal))
                .collect();

            let tvec: Vec<(u32, f32)> = target.iter().map(|(x, p)| (*x, *p as f32)).collect();
            let qvec: Vec<(u32, f32)> = draft.iter().map(|(x, p)| (*x, *p as f32)).collect();

            let mut st = 0x9E3779B97F4A7C15u64;
            let mut next = || {
                st ^= st << 13;
                st ^= st >> 7;
                st ^= st << 17;
                (st >> 11) as f64 / (1u64 << 53) as f64
            };

            let mut counts: HashMap<u32, usize> = HashMap::new();
            for _ in 0..ROUNDS {
                // `b` candidates drawn i.i.d. from the draft — which is what a
                // tree's siblings are, all sampled from their parent's `q`.
                let cands: Vec<u32> = (0..b)
                    .map(|_| {
                        let mut r = next();
                        let mut pick = draft.last().unwrap().0;
                        for (tok, p) in &draft {
                            r -= p;
                            if r <= 0.0 {
                                pick = *tok;
                                break;
                            }
                        }
                        pick
                    })
                    .collect();
                let draws: Vec<f64> = (0..b).map(|_| next()).collect();
                let (_, emitted) = tuili_model::Model::accept_multi(
                    &tvec,
                    1.0,
                    &qvec,
                    &cands,
                    &draws,
                    next(),
                );
                *counts.entry(emitted).or_default() += 1;
            }

            // The same bound the single-candidate test uses: at 400k rounds the
            // standard error on a probability is under 0.0008, so 0.006 is seven
            // sigma on the largest bin.
            let mut worst = 0.0f64;
            let mut worst_tok = 0u32;
            for (tok, p) in &target {
                let seen = *counts.get(tok).unwrap_or(&0) as f64 / ROUNDS as f64;
                if (seen - p).abs() > worst {
                    worst = (seen - p).abs();
                    worst_tok = *tok;
                }
            }
            assert!(
                worst < 0.006,
                "b={b} t={t} top_p={top_p} top_k={top_k}: token {worst_tok} came \
                 out {worst:.4} away from the target's probability; the \
                 multi-candidate composition is not sampling from the target"
            );

            // And nothing outside the target's support may ever come out.
            for tok in counts.keys() {
                assert!(
                    target.iter().any(|(t2, _)| t2 == tok),
                    "b={b}: token {tok} was emitted but is outside the target's \
                     support"
                );
            }

            // And the rule this is *not*: test each candidate against the
            // original `p` and keep the first that passes. It looks like the
            // same thing and is the obvious way to extend a tree, so the reason
            // not to write it should be a measurement rather than a remark.
            if b > 1 {
                let mut st2 = 0x9E3779B97F4A7C15u64;
                let mut next2 = || {
                    st2 ^= st2 << 13;
                    st2 ^= st2 >> 7;
                    st2 ^= st2 << 17;
                    (st2 >> 11) as f64 / (1u64 << 53) as f64
                };
                let p_of = |d: &[(u32, f64)], tok: u32| {
                    d.iter().find(|(t2, _)| *t2 == tok).map(|(_, p)| *p).unwrap_or(0.0)
                };
                let mut naive: HashMap<u32, usize> = HashMap::new();
                for _ in 0..ROUNDS {
                    let cands: Vec<u32> = (0..b)
                        .map(|_| {
                            let mut r = next2();
                            let mut pick = draft.last().unwrap().0;
                            for (tok, p) in &draft {
                                r -= p;
                                if r <= 0.0 {
                                    pick = *tok;
                                    break;
                                }
                            }
                            pick
                        })
                        .collect();
                    let mut emitted = None;
                    for x in &cands {
                        let q = p_of(&draft, *x);
                        let p = p_of(&target, *x);
                        if q > 0.0 && p / q >= next2() {
                            emitted = Some(*x);
                            break;
                        }
                    }
                    let e = emitted.unwrap_or_else(|| {
                        tuili_model::Model::draw_residual(&tvec, 1.0, &qvec, next2())
                    });
                    *naive.entry(e).or_default() += 1;
                }
                let mut nworst = 0.0f64;
                for (tok, p) in &target {
                    let seen = *naive.get(tok).unwrap_or(&0) as f64 / ROUNDS as f64;
                    nworst = nworst.max((seen - p).abs());
                }
                assert!(
                    nworst > 0.006,
                    "b={b} t={t}: the naive rule came within {nworst:.4} of the \
                     target, so this test no longer distinguishes it from the \
                     residual-carrying one and the comment above is unearned"
                );
            }
        }
    }
}
