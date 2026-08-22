//! `Sampler::distribution` is the same pipeline `sample_with_draw` runs.
//!
//! Speculative decoding needs the probability the request's own transformation
//! assigns to a token — temperature, top-k, top-p and repetition penalty all
//! applied. Writing that pipeline a second time for the speculative path would
//! be a second chance to get it subtly wrong, and the failure mode is not a
//! crash or visibly worse text: it is an output distribution that is no longer
//! the one the caller asked for.
//!
//! So `distribution` and `sample_with_draw` share one implementation, and these
//! tests are what says the extraction did not change what the sampler does.

use tuili_model::{Sampler, SamplingParams};

/// Deterministic logits with a wide dynamic range, so top-k and top-p both bite.
fn logits(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 23) as f32 - 1.0) * 8.0
        })
        .collect()
}

fn params(temperature: f32, top_p: f32, top_k: usize, rep: f32) -> SamplingParams {
    SamplingParams {
        temperature,
        top_p,
        top_k,
        repetition_penalty: rep,
        ..Default::default()
    }
}

/// Drawing from the extracted distribution picks exactly what the sampler picks,
/// for every combination that changes the pipeline's shape.
#[test]
fn drawing_from_the_distribution_picks_what_the_sampler_picks() {
    let vocab = 4096;
    let l = logits(vocab, 0xd15);
    // A history with repeats, so the penalty has something to act on.
    let history: Vec<u32> = (0..64).map(|i| (i * 37 % vocab as u32)).collect();

    for (t, p, k, rep) in [
        (1.0f32, 1.0f32, usize::MAX, 1.0f32),
        (0.7, 1.0, usize::MAX, 1.0),
        (0.7, 0.9, usize::MAX, 1.0),
        (0.7, 0.9, 40, 1.0),
        (0.7, 0.9, 40, 1.05),
        (1.3, 0.5, 8, 1.2),
        (0.05, 0.99, 1000, 1.05),
    ] {
        for draw in [0.0f64, 0.001, 0.25, 0.5, 0.75, 0.999_999] {
            let sp = params(t, p, k, rep);
            let mut a = Sampler::new(sp.clone());
            let want = a.sample_with_draw(&l, &history, draw);

            let mut b = Sampler::new(sp.clone());
            let (dist, total) = b.distribution(&l, &history);
            let got = Sampler::pick(dist, total, draw);

            assert_eq!(
                got, want,
                "t={t} top_p={p} top_k={k} rep={rep} draw={draw}: the \
                 distribution picked {got} where the sampler picked {want}"
            );
        }
    }
}

/// The distribution's weights sum to its total, and every token in it is a real
/// token. A ratio-based acceptance rule divides by these, so a stale total or a
/// weight from a truncated tail would corrupt the ratio rather than fail.
#[test]
fn the_distribution_is_normalizable_and_its_support_is_sane() {
    let vocab = 2048;
    let l = logits(vocab, 0x5a5a);
    let mut s = Sampler::new(params(0.7, 0.9, 50, 1.05));
    let (dist, total) = s.distribution(&l, &[]);

    assert!(!dist.is_empty(), "an empty support cannot be drawn from");
    assert!(dist.len() <= 50, "top_k = 50 should bound the support");
    let sum: f64 = dist.iter().map(|c| c.1 as f64).sum();
    assert!(
        (sum - total).abs() <= 1e-6 * total.max(1e-30),
        "the weights sum to {sum} against a reported total of {total}"
    );
    assert!(
        dist.iter().all(|c| c.1 > 0.0 && (c.0 as usize) < vocab),
        "every entry must be a positive weight on a real token"
    );
    // Sorted descending, which the nucleus truncation relies on.
    assert!(
        dist.windows(2).all(|w| w[0].1 >= w[1].1),
        "the support has to be in descending weight order"
    );
    // And the mass is at most top_p of what it was before truncation, give or
    // take the one entry that crosses the threshold.
    let mut all = Sampler::new(params(0.7, 1.0, 50, 1.05));
    let (_, full) = all.distribution(&l, &[]);
    assert!(total <= full, "truncation cannot add mass");
}

/// A greedy sampler has no distribution, and asking for one is a caller bug
/// rather than something to paper over: rejection sampling against a point mass
/// is a different rule, and the speculative path has to choose it deliberately.
#[test]
#[should_panic(expected = "greedy sampler has no distribution")]
fn a_greedy_sampler_refuses_to_hand_out_a_distribution() {
    let mut s = Sampler::new(SamplingParams::greedy());
    let l = logits(64, 1);
    let _ = s.distribution(&l, &[]);
}
