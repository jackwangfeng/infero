//! Turning logits into a token.
//!
//! Sampling runs on the host: the logit vector is one 600 KB transfer per
//! token, which is a rounding error next to the forward pass, and it keeps the
//! penalty bookkeeping in ordinary Rust.

use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

#[derive(Debug, Clone)]
pub struct SamplingParams {
    /// 0.0 means greedy: always take the argmax.
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    /// Divides the logit of tokens already seen. 1.0 disables.
    pub repetition_penalty: f32,
    /// How far back the repetition penalty looks.
    pub repetition_window: usize,
    pub seed: Option<u64>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.8,
            top_k: 20,
            repetition_penalty: 1.05,
            repetition_window: 256,
            seed: None,
        }
    }
}

impl SamplingParams {
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            ..Default::default()
        }
    }

    pub fn is_greedy(&self) -> bool {
        self.temperature <= 0.0 || self.top_k == 1
    }
}

pub struct Sampler {
    params: SamplingParams,
    rng: StdRng,
    /// Scratch, reused across tokens to keep sampling allocation-free.
    candidates: Vec<(u32, f32)>,
    /// One bit per vocabulary entry, marking what the penalty touches.
    penalized: Vec<u64>,
}

impl Sampler {
    pub fn new(params: SamplingParams) -> Self {
        let rng = match params.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_os_rng(),
        };
        Self {
            params,
            rng,
            candidates: Vec::new(),
            penalized: Vec::new(),
        }
    }

    /// The slice of `history` the repetition penalty actually reads.
    pub fn window<'a>(&self, history: &'a [u32]) -> &'a [u32] {
        if self.params.repetition_penalty == 1.0 {
            return &[];
        }
        &history[history.len().saturating_sub(self.params.repetition_window)..]
    }

    pub fn params(&self) -> &SamplingParams {
        &self.params
    }

    /// Pick the next token. `history` is the tokens generated so far, used for
    /// the repetition penalty.
    pub fn sample(&mut self, logits: &[f32], history: &[u32]) -> u32 {
        let r = self.next_draw();
        self.sample_with_draw(logits, history, r)
    }

    /// The uniform draw this sequence's generator would use next.
    ///
    /// Split out because the device sampler needs the number on the host: the
    /// generator, its seed and its sequence stay exactly where they were, and
    /// only the selection moves. A greedy sampler never consumes it, so
    /// pulling one unconditionally costs a `u64` and changes no output.
    pub fn next_draw(&mut self) -> f64 {
        self.rng.random_range(0.0..1.0f64)
    }

    /// [`Sampler::sample`] with the draw supplied rather than taken.
    pub fn sample_with_draw(&mut self, logits: &[f32], history: &[u32], draw: f64) -> u32 {
        debug_assert!(!logits.is_empty());
        if self.params.is_greedy() {
            return self.greedy(logits, history);
        }
        let total = self.build_distribution(logits, history);
        Self::pick(&self.candidates, total, draw)
    }

    /// The distribution this sampler would draw from, and its total.
    ///
    /// `(token, weight)` pairs over the surviving support, *unnormalized* — a
    /// weight divided by the returned total is the token's probability. Left
    /// unnormalized so that [`Sampler::pick`] compares against the same numbers
    /// it always did; normalizing first is the same arithmetic in exact terms
    /// and not in float, and this function exists to be the single definition of
    /// what the sampler's distribution *is*.
    ///
    /// Speculative decoding needs exactly this. Its acceptance rule is a ratio
    /// of the target's probability to the draft's, and both have to be measured
    /// under the transformation the request asked for — temperature, top-k,
    /// top-p, repetition penalty. A second implementation of that pipeline for
    /// the speculative path would be a second chance to get it subtly wrong, and
    /// the failure would be a shifted output distribution: not a crash, not
    /// obviously worse text, just no longer the distribution the caller
    /// specified.
    ///
    /// Greedy has no distribution in this sense; callers must handle it
    /// separately, as `sample_with_draw` does.
    pub fn distribution(&mut self, logits: &[f32], history: &[u32]) -> (&[(u32, f32)], f64) {
        // A real assertion, not a debug one. Rejection sampling against a point
        // mass is a different acceptance rule, so a speculative path that
        // reached here with a greedy sampler would be applying the wrong rule —
        // and a `debug_assert` would let exactly that through in the build that
        // serves requests.
        assert!(
            !self.params.is_greedy(),
            "a greedy sampler has no distribution to draw from; the caller has \
             to choose the point-mass acceptance rule deliberately"
        );
        let total = self.build_distribution(logits, history);
        (&self.candidates, total)
    }

    /// Everything from the raw logits to the truncated, weighted support.
    fn build_distribution(&mut self, logits: &[f32], history: &[u32]) -> f64 {
        self.candidates.clear();
        self.candidates.extend(
            logits
                .iter()
                .copied()
                .enumerate()
                .map(|(i, l)| (i as u32, l)),
        );

        self.apply_repetition_penalty(history);

        // Partition, then sort only the survivors.
        //
        // A full sort here is O(V log V) over a 150k-entry vocabulary — around
        // 2.5M comparisons per token, per sequence. At a batch of 32 that is
        // more CPU time than the whole forward pass takes on the GPU, and it
        // is what caps continuous batching's throughput if left in.
        let k = self.params.top_k.clamp(1, self.candidates.len());
        if k < self.candidates.len() {
            self.candidates
                .select_nth_unstable_by(k - 1, |a, b| b.1.total_cmp(&a.1));
            self.candidates.truncate(k);
        }
        self.candidates.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

        // Softmax over the surviving logits, at temperature.
        let inv_t = 1.0 / self.params.temperature.max(1e-5);
        let max = self.candidates[0].1;
        let mut total = 0.0f64;
        for c in &mut self.candidates {
            let p = (((c.1 - max) * inv_t) as f64).exp();
            c.1 = p as f32;
            total += p;
        }

        // Nucleus: keep the shortest prefix whose mass reaches top_p.
        if self.params.top_p < 1.0 {
            let target = total * self.params.top_p.clamp(1e-4, 1.0) as f64;
            let mut acc = 0.0f64;
            let mut keep = 0usize;
            for c in &self.candidates {
                acc += c.1 as f64;
                keep += 1;
                if acc >= target {
                    break;
                }
            }
            self.candidates.truncate(keep.max(1));
            total = self.candidates.iter().map(|c| c.1 as f64).sum();
        }
        total
    }

    /// Choose from a distribution built by [`Sampler::distribution`].
    pub fn pick(dist: &[(u32, f32)], total: f64, draw: f64) -> u32 {
        let mut r = draw * total;
        for c in dist {
            r -= c.1 as f64;
            if r <= 0.0 {
                return c.0;
            }
        }
        // Float drift can leave `r` marginally positive; the last candidate is
        // the correct answer in that case.
        dist.last().map(|c| c.0).unwrap_or(0)
    }

    /// Argmax with the repetition penalty applied, without materializing the
    /// vocabulary.
    ///
    /// The general path below copies every logit into an indexed vector so the
    /// penalty can be written in place — a megabyte of writes per token per
    /// sequence for a penalty that touches a few dozen entries. At a batch of
    /// 32 that was 16.8 ms per step against a 44 ms forward pass, and the
    /// greedy fast path never ran because it required a penalty of exactly 1.0
    /// while the server's default is 1.05.
    ///
    /// The penalty only ever lowers a logit — positive ones are divided,
    /// negative ones multiplied — so a scan that substitutes the penalized
    /// value for penalized tokens gives the same answer as penalizing first.
    fn greedy(&mut self, logits: &[f32], history: &[u32]) -> u32 {
        let penalty = self.params.repetition_penalty;
        if penalty == 1.0 || history.is_empty() {
            return argmax(logits);
        }
        let start = history.len().saturating_sub(self.params.repetition_window);
        let recent = &history[start..];

        self.penalized.clear();
        self.penalized.resize(logits.len().div_ceil(64), 0u64);
        for &tok in recent {
            let t = tok as usize;
            if t < logits.len() {
                self.penalized[t / 64] |= 1u64 << (t % 64);
            }
        }

        let mut best = (0u32, f32::NEG_INFINITY);
        for (i, &l) in logits.iter().enumerate() {
            let v = if self.penalized[i / 64] & (1u64 << (i % 64)) != 0 {
                if l > 0.0 { l / penalty } else { l * penalty }
            } else {
                l
            };
            if v > best.1 {
                best = (i as u32, v);
            }
        }
        best.0
    }

    fn apply_repetition_penalty(&mut self, history: &[u32]) {
        let penalty = self.params.repetition_penalty;
        if penalty == 1.0 || history.is_empty() {
            return;
        }
        let start = history.len().saturating_sub(self.params.repetition_window);
        for &tok in &history[start..] {
            if let Some(c) = self.candidates.get_mut(tok as usize) {
                debug_assert_eq!(c.0, tok);
                // Dividing a negative logit by the penalty would reward the
                // token, so the sign decides the direction.
                c.1 = if c.1 > 0.0 {
                    c.1 / penalty
                } else {
                    c.1 * penalty
                };
            }
        }
    }
}

fn argmax(logits: &[f32]) -> u32 {
    let mut best = (0u32, f32::NEG_INFINITY);
    for (i, &l) in logits.iter().enumerate() {
        if l > best.1 {
            best = (i as u32, l);
        }
    }
    best.0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Selection must agree with a full sort, since the nucleus cut depends on
    /// the survivors being the genuine top-k in the right order.
    #[test]
    fn selection_picks_the_same_top_k_as_a_sort() {
        let logits: Vec<f32> = (0..5000)
            .map(|i| ((i * 7919) % 5000) as f32 * 0.01)
            .collect();
        let mut want: Vec<(u32, f32)> = logits
            .iter()
            .copied()
            .enumerate()
            .map(|(i, l)| (i as u32, l))
            .collect();
        want.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        want.truncate(20);

        let mut s = Sampler::new(SamplingParams {
            temperature: 1.0,
            top_k: 20,
            top_p: 1.0,
            repetition_penalty: 1.0,
            seed: Some(9),
            ..Default::default()
        });
        // Sampling 200 times can only ever return one of the true top 20.
        let allowed: std::collections::HashSet<u32> = want.iter().map(|c| c.0).collect();
        for _ in 0..200 {
            let got = s.sample(&logits, &[]);
            assert!(
                allowed.contains(&got),
                "sampled {got}, outside the true top 20"
            );
        }
    }

    #[test]
    fn greedy_takes_the_argmax() {
        let mut s = Sampler::new(SamplingParams::greedy());
        let logits = vec![0.1, 5.0, -2.0, 4.9];
        assert_eq!(s.sample(&logits, &[]), 1);
    }

    #[test]
    fn top_k_1_is_greedy_even_at_temperature() {
        let mut s = Sampler::new(SamplingParams {
            temperature: 2.0,
            top_k: 1,
            ..Default::default()
        });
        let logits = vec![0.1, 5.0, -2.0];
        for _ in 0..20 {
            assert_eq!(s.sample(&logits, &[]), 1);
        }
    }

    #[test]
    fn a_seed_makes_sampling_reproducible() {
        let logits: Vec<f32> = (0..64).map(|i| (i as f32) * 0.1).collect();
        let params = SamplingParams {
            seed: Some(42),
            temperature: 1.0,
            top_k: 64,
            top_p: 1.0,
            repetition_penalty: 1.0,
            ..Default::default()
        };
        let run = || {
            let mut s = Sampler::new(params.clone());
            (0..32).map(|_| s.sample(&logits, &[])).collect::<Vec<_>>()
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn top_p_confines_sampling_to_the_nucleus() {
        // One token holds nearly all the mass.
        let mut logits = vec![0.0f32; 100];
        logits[7] = 20.0;
        let mut s = Sampler::new(SamplingParams {
            temperature: 1.0,
            top_p: 0.9,
            top_k: 100,
            repetition_penalty: 1.0,
            seed: Some(1),
            ..Default::default()
        });
        for _ in 0..50 {
            assert_eq!(s.sample(&logits, &[]), 7);
        }
    }

    #[test]
    fn repetition_penalty_demotes_recent_tokens() {
        let logits = vec![1.0, 1.1, 1.0];
        let mut s = Sampler::new(SamplingParams {
            temperature: 0.0,
            repetition_penalty: 2.0,
            repetition_window: 8,
            ..Default::default()
        });
        // Without history the argmax is token 1; penalising it hands the pick
        // to a tie-broken neighbour.
        assert_eq!(s.sample(&logits, &[]), 1);
        assert_ne!(s.sample(&logits, &[1]), 1);
    }

    #[test]
    fn negative_logits_are_penalised_downward() {
        let logits = vec![-1.0, -1.5];
        let mut s = Sampler::new(SamplingParams {
            temperature: 0.0,
            repetition_penalty: 2.0,
            ..Default::default()
        });
        // Token 0 starts ahead. Penalising a negative logit has to move it
        // further from zero (-1.0 -> -2.0), not closer, so it must fall behind.
        assert_eq!(s.sample(&logits, &[]), 0);
        assert_eq!(s.sample(&logits, &[0]), 1);
    }
}
