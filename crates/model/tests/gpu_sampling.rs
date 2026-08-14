//! The device sampler against the host one it replaces.
//!
//! Sampling moved to the GPU to stop copying 16 MiB of logits per step, which
//! is only worth anything if it picks the same token. That is not a given: the
//! host has two repetition-penalty paths that disagree with each other — greedy
//! penalizes a distinct token once, the general path once per occurrence — and
//! reproducing the wrong one would show up as a quality change nobody could
//! attribute to a sampling rewrite.
//!
//! Both sides get the same uniform draw, so a disagreement is a disagreement
//! about selection rather than about randomness.

use anyhow::Result;
use tuili_cuda::Device;
use tuili_kernels::Kernels;
use tuili_model::{Sampler, SamplingParams};

/// A vocabulary big enough for the bitset to span many words and for the
/// strided block scan to wrap, without making the test slow.
const VOCAB: usize = 32_000;

fn logits_for(row: usize, n: usize) -> Vec<f32> {
    // A bijection mod `n`, so every logit is distinct.
    //
    // Ties are deliberately absent here. The host breaks them with
    // `select_nth_unstable_by` followed by an unstable sort, which does not
    // define which of two equal logits survives the top-k cut, so a tie
    // straddling that boundary makes the two implementations legitimately
    // disagree. Greedy is the case where the host *does* pin the tie — first
    // index wins — and `the_device_greedy_tie_break_matches` covers it.
    (0..n)
        .map(|i| {
            let x = (i * 7919 + row * 104_729) % n;
            (x as f32 / n as f32 - 0.5) * 12.0
        })
        .collect()
}

/// Sorted unique ids and counts, laid out the way the kernel reads them.
fn window_tables(windows: &[Vec<u32>], stride: usize) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
    let mut tok = vec![0i32; windows.len() * stride];
    let mut cnt = vec![0i32; windows.len() * stride];
    let mut len = vec![0i32; windows.len()];
    for (i, w) in windows.iter().enumerate() {
        let mut s = w.clone();
        s.sort_unstable();
        let (mut m, mut j) = (0usize, 0usize);
        while j < s.len() {
            let t = s[j];
            let mut c = 0i32;
            while j < s.len() && s[j] == t {
                c += 1;
                j += 1;
            }
            tok[i * stride + m] = t as i32;
            cnt[i * stride + m] = c;
            m += 1;
        }
        len[i] = m as i32;
    }
    (tok, cnt, len)
}

#[test]
fn the_device_sampler_picks_what_the_host_sampler_picks() -> Result<()> {
    let Ok(dev) = Device::new(0) else {
        eprintln!("no CUDA device; skipping");
        return Ok(());
    };
    let kern = Kernels::new(dev.clone());
    let stream = dev.stream().clone();

    // Greedy and non-greedy, with and without a penalty, and a nucleus tight
    // enough to actually cut. `top_k = 1` is the other way `is_greedy` is true.
    let cases = [
        SamplingParams {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 20,
            repetition_penalty: 1.0,
            repetition_window: 256,
            seed: Some(1),
        },
        SamplingParams {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 20,
            repetition_penalty: 1.05,
            repetition_window: 256,
            seed: Some(2),
        },
        SamplingParams {
            temperature: 0.7,
            top_p: 0.8,
            top_k: 20,
            repetition_penalty: 1.05,
            repetition_window: 256,
            seed: Some(3),
        },
        SamplingParams {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 64,
            repetition_penalty: 1.3,
            repetition_window: 32,
            seed: Some(4),
        },
        SamplingParams {
            temperature: 0.9,
            top_p: 0.95,
            top_k: 1,
            repetition_penalty: 1.1,
            repetition_window: 256,
            seed: Some(5),
        },
    ];

    for (case, params) in cases.iter().enumerate() {
        let rows = 4usize;
        let mut all: Vec<f32> = Vec::with_capacity(rows * VOCAB);
        let mut windows: Vec<Vec<u32>> = Vec::new();
        for r in 0..rows {
            all.extend_from_slice(&logits_for(case * 8 + r, VOCAB));
            // Deliberate repeats, so the once-per-occurrence path differs from
            // the once-per-token one and the test can tell them apart.
            let w: Vec<u32> = (0..40u32)
                .map(|i| ((i * 13 + r as u32 * 7) % 200) * 11)
                .chain((0..25u32).map(|i| (i % 5) * 11))
                .collect();
            windows.push(w);
        }

        let mut samplers: Vec<Sampler> = (0..rows).map(|_| Sampler::new(params.clone())).collect();
        let draws: Vec<f64> = samplers.iter_mut().map(|s| s.next_draw()).collect();

        let want: Vec<u32> = (0..rows)
            .map(|r| {
                samplers[r].sample_with_draw(&all[r * VOCAB..(r + 1) * VOCAB], &windows[r], draws[r])
            })
            .collect();

        let stride = windows.iter().map(|w| w.len()).max().unwrap().max(1);
        // The host only penalizes inside the window, and not at all at 1.0.
        let effective: Vec<Vec<u32>> = windows
            .iter()
            .map(|w| {
                if params.repetition_penalty == 1.0 {
                    Vec::new()
                } else {
                    w[w.len().saturating_sub(params.repetition_window)..].to_vec()
                }
            })
            .collect();
        let (tok, cnt, len) = window_tables(&effective, stride);

        let mut pv = vec![0f32; rows * 4];
        for p in 0..rows {
            pv[p * 4] = params.temperature;
            pv[p * 4 + 1] = params.top_p;
            pv[p * 4 + 2] = f32::from_bits(params.top_k as u32);
            pv[p * 4 + 3] = params.repetition_penalty;
        }

        let d_logits = stream.clone_htod(&all)?;
        let d_params = stream.clone_htod(&pv)?;
        let d_tok = stream.clone_htod(&tok)?;
        let d_cnt = stream.clone_htod(&cnt)?;
        let d_len = stream.clone_htod(&len)?;
        let d_rnd = stream.clone_htod(&draws)?;
        let mut d_out = stream.alloc_zeros::<u32>(rows)?;

        kern.sample_rows(
            &mut d_out.as_view_mut(),
            &d_logits.as_view(),
            &d_params.as_view(),
            &d_tok.as_view(),
            &d_cnt.as_view(),
            &d_len.as_view(),
            &d_rnd.as_view(),
            rows,
            VOCAB,
            stride,
        )?;
        let got = stream.clone_dtoh(&d_out)?;
        dev.synchronize()?;

        for r in 0..rows {
            assert_eq!(
                got[r], want[r],
                "case {case} row {r}: device picked {} where the host picked {} \
                 (temp {}, top_p {}, top_k {}, penalty {})",
                got[r],
                want[r],
                params.temperature,
                params.top_p,
                params.top_k,
                params.repetition_penalty
            );
        }
    }
    Ok(())
}
