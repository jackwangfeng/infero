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
        
            None,
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

/// The split greedy argmax against the single-block kernel it replaces.
///
/// One block a row is 2% of a big card, so the greedy path now gives each row
/// thirty-two blocks and reduces the slice winners. It has to pick the same
/// token, and "the same" includes the tie-break: both passes order candidates
/// with `samp_better`, so the lowest index wins, and this test feeds duplicate
/// logits on purpose to say so.
#[test]
fn the_split_argmax_picks_what_one_block_picks() -> Result<()> {
    let dev = match Device::new(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skipping: no cuda device ({e})");
            return Ok(());
        }
    };
    let kern = Kernels::new(dev.clone());
    let stream = dev.stream();
    let rows = 5usize;
    let stride = 8usize;

    for (case, penalty) in [(0usize, 1.0f32), (1, 1.3)] {
        let mut all = Vec::with_capacity(rows * VOCAB);
        for r in 0..rows {
            let mut row = logits_for(r, VOCAB);
            // A plateau of equal maxima, so the tie-break is exercised: the
            // lowest index has to win whichever slice each copy lands in.
            let top = row.iter().cloned().fold(f32::MIN, f32::max);
            for j in [3usize, VOCAB / 3, VOCAB / 2, VOCAB - 7] {
                row[(j + r) % VOCAB] = top;
            }
            all.extend_from_slice(&row);
        }
        // A window whose tokens include some of the plateau, so the penalty
        // moves the answer between the two cases.
        let windows: Vec<Vec<u32>> = (0..rows)
            .map(|r| vec![3u32 + r as u32, (VOCAB / 2 + r) as u32, 17])
            .collect();
        let (tok, cnt, len) = window_tables(&windows, stride);

        let mut pv = vec![0f32; rows * 4];
        for p in 0..rows {
            pv[p * 4] = 0.0; // greedy
            pv[p * 4 + 1] = 1.0;
            pv[p * 4 + 2] = f32::from_bits(1);
            pv[p * 4 + 3] = penalty;
        }

        let d_logits = stream.clone_htod(&all)?;
        let d_params = stream.clone_htod(&pv)?;
        let d_tok = stream.clone_htod(&tok)?;
        let d_cnt = stream.clone_htod(&cnt)?;
        let d_len = stream.clone_htod(&len)?;
        let d_rnd = stream.clone_htod(&vec![0.5f64; rows])?;

        let mut one = stream.alloc_zeros::<u32>(rows)?;
        kern.sample_rows(
            &mut one.as_view_mut(),
            &d_logits.as_view(),
            &d_params.as_view(),
            &d_tok.as_view(),
            &d_cnt.as_view(),
            &d_len.as_view(),
            &d_rnd.as_view(),
            rows,
            VOCAB,
            stride,
            None,
        )?;

        let mut split = stream.alloc_zeros::<u32>(rows)?;
        let mut av = stream.alloc_zeros::<f32>(rows * Kernels::ARGMAX_SPLITS)?;
        let mut ai = stream.alloc_zeros::<i32>(rows * Kernels::ARGMAX_SPLITS)?;
        kern.sample_rows_greedy(
            &mut split.as_view_mut(),
            &mut av.as_view_mut(),
            &mut ai.as_view_mut(),
            &d_logits.as_view(),
            &d_params.as_view(),
            &d_tok.as_view(),
            &d_cnt.as_view(),
            &d_len.as_view(),
            rows,
            VOCAB,
            stride,
        )?;
        let (a, b) = (stream.clone_dtoh(&one)?, stream.clone_dtoh(&split)?);
        dev.synchronize()?;
        assert_eq!(
            b, a,
            "case {case} (penalty {penalty}): split argmax and one-block \
             disagree"
        );
    }
    Ok(())
}

/// Does the distribution the device drew from match the one the host builds?
///
/// The token alone is not enough for speculative decoding. The acceptance rule
/// is `min(1, p(x)/q(x))` and a rejection draws from the normalized `(p - q)+`,
/// so `q` has to be right as *numbers* over its whole truncated support — an
/// error there does not crash, it quietly stops reproducing the target's
/// distribution. That is why the draft now asks the kernel for the survivors
/// instead of redoing the sampling on the host, and why they need checking.
#[test]
fn the_device_survivors_match_the_host_distribution() -> Result<()> {
    let Ok(dev) = Device::new(0) else {
        eprintln!("no CUDA device; skipping");
        return Ok(());
    };
    let kern = Kernels::new(dev.clone());
    let stream = dev.stream().clone();

    // Non-greedy only: the survivors are what a *sampled* draft composes with,
    // and a greedy request takes the greedy acceptance rule instead.
    let cases = [
        SamplingParams {
            temperature: 0.7,
            top_p: 1.0,
            top_k: 20,
            repetition_penalty: 1.0,
            repetition_window: 256,
            seed: Some(11),
        },
        // A nucleus tight enough to cut, so `keep < top_k` is exercised.
        SamplingParams {
            temperature: 0.7,
            top_p: 0.6,
            top_k: 40,
            repetition_penalty: 1.05,
            repetition_window: 64,
            seed: Some(12),
        },
        SamplingParams {
            temperature: 1.2,
            top_p: 0.95,
            top_k: 64,
            repetition_penalty: 1.3,
            repetition_window: 32,
            seed: Some(13),
        },
    ];

    for (case, params) in cases.iter().enumerate() {
        let rows = 3usize;
        let mut all: Vec<f32> = Vec::with_capacity(rows * VOCAB);
        let mut windows: Vec<Vec<u32>> = Vec::new();
        for r in 0..rows {
            all.extend_from_slice(&logits_for(case * 16 + r + 3, VOCAB));
            let w: Vec<u32> = (0..40u32)
                .map(|i| ((i * 17 + r as u32 * 5) % 300) * 7)
                .chain((0..20u32).map(|i| (i % 4) * 7))
                .collect();
            windows.push(w);
        }

        let mut samplers: Vec<Sampler> = (0..rows).map(|_| Sampler::new(params.clone())).collect();
        let draws: Vec<f64> = samplers.iter_mut().map(|s| s.next_draw()).collect();

        let stride = windows.iter().map(|w| w.len()).max().unwrap().max(1);
        let effective: Vec<Vec<u32>> = windows
            .iter()
            .map(|w| {
                let n = params.repetition_window.min(w.len());
                w[w.len() - n..].to_vec()
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
        // The survivor buffers are `top_k` wide, which is what the draft sizes
        // them to.
        let sstride = params.top_k;
        let mut d_id = stream.alloc_zeros::<u32>(rows * sstride)?;
        let mut d_p = stream.alloc_zeros::<f32>(rows * sstride)?;
        let mut d_slen = stream.alloc_zeros::<i32>(rows)?;
        {
            let mut out_v = d_out.as_view_mut();
            let mut id_v = d_id.as_view_mut();
            let mut p_v = d_p.as_view_mut();
            let mut l_v = d_slen.as_view_mut();
            kern.sample_rows(
                &mut out_v,
                &d_logits.as_view(),
                &d_params.as_view(),
                &d_tok.as_view(),
                &d_cnt.as_view(),
                &d_len.as_view(),
                &d_rnd.as_view(),
                rows,
                VOCAB,
                stride,
                Some(tuili_kernels::Survivors {
                    id: &mut id_v,
                    p: &mut p_v,
                    len: &mut l_v,
                    stride: sstride,
                }),
            )?;
        }
        dev.synchronize()?;
        let got_id = stream.clone_dtoh(&d_id)?;
        let got_p = stream.clone_dtoh(&d_p)?;
        let got_len = stream.clone_dtoh(&d_slen)?;

        for r in 0..rows {
            let mut s = Sampler::new(params.clone());
            let _ = s.next_draw();
            let logits = &all[r * VOCAB..(r + 1) * VOCAB];
            let (dist, total) = s.distribution(logits, &windows[r]);
            let want: Vec<(u32, f32)> = dist
                .iter()
                .map(|(t, w)| (*t, (*w as f64 / total) as f32))
                .collect();

            let keep = got_len[r] as usize;
            assert_eq!(
                keep,
                want.len(),
                "case {case} row {r}: the device kept {keep} survivors, the host {}",
                want.len()
            );
            let mut mass = 0.0f64;
            for j in 0..keep {
                let (gi, gp) = (got_id[r * sstride + j], got_p[r * sstride + j]);
                let (wi, wp) = want[j];
                assert_eq!(
                    gi, wi,
                    "case {case} row {r} rank {j}: device id {gi}, host {wi}"
                );
                // Both normalize in f64 over the same survivors, so this is a
                // rounding difference and nothing else.
                assert!(
                    (gp - wp).abs() <= 1e-6 + 1e-5 * wp,
                    "case {case} row {r} rank {j} (id {gi}): device p {gp}, host {wp}"
                );
                mass += gp as f64;
            }
            assert!(
                (mass - 1.0).abs() < 1e-4,
                "case {case} row {r}: the survivors sum to {mass}, not one"
            );
        }
    }
    Ok(())
}

/// Does the split top-k pick and normalize what the single-block scan does?
///
/// The split version exists because the single-block one scans the vocabulary
/// once per survivor — forty passes over 248320 tokens at a typical `top_k`. The
/// claim it rests on is that a token in the global top-k is in its own slice's
/// top-k, so `top_k` candidates a slice loses nothing. If that were wrong the
/// symptom would be a survivor missing from deep in the tail, which changes the
/// distribution slightly and nothing else — so the two are compared entry for
/// entry rather than by their sampled token alone.
#[test]
fn the_split_sampler_agrees_with_the_single_block_one() -> Result<()> {
    let Ok(dev) = Device::new(0) else {
        eprintln!("no CUDA device; skipping");
        return Ok(());
    };
    let kern = Kernels::new(dev.clone());
    let stream = dev.stream().clone();

    let cases = [
        SamplingParams {
            temperature: 0.7,
            top_p: 1.0,
            top_k: 40,
            repetition_penalty: 1.0,
            repetition_window: 256,
            seed: Some(21),
        },
        // A penalty heavy enough to reorder the head of the distribution, which
        // is where a slice-local bitset could go wrong.
        SamplingParams {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            repetition_penalty: 1.4,
            repetition_window: 256,
            seed: Some(22),
        },
        // `top_k` above what any one slice holds is the padding path.
        SamplingParams {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 200,
            repetition_penalty: 1.05,
            repetition_window: 64,
            seed: Some(23),
        },
    ];

    for (case, params) in cases.iter().enumerate() {
        let rows = 3usize;
        let mut all: Vec<f32> = Vec::with_capacity(rows * VOCAB);
        let mut windows: Vec<Vec<u32>> = Vec::new();
        for r in 0..rows {
            all.extend_from_slice(&logits_for(case * 32 + r + 7, VOCAB));
            // A window that includes some of the very top logits, so the
            // penalty actually moves the ranking.
            let w: Vec<u32> = (0..60u32)
                .map(|i| ((i * 7 + r as u32 * 3) % 400) * 5)
                .chain((0..30u32).map(|i| (i % 6) * 5))
                .collect();
            windows.push(w);
        }
        let mut samplers: Vec<Sampler> = (0..rows).map(|_| Sampler::new(params.clone())).collect();
        let draws: Vec<f64> = samplers.iter_mut().map(|s| s.next_draw()).collect();

        let stride = windows.iter().map(|w| w.len()).max().unwrap().max(1);
        let effective: Vec<Vec<u32>> = windows
            .iter()
            .map(|w| {
                let n = params.repetition_window.min(w.len());
                w[w.len() - n..].to_vec()
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
        let ss = params.top_k;

        // Both paths, same inputs, same draws.
        let mut want_tok = stream.alloc_zeros::<u32>(rows)?;
        let mut want_id = stream.alloc_zeros::<u32>(rows * ss)?;
        let mut want_p = stream.alloc_zeros::<f32>(rows * ss)?;
        let mut want_len = stream.alloc_zeros::<i32>(rows)?;
        {
            let (mut o, mut i, mut p, mut l) = (
                want_tok.as_view_mut(),
                want_id.as_view_mut(),
                want_p.as_view_mut(),
                want_len.as_view_mut(),
            );
            kern.sample_rows(
                &mut o,
                &d_logits.as_view(),
                &d_params.as_view(),
                &d_tok.as_view(),
                &d_cnt.as_view(),
                &d_len.as_view(),
                &d_rnd.as_view(),
                rows,
                VOCAB,
                stride,
                Some(tuili_kernels::Survivors {
                    id: &mut i,
                    p: &mut p,
                    len: &mut l,
                    stride: ss,
                }),
            )?;
        }

        let mut got_tok = stream.alloc_zeros::<u32>(rows)?;
        let mut got_id = stream.alloc_zeros::<u32>(rows * ss)?;
        let mut got_p = stream.alloc_zeros::<f32>(rows * ss)?;
        let mut got_len = stream.alloc_zeros::<i32>(rows)?;
        let ck = params.top_k.max(1);
        let mut cv = stream.alloc_zeros::<f32>(rows * Kernels::SAMPLE_SPLITS * ck)?;
        let mut ci = stream.alloc_zeros::<i32>(rows * Kernels::SAMPLE_SPLITS * ck)?;
        {
            let (mut o, mut i, mut p, mut l) = (
                got_tok.as_view_mut(),
                got_id.as_view_mut(),
                got_p.as_view_mut(),
                got_len.as_view_mut(),
            );
            let (mut a, mut b) = (cv.as_view_mut(), ci.as_view_mut());
            kern.sample_rows_split(
                &mut o,
                &mut a,
                &mut b,
                &d_logits.as_view(),
                &d_params.as_view(),
                &d_tok.as_view(),
                &d_cnt.as_view(),
                &d_len.as_view(),
                &d_rnd.as_view(),
                rows,
                VOCAB,
                stride,
                params.top_k,
                Some(tuili_kernels::Survivors {
                    id: &mut i,
                    p: &mut p,
                    len: &mut l,
                    stride: ss,
                }),
            )?;
        }
        dev.synchronize()?;

        let (wt, gt) = (stream.clone_dtoh(&want_tok)?, stream.clone_dtoh(&got_tok)?);
        let (wi, gi) = (stream.clone_dtoh(&want_id)?, stream.clone_dtoh(&got_id)?);
        let (wp, gp) = (stream.clone_dtoh(&want_p)?, stream.clone_dtoh(&got_p)?);
        let (wl, gl) = (stream.clone_dtoh(&want_len)?, stream.clone_dtoh(&got_len)?);
        for r in 0..rows {
            assert_eq!(
                gl[r], wl[r],
                "case {case} row {r}: split kept {}, single block kept {}",
                gl[r], wl[r]
            );
            let keep = wl[r] as usize;
            for j in 0..keep {
                assert_eq!(
                    gi[r * ss + j],
                    wi[r * ss + j],
                    "case {case} row {r} rank {j}: split id {}, single block {}",
                    gi[r * ss + j],
                    wi[r * ss + j]
                );
                let (a, b) = (gp[r * ss + j], wp[r * ss + j]);
                assert!(
                    (a - b).abs() <= 1e-6 + 1e-5 * b,
                    "case {case} row {r} rank {j}: split p {a}, single block {b}"
                );
            }
            assert_eq!(
                gt[r], wt[r],
                "case {case} row {r}: split sampled {}, single block {}",
                gt[r], wt[r]
            );
        }
    }
    Ok(())
}
