//! `sample_rows_topk_forced`: the device-resident draft loop's own top-`k`
//! extraction over already-scaled logits, with a forced-inclusion token --
//! see `sample_rows_topk_forced_f32`'s own doc comment in `sample.cu` for
//! why the forced token can fall outside the natural top-`k` and why it is
//! safe to force it in regardless. Checked against a plain host reference
//! computed directly from the same synthetic logits, in three shapes: the
//! forced token already inside the natural top-`k` (no replacement should
//! happen), deliberately outside it (replacement must happen and must not
//! disturb the row's own softmax anchor), and the real checkpoint's own
//! vocab width (248320) at a real batch width, so the two-stage split+merge
//! actually exercises more than one candidate slice.

use anyhow::Result;
use infero_cuda::Device;
use infero_kernels::{Kernels, Survivors};

fn synthetic_logits(n_rows: usize, vocab: usize, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n_rows * vocab)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (((s >> 40) as f32 / (1u64 << 23) as f32) - 1.0) * 8.0
        })
        .collect()
}

/// The host's own reading of what the kernel should produce: top-`k` by raw
/// value, with `forced` guaranteed present (replacing the weakest kept
/// entry if it was not already there), then a temperature-1 softmax over
/// exactly those `k` entries.
fn reference_topk_forced(
    row_logits: &[f32],
    forced: usize,
    top_k: usize,
) -> Vec<(u32, f32)> {
    let mut idx: Vec<usize> = (0..row_logits.len()).collect();
    idx.sort_by(|&a, &b| {
        row_logits[b]
            .partial_cmp(&row_logits[a])
            .unwrap()
            .then(a.cmp(&b))
    });
    let mut kept: Vec<usize> = idx.into_iter().take(top_k).collect();
    if !kept.contains(&forced) {
        *kept.last_mut().unwrap() = forced;
    }
    let mx = kept.iter().map(|&i| row_logits[i]).fold(f32::MIN, f32::max);
    let exps: Vec<f64> = kept.iter().map(|&i| ((row_logits[i] - mx) as f64).exp()).collect();
    let total: f64 = exps.iter().sum();
    kept.iter()
        .zip(&exps)
        .map(|(&i, &e)| (i as u32, (e / total) as f32))
        .collect()
}

fn run_case(kern: &Kernels, dev: &Device, n_rows: usize, vocab: usize, top_k: usize, seed: u64, forced_ids: &[i32]) -> Result<()> {
    let stream = dev.stream();
    let logits = synthetic_logits(n_rows, vocab, seed);
    let d_logits = stream.clone_htod(&logits)?;
    let d_forced = stream.clone_htod(forced_ids)?;

    // temperature=1.0 (already-scaled input), top_k as given, rep_penalty
    // irrelevant (pen_len is zero for every row).
    let mut params = vec![0f32; n_rows * 4];
    for r in 0..n_rows {
        params[r * 4] = 1.0; // temperature
        params[r * 4 + 1] = 1.0; // top_p (keep all k)
        params[r * 4 + 2] = f32::from_bits(top_k as u32);
        params[r * 4 + 3] = 1.0; // rep_penalty (no-op at 1.0)
    }
    let d_params = stream.clone_htod(&params)?;
    let pen_stride = 1usize;
    let zeros_i32 = vec![0i32; n_rows * pen_stride];
    let d_pen_tok = stream.clone_htod(&zeros_i32)?;
    let d_pen_cnt = stream.clone_htod(&zeros_i32)?;
    let zeros_len = vec![0i32; n_rows];
    let d_pen_len = stream.clone_htod(&zeros_len)?;

    let mut d_cand_v = stream.alloc_zeros::<f32>(n_rows * Kernels::SAMPLE_SPLITS * top_k)?;
    let mut d_cand_i = stream.alloc_zeros::<i32>(n_rows * Kernels::SAMPLE_SPLITS * top_k)?;
    let mut d_surv_id = stream.alloc_zeros::<u32>(n_rows * top_k)?;
    let mut d_surv_p = stream.alloc_zeros::<f32>(n_rows * top_k)?;
    let mut d_surv_len = stream.alloc_zeros::<i32>(n_rows)?;

    kern.sample_rows_topk_forced(
        &mut d_cand_v.as_view_mut(),
        &mut d_cand_i.as_view_mut(),
        &d_logits.as_view(),
        &d_forced.as_view(),
        &d_params.as_view(),
        &d_pen_tok.as_view(),
        &d_pen_cnt.as_view(),
        &d_pen_len.as_view(),
        n_rows,
        vocab,
        pen_stride,
        top_k,
        Survivors {
            id: &mut d_surv_id.as_view_mut(),
            p: &mut d_surv_p.as_view_mut(),
            len: &mut d_surv_len.as_view_mut(),
            stride: top_k,
        },
    )?;

    let got_id = stream.clone_dtoh(&d_surv_id)?;
    let got_p = stream.clone_dtoh(&d_surv_p)?;
    let got_len = stream.clone_dtoh(&d_surv_len)?;
    dev.synchronize()?;

    for r in 0..n_rows {
        let row_logits = &logits[r * vocab..(r + 1) * vocab];
        let want = reference_topk_forced(row_logits, forced_ids[r] as usize, top_k);
        assert_eq!(got_len[r] as usize, top_k, "row {r}: survivor count");

        let mut got: Vec<(u32, f32)> = (0..top_k)
            .map(|j| (got_id[r * top_k + j], got_p[r * top_k + j]))
            .collect();
        got.sort_by_key(|(id, _)| *id);
        let mut want_sorted = want.clone();
        want_sorted.sort_by_key(|(id, _)| *id);

        assert_eq!(
            got.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
            want_sorted.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
            "row {r}: survivor ids (forced={})",
            forced_ids[r]
        );
        for ((gid, gp), (wid, wp)) in got.iter().zip(&want_sorted) {
            assert_eq!(gid, wid);
            assert!(
                (gp - wp).abs() <= 1e-4,
                "row {r} token {gid}: prob {gp} vs reference {wp}"
            );
        }

        // The real invariant this kernel exists to guarantee: the forced
        // token always carries positive weight in its own row.
        let forced = forced_ids[r] as u32;
        assert!(
            got.iter().any(|(id, p)| *id == forced && *p > 0.0),
            "row {r}: forced token {forced} missing positive weight, got {got:?}"
        );
    }
    Ok(())
}

#[test]
fn forced_token_already_in_natural_topk_is_unchanged() -> Result<()> {
    let dev = match Device::new(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skipping: no cuda device ({e})");
            return Ok(());
        }
    };
    let kern = Kernels::new(dev.clone());
    let (n_rows, vocab, top_k) = (4usize, 512usize, 8usize);
    let logits = synthetic_logits(n_rows, vocab, 0xF00D);
    // Force each row's own true top-1 (argmax) -- guaranteed already inside
    // any real top-k, so the kernel's "found" branch (no replacement) runs.
    let forced: Vec<i32> = (0..n_rows)
        .map(|r| {
            let row = &logits[r * vocab..(r + 1) * vocab];
            row.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i as i32)
                .unwrap()
        })
        .collect();
    run_case(&kern, &dev, n_rows, vocab, top_k, 0xF00D, &forced)
}

#[test]
fn forced_token_outside_natural_topk_gets_inserted() -> Result<()> {
    let dev = match Device::new(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skipping: no cuda device ({e})");
            return Ok(());
        }
    };
    let kern = Kernels::new(dev.clone());
    let (n_rows, vocab, top_k) = (4usize, 512usize, 8usize);
    let mut logits = synthetic_logits(n_rows, vocab, 0xACE0);
    // Pin a specific, deliberately-low-valued token per row as the "forced"
    // one -- well outside any real top-8 by construction.
    let forced: Vec<i32> = (0..n_rows).map(|r| (17 + r * 3) as i32).collect();
    for (r, &f) in forced.iter().enumerate() {
        logits[r * vocab + f as usize] = -100.0 - r as f32;
    }
    run_case(&kern, &dev, n_rows, vocab, top_k, 0xACE0, &forced)
}

#[test]
fn real_vocab_width_multi_split_shape() -> Result<()> {
    let dev = match Device::new(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skipping: no cuda device ({e})");
            return Ok(());
        }
    };
    let kern = Kernels::new(dev.clone());
    // The checkpoint's own real vocab (248320) and a real batch*k width
    // (n=16 sequences, k up to 3 draft steps accumulated -> 48 rows).
    let (n_rows, vocab, top_k) = (48usize, 248320usize, 40usize);
    let mut logits = synthetic_logits(n_rows, vocab, 0x5EED);
    let forced: Vec<i32> = (0..n_rows).map(|r| (100 + r * 37) as i32 % vocab as i32).collect();
    for (r, &f) in forced.iter().enumerate() {
        // Half the rows: forced token deliberately outside the top-k. Half:
        // leave it wherever it naturally lands (exercises both branches at
        // the real shape in one pass).
        if r % 2 == 0 {
            logits[r * vocab + f as usize] = -1000.0;
        }
    }
    run_case(&kern, &dev, n_rows, vocab, top_k, 0x5EED, &forced)
}
