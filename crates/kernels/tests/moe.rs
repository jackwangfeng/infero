//! The MoE kernels: the router's top-k, and the mat-vec that selects an expert
//! by a device-side index.
//!
//! `mmvq_moe` is checked against `mmvq` on the same bytes rather than against a
//! reimplementation of the dot product. That is the sharper check available
//! here: the two share `tq_dot_*`, so anything they disagree about is the
//! expert offset — the one thing this kernel adds — and the answer has to be
//! bit-identical, not merely close.

use anyhow::Result;
use tuili_cuda::Device;
use tuili_kernels::awq::AwqTensor;
use tuili_kernels::{Kernels, WeightType};

/// AWQ codes that vary with the expert as well as with the position, so an
/// expert read at the wrong offset produces different numbers rather than
/// coincidentally similar ones.
fn synthetic(k: usize, n: usize, group: usize, salt: usize) -> (Vec<i32>, Vec<i32>, Vec<half::f16>) {
    let mut qweight = vec![0i32; k * n / 8];
    let mut qzeros = vec![0i32; k / group * n / 8];
    let mut scales = vec![half::f16::ZERO; k / group * n];
    const ORDER: [usize; 8] = [0, 2, 4, 6, 1, 3, 5, 7];

    for kk in 0..k {
        for col in 0..n / 8 {
            let mut word = 0u32;
            for (i, off) in ORDER.iter().enumerate() {
                let out = col * 8 + off;
                let code = ((kk * 7 + out * 3 + salt * 5 + kk / group) % 16) as u32;
                word |= code << (4 * i);
            }
            qweight[kk * (n / 8) + col] = word as i32;
        }
    }
    for g in 0..k / group {
        for col in 0..n / 8 {
            let mut word = 0u32;
            for (i, off) in ORDER.iter().enumerate() {
                let out = col * 8 + off;
                word |= (((g + out + salt) % 16) as u32) << (4 * i);
            }
            qzeros[g * (n / 8) + col] = word as i32;
        }
        for out in 0..n {
            scales[g * n + out] =
                half::f16::from_f32(0.002 + ((g * 5 + out + salt) % 23) as f32 * 0.0007);
        }
    }
    (qweight, qzeros, scales)
}

fn device() -> Option<Device> {
    match Device::new(0) {
        Ok(d) => Some(d),
        Err(e) => {
            eprintln!("skipping: no cuda device ({e})");
            None
        }
    }
}

#[test]
fn an_expert_is_the_dense_matvec_at_an_offset() -> Result<()> {
    let Some(dev) = device() else { return Ok(()) };
    let kern = Kernels::new(dev.clone());
    let stream = dev.stream();

    // The shapes an expert of Qwen3-30B-A3B actually has, plus a small one.
    for (k, n) in [(2048usize, 768usize), (768, 2048), (128, 64)] {
        for ty in [WeightType::Q4G128, WeightType::Q4G128T] {
            const N_EXPERTS: usize = 5;
            // Build each expert separately, then concatenate — which is what
            // the loader does, so the stride under test is the loader's.
            let mut per_expert: Vec<Vec<u8>> = Vec::new();
            for e in 0..N_EXPERTS {
                let (qweight, qzeros, scales) = synthetic(k, n, 128, e);
                let packed = AwqTensor {
                    qweight: &qweight,
                    qzeros: &qzeros,
                    scales: &scales,
                    in_features: k,
                    out_features: n,
                }
                .repack()?;
                per_expert.push(match ty {
                    WeightType::Q4G128T => tuili_kernels::awq::transpose_words(&packed, k, n),
                    _ => packed,
                });
            }
            let stride = per_expert[0].len();
            assert!(
                per_expert.iter().all(|b| b.len() == stride),
                "experts must encode to equal sizes or the stride is a lie"
            );
            let all: Vec<u8> = per_expert.concat();
            let d_all = stream.clone_htod(&all)?;

            let x: Vec<f32> = (0..k)
                .map(|i| ((i * 2654435761usize) % 401) as f32 / 200.0 - 1.0)
                .collect();
            let dx = stream.clone_htod(&x)?;
            let mut q8 = stream.alloc_zeros::<u8>(Kernels::q8_1_bytes(k))?;
            kern.quantize_q8_1(&mut q8.as_view_mut(), &dx.as_view(), k)?;

            // Route to a subset, out of order and with the last expert first,
            // so a kernel that ignored `expert_ids` and used `blockIdx.y`
            // directly would disagree.
            let ids: Vec<i32> = vec![4, 0, 3, 1];
            let d_ids = stream.clone_htod(&ids)?;
            let mut got = stream.alloc_zeros::<f32>(ids.len() * n)?;
            kern.mmvq_moe(
                &mut got.as_view_mut(),
                &d_all.as_view(),
                ty,
                &d_ids.as_view(),
                &q8.as_view(),
                k,
                n,
                ids.len(),
                stride,
                // Every slot reads the same activation row, which is the decode
                // case; `y_group = 1` would ask for four rows and read past the
                // one this test quantized.
                ids.len(),
            )?;
            let got = stream.clone_dtoh(&got)?;

            for (slot, &e) in ids.iter().enumerate() {
                let d_one = stream.clone_htod(&per_expert[e as usize])?;
                let mut want = stream.alloc_zeros::<f32>(n)?;
                kern.mmvq(
                    &mut want.as_view_mut(),
                    &d_one.as_view(),
                    ty,
                    &q8.as_view(),
                    k,
                    n,
                )?;
                let want = stream.clone_dtoh(&want)?;
                dev.synchronize()?;
                for r in 0..n {
                    assert_eq!(
                        got[slot * n + r].to_bits(),
                        want[r].to_bits(),
                        "{ty} k={k} n={n} slot {slot} (expert {e}) row {r}: \
                         {} against the dense mat-vec's {}",
                        got[slot * n + r],
                        want[r]
                    );
                }
            }
        }
    }
    Ok(())
}

/// The router, against a host softmax and sort.
#[test]
fn the_router_takes_the_top_k_of_a_softmax() -> Result<()> {
    let Some(dev) = device() else { return Ok(()) };
    let kern = Kernels::new(dev.clone());
    let stream = dev.stream();

    const N_EXPERTS: usize = 128;
    const K: usize = 8;
    const TOKENS: usize = 3;

    let logits: Vec<f32> = (0..TOKENS * N_EXPERTS)
        .map(|i| (((i * 37) % 71) as f32 / 7.0) - 5.0)
        .collect();
    let d_logits = stream.clone_htod(&logits)?;

    for norm in [true, false] {
        let mut d_ids = stream.alloc_zeros::<i32>(TOKENS * K)?;
        let mut d_w = stream.alloc_zeros::<f32>(TOKENS * K)?;
        kern.moe_topk(
            &mut d_ids.as_view_mut(),
            &mut d_w.as_view_mut(),
            &d_logits.as_view(),
            N_EXPERTS,
            K,
            TOKENS,
            norm,
        )?;
        let ids = stream.clone_dtoh(&d_ids)?;
        let w = stream.clone_dtoh(&d_w)?;
        dev.synchronize()?;

        for t in 0..TOKENS {
            let row = &logits[t * N_EXPERTS..(t + 1) * N_EXPERTS];
            let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = row.iter().map(|v| (v - max).exp()).collect();
            let sum_all: f32 = exps.iter().sum();
            // `softmax` then `topk`, which on a monotone transform is the same
            // order as `topk` on the logits — stated here because the kernel
            // relies on it.
            let mut order: Vec<usize> = (0..N_EXPERTS).collect();
            order.sort_by(|&a, &b| {
                row[b].partial_cmp(&row[a]).unwrap().then(a.cmp(&b))
            });
            let top = &order[..K];
            let sum_top: f32 = top.iter().map(|&e| exps[e]).sum();
            let denom = if norm { sum_top } else { sum_all };

            for a in 0..K {
                assert_eq!(
                    ids[t * K + a] as usize, top[a],
                    "norm={norm} token {t} slot {a}: expert {} against {}",
                    ids[t * K + a], top[a]
                );
                let want = exps[top[a]] / denom;
                assert!(
                    (w[t * K + a] - want).abs() < 1e-6,
                    "norm={norm} token {t} slot {a}: weight {} against {want}",
                    w[t * K + a]
                );
            }
            if norm {
                let total: f32 = (0..K).map(|a| w[t * K + a]).sum();
                assert!(
                    (total - 1.0).abs() < 1e-5,
                    "renormalized weights sum to {total}"
                );
            }
        }
    }
    Ok(())
}

/// The combine, and that it is a plain weighted sum rather than an average.
#[test]
fn the_combine_is_a_weighted_sum_over_the_active_experts() -> Result<()> {
    let Some(dev) = device() else { return Ok(()) };
    let kern = Kernels::new(dev.clone());
    let stream = dev.stream();

    const D: usize = 96;
    const K: usize = 4;
    const TOKENS: usize = 2;

    let partials: Vec<f32> = (0..TOKENS * K * D)
        .map(|i| ((i % 17) as f32) * 0.25 - 2.0)
        .collect();
    let weights: Vec<f32> = (0..TOKENS * K).map(|i| 0.1 + (i as f32) * 0.05).collect();

    let d_p = stream.clone_htod(&partials)?;
    let d_w = stream.clone_htod(&weights)?;
    let mut d_out = stream.alloc_zeros::<f32>(TOKENS * D)?;
    kern.moe_combine(
        &mut d_out.as_view_mut(),
        &d_p.as_view(),
        &d_w.as_view(),
        D,
        K,
        TOKENS,
    )?;
    let got = stream.clone_dtoh(&d_out)?;
    dev.synchronize()?;

    for t in 0..TOKENS {
        for c in 0..D {
            let want: f32 = (0..K)
                .map(|a| weights[t * K + a] * partials[(t * K + a) * D + c])
                .sum();
            assert!(
                (got[t * D + c] - want).abs() < 1e-5,
                "token {t} col {c}: {} against {want}",
                got[t * D + c]
            );
        }
    }
    Ok(())
}
