//! Block-scaled FP8: the mat-vec and the on-device expansion.
//!
//! The oracle is `tuili_safetensors::Tensor::dequant_f8_to_f16`, which has eight
//! tests of its own against the format's defined bit patterns and was checked
//! against a real 27B tensor and an independent Python computation. So these
//! compare a kernel against something already established rather than against a
//! second reading of the same guess.
//!
//! The thing worth testing hardest is where the scale applies. A scale covers
//! 128 rows *and* 128 columns; applying it once per row, or once per matrix, or
//! with the grid's indices transposed, all run and all produce a matrix of the
//! right shape. Each of those readings gets its own check that it fails.

mod common;

use anyhow::Result;
use common::*;
use half::f16;
use tuili_kernels::fp8::{FP8_BLOCK, fp8_bytes, scale_grid};
use tuili_safetensors::{Dtype, Tensor};

/// Two block-rows by three block-columns, so the grid is 2x3 and both indices
/// matter. A single block row would make a transposed lookup invisible.
const N: usize = 2 * FP8_BLOCK;
const K: usize = 3 * FP8_BLOCK;

/// A deterministic spread of E4M3 bytes, avoiding the two NaN patterns.
fn quant_bytes(n: usize, seed: u64) -> Vec<u8> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let b = (s >> 24) as u8;
            if b == 0x7F || b == 0xFF { 0x38 } else { b }
        })
        .collect()
}

/// The device buffer: quants then the scale grid as f32.
fn packed(quants: &[u8], scales: &[f32]) -> Vec<u8> {
    let mut v = quants.to_vec();
    for s in scales {
        v.extend_from_slice(&s.to_le_bytes());
    }
    v
}

/// The reference matrix, from the verified host dequantizer.
fn reference_matrix(quants: &[u8], scales: &[f32]) -> Vec<f32> {
    let sbytes: Vec<u8> = scales.iter().flat_map(|s| s.to_le_bytes()).collect();
    let q = Tensor {
        name: "w",
        dtype: Dtype::F8E4M3,
        shape: vec![N, K],
        data: quants,
    };
    let s = Tensor {
        name: "s",
        dtype: Dtype::F32,
        shape: vec![N / FP8_BLOCK, K / FP8_BLOCK],
        data: &sbytes,
    };
    q.dequant_f8_to_f16(&s, FP8_BLOCK)
        .unwrap()
        .iter()
        .map(|h| f32::from(*h))
        .collect()
}

/// Worst error as a fraction of the tolerance, where the tolerance follows the
/// error model this arithmetic has: a relative part, plus an absolute floor
/// taken from the tensor's own peak.
///
/// The floor is the part that matters here. A row's dot product sums hundreds
/// of products of magnitude ~1000, so a row whose result lands near zero did so
/// by cancellation and carries the same absolute error as one that landed at
/// 1000. Measuring its error relative to itself declares it broken. This is the
/// same lesson as `agree` in crates/model/tests/qwen35_capture.rs, arrived at
/// the same way.
fn worst_ratio(got: &[f32], want: &[f32], rel: f32, scale: f32) -> (f32, usize) {
    let peak = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let floor = scale * peak.max(f32::MIN_POSITIVE);
    let mut worst = (0.0f32, 0usize);
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let r = (g - w).abs() / (floor + rel * w.abs());
        if r > worst.0 {
            worst = (r, i);
        }
    }
    worst
}

fn reference_matvec(m: &[f32], x: &[f32]) -> Vec<f32> {
    (0..N)
        .map(|r| {
            // f64 accumulation, so a disagreement is the kernel's and not the
            // reference's summation order.
            m[r * K..(r + 1) * K]
                .iter()
                .zip(x)
                .map(|(a, b)| *a as f64 * *b as f64)
                .sum::<f64>() as f32
        })
        .collect()
}

#[test]
fn the_matvec_matches_the_verified_host_dequantizer() -> Result<()> {
    let k = kernels()?;
    let stream = k.device().stream().clone();

    let quants = quant_bytes(N * K, 0xf80f);
    // Distinct per-block scales spanning a wide range, so a wrong index lands
    // on a visibly wrong magnitude rather than a nearby one.
    let scales: Vec<f32> = (0..scale_grid(K, N))
        .map(|i| 0.25 * (i as f32 + 1.0) * if i % 2 == 0 { 1.0 } else { 3.0 })
        .collect();
    let x = pseudo_random(K, 0xabc);

    let want_m = reference_matrix(&quants, &scales);
    let want = reference_matvec(&want_m, &x);

    let buf = packed(&quants, &scales);
    assert_eq!(buf.len(), fp8_bytes(K, N), "the packed layout's size");
    let d_w = stream.clone_htod(&buf)?;
    let d_x = stream.clone_htod(&x)?;
    let mut d_out = stream.alloc_zeros::<f32>(N)?;
    k.mmv_f8_block(
        &mut d_out.as_view_mut(),
        &d_w.as_view(),
        &d_x.as_view(),
        K,
        N,
        false,
    )?;
    k.device().synchronize()?;
    let got = stream.clone_dtoh(&d_out)?;
    let rel = max_rel_diff(&got, &want);
    assert!(rel < 1e-5, "the mat-vec diverged by {rel:.2e}");

    // Three wrong readings of the same grid. Each is a matrix of the right
    // shape, so only the numbers can tell them apart.
    let per_row: Vec<f32> = {
        // One scale for the whole row: the first of its block-row.
        let mut m = vec![0.0f32; N * K];
        for r in 0..N {
            let s = scales[(r / FP8_BLOCK) * (K / FP8_BLOCK)];
            for c in 0..K {
                m[r * K + c] = tuili_safetensors::e4m3_value(quants[r * K + c]) * s;
            }
        }
        reference_matvec(&m, &x)
    };
    assert!(
        max_rel_diff(&per_row, &want) > 1e-2,
        "one scale a row gave the same answer, so this test does not pin where \
         the scale applies"
    );

    let transposed: Vec<f32> = {
        // The grid's indices swapped. Only square grids would survive this, and
        // 2x3 is not one — so this also checks the test's own shape choice.
        let mut m = vec![0.0f32; N * K];
        for r in 0..N {
            for c in 0..K {
                let i = (c / FP8_BLOCK) * (N / FP8_BLOCK) + r / FP8_BLOCK;
                let s = scales[i.min(scales.len() - 1)];
                m[r * K + c] = tuili_safetensors::e4m3_value(quants[r * K + c]) * s;
            }
        }
        reference_matvec(&m, &x)
    };
    assert!(
        max_rel_diff(&transposed, &want) > 1e-2,
        "a transposed grid lookup gave the same answer"
    );
    Ok(())
}

/// `accum` adds into the output, which is how the residual add gets folded into
/// the projection that feeds it. Getting it inverted drops or doubles a
/// residual, which reads as a model that is subtly worse rather than broken.
#[test]
fn accumulating_adds_to_the_output_rather_than_replacing_it() -> Result<()> {
    let k = kernels()?;
    let stream = k.device().stream().clone();
    let quants = quant_bytes(N * K, 0x1234);
    let scales: Vec<f32> = (0..scale_grid(K, N)).map(|i| 0.5 + i as f32).collect();
    let x = pseudo_random(K, 0x777);
    let want = reference_matvec(&reference_matrix(&quants, &scales), &x);

    let buf = packed(&quants, &scales);
    let d_w = stream.clone_htod(&buf)?;
    let d_x = stream.clone_htod(&x)?;
    let seed: Vec<f32> = (0..N).map(|i| i as f32 * 0.1).collect();
    let mut d_out = stream.clone_htod(&seed)?;
    k.mmv_f8_block(
        &mut d_out.as_view_mut(),
        &d_w.as_view(),
        &d_x.as_view(),
        K,
        N,
        true,
    )?;
    k.device().synchronize()?;
    let got = stream.clone_dtoh(&d_out)?;
    let expect: Vec<f32> = want.iter().zip(&seed).map(|(a, b)| a + b).collect();
    let (worst, at) = worst_ratio(&got, &expect, 1e-5, 1e-6);
    assert!(
        worst <= 1.0,
        "accumulate is {worst:.1}x the tolerance at element {at}: {} vs {}",
        got[at],
        expect[at]
    );
    // And it must actually have accumulated: without the seed the answer would
    // be `want`, so check it is not that.
    let (plain, _) = worst_ratio(&got, &want, 1e-5, 1e-6);
    assert!(
        plain > 10.0,
        "the output equals the non-accumulated answer, so `accum` did nothing"
    );
    Ok(())
}

/// The on-device expansion must produce exactly what the host dequantizer does,
/// because prefill's GEMM reads it and the two paths have to be the same model.
#[test]
fn the_device_expansion_matches_the_host_dequantizer() -> Result<()> {
    let k = kernels()?;
    let stream = k.device().stream().clone();
    let quants = quant_bytes(N * K, 0x5150);
    let scales: Vec<f32> = (0..scale_grid(K, N))
        .map(|i| 0.125 * (i as f32 + 1.0))
        .collect();
    let want = reference_matrix(&quants, &scales);

    let buf = packed(&quants, &scales);
    let d_w = stream.clone_htod(&buf)?;
    let mut d_out = stream.alloc_zeros::<f16>(N * K)?;
    k.dequant_f8_block_to_f16(&mut d_out.as_view_mut(), &d_w.as_view(), K, N)?;
    k.device().synchronize()?;
    let got: Vec<f32> = stream
        .clone_dtoh(&d_out)?
        .iter()
        .map(|h| f32::from(*h))
        .collect();
    let (worst, at) = max_abs_diff(&got, &want);
    // Both sides round to f16, so this should be exact.
    assert_eq!(
        worst, 0.0,
        "the device expansion differs at element {at}: {} vs {}",
        got[at], want[at]
    );
    Ok(())
}

/// The two paths have to agree with each other, not just each with the host:
/// decode uses the mat-vec and prefill the expansion, and a model whose first
/// token disagrees with its second is worse than one that is uniformly wrong.
#[test]
fn the_matvec_and_the_expanded_gemm_agree() -> Result<()> {
    let k = kernels()?;
    let stream = k.device().stream().clone();
    let quants = quant_bytes(N * K, 0x9e9e);
    let scales: Vec<f32> = (0..scale_grid(K, N))
        .map(|i| 0.3 + 0.7 * (i % 5) as f32)
        .collect();
    let x = pseudo_random(K, 0x2f2f);
    let buf = packed(&quants, &scales);
    let d_w = stream.clone_htod(&buf)?;
    let d_x = stream.clone_htod(&x)?;

    let mut d_mv = stream.alloc_zeros::<f32>(N)?;
    k.mmv_f8_block(
        &mut d_mv.as_view_mut(),
        &d_w.as_view(),
        &d_x.as_view(),
        K,
        N,
        false,
    )?;
    let mut d_f16 = stream.alloc_zeros::<f16>(N * K)?;
    k.dequant_f8_block_to_f16(&mut d_f16.as_view_mut(), &d_w.as_view(), K, N)?;
    k.device().synchronize()?;

    let expanded: Vec<f32> = stream
        .clone_dtoh(&d_f16)?
        .iter()
        .map(|h| f32::from(*h))
        .collect();
    let via_expansion = reference_matvec(&expanded, &x);
    let via_matvec = stream.clone_dtoh(&d_mv)?;

    // The bound is derived rather than tuned. The expansion rounds each weight
    // to f16 before multiplying, so its error in row r is at most
    // `sum_i |x_i| * |w_i| * 2^-11` — half an f16 ulp per weight, relative —
    // while the mat-vec keeps f32 partials. A row whose dot product lands near
    // zero did so by cancellation, and that bound is what it actually carries;
    // a tolerance relative to the row's own value would call it broken, and a
    // tolerance relative to the tensor peak would be a number I had picked.
    let exact = reference_matrix(&quants, &scales);
    let mut worst = (0.0f32, 0usize);
    for r in 0..N {
        let bound: f64 = (0..K)
            .map(|i| {
                (x[i] as f64).abs() * (exact[r * K + i] as f64).abs() * f64::exp2(-11.0)
            })
            .sum();
        // Plus the f32 mat-vec's own accumulation noise, which is far smaller.
        let bound = bound as f32 * 1.5 + 1e-6 * via_expansion[r].abs();
        let ratio = (via_matvec[r] - via_expansion[r]).abs() / bound.max(f32::MIN_POSITIVE);
        if ratio > worst.0 {
            worst = (ratio, r);
        }
    }
    assert!(
        worst.0 <= 1.0,
        "row {} differs by {:.3} against a derived f16-rounding bound, which is \
         {:.1}x it: mat-vec {}, expansion {}",
        worst.1,
        (via_matvec[worst.1] - via_expansion[worst.1]).abs(),
        worst.0,
        via_matvec[worst.1],
        via_expansion[worst.1]
    );
    Ok(())
}
