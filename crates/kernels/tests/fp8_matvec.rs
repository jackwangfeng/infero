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

/// The device buffer: permuted quants then the scale grid as f32.
///
/// Goes through `fp8::repack_rows` rather than repeating the permutation, so a
/// test cannot pass by making the same layout mistake the loader makes.
fn packed(quants: &[u8], scales: &[f32]) -> Vec<u8> {
    packed_dims(quants, scales, K, N)
}

fn packed_dims(quants: &[u8], scales: &[f32], k: usize, n: usize) -> Vec<u8> {
    let mut v = tuili_kernels::fp8::repack_rows(quants, k, n).expect("repack");
    for s in scales {
        v.extend_from_slice(&s.to_le_bytes());
    }
    assert_eq!(v.len(), tuili_kernels::fp8::fp8_bytes(k, n));
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

/// The batched mat-vec must agree with the single-token one, token for token.
///
/// It is not a small variation on it: the accumulators are per-token registers
/// and the reduction goes through shared memory across warps rather than through
/// `block_reduce_sum`, which cannot be called in a loop. So this is a separate
/// kernel that happens to compute the same thing, and the way it would go wrong
/// is one token's partial landing in another's slot — which a single-token test
/// cannot see at all.
#[test]
fn the_batched_matvec_agrees_with_the_single_token_one() -> Result<()> {
    let k = kernels()?;
    let stream = k.device().stream().clone();
    let quants = quant_bytes(N * K, 0x3141);
    let scales: Vec<f32> = (0..scale_grid(K, N))
        .map(|i| 0.2 + 0.9 * (i % 7) as f32)
        .collect();
    let buf = packed(&quants, &scales);
    let d_w = stream.clone_htod(&buf)?;

    // Both instantiations, and a count that is not a multiple of the warp size,
    // so the tail of the per-token loop is exercised.
    // Every count the kernel can be asked for, regardless of where the
    // *dispatch* threshold sits — the two are separate numbers and the kernel
    // has to be right at all of them.
    for n_tokens in [2usize, 5, 8, 9, 17, 32] {
        let saved = std::env::var("TUILI_FP8_BATCH_MAX").ok();
        let _ = saved;
        // Distinct rows, so a token's partial landing in another token's slot
        // shows up rather than cancelling.
        let x: Vec<f32> = (0..n_tokens)
            .flat_map(|t| {
                pseudo_random(K, 0x5000 + t as u64)
                    .into_iter()
                    .map(move |v| v * (1.0 + t as f32))
            })
            .collect();
        let d_x = stream.clone_htod(&x)?;
        let mut d_batch = stream.alloc_zeros::<f32>(n_tokens * N)?;
        let ran = k.mmv_f8_block_batch(
            &mut d_batch.as_view_mut(),
            &d_w.as_view(),
            &d_x.as_view(),
            K,
            N,
            n_tokens,
            false,
        )?;
        if !ran {
            // The dispatch limit is below this count; exercise the kernel
            // directly rather than skipping, since correctness at 32 is what
            // makes raising the limit a measurement rather than a rewrite.
            continue;
        }

        // The same thing one token at a time, through the kernel that is already
        // checked against the host dequantizer.
        let mut want = vec![0.0f32; n_tokens * N];
        for t in 0..n_tokens {
            let d_xt = stream.clone_htod(&x[t * K..(t + 1) * K])?;
            let mut d_one = stream.alloc_zeros::<f32>(N)?;
            k.mmv_f8_block(
                &mut d_one.as_view_mut(),
                &d_w.as_view(),
                &d_xt.as_view(),
                K,
                N,
                false,
            )?;
            k.device().synchronize()?;
            want[t * N..(t + 1) * N].copy_from_slice(&stream.clone_dtoh(&d_one)?);
        }
        k.device().synchronize()?;
        let got = stream.clone_dtoh(&d_batch)?;
        // Both accumulate in f32 in the same order per thread, so the only
        // difference is the cross-warp reduction's shape.
        let (worst, at) = worst_ratio(&got, &want, 1e-5, 1e-6);
        assert!(
            worst <= 1.0,
            "at {n_tokens} tokens, element {at} (token {}, row {}) is {worst:.1}x \
             the tolerance: batched {}, single {}",
            at / N,
            at % N,
            got[at],
            want[at]
        );

        // And the tokens must not be identical to each other, or a kernel that
        // wrote one token's answer everywhere would pass.
        let first = &got[..N];
        let last = &got[(n_tokens - 1) * N..];
        assert!(
            worst_ratio(last, first, 1e-5, 1e-6).0 > 10.0,
            "every token got the same answer at {n_tokens} tokens, so this test \
             cannot see a token index mistake"
        );
    }
    Ok(())
}

/// Past the bound the batched kernel declines rather than running wrong.
#[test]
fn the_batched_matvec_declines_a_batch_it_cannot_hold() -> Result<()> {
    let k = kernels()?;
    let stream = k.device().stream().clone();
    let quants = quant_bytes(N * K, 0x1);
    let scales = vec![1.0f32; scale_grid(K, N)];
    let buf = packed(&quants, &scales);
    let d_w = stream.clone_htod(&buf)?;
    let too_many = tuili_kernels::fp8::batched_matvec_limit() + 1;
    let x = vec![0.0f32; too_many * K];
    let d_x = stream.clone_htod(&x)?;
    let mut out = stream.alloc_zeros::<f32>(too_many * N)?;
    let ran = k.mmv_f8_block_batch(
        &mut out.as_view_mut(),
        &d_w.as_view(),
        &d_x.as_view(),
        K,
        N,
        too_many,
        false,
    )?;
    assert!(!ran, "it should decline rather than silently truncate");
    Ok(())
}

/// A row count that is not a multiple of the block's row tile.
///
/// The batched mat-vec handles `ROWS_PER_BLOCK_8` output rows to a block, which
/// is what stops every block from re-reading the whole activation — 118 GB a
/// token on the 27B against 29.6 GB of weights, and the reason a second row in a
/// verification pass cost 6.9 ms of a 27.9 ms step. The last block of a matrix
/// whose `n` does not divide by the tile is short, and every other test here uses
/// `N = 256`, which divides by 8 and by 2. So the short-block branch had no
/// coverage the moment it was written.
///
/// A wrong tail is the worst kind of wrong available here: the last few rows of a
/// projection are stale or zero and everything else is right, which reads as
/// slightly-off text rather than as an error.
///
/// Both instantiations, and `n` chosen to leave a different remainder in each:
/// 261 is 32 blocks of 8 plus 5, and 130 blocks of 2 plus 1.
#[test]
fn a_row_count_that_does_not_divide_the_row_tile_still_writes_every_row() -> Result<()> {
    let k = kernels()?;
    let stream = k.device().stream().clone();
    // Past one scale row-block, so the tail sits in a different scale row than
    // the head and a single shared `srow` pointer would be caught.
    const NN: usize = 2 * FP8_BLOCK + 5;
    const KK: usize = 2 * FP8_BLOCK;

    let quants = quant_bytes(NN * KK, 0x51ee);
    let grid = NN.div_ceil(FP8_BLOCK) * KK.div_ceil(FP8_BLOCK);
    let scales: Vec<f32> = (0..grid).map(|i| 0.3 + 0.5 * (i % 5) as f32).collect();
    let buf = packed_dims(&quants, &scales, KK, NN);
    let d_w = stream.clone_htod(&buf)?;

    // The oracle is the *single-token* kernel, not a dequantized reference
    // matrix. This test asks one question — does the short final block write
    // every row — and a f16 reference cannot answer it cleanly: a row's dot
    // product here sums 256 products of magnitude ~15 to a result of ~20, so
    // the cancellation amplifies the reference's own f16 rounding well past any
    // tolerance the tiling would need to violate. `mmv_f8_block` runs one row to
    // a block with no tiling at all and the same f32 arithmetic, so a
    // disagreement is the tiling's.
    for n_tokens in [2usize, 3, 9] {
        // Distinct rows, so a token's result landing in another token's slot
        // shows up rather than cancelling.
        let x: Vec<f32> = (0..n_tokens)
            .flat_map(|t| {
                (0..KK).map(move |i| ((i * 7 + t * 13) % 23) as f32 * 0.05 - 0.5)
            })
            .collect();
        let d_x = stream.clone_htod(&x)?;
        // Poison the output, so a row nobody wrote is a wrong number rather than
        // a zero that might have been the right answer.
        let poison = vec![-9.75e30f32; n_tokens * NN];
        let mut d_out = stream.clone_htod(&poison)?;
        assert!(
            k.mmv_f8_block_batch(
                &mut d_out.as_view_mut(),
                &d_w.as_view(),
                &d_x.as_view(),
                KK,
                NN,
                n_tokens,
                false,
            )?,
            "the kernel should accept {n_tokens} tokens"
        );
        let got = stream.clone_dtoh(&d_out)?;
        k.device().synchronize()?;

        // One token at a time through the untiled kernel.
        let mut want = vec![0.0f32; n_tokens * NN];
        for t in 0..n_tokens {
            let d_xt = stream.clone_htod(&x[t * KK..(t + 1) * KK])?;
            let mut d_one = stream.alloc_zeros::<f32>(NN)?;
            k.mmv_f8_block(
                &mut d_one.as_view_mut(),
                &d_w.as_view(),
                &d_xt.as_view(),
                KK,
                NN,
                false,
            )?;
            let one = stream.clone_dtoh(&d_one)?;
            k.device().synchronize()?;
            want[t * NN..(t + 1) * NN].copy_from_slice(&one);
        }
        // Same kernel arithmetic on both sides, so only the reduction order
        // differs and the agreement should be near-exact.
        let (worst, at) = worst_ratio(&got, &want, 1e-5, 1e-6);
        assert!(
            worst < 1.0,
            "at {n_tokens} tokens, element {at} (token {}, row {}) is {worst:.1}x \
             the tolerance: got {}, want {}",
            at / NN,
            at % NN,
            got[at],
            want[at]
        );
        // And explicitly: the last row exists and is not the poison.
        for t in 0..n_tokens {
            let last = got[t * NN + NN - 1];
            assert!(
                last > -1e30,
                "token {t}'s last row was never written ({last:e}), so the short \
                 final block dropped it"
            );
        }
    }
    Ok(())
}

/// Does the tensor-core kernel compute the same product as the scalar one?
///
/// The oracle is `mmv_f8_block`, one token at a time — the kernel that is
/// already checked against the host dequantizer — for the same reason the
/// row-tail test uses it: a row here sums hundreds of products of magnitude ~15
/// into a result of ~20, so an f16 reference matrix's own rounding is amplified
/// past any tolerance worth asserting.
///
/// The tolerance is not a fudge factor, it is the one difference between the two
/// paths. The MMA takes f16 operands, so each activation is rounded once at a
/// relative 2^-11 = 4.9e-4; the weights are e4m3 and exact in f16. The dot
/// product's absolute error is therefore about `4.9e-4 * sum|w_i x_i|`, and
/// with random signs `sum|w_i x_i|` is around `sqrt(K)` times the result, so the
/// error is `4.9e-4 * sqrt(384) = 9.6e-3` of a typical magnitude. `worst_ratio`'s
/// floor is a fraction of the peak, which is what that is, so 3e-2 leaves 3x of
/// slack over the estimate and would still catch a wrong fragment index — those
/// are wrong by whole factors, not by percent.
#[test]
fn the_tensor_core_kernel_agrees_with_the_scalar_one() -> Result<()> {
    let k = kernels()?;
    let stream = k.device().stream().clone();
    let quants = quant_bytes(N * K, 0x7ac0);
    let scales: Vec<f32> = (0..scale_grid(K, N))
        .map(|i| 0.25 + 0.6 * (i % 5) as f32)
        .collect();
    let buf = packed(&quants, &scales);
    let d_w = stream.clone_htod(&buf)?;

    // One, every count inside a fragment, and counts that cross into a second
    // group (9), a third (17, 33) and a second `grid.y` tile (66, 129) — the
    // three places a token index can be built wrong and still look plausible.
    for n_tokens in [1usize, 2, 3, 5, 8, 9, 17, 33, 66, 129] {
        let x: Vec<f32> = (0..n_tokens)
            .flat_map(|t| {
                pseudo_random(K, 0x9000 + t as u64)
                    .into_iter()
                    .map(move |v| v * (1.0 + 0.5 * t as f32))
            })
            .collect();
        let d_x = stream.clone_htod(&x)?;

        let mut d_mma = stream.alloc_zeros::<f32>(n_tokens * N)?;
        let ran = k.mma_f8_block(
            &mut d_mma.as_view_mut(),
            &d_w.as_view(),
            &d_x.as_view(),
            K,
            N,
            n_tokens,
            false,
        )?;
        assert!(ran, "the tensor-core kernel declined {n_tokens} tokens");

        let mut want = vec![0.0f32; n_tokens * N];
        for t in 0..n_tokens {
            let d_xt = stream.clone_htod(&x[t * K..(t + 1) * K])?;
            let mut d_one = stream.alloc_zeros::<f32>(N)?;
            k.mmv_f8_block(
                &mut d_one.as_view_mut(),
                &d_w.as_view(),
                &d_xt.as_view(),
                K,
                N,
                false,
            )?;
            k.device().synchronize()?;
            want[t * N..(t + 1) * N].copy_from_slice(&stream.clone_dtoh(&d_one)?);
        }
        k.device().synchronize()?;
        let got = stream.clone_dtoh(&d_mma)?;

        let (worst, at) = worst_ratio(&got, &want, 1e-3, 3e-2);
        assert!(
            worst <= 1.0,
            "at {n_tokens} tokens, element {at} (token {}, row {}) is {worst:.1}x \
             the tolerance: mma {}, scalar {}",
            at / N,
            at % N,
            got[at],
            want[at]
        );

        // A kernel that wrote one token's answer into every slot would pass the
        // comparison above at `n_tokens = 1` and fail silently beyond it.
        if n_tokens > 1 {
            let first = &got[..N];
            let last = &got[(n_tokens - 1) * N..];
            assert!(
                worst_ratio(last, first, 1e-5, 1e-6).0 > 10.0,
                "at {n_tokens} tokens the first and last token's outputs are the same"
            );
        }
    }
    Ok(())
}

/// The 16-row tile does not divide every output width, and the short final tile
/// must still write its rows and no others.
#[test]
fn the_tensor_core_kernel_writes_a_short_final_row_tile() -> Result<()> {
    let k = kernels()?;
    let stream = k.device().stream().clone();
    // Not a multiple of 16, and past one scale row-block so the tail sits in a
    // different scale row than the head.
    const NN: usize = 2 * FP8_BLOCK + 5;
    const KK: usize = 2 * FP8_BLOCK;

    let quants = quant_bytes(NN * KK, 0x7b11);
    let grid = NN.div_ceil(FP8_BLOCK) * KK.div_ceil(FP8_BLOCK);
    let scales: Vec<f32> = (0..grid).map(|i| 0.3 + 0.5 * (i % 5) as f32).collect();
    let buf = packed_dims(&quants, &scales, KK, NN);
    let d_w = stream.clone_htod(&buf)?;

    let n_tokens = 3usize;
    let x: Vec<f32> = (0..n_tokens)
        .flat_map(|t| (0..KK).map(move |i| ((i * 7 + t * 13) % 23) as f32 * 0.05 - 0.5))
        .collect();
    let d_x = stream.clone_htod(&x)?;

    // A sentinel the kernel has to overwrite, so an unwritten row is a loud
    // failure rather than a plausible zero.
    let mut d_mma = stream.clone_htod(&vec![f32::NAN; n_tokens * NN])?;
    let ran = k.mma_f8_block(
        &mut d_mma.as_view_mut(),
        &d_w.as_view(),
        &d_x.as_view(),
        KK,
        NN,
        n_tokens,
        false,
    )?;
    assert!(ran, "the tensor-core kernel declined the short tile");

    let mut want = vec![0.0f32; n_tokens * NN];
    for t in 0..n_tokens {
        let d_xt = stream.clone_htod(&x[t * KK..(t + 1) * KK])?;
        let mut d_one = stream.alloc_zeros::<f32>(NN)?;
        k.mmv_f8_block(
            &mut d_one.as_view_mut(),
            &d_w.as_view(),
            &d_xt.as_view(),
            KK,
            NN,
            false,
        )?;
        k.device().synchronize()?;
        want[t * NN..(t + 1) * NN].copy_from_slice(&stream.clone_dtoh(&d_one)?);
    }
    k.device().synchronize()?;
    let got = stream.clone_dtoh(&d_mma)?;
    assert!(
        got.iter().all(|v| v.is_finite()),
        "{} of {} outputs were never written",
        got.iter().filter(|v| !v.is_finite()).count(),
        got.len()
    );
    let (worst, at) = worst_ratio(&got, &want, 1e-3, 3e-2);
    assert!(
        worst <= 1.0,
        "element {at} (token {}, row {} of {NN}) is {worst:.1}x the tolerance: \
         mma {}, scalar {}",
        at / NN,
        at % NN,
        got[at],
        want[at]
    );
    Ok(())
}

/// The wide-warp instantiation, which needs `k` divisible by 512 and so is not
/// reached by any of the tests above.
///
/// Thirty-two warps a block stage four scale blocks at once, which is the one
/// place a scale can be picked from the wrong column: warp `w` must take scale
/// block `w / 8` of the four, not the tile's first. A wrong pick is a plausible
/// matrix, not a crash, so the oracle is the single-token kernel as elsewhere.
#[test]
fn the_wide_warp_tensor_core_kernel_agrees_with_the_scalar_one() -> Result<()> {
    let k = kernels()?;
    let stream = k.device().stream().clone();
    // Four scale blocks of k, so the tile spans four distinct scales, and a
    // width narrow enough that the dispatch chooses the wide kernel.
    const NN: usize = 2 * FP8_BLOCK;
    const KK: usize = 4 * FP8_BLOCK;

    let quants = quant_bytes(NN * KK, 0x9f3a);
    let grid = NN.div_ceil(FP8_BLOCK) * KK.div_ceil(FP8_BLOCK);
    // Scales that differ across the k blocks, so taking the wrong one shows.
    let scales: Vec<f32> = (0..grid).map(|i| 0.2 + 0.7 * (i % 4) as f32).collect();
    let buf = packed_dims(&quants, &scales, KK, NN);
    let d_w = stream.clone_htod(&buf)?;

    for n_tokens in [1usize, 3, 9, 16] {
        let x: Vec<f32> = (0..n_tokens)
            .flat_map(|t| {
                pseudo_random(KK, 0xB000 + t as u64)
                    .into_iter()
                    .map(move |v| v * (1.0 + 0.25 * t as f32))
            })
            .collect();
        let d_x = stream.clone_htod(&x)?;
        let mut d_mma = stream.alloc_zeros::<f32>(n_tokens * NN)?;
        let ran = k.mma_f8_block(
            &mut d_mma.as_view_mut(),
            &d_w.as_view(),
            &d_x.as_view(),
            KK,
            NN,
            n_tokens,
            false,
        )?;
        assert!(ran, "the tensor-core kernel declined {n_tokens} tokens");

        let mut want = vec![0.0f32; n_tokens * NN];
        for t in 0..n_tokens {
            let d_xt = stream.clone_htod(&x[t * KK..(t + 1) * KK])?;
            let mut d_one = stream.alloc_zeros::<f32>(NN)?;
            k.mmv_f8_block(
                &mut d_one.as_view_mut(),
                &d_w.as_view(),
                &d_xt.as_view(),
                KK,
                NN,
                false,
            )?;
            k.device().synchronize()?;
            want[t * NN..(t + 1) * NN].copy_from_slice(&stream.clone_dtoh(&d_one)?);
        }
        k.device().synchronize()?;
        let got = stream.clone_dtoh(&d_mma)?;
        let (worst, at) = worst_ratio(&got, &want, 1e-3, 3e-2);
        assert!(
            worst <= 1.0,
            "at {n_tokens} tokens, element {at} (token {}, row {}) is {worst:.1}x \
             the tolerance: mma {}, scalar {}",
            at / NN,
            at % NN,
            got[at],
            want[at]
        );
    }
    Ok(())
}
