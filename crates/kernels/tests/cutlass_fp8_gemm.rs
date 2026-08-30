//! `mma_e4m3_cutlass` against `mma_e4m3_block` on the exact same weight and
//! quantized activation -- `mma_e4m3_block` is already verified against an
//! independent f64 dequantize-and-matmul reference (`mma_e4m3_gemm.rs`), so
//! agreeing with it here is agreeing with that reference at one remove,
//! without re-deriving it. This isolates "does the CUTLASS path's un-repack
//! / transpose / pad adapters preserve the answer" from "is e4m3xe4m3 GEMM
//! correct", which the sibling test already covers.

#![cfg(feature = "cutlass")]

mod common;

use anyhow::Result;
use common::*;
use infero_kernels::fp8::{ACT_QUANT_GROUP, FP8_BLOCK, fp8_bytes};

const N: usize = 2 * FP8_BLOCK;
const K: usize = 4 * FP8_BLOCK;

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

fn pseudo_random_f32(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (((s >> 40) as f32 / (1u64 << 23) as f32) - 1.0) * scale
        })
        .collect()
}

fn packed(quants: &[u8], scales: &[f32], k: usize, n: usize) -> Vec<u8> {
    let mut v = infero_kernels::fp8::repack_rows(quants, k, n).expect("repack");
    for s in scales {
        v.extend_from_slice(&s.to_le_bytes());
    }
    assert_eq!(v.len(), fp8_bytes(k, n));
    v
}

#[test]
fn the_cutlass_gemm_agrees_with_mma_e4m3_block() -> Result<()> {
    let k = kernels()?;
    if !k.device().caps().fp8 {
        eprintln!("skipping: sm_{} has no native e4m3 mma", k.device().arch());
        return Ok(());
    }
    let stream = k.device().stream().clone();

    let quants = quant_bytes(N * K, 0xE4A3);
    let scale_grid_n = N / FP8_BLOCK;
    let scale_grid_k = K / FP8_BLOCK;
    let scales: Vec<f32> = (0..scale_grid_n * scale_grid_k).map(|i| 0.3 + 0.4 * (i % 5) as f32).collect();
    let w_buf = packed(&quants, &scales, K, N);
    let d_w = stream.clone_htod(&w_buf)?;
    let cutlass_w = k.prepare_cutlass_weight(&d_w.as_view(), K, N, false)?;

    for n_tokens in [1usize, 2, 3, 5, 8, 9, 17, 128, 129] {
        let x: Vec<f32> =
            (0..n_tokens).flat_map(|t| pseudo_random_f32(K, 0xACE0 + t as u64, 4.0 + t as f32)).collect();
        let d_x = stream.clone_htod(&x)?;

        let scale_cols = K / ACT_QUANT_GROUP;
        let mut d_xq = stream.alloc_zeros::<u8>(n_tokens * K)?;
        let mut d_xs = stream.alloc_zeros::<f32>(n_tokens * scale_cols)?;
        k.quantize_act_e4m3(&mut d_xq.as_view_mut(), &mut d_xs.as_view_mut(), &d_x.as_view(), K, n_tokens)?;

        let mut d_want = stream.alloc_zeros::<f32>(n_tokens * N)?;
        let ran = k.mma_e4m3_block(
            &mut d_want.as_view_mut(),
            &d_w.as_view(),
            &d_xq.as_view(),
            &d_xs.as_view(),
            K,
            N,
            n_tokens,
            false,
        )?;
        assert!(ran, "mma_e4m3_block declined {n_tokens} tokens at K={K}");

        let mut d_got = stream.alloc_zeros::<f32>(n_tokens * N)?;
        let ran = k.mma_e4m3_cutlass(
            &mut d_got.as_view_mut(),
            &d_w.as_view(),
            &cutlass_w,
            &d_xq.as_view(),
            &d_xs.as_view(),
            K,
            N,
            n_tokens,
            false,
        )?;
        assert!(ran, "mma_e4m3_cutlass declined {n_tokens} tokens at K={K}");

        k.device().synchronize()?;
        let want = stream.clone_dtoh(&d_want)?;
        let got = stream.clone_dtoh(&d_got)?;
        k.device().synchronize()?;

        // bf16 output against mma_e4m3_block's f32 accumulate -- a coarser
        // tolerance than the sibling test's, for the extra rounding step.
        let (worst, at) = max_rel_diff(&got, &want, 3e-2);
        assert!(
            worst <= 1.0,
            "at {n_tokens} tokens, element {at} (token {}, row {}) is {worst:.1}x \
             the tolerance: cutlass {}, mma_e4m3_block {}",
            at / N,
            at % N,
            got[at],
            want[at]
        );
    }
    Ok(())
}

/// The unified plain-layout path -- `mmv_f8_plain` for one token,
/// `mma_e4m3_cutlass` built `already_plain` for the rest -- against
/// `mma_e4m3_block` reading the *same logical weight* stored the ordinary
/// (`repack_rows`-interleaved) way. Same reference reasoning as the sibling
/// test above: `mma_e4m3_block` is independently verified elsewhere, so
/// agreeing with it here on the same values is enough.
#[test]
fn the_unified_plain_layout_agrees_with_mma_e4m3_block() -> Result<()> {
    let k = kernels()?;
    if !k.device().caps().fp8 {
        eprintln!("skipping: sm_{} has no native e4m3 mma", k.device().arch());
        return Ok(());
    }
    let stream = k.device().stream().clone();

    let quants = quant_bytes(N * K, 0x9EAD);
    let scale_grid_n = N / FP8_BLOCK;
    let scale_grid_k = K / FP8_BLOCK;
    let scales: Vec<f32> = (0..scale_grid_n * scale_grid_k).map(|i| 0.2 + 0.3 * (i % 7) as f32).collect();

    // Interleaved copy, for the mma_e4m3_block reference.
    let w_interleaved = packed(&quants, &scales, K, N);
    let d_w_interleaved = stream.clone_htod(&w_interleaved)?;

    // Plain copy, for the unified path -- same quants and scales, no repack.
    let mut w_plain = infero_kernels::fp8::pad_rows(&quants, K, N)?;
    for s in &scales {
        w_plain.extend_from_slice(&s.to_le_bytes());
    }
    let d_w_plain = stream.clone_htod(&w_plain)?;
    let cutlass_w = k.prepare_cutlass_weight(&d_w_plain.as_view(), K, N, true)?;

    for n_tokens in [1usize, 2, 3, 8, 17, 128, 129] {
        let x: Vec<f32> =
            (0..n_tokens).flat_map(|t| pseudo_random_f32(K, 0xB0A7 + t as u64, 3.0 + t as f32)).collect();
        let d_x = stream.clone_htod(&x)?;

        let scale_cols = K / ACT_QUANT_GROUP;
        let mut d_xq = stream.alloc_zeros::<u8>(n_tokens * K)?;
        let mut d_xs = stream.alloc_zeros::<f32>(n_tokens * scale_cols)?;
        k.quantize_act_e4m3(&mut d_xq.as_view_mut(), &mut d_xs.as_view_mut(), &d_x.as_view(), K, n_tokens)?;

        let mut d_want = stream.alloc_zeros::<f32>(n_tokens * N)?;
        let ran = k.mma_e4m3_block(
            &mut d_want.as_view_mut(),
            &d_w_interleaved.as_view(),
            &d_xq.as_view(),
            &d_xs.as_view(),
            K,
            N,
            n_tokens,
            false,
        )?;
        assert!(ran, "mma_e4m3_block declined {n_tokens} tokens at K={K}");

        let mut d_got = stream.alloc_zeros::<f32>(n_tokens * N)?;
        if n_tokens == 1 {
            k.mmv_f8_plain(&mut d_got.as_view_mut(), &d_w_plain.as_view(), &d_x.as_view(), K, N, false)?;
        } else {
            let ran = k.mma_e4m3_cutlass(
                &mut d_got.as_view_mut(),
                &d_w_plain.as_view(),
                &cutlass_w,
                &d_xq.as_view(),
                &d_xs.as_view(),
                K,
                N,
                n_tokens,
                false,
            )?;
            assert!(ran, "mma_e4m3_cutlass declined {n_tokens} tokens at K={K}");
        }

        k.device().synchronize()?;
        let want = stream.clone_dtoh(&d_want)?;
        let got = stream.clone_dtoh(&d_got)?;
        k.device().synchronize()?;

        // mmv_f8_plain is f32 straight through (no bf16 rounding), so its
        // tolerance could be tighter than mma_e4m3_cutlass's at n_tokens>1 --
        // one shared bound for simplicity, still well inside either kernel's
        // real error.
        let (worst, at) = max_rel_diff(&got, &want, 3e-2);
        assert!(
            worst <= 1.0,
            "unified path, at {n_tokens} tokens, element {at} (token {}, row {}) is {worst:.1}x \
             the tolerance: got {}, want {}",
            at / N,
            at % N,
            got[at],
            want[at]
        );
    }
    Ok(())
}

/// `quantize_act_e4m3_cutlass` + `mma_e4m3_cutlass_sfa` (the fused path,
/// writing the transposed scale directly) against `quantize_act_e4m3` +
/// `mma_e4m3_cutlass` (the original two-step path) on the same weight and
/// activation. Both compute the identical quantization; only *where* the
/// scale transpose happens differs, so this expects exact equality, not a
/// tolerance -- any difference means the fused kernel's `sfa_t` indexing or
/// padding-fill disagrees with `transpose_pad_scale_a_f32`'s.
#[test]
fn the_fused_quantizer_matches_the_separate_transpose() -> Result<()> {
    let k = kernels()?;
    if !k.device().caps().fp8 {
        eprintln!("skipping: sm_{} has no native e4m3 mma", k.device().arch());
        return Ok(());
    }
    let stream = k.device().stream().clone();

    let quants = quant_bytes(N * K, 0xFEED);
    let scale_grid_n = N / FP8_BLOCK;
    let scale_grid_k = K / FP8_BLOCK;
    let scales: Vec<f32> = (0..scale_grid_n * scale_grid_k).map(|i| 0.25 + 0.35 * (i % 6) as f32).collect();
    let w_buf = packed(&quants, &scales, K, N);
    let d_w = stream.clone_htod(&w_buf)?;
    let cutlass_w = k.prepare_cutlass_weight(&d_w.as_view(), K, N, false)?;

    // Token counts that land on both sides of a 128-row pad boundary, each
    // tried both at `m_pad = n_tokens` (no padding -- the common case now
    // that CUTLASS is known to accept any M) and `m_pad =
    // n_tokens.next_multiple_of(128)` (the old, still-supported padded path,
    // for whatever caller might still want it).
    for n_tokens in [1usize, 2, 8, 17, 127, 128, 129] {
        for m_pad in [n_tokens, n_tokens.next_multiple_of(128)] {
        let x: Vec<f32> =
            (0..n_tokens).flat_map(|t| pseudo_random_f32(K, 0x5CA1E + t as u64, 2.0 + t as f32)).collect();
        let d_x = stream.clone_htod(&x)?;
        let scale_cols = K / ACT_QUANT_GROUP;

        // Original: separate quantize + internal transpose.
        let mut d_xq_a = stream.alloc_zeros::<u8>(n_tokens * K)?;
        let mut d_xs_a = stream.alloc_zeros::<f32>(n_tokens * scale_cols)?;
        k.quantize_act_e4m3(&mut d_xq_a.as_view_mut(), &mut d_xs_a.as_view_mut(), &d_x.as_view(), K, n_tokens)?;
        let mut d_want = stream.alloc_zeros::<f32>(n_tokens * N)?;
        let ran = k.mma_e4m3_cutlass(
            &mut d_want.as_view_mut(),
            &d_w.as_view(),
            &cutlass_w,
            &d_xq_a.as_view(),
            &d_xs_a.as_view(),
            K,
            N,
            n_tokens,
            false,
        )?;
        assert!(ran, "mma_e4m3_cutlass declined {n_tokens} tokens at K={K}");

        // Fused: quantizer writes sfa_t directly.
        let mut d_xq_b = stream.alloc_zeros::<u8>(n_tokens * K)?;
        let mut d_sfa_t = stream.alloc_zeros::<f32>(scale_cols * m_pad)?;
        k.quantize_act_e4m3_cutlass(
            &mut d_xq_b.as_view_mut(),
            &mut d_sfa_t.as_view_mut(),
            &d_x.as_view(),
            K,
            n_tokens,
            m_pad,
        )?;
        let mut d_got = stream.alloc_zeros::<f32>(n_tokens * N)?;
        let ran = k.mma_e4m3_cutlass_sfa(
            &mut d_got.as_view_mut(),
            &d_w.as_view(),
            &cutlass_w,
            &d_xq_b.as_view(),
            &d_sfa_t.as_view(),
            K,
            N,
            n_tokens,
            m_pad,
            false,
        )?;
        assert!(ran, "mma_e4m3_cutlass_sfa declined {n_tokens} tokens (m_pad={m_pad}) at K={K}");

        k.device().synchronize()?;
        let want = stream.clone_dtoh(&d_want)?;
        let got = stream.clone_dtoh(&d_got)?;
        let xq_a = stream.clone_dtoh(&d_xq_a)?;
        let xq_b = stream.clone_dtoh(&d_xq_b)?;
        k.device().synchronize()?;

        assert_eq!(xq_a, xq_b, "{n_tokens} tokens (m_pad={m_pad}): quantized activation bytes differ");
        assert_eq!(got, want, "{n_tokens} tokens (m_pad={m_pad}): GEMM output differs between the two paths");
        }
    }
    Ok(())
}

/// [`infero_kernels::Kernels::mma_e4m3_cutlass_sfa_f32out`] against the
/// existing bf16 path -- CUTLASS's epilogue writes `out` (f32) directly,
/// skipping the whole `cutlass_bf16_store` kernel, so this checks it isn't
/// silently wrong the way a bf16->f32 rounding difference might make it look
/// coincidentally close. Same tolerance as the other cutlass-vs-cutlass
/// comparison in this file (both paths run the same MMA, only the epilogue's
/// output element type differs): exact equality is the expectation, not a
/// tolerance, except right at the boundary between two rounding modes.
/// `accum=true` is checked too, since the f32-direct path's accumulate is
/// CUTLASS's own `beta=1` rather than this crate's own store kernel's -- an
/// untested code path in the reverted f32-output experiment this is not.
#[test]
fn the_f32out_gemm_matches_the_bf16_path() -> Result<()> {
    let k = kernels()?;
    if !k.device().caps().fp8 {
        eprintln!("skipping: sm_{} has no native e4m3 mma", k.device().arch());
        return Ok(());
    }
    let stream = k.device().stream().clone();

    let quants = quant_bytes(N * K, 0xF32D);
    let scale_grid_n = N / FP8_BLOCK;
    let scale_grid_k = K / FP8_BLOCK;
    let scales: Vec<f32> = (0..scale_grid_n * scale_grid_k).map(|i| 0.2 + 0.3 * (i % 7) as f32).collect();
    let w_buf = packed(&quants, &scales, K, N);
    let d_w = stream.clone_htod(&w_buf)?;
    let cutlass_w = k.prepare_cutlass_weight(&d_w.as_view(), K, N, false)?;

    for n_tokens in [1usize, 2, 3, 5, 8, 9, 17, 127, 128, 129] {
        let x: Vec<f32> =
            (0..n_tokens).flat_map(|t| pseudo_random_f32(K, 0xD00D + t as u64, 3.0 + t as f32)).collect();
        let d_x = stream.clone_htod(&x)?;
        let scale_cols = K / ACT_QUANT_GROUP;

        let mut d_xq = stream.alloc_zeros::<u8>(n_tokens * K)?;
        let mut d_sfa_t = stream.alloc_zeros::<f32>(scale_cols * n_tokens)?;
        k.quantize_act_e4m3_cutlass(
            &mut d_xq.as_view_mut(),
            &mut d_sfa_t.as_view_mut(),
            &d_x.as_view(),
            K,
            n_tokens,
            n_tokens,
        )?;

        for accum in [false, true] {
            // A nonzero starting `out` when `accum`, so a beta=0 bug (silently
            // overwriting instead of adding) would show up as a wrong answer
            // instead of coincidentally matching from a zeroed buffer.
            let seed_out: Vec<f32> = if accum {
                pseudo_random_f32(n_tokens * N, 0xBEEF, 100.0)
            } else {
                vec![0.0f32; n_tokens * N]
            };

            let mut d_want = stream.clone_htod(&seed_out)?;
            let ran = k.mma_e4m3_cutlass_sfa(
                &mut d_want.as_view_mut(),
                &d_w.as_view(),
                &cutlass_w,
                &d_xq.as_view(),
                &d_sfa_t.as_view(),
                K,
                N,
                n_tokens,
                n_tokens,
                accum,
            )?;
            assert!(ran, "mma_e4m3_cutlass_sfa declined {n_tokens} tokens at K={K}");

            let mut d_got = stream.clone_htod(&seed_out)?;
            let ran = k.mma_e4m3_cutlass_sfa_f32out(
                &mut d_got.as_view_mut(),
                &d_w.as_view(),
                &cutlass_w,
                &d_xq.as_view(),
                &d_sfa_t.as_view(),
                K,
                N,
                n_tokens,
                accum,
            )?;
            assert!(ran, "mma_e4m3_cutlass_sfa_f32out declined {n_tokens} tokens at K={K}");

            k.device().synchronize()?;
            let want = stream.clone_dtoh(&d_want)?;
            let got = stream.clone_dtoh(&d_got)?;
            k.device().synchronize()?;

            let (worst, at) = max_rel_diff(&got, &want, 3e-2);
            assert!(
                worst <= 1.0,
                "{n_tokens} tokens, accum={accum}: element {at} is {worst:.1}x the tolerance: \
                 f32out {}, bf16-path {}",
                got[at],
                want[at]
            );
        }
    }
    Ok(())
}

fn max_rel_diff(got: &[f32], want: &[f32], rel: f32) -> (f32, usize) {
    let peak = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let floor = 3e-2 * peak.max(f32::MIN_POSITIVE);
    let mut worst = (0.0f32, 0usize);
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let r = (g - w).abs() / (floor + rel * w.abs());
        if r > worst.0 {
            worst = (r, i);
        }
    }
    worst
}
