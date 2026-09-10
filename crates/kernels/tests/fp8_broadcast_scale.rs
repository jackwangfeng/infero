//! F8E4M3's block-scaled kernels (`mma_e4m3_block`, `mma_f8_block`) fed a
//! scale grid that is a BROADCAST of one scalar into every block, instead of
//! the varying-per-block grid `mma_e4m3_gemm.rs`'s own sibling test (and
//! every other F8E4M3 test in this crate) uses.
//!
//! This is the real shape `weights.rs::f8e4m3_scale_grid`'s new fallback
//! (commit `25bd025`) produces for a checkpoint tensor that carries only a
//! single scalar `.weight_scale` rather than a `.weight_scale_inv` block
//! grid: `vec![scalar; scale_grid(k, n)]`, every entry identical. That
//! fallback is real, shipped, unmodified code, and — unlike every existing
//! F8E4M3 kernel test in this crate — no existing test has ever fed its
//! output (a uniform grid) to these kernels; every one of them builds a
//! grid that varies block to block. If either kernel has an internal
//! optimization, cache, or indexing path that quietly assumes variation (or,
//! more mundanely, if the *loader*'s broadcast convention turns out not to
//! be the plain per-element multiplier the existing block-scale grid already
//! is), this is where it would show up.
//!
//! Shape: `k=5120, n=10240` — this plan's real checkpoint's own GDN
//! `in_proj_qkv` (per `progress.md`'s real header inspection: production's
//! `in_proj_qkv.weight_scale_inv` is BF16 `[80, 40]`, i.e.
//! `n/128=80, k/128=40` -> `n=10240, k=5120`) — the exact real tensor whose
//! *absence* of a block grid is what triggered the loader fallback this test
//! is chasing, not an arbitrary GDN-shaped stand-in.
//!
//! Methodology mirrors `mma_e4m3_gemm.rs` exactly (same helpers, same
//! `dequant_weight`/`e4m3_to_f32`/`worst_ratio` shapes) — the one existing
//! test in this crate with an independent f64 host reference for F8E4M3
//! (not a kernel-vs-kernel comparison), which is what "does a genuinely
//! broadcast-uniform scale change the answer" needs: two kernels that both
//! happened to share the same broadcast-blind bug would still agree with
//! each other and pass a kernel-vs-kernel check.
//!
//! Which kernels: read `Model::matmul_pre` in `crates/model/src/lib.rs`
//! (the real dispatch this checkpoint's F8E4M3 tensors go through) rather
//! than guessing. Under the *default*, non-unified, non-CUTLASS-opt-in FP8
//! path (the one this checkpoint's `in_proj_qkv` etc. actually take, since
//! `INFERO_FP8_UNIFIED` and the CUTLASS FFN opt-in are both off by default),
//! the real dispatch is: `n_tokens == 1` -> `mma_f8_block` (raw f32
//! activation, no activation quantization); `n_tokens >= 2` -> (CUTLASS
//! opt-in, skipped by default) then `mma_e4m3_block` (quantized e4m3
//! activation). Both are exercised below, at their own real call shape, not
//! just one.

mod common;

use anyhow::Result;
use common::*;
use infero_kernels::fp8::{ACT_QUANT_GROUP, FP8_BLOCK, fp8_bytes};
use infero_safetensors::{Dtype, Tensor};

const K: usize = 5120;
const N: usize = 10240;

// One nonzero, non-power-of-two scalar: not 1.0 (so a bug that silently
// drops the scale entirely would show up as a wrong answer, not a
// coincidental match) and not a "nice" binary value (so a bug that only
// breaks on round scale values would show up too).
const BROADCAST_SCALE: f32 = 0.4375;

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

/// The weight, dequantized by the same oracle `fp8_matvec.rs`/
/// `mma_e4m3_gemm.rs` trust — `Tensor::dequant_f8_to_f16`, real production
/// code, fed a real broadcast-uniform scale tensor here instead of a varying
/// one.
fn dequant_weight(quants: &[u8], scales: &[f32], k: usize, n: usize) -> Vec<f32> {
    let sbytes: Vec<u8> = scales.iter().flat_map(|s| s.to_le_bytes()).collect();
    let q = Tensor { name: "w", dtype: Dtype::F8E4M3, shape: vec![n, k], data: quants };
    let s = Tensor {
        name: "s",
        dtype: Dtype::F32,
        shape: vec![n / FP8_BLOCK, k / FP8_BLOCK],
        data: &sbytes,
    };
    q.dequant_f8_to_f16(&s, FP8_BLOCK)
        .unwrap()
        .iter()
        .map(|h| f32::from(*h))
        .collect()
}

/// Bit for bit `e4m3_to_f32` in `fp8.cu` — same copy `mma_e4m3_gemm.rs`
/// keeps, needed to read the *quantized activation* back, independent of
/// `dequant_weight`'s own decoder.
fn e4m3_to_f32(b: u8) -> f32 {
    let sign = (b & 0x80) != 0;
    let exp = (b >> 3) & 0x0F;
    let man = (b & 0x07) as f32;
    let mag = if exp == 0 { man / 512.0 } else { (1.0 + man / 8.0) * 2f32.powi(exp as i32 - 7) };
    if sign { -mag } else { mag }
}

fn worst_ratio(got: &[f32], want: &[f32], rel: f32, floor_scale: f32) -> (f32, usize) {
    let peak = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let floor = floor_scale * peak.max(f32::MIN_POSITIVE);
    let mut worst = (0.0f32, 0usize);
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let r = (g - w).abs() / (floor + rel * w.abs());
        if r > worst.0 {
            worst = (r, i);
        }
    }
    worst
}

/// [`infero_kernels::Kernels::mma_e4m3_block`] — the real kernel
/// `Model::matmul_pre` dispatches to for this checkpoint's non-unified,
/// non-CUTLASS-opt-in F8E4M3 path at `n_tokens >= 2` (`crates/model/src/lib.rs`,
/// the `if n_tokens >= 2 { ... mma_e4m3_block ... }` arm inside the
/// `w.ty == F8E4M3` branch) — against a broadcast-uniform scale grid.
#[test]
fn the_e4m3_block_mma_agrees_with_a_broadcast_scale_grid() -> Result<()> {
    let k = kernels()?;
    if !k.device().caps().fp8 {
        eprintln!("skipping: sm_{} has no native e4m3 mma", k.device().arch());
        return Ok(());
    }
    let stream = k.device().stream().clone();

    let quants = quant_bytes(N * K, 0xB4CA);
    let scale_grid_n = N / FP8_BLOCK;
    let scale_grid_k = K / FP8_BLOCK;
    let scales: Vec<f32> = vec![BROADCAST_SCALE; scale_grid_n * scale_grid_k];
    let w_buf = packed(&quants, &scales, K, N);
    let d_w = stream.clone_htod(&w_buf)?;
    let w_ref = dequant_weight(&quants, &scales, K, N);

    for n_tokens in [1usize, 2, 3, 5, 8, 9, 17] {
        let x: Vec<f32> =
            (0..n_tokens).flat_map(|t| pseudo_random_f32(K, 0xACE0 + t as u64, 4.0 + t as f32)).collect();
        let d_x = stream.clone_htod(&x)?;

        let scale_cols = K / ACT_QUANT_GROUP;
        let mut d_xq = stream.alloc_zeros::<u8>(n_tokens * K)?;
        let mut d_xs = stream.alloc_zeros::<f32>(n_tokens * scale_cols)?;
        k.quantize_act_e4m3(&mut d_xq.as_view_mut(), &mut d_xs.as_view_mut(), &d_x.as_view(), K, n_tokens)?;

        let mut d_out = stream.alloc_zeros::<f32>(n_tokens * N)?;
        let ran = k.mma_e4m3_block(
            &mut d_out.as_view_mut(),
            &d_w.as_view(),
            &d_xq.as_view(),
            &d_xs.as_view(),
            K,
            N,
            n_tokens,
            false,
        )?;
        assert!(ran, "mma_e4m3_block declined {n_tokens} tokens at K={K}");

        let xq = stream.clone_dtoh(&d_xq)?;
        let xs = stream.clone_dtoh(&d_xs)?;
        k.device().synchronize()?;
        let got = stream.clone_dtoh(&d_out)?;
        k.device().synchronize()?;

        // The reference activation is the quantized-and-read-back one, not
        // the original `x` — same reasoning as `mma_e4m3_gemm.rs`: this test
        // is about the GEMM's handling of the broadcast scale, not about how
        // much error e4m3 activation quantization itself introduces.
        let x_dq: Vec<f32> = (0..n_tokens * K)
            .map(|i| {
                let (t, kk) = (i / K, i % K);
                e4m3_to_f32(xq[i]) * xs[t * scale_cols + kk / ACT_QUANT_GROUP]
            })
            .collect();

        let mut want = vec![0.0f32; n_tokens * N];
        for t in 0..n_tokens {
            for r in 0..N {
                let mut acc = 0.0f64;
                for kk in 0..K {
                    acc += w_ref[r * K + kk] as f64 * x_dq[t * K + kk] as f64;
                }
                want[t * N + r] = acc as f32;
            }
        }

        let (worst, at) = worst_ratio(&got, &want, 2e-3, 3e-2);
        assert!(
            worst <= 1.0,
            "broadcast scale, at {n_tokens} tokens, element {at} (token {}, row {}) is {worst:.1}x \
             the tolerance: got {}, want {}",
            at / N,
            at % N,
            got[at],
            want[at]
        );
    }
    Ok(())
}

/// [`infero_kernels::Kernels::mma_f8_block`] — the real kernel
/// `Model::matmul_pre` dispatches to for this checkpoint's F8E4M3 path at
/// `n_tokens == 1` (the real single-token decode step, the exact case that
/// produced the garbage output), which unlike `mma_e4m3_block` takes the
/// activation as plain `f32` — no activation quantization at all, see that
/// function's own doc comment ("the activation stays f32, because the
/// weights carry a per-block scale rather than a per-block quantization of
/// the input") — so this reference needs no activation dequant step, only
/// the weight's, against the same broadcast-uniform scale grid.
#[test]
fn the_f8_block_mma_agrees_with_a_broadcast_scale_grid() -> Result<()> {
    let k = kernels()?;
    if !k.device().caps().fp8 {
        eprintln!("skipping: sm_{} has no native e4m3 mma", k.device().arch());
        return Ok(());
    }
    let stream = k.device().stream().clone();

    let quants = quant_bytes(N * K, 0xF8B0);
    let scale_grid_n = N / FP8_BLOCK;
    let scale_grid_k = K / FP8_BLOCK;
    let scales: Vec<f32> = vec![BROADCAST_SCALE; scale_grid_n * scale_grid_k];
    let w_buf = packed(&quants, &scales, K, N);
    let d_w = stream.clone_htod(&w_buf)?;
    let w_ref = dequant_weight(&quants, &scales, K, N);

    for n_tokens in [1usize, 2, 3] {
        let x: Vec<f32> =
            (0..n_tokens).flat_map(|t| pseudo_random_f32(K, 0x0DEC + t as u64, 3.0 + t as f32)).collect();
        let d_x = stream.clone_htod(&x)?;

        let mut d_out = stream.alloc_zeros::<f32>(n_tokens * N)?;
        let ran = k.mma_f8_block(&mut d_out.as_view_mut(), &d_w.as_view(), &d_x.as_view(), K, N, n_tokens, false)?;
        assert!(ran, "mma_f8_block declined {n_tokens} tokens at K={K}");

        k.device().synchronize()?;
        let got = stream.clone_dtoh(&d_out)?;
        k.device().synchronize()?;

        let mut want = vec![0.0f32; n_tokens * N];
        for t in 0..n_tokens {
            for r in 0..N {
                let mut acc = 0.0f64;
                for kk in 0..K {
                    acc += w_ref[r * K + kk] as f64 * x[t * K + kk] as f64;
                }
                want[t * N + r] = acc as f32;
            }
        }

        let (worst, at) = worst_ratio(&got, &want, 2e-3, 3e-2);
        assert!(
            worst <= 1.0,
            "broadcast scale (mma_f8_block), at {n_tokens} tokens, element {at} (token {}, row {}) is \
             {worst:.1}x the tolerance: got {}, want {}",
            at / N,
            at % N,
            got[at],
            want[at]
        );
    }
    Ok(())
}
