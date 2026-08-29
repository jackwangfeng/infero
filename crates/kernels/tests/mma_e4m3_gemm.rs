//! `mma_e4m3_block` end to end: quantize a real activation, run the native
//! e4m3xe4m3 GEMM, and check it against the weight's own verified dequantizer
//! composed with the quantized activation read back and dequantized the same
//! way — so this isolates "does the kernel implement e4m3xe4m3 GEMM with the
//! weight and activation scales combined correctly" from "how much error does
//! FP8 activation quantization introduce", which is a property of the format
//! and not this kernel.

mod common;

use anyhow::Result;
use common::*;
use infero_kernels::fp8::{ACT_QUANT_GROUP, FP8_BLOCK, fp8_bytes};
use infero_safetensors::{Dtype, Tensor};

// Two 128-row scale blocks by four 128-column scale blocks, and two k-tiles
// of `mma_e4m3_block`'s own `K_TILE = 256` — enough for the scale-grid
// indexing and the multi-tile loop to both have somewhere to go wrong.
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

/// The weight, dequantized by the same oracle `fp8_matvec.rs` trusts.
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

/// Bit for bit `e4m3_to_f32` in `fp8.cu` — reading the quantized activation
/// back needs the same decoder the GEMM's own scale application implies.
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

#[test]
fn the_e4m3_gemm_agrees_with_its_own_dequantized_inputs() -> Result<()> {
    let k = kernels()?;
    if !k.device().caps().fp8 {
        eprintln!("skipping: sm_{} has no native e4m3 mma", k.device().arch());
        return Ok(());
    }
    let stream = k.device().stream().clone();

    let quants = quant_bytes(N * K, 0xE4A3);
    let scale_grid_n = N / FP8_BLOCK;
    let scale_grid_k = K / FP8_BLOCK;
    let scales: Vec<f32> = (0..scale_grid_n * scale_grid_k)
        .map(|i| 0.3 + 0.4 * (i % 5) as f32)
        .collect();
    let w_buf = packed(&quants, &scales, K, N);
    let d_w = stream.clone_htod(&w_buf)?;
    let w_ref = dequant_weight(&quants, &scales, K, N);

    for n_tokens in [1usize, 2, 3, 5, 8, 9, 17] {
        let x: Vec<f32> = (0..n_tokens)
            .flat_map(|t| pseudo_random_f32(K, 0xACE0 + t as u64, 4.0 + t as f32))
            .collect();
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
        // the original `x` — this test is about the GEMM, not about how much
        // error quantization itself introduces.
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

        // f32 accumulation over K=512 terms of an MMA whose internal
        // reduction order this test's f64 reference does not replicate — a
        // relative tolerance, the same reasoning `fp8_matvec.rs`'s own
        // `worst_ratio` uses for its GEMM comparisons.
        let (worst, at) = worst_ratio(&got, &want, 2e-3, 3e-2);
        assert!(
            worst <= 1.0,
            "at {n_tokens} tokens, element {at} (token {}, row {}) is {worst:.1}x \
             the tolerance: got {}, want {}",
            at / N,
            at % N,
            got[at],
            want[at]
        );
    }
    Ok(())
}
