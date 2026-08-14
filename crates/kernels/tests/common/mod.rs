//! Shared test scaffolding: a device, CPU references, and error metrics.
//!
//! Included by several test binaries, each of which uses a different subset.
#![allow(dead_code)]

use anyhow::Result;
use tuili_cuda::Device;
use tuili_kernels::Kernels;

pub fn kernels() -> Result<Kernels> {
    // cudarc hands out the device's primary context, so tests running in
    // parallel share one context and one stream rather than fighting over
    // separate ones.
    Ok(Kernels::new(Device::new(0)?))
}

/// Largest absolute difference, and where it happened.
pub fn max_abs_diff(a: &[f32], b: &[f32]) -> (f32, usize) {
    assert_eq!(a.len(), b.len());
    let mut worst = (0.0f32, 0usize);
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        let d = (x - y).abs();
        if d > worst.0 {
            worst = (d, i);
        }
    }
    worst
}

/// Relative error against the magnitude of the reference, which is the right
/// measure for logits and activations that span several orders of magnitude.
pub fn max_rel_diff(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len());
    got.iter()
        .zip(want)
        .map(|(g, w)| (g - w).abs() / (w.abs().max(1e-3)))
        .fold(0.0f32, f32::max)
}

pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
    let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    (dot / (na * nb + 1e-30)) as f32
}

/// Deterministic pseudo-random values in [-1, 1); a fixed stream keeps a
/// failing test reproducible.
pub fn pseudo_random(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 23) as f32) - 1.0
        })
        .collect()
}

// ---- CPU references -----------------------------------------------------

pub fn rms_norm_ref(x: &[f32], weight: &[f32], n_tokens: usize, d: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0.0; n_tokens * d];
    for t in 0..n_tokens {
        let row = &x[t * d..(t + 1) * d];
        let mean_sq = row.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / d as f64;
        let scale = 1.0 / (mean_sq + eps as f64).sqrt();
        for i in 0..d {
            out[t * d + i] = (row[i] as f64 * scale) as f32 * weight[i];
        }
    }
    out
}

/// NeoX / rotate-half rotary embedding.
pub fn rope_ref(
    x: &[f32],
    positions: &[i32],
    n_tokens: usize,
    n_heads: usize,
    d_head: usize,
    theta_base: f32,
) -> Vec<f32> {
    let mut out = x.to_vec();
    let half = d_head / 2;
    for (t, &pos) in positions.iter().enumerate().take(n_tokens) {
        for h in 0..n_heads {
            let base = (t * n_heads + h) * d_head;
            for i in 0..half {
                let inv_freq = (theta_base as f64).powf(-2.0 * i as f64 / d_head as f64);
                let angle = pos as f64 * inv_freq;
                let (sin, cos) = angle.sin_cos();
                let a = x[base + i] as f64;
                let b = x[base + i + half] as f64;
                out[base + i] = (a * cos - b * sin) as f32;
                out[base + i + half] = (a * sin + b * cos) as f32;
            }
        }
    }
    out
}

pub fn silu_mul_ref(gate: &[f32], up: &[f32]) -> Vec<f32> {
    gate.iter()
        .zip(up)
        .map(|(g, u)| {
            let g = *g as f64;
            ((g / (1.0 + (-g).exp())) * *u as f64) as f32
        })
        .collect()
}

pub fn softmax_ref(row: &[f32]) -> Vec<f32> {
    let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f64> = row.iter().map(|v| ((*v - m) as f64).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|e| (e / sum) as f32).collect()
}
