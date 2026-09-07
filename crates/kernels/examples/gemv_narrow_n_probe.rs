//! Is `gemv`'s real cost at GDN's own `in_proj_ba` shape (K=5120, N=96,
//! F16 -- the fused `in_proj_a`+`in_proj_b`) bandwidth or launch/reduction
//! overhead?
//!
//! Measured: flat at ~24.6us from N=48 through N=512, only starting to
//! scale with N around N=1024 -- confirms bandwidth is not the constraint
//! at this shape. `examples/launch_overhead.rs`'s own real floor on this box
//! is ~2.45us/launch with no sync, ~8.61us with one -- an order of magnitude
//! below the measured 24.6us. `ncu` (this box's own hardware counters,
//! `sudo`-unblocked, see project memory for the earlier session that found
//! it blocked and never re-tried `sudo`) explained why: the grid for this
//! launch is `n` blocks (one output row a block), and at N=96 against this
//! GPU's 188 SMs, "0.0 full waves" -- ncu's own words. `gemv_f16_ksplit`
//! splits the reduction dimension across more blocks to recover it.
//!
//! Real production always runs with speculation on (`INFERO_SPEC_K=3`), so
//! GDN's real `in_proj_ba` call is never `n_tokens=1` -- it's the verify
//! pass's own `k+1` rows (4, confirmed against a real running trace). An
//! earlier version of this probe (and the kernel/dispatch it validated) only
//! covered `n_tokens=1`, real, tested, and dead code for every request this
//! server has ever actually served -- this one covers the real shape.
//!
//!   cargo run --release -p infero-kernels --features cuda --example gemv_narrow_n_probe

use std::time::Instant;

use anyhow::Result;
use infero_gpu::Device;
use infero_kernels::{Kernels, WeightType};

fn pseudo_random_f16(n: usize, seed: u64) -> Vec<half::f16> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            half::f16::from_f32((((s >> 40) as f32 / (1u64 << 23) as f32) - 1.0) * 3.0)
        })
        .collect()
}

fn pseudo_random_f32(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (((s >> 40) as f32 / (1u64 << 23) as f32) - 1.0) * 3.0
        })
        .collect()
}

fn bench(k: &Kernels, k_dim: usize, n_dim: usize, reps: usize) -> Result<()> {
    let stream = k.device().stream().clone();
    let w: Vec<half::f16> = pseudo_random_f16(n_dim * k_dim, 0xF16F);
    let w_bytes: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();
    let d_w = stream.clone_htod(&w_bytes)?;
    let x: Vec<f32> = pseudo_random_f32(k_dim, 0xACE0);
    let d_x = stream.clone_htod(&x)?;
    let mut d_out = stream.alloc_zeros::<f32>(n_dim)?;

    for _ in 0..3 {
        k.gemv(&mut d_out.as_view_mut(), &d_w.as_view(), WeightType::F16, &d_x.as_view(), k_dim, n_dim, 1)?;
    }
    k.device().synchronize()?;
    let t0 = Instant::now();
    for _ in 0..reps {
        k.gemv(&mut d_out.as_view_mut(), &d_w.as_view(), WeightType::F16, &d_x.as_view(), k_dim, n_dim, 1)?;
    }
    k.device().synchronize()?;
    let ms = t0.elapsed().as_secs_f64() * 1000.0 / reps as f64;
    let bytes = (n_dim * k_dim * 2) as f64; // F16 = 2 bytes/elem
    let gbps = bytes / (ms / 1000.0) / 1e9;
    println!("K={k_dim:6} N={n_dim:6}  {ms:9.5} ms  {gbps:8.2} GB/s  ({:.1} us/call)", ms * 1000.0);
    Ok(())
}

fn reference_gemv(w: &[half::f16], x: &[f32], k_dim: usize, n_dim: usize) -> Vec<f32> {
    (0..n_dim)
        .map(|r| {
            (0..k_dim)
                .map(|i| f32::from(w[r * k_dim + i]) as f64 * x[i] as f64)
                .sum::<f64>() as f32
        })
        .collect()
}

fn max_rel_diff(got: &[f32], want: &[f32]) -> f32 {
    got.iter()
        .zip(want)
        .map(|(g, w)| (g - w).abs() / w.abs().max(1e-3))
        .fold(0.0f32, f32::max)
}

/// Correctness AND determinism: `gemv_f16_ksplit` at several `(n_tokens,
/// ksplit)` combinations against a host f64-accumulated reference, at the
/// real GDN shape -- plus a direct repeated-call bit-identical check, the
/// property the first, atomic-combining version of this kernel real-tested
/// and failed (`mrope_on_is_bit_identical_to_mrope_off_for_plain_text`,
/// ~1e-5 relative, the signature of summation-order noise from a
/// nondeterministic combine).
fn check_ksplit_correctness(k: &Kernels, k_dim: usize, n_dim: usize, n_tokens: usize) -> Result<()> {
    let stream = k.device().stream().clone();
    let w: Vec<half::f16> = pseudo_random_f16(n_dim * k_dim, 0xF16F);
    let w_bytes: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();
    let d_w = stream.clone_htod(&w_bytes)?;
    // Each token its own independent activation vector -- a bug that mixes
    // up which token a row reads shows up as a wrong per-token answer, not
    // a shape mismatch.
    let xs: Vec<Vec<f32>> = (0..n_tokens).map(|t| pseudo_random_f32(k_dim, 0xACE0 + t as u64)).collect();
    let x: Vec<f32> = xs.iter().flatten().copied().collect();
    let d_x = stream.clone_htod(&x)?;
    let wants: Vec<Vec<f32>> = xs.iter().map(|xt| reference_gemv(&w, xt, k_dim, n_dim)).collect();

    for ksplit in [1, 2, 4, 8, 16] {
        let mut d_out = stream.alloc_zeros::<f32>(n_tokens * n_dim)?;
        let mut d_partial = stream.alloc_zeros::<f32>(ksplit * n_tokens * n_dim)?;
        let ran = k.gemv_f16_ksplit(
            &mut d_out.as_view_mut(), &mut d_partial.as_view_mut(), &d_w.as_view(), &d_x.as_view(), k_dim, n_dim,
            n_tokens, ksplit,
        )?;
        assert!(ran, "ksplit={ksplit} n_tokens={n_tokens} declined, expected it to run");
        k.device().synchronize()?;
        let got = stream.clone_dtoh(&d_out)?;
        for (t, want) in wants.iter().enumerate() {
            let got_t = &got[t * n_dim..(t + 1) * n_dim];
            let rel = max_rel_diff(got_t, want);
            assert!(rel < 1e-4, "ksplit={ksplit} token={t} diverged from the f64 reference by {rel:.2e}");
        }
        println!("  n_tokens={n_tokens} ksplit={ksplit:3}  ok");

        // Determinism: ten repeated calls, same inputs, must be bit-identical
        // to the first -- not just "close". `PartialEq` on `f32`, not a
        // tolerance, is the point of this check.
        let mut d_out2 = stream.alloc_zeros::<f32>(n_tokens * n_dim)?;
        for rep in 0..10 {
            k.gemv_f16_ksplit(
                &mut d_out2.as_view_mut(), &mut d_partial.as_view_mut(), &d_w.as_view(), &d_x.as_view(), k_dim, n_dim,
                n_tokens, ksplit,
            )?;
            k.device().synchronize()?;
            let got2 = stream.clone_dtoh(&d_out2)?;
            assert_eq!(got, got2, "n_tokens={n_tokens} ksplit={ksplit} rep={rep}: not bit-identical to the first call");
        }
    }
    Ok(())
}

fn bench_ksplit(k: &Kernels, k_dim: usize, n_dim: usize, n_tokens: usize, ksplit: usize, reps: usize) -> Result<()> {
    let stream = k.device().stream().clone();
    let w: Vec<half::f16> = pseudo_random_f16(n_dim * k_dim, 0xF16F);
    let w_bytes: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();
    let d_w = stream.clone_htod(&w_bytes)?;
    let x: Vec<f32> = pseudo_random_f32(n_tokens * k_dim, 0xACE0);
    let d_x = stream.clone_htod(&x)?;
    let mut d_out = stream.alloc_zeros::<f32>(n_tokens * n_dim)?;
    let mut d_partial = stream.alloc_zeros::<f32>(ksplit * n_tokens * n_dim)?;

    for _ in 0..3 {
        k.gemv_f16_ksplit(
            &mut d_out.as_view_mut(), &mut d_partial.as_view_mut(), &d_w.as_view(), &d_x.as_view(), k_dim, n_dim,
            n_tokens, ksplit,
        )?;
    }
    k.device().synchronize()?;
    let t0 = Instant::now();
    for _ in 0..reps {
        k.gemv_f16_ksplit(
            &mut d_out.as_view_mut(), &mut d_partial.as_view_mut(), &d_w.as_view(), &d_x.as_view(), k_dim, n_dim,
            n_tokens, ksplit,
        )?;
    }
    k.device().synchronize()?;
    let ms = t0.elapsed().as_secs_f64() * 1000.0 / reps as f64;
    println!(
        "K={k_dim:6} N={n_dim:6} n_tokens={n_tokens:2} ksplit={ksplit:3}  {:.2} us/call  ({} blocks)",
        ms * 1000.0,
        n_dim * ksplit,
    );
    Ok(())
}

fn main() -> Result<()> {
    let k = Kernels::new(Device::new(0)?);
    println!("-- real GDN in_proj_a/b shape (K=5120, N=48) vs larger N, same K, n_tokens=1 --");
    for n in [48, 96, 256, 512, 1024, 4096, 17408] {
        bench(&k, 5120, n, 500)?;
    }
    println!("-- gemv_f16_ksplit correctness+determinism (K=5120, N=96, the REAL fused in_proj_ba shape) --");
    println!("   n_tokens=1 (plain decode step, no speculation):");
    check_ksplit_correctness(&k, 5120, 96, 1)?;
    println!("   n_tokens=4 (the REAL production shape: a speculative verify pass at spec_k=3, k+1 rows):");
    check_ksplit_correctness(&k, 5120, 96, 4)?;
    println!("   n_tokens=8 (gemv_f16_ksplit's own upper cap):");
    check_ksplit_correctness(&k, 5120, 96, 8)?;
    println!("-- gemv_f16_ksplit sweep at the REAL fused shape (K=5120, N=96), REAL n_tokens=4, 188 SMs --");
    for ksplit in [1, 2, 4, 6, 8, 12, 16] {
        bench_ksplit(&k, 5120, 96, 4, ksplit, 500)?;
    }
    println!("-- for comparison: n_tokens=1 sweep --");
    for ksplit in [1, 2, 4, 6, 8, 12, 16] {
        bench_ksplit(&k, 5120, 96, 1, ksplit, 500)?;
    }
    Ok(())
}
