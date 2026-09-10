//! Real A/B at the real attention K/V projection shape (K=5120, N=1024,
//! real batch=16 decode) between the CURRENT SHIPPED dispatch
//! (`mma_e4m3_cutlass_sfa_f32out`'s real dispatch -- at n_tokens=16 this
//! picks the `small_m_swap` tile, `SWAP_AB_MAX_TOKENS=32`) and
//! `mmv_f8_plain_multi8` (declines above 8 tokens, so n_tokens=16 needs two
//! back-to-back launches of 8 each -- exactly what a real dispatch branch
//! would do if wired in).
//!
//! Why this shape specifically: the module-split roofline analysis found
//! `attn_kv` (K=5120,N=1024) at only 15.8% of this GPU's real bandwidth-bound
//! ceiling at batch=16 -- dramatically worse than every other real projection
//! shape in the model (57-75%). A prior session's rejection of
//! `mmv_f8_plain_multi2`/`multi8` for this shape used weight-BYTE share
//! (~3.4% of a decode step's FP8 weight bytes) as the metric, not wall-clock
//! time share -- this probe re-checks with a real, direct, same-shape timing
//! comparison instead.
//!
//!   cargo run --release -p infero-kernels --features cutlass --example attn_kv_narrow_probe

#![cfg(feature = "cutlass")]

use std::time::Instant;

use anyhow::Result;
use infero_cuda::Device;
use infero_gpu::View;
use infero_kernels::fp8::{ACT_QUANT_GROUP, FP8_BLOCK};
use infero_kernels::Kernels;

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

fn packed(quants: &[u8], scales: &[f32], k: usize, n: usize) -> Vec<u8> {
    let mut v = infero_kernels::fp8::repack_rows(quants, k, n).expect("repack");
    for s in scales {
        v.extend_from_slice(&s.to_le_bytes());
    }
    v
}

fn bench_cutlass_real_dispatch(
    k: &Kernels,
    d_w: &View<'_, u8>,
    cutlass_w: &infero_kernels::CutlassWeight,
    kk: usize,
    n: usize,
    n_tokens: usize,
    reps: usize,
) -> Result<f64> {
    let stream = k.device().stream().clone();
    let x: Vec<f32> = pseudo_random_f32(n_tokens * kk, 0xACE0);
    let d_x = stream.clone_htod(&x)?;
    let scale_cols = kk / ACT_QUANT_GROUP;
    let mut d_xq = stream.alloc_zeros::<u8>(n_tokens * kk)?;
    let mut d_sfa_t = stream.alloc_zeros::<f32>(scale_cols * n_tokens)?;
    k.quantize_act_e4m3_cutlass(
        &mut d_xq.as_view_mut(),
        &mut d_sfa_t.as_view_mut(),
        &d_x.as_view(),
        kk,
        n_tokens,
        n_tokens,
    )?;
    let mut d_out = stream.alloc_zeros::<f32>(n_tokens * n)?;

    for _ in 0..3 {
        k.mma_e4m3_cutlass_sfa_f32out(
            &mut d_out.as_view_mut(),
            d_w,
            cutlass_w,
            &d_xq.as_view(),
            &d_sfa_t.as_view(),
            kk,
            n,
            n_tokens,
            false,
        )?;
    }
    k.device().synchronize()?;
    let t0 = Instant::now();
    for _ in 0..reps {
        k.mma_e4m3_cutlass_sfa_f32out(
            &mut d_out.as_view_mut(),
            d_w,
            cutlass_w,
            &d_xq.as_view(),
            &d_sfa_t.as_view(),
            kk,
            n,
            n_tokens,
            false,
        )?;
    }
    k.device().synchronize()?;
    Ok(t0.elapsed().as_secs_f64() * 1000.0 / reps as f64)
}

/// n_tokens=16 as two back-to-back `mmv_f8_plain_multi8` launches (8+8) on
/// the same stream -- what a real dispatch branch would issue for this
/// n_tokens, since the kernel itself declines above 8.
fn bench_plain_multi8_x2(
    k: &Kernels,
    d_w: &View<'_, u8>,
    kk: usize,
    n: usize,
    reps: usize,
) -> Result<f64> {
    let stream = k.device().stream().clone();
    let n_tokens = 16usize;
    let x: Vec<f32> = pseudo_random_f32(n_tokens * kk, 0xACE0);
    let d_x = stream.clone_htod(&x)?;
    let mut d_out = stream.alloc_zeros::<f32>(n_tokens * n)?;

    let run = |d_x: &infero_gpu::Buf<f32>, d_out: &mut infero_gpu::Buf<f32>| -> Result<()> {
        let x0 = d_x.slice(0..8 * kk);
        let mut o0 = d_out.slice_mut(0..8 * n);
        k.mmv_f8_plain_multi8(&mut o0, d_w, &x0, kk, n, 8, false)?;
        drop(o0);
        let x1 = d_x.slice(8 * kk..16 * kk);
        let mut o1 = d_out.slice_mut(8 * n..16 * n);
        k.mmv_f8_plain_multi8(&mut o1, d_w, &x1, kk, n, 8, false)?;
        Ok(())
    };

    for _ in 0..3 {
        run(&d_x, &mut d_out)?;
    }
    k.device().synchronize()?;
    let t0 = Instant::now();
    for _ in 0..reps {
        run(&d_x, &mut d_out)?;
    }
    k.device().synchronize()?;
    Ok(t0.elapsed().as_secs_f64() * 1000.0 / reps as f64)
}

fn bench_plain_multi8_single(
    k: &Kernels,
    d_w: &View<'_, u8>,
    kk: usize,
    n: usize,
    reps: usize,
) -> Result<f64> {
    let stream = k.device().stream().clone();
    let x: Vec<f32> = pseudo_random_f32(8 * kk, 0xACE0);
    let d_x = stream.clone_htod(&x)?;
    let mut d_out = stream.alloc_zeros::<f32>(8 * n)?;

    for _ in 0..3 {
        k.mmv_f8_plain_multi8(&mut d_out.as_view_mut(), d_w, &d_x.as_view(), kk, n, 8, false)?;
    }
    k.device().synchronize()?;
    let t0 = Instant::now();
    for _ in 0..reps {
        k.mmv_f8_plain_multi8(&mut d_out.as_view_mut(), d_w, &d_x.as_view(), kk, n, 8, false)?;
    }
    k.device().synchronize()?;
    Ok(t0.elapsed().as_secs_f64() * 1000.0 / reps as f64)
}

fn main() -> Result<()> {
    let k = Kernels::new(Device::new(0)?);
    let (kk, n) = (5120usize, 1024usize);
    let reps = 300;

    let quants = quant_bytes(n * kk, 0xE4A3);
    let scale_n = n / FP8_BLOCK;
    let scale_k = kk / FP8_BLOCK;
    let scales: Vec<f32> = (0..scale_n * scale_k).map(|i| 0.3 + 0.4 * (i % 5) as f32).collect();
    let w_buf = packed(&quants, &scales, kk, n);
    let stream = k.device().stream().clone();
    let d_w = stream.clone_htod(&w_buf)?;
    let cutlass_w = k.prepare_cutlass_weight(&d_w.as_view(), kk, n, false)?;

    println!("K={kk} N={n} n_tokens=16 (real attn_kv shape, real batch=16 decode)");

    let cutlass_ms = bench_cutlass_real_dispatch(&k, &d_w.as_view(), &cutlass_w, kk, n, 16, reps)?;
    let bytes = (kk * n + scale_n * scale_k * 4) as f64; // weight + scale bytes, read once regardless of n_tokens
    let cutlass_gbps = bytes / (cutlass_ms / 1000.0) / 1e9;
    println!("  real dispatch (small_m_swap tile):  {cutlass_ms:8.4} ms  {cutlass_gbps:7.1} GB/s");

    let multi8x2_ms = bench_plain_multi8_x2(&k, &d_w.as_view(), kk, n, reps)?;
    let multi8x2_gbps = (bytes * 2.0) / (multi8x2_ms / 1000.0) / 1e9; // weight read twice (once per 8-token launch)
    println!("  mmv_f8_plain_multi8 x2 (8+8):        {multi8x2_ms:8.4} ms  {multi8x2_gbps:7.1} GB/s (weight re-read 2x)");

    println!();
    println!("ratio (cutlass_ms / multi8x2_ms): {:.3}  ({} wins)", cutlass_ms / multi8x2_ms, if multi8x2_ms < cutlass_ms { "multi8x2" } else { "cutlass" });

    println!();
    println!("-- isolating the forced-split handicap: n_tokens=8 single launch, no doubling --");
    let cutlass8_ms = bench_cutlass_real_dispatch(&k, &d_w.as_view(), &cutlass_w, kk, n, 8, reps)?;
    let cutlass8_gbps = bytes / (cutlass8_ms / 1000.0) / 1e9;
    println!("  real dispatch @8:                   {cutlass8_ms:8.4} ms  {cutlass8_gbps:7.1} GB/s");
    let multi8_single_ms = bench_plain_multi8_single(&k, &d_w.as_view(), kk, n, reps)?;
    let multi8_single_gbps = bytes / (multi8_single_ms / 1000.0) / 1e9;
    println!("  mmv_f8_plain_multi8 @8 (1 launch):   {multi8_single_ms:8.4} ms  {multi8_single_gbps:7.1} GB/s");
    println!("  ratio (cutlass8_ms / multi8_single_ms): {:.3}  ({} wins)", cutlass8_ms / multi8_single_ms, if multi8_single_ms < cutlass8_ms { "multi8_single" } else { "cutlass" });

    Ok(())
}
