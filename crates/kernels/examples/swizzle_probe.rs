//! Real A/B: does CTA-rasterization swizzle (`TileSchedulerArguments::
//! max_swizzle_size`, never set anywhere in this file before) help the one
//! real shape whose weight exceeds this GPU's L2 -- the fused FFN gate/up
//! projection (`INFERO_FUSE_FFN=1`, K=5120,N=34816, ~178 MiB weight vs this
//! GPU's real 96-128 MiB L2, per vLLM's own real dispatch comment,
//! `scaled_mm_blockwise_sm120_fp8.cu`, fetched 2026-09-10)?
//!
//! vLLM's own logic: weight fits in L2 -> swizzle=1 (its own "no swizzle"
//! baseline); weight exceeds L2 -> swizzle=8. This file's own kernels have
//! never set this field at all, i.e. always run at the class default (`0`,
//! confirmed by reading `sm100_tile_scheduler.hpp` directly -- `0` and `1`
//! both resolve to `log_swizzle_size=0`, no actual raster change, so `0`
//! behaves identically to vLLM's own "no swizzle" baseline).
//!
//! Also checked at the un-fused gate/up shape (K=5120,N=17408, ~85 MiB,
//! under L2) as a control -- vLLM's own logic says this one should NOT
//! benefit from swizzling, so if it does, the L2-size-based theory itself
//! is wrong or this GPU's real L2 differs from vLLM's comment's 96-128 MiB
//! range.
//!
//!   cargo run --release -p infero-kernels --features cutlass --example swizzle_probe

#![cfg(feature = "cutlass")]

use std::time::Instant;

use anyhow::Result;
use infero_cuda::Device;
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

fn bench_swizzle(
    k: &Kernels,
    d_w: &infero_gpu::View<'_, u8>,
    cw: &infero_kernels::CutlassWeight,
    kk: usize,
    n: usize,
    n_tokens: usize,
    max_swizzle_size: i32,
    reps: usize,
) -> Result<f64> {
    let stream = k.device().stream().clone();
    let x: Vec<f32> = pseudo_random_f32(n_tokens * kk, 0xACE0);
    let d_x = stream.clone_htod(&x)?;
    let scale_cols = kk / ACT_QUANT_GROUP;
    let mut d_xq = stream.alloc_zeros::<u8>(n_tokens * kk)?;
    let mut d_sfa_t = stream.alloc_zeros::<f32>(scale_cols * n_tokens)?;
    k.quantize_act_e4m3_cutlass(&mut d_xq.as_view_mut(), &mut d_sfa_t.as_view_mut(), &d_x.as_view(), kk, n_tokens, n_tokens)?;
    let mut d_out = stream.alloc_zeros::<f32>(n_tokens * n)?;

    for _ in 0..3 {
        k.mma_e4m3_cutlass_sfa_f32out_swizzle_bench(
            &mut d_out.as_view_mut(), d_w, cw, &d_xq.as_view(), &d_sfa_t.as_view(), kk, n, n_tokens, false, max_swizzle_size,
        )?;
    }
    k.device().synchronize()?;
    let t0 = Instant::now();
    for _ in 0..reps {
        k.mma_e4m3_cutlass_sfa_f32out_swizzle_bench(
            &mut d_out.as_view_mut(), d_w, cw, &d_xq.as_view(), &d_sfa_t.as_view(), kk, n, n_tokens, false, max_swizzle_size,
        )?;
    }
    k.device().synchronize()?;
    Ok(t0.elapsed().as_secs_f64() * 1000.0 / reps as f64)
}

fn run_shape(k: &Kernels, kk: usize, n: usize, label: &str) -> Result<()> {
    let stream = k.device().stream().clone();
    let quants = quant_bytes(n * kk, 0xE4A3);
    let scale_n = n / FP8_BLOCK;
    let scale_k = kk / FP8_BLOCK;
    let scales: Vec<f32> = (0..scale_n * scale_k).map(|i| 0.3 + 0.4 * (i % 5) as f32).collect();
    let w_buf = packed(&quants, &scales, kk, n);
    let d_w = stream.clone_htod(&w_buf)?;
    let cutlass_w = k.prepare_cutlass_weight(&d_w.as_view(), kk, n, false)?;
    let weight_mib = (kk * n) as f64 / (1 << 20) as f64;

    println!("K={kk} N={n} ({label}, weight={weight_mib:.1} MiB)");
    println!("{:>8}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}", "tokens", "sw=0(ms)", "sw=1(ms)", "sw=2(ms)", "sw=4(ms)", "sw=8(ms)");
    for n_tokens in [16usize, 32, 64, 128] {
        let reps = 200;
        let mut times = vec![];
        for sw in [0i32, 1, 2, 4, 8] {
            times.push(bench_swizzle(k, &d_w.as_view(), &cutlass_w, kk, n, n_tokens, sw, reps)?);
        }
        println!("{n_tokens:>8}  {:>10.4}  {:>10.4}  {:>10.4}  {:>10.4}  {:>10.4}", times[0], times[1], times[2], times[3], times[4]);
    }
    Ok(())
}

fn main() -> Result<()> {
    let k = Kernels::new(Device::new(0)?);
    run_shape(&k, 5120, 34816, "real fused gate_up, INFERO_FUSE_FFN=1 -- exceeds this GPU's L2")?;
    run_shape(&k, 5120, 17408, "real un-fused gate/up -- under this GPU's L2 (control)")?;
    Ok(())
}
