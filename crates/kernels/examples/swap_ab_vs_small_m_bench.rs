//! Head-to-head timing: the operand-swapped small-M CUTLASS tile
//! (`fp8_bw_gemm.cu`'s `small_m_swap`, `<128,32,128>`) against the plain
//! small-M tile it replaced in the real dispatch (`small_m`, `<64,128,128>`)
//! and the wide default (`<128,128,128>`), at both real qwen38-27b-fp8
//! dense-FFN shapes (gate/up: K=5120,N=17408; down: K=17408,N=5120 -- the
//! transpose of gate/up, checking whether the swap threshold this file's
//! own dispatch uses (measured only at gate/up) generalizes to a shape with
//! the wide/narrow roles reversed) across the n_tokens range that matters
//! for the batch=16-decode investigation this kernel was built for.
//! Uses [`infero_kernels::Kernels::mma_e4m3_cutlass_sfa_f32out_bench`],
//! which bypasses the real `SMALL_M_MAX_TOKENS`/`SWAP_AB_MAX_TOKENS`
//! dispatch to force one specific tile per call -- not a path any
//! production code takes.
#![cfg(feature = "cutlass")]

use std::time::Instant;

use anyhow::Result;
use infero_cuda::Device;
use infero_gpu::View;
use infero_kernels::cutlass_fp8::BenchTile;
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

#[allow(clippy::too_many_arguments)]
fn bench_tile(
    k: &Kernels,
    d_w: &View<'_, u8>,
    cutlass_w: &infero_kernels::CutlassWeight,
    kk: usize,
    n: usize,
    n_tokens: usize,
    tile: BenchTile,
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
        k.mma_e4m3_cutlass_sfa_f32out_bench(
            &mut d_out.as_view_mut(),
            d_w,
            cutlass_w,
            &d_xq.as_view(),
            &d_sfa_t.as_view(),
            kk,
            n,
            n_tokens,
            false,
            tile,
        )?;
    }
    k.device().synchronize()?;
    let t0 = Instant::now();
    for _ in 0..reps {
        k.mma_e4m3_cutlass_sfa_f32out_bench(
            &mut d_out.as_view_mut(),
            d_w,
            cutlass_w,
            &d_xq.as_view(),
            &d_sfa_t.as_view(),
            kk,
            n,
            n_tokens,
            false,
            tile,
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

    println!("K={kk} N={n} ({label})");
    println!(
        "{:>8}  {:>12}  {:>12}  {:>12}  {:>12}  {:>14}  {:>10}  {:>10}",
        "tokens", "small_m(ms)", "swap(ms)", "default(ms)", "stream_k(ms)", "swap_streamk(ms)", "swap/small_m", "swap/default"
    );
    for n_tokens in [1usize, 2, 4, 8, 16, 24, 32, 48, 64] {
        let reps = if n_tokens <= 16 { 200 } else { 100 };
        let small_m = bench_tile(k, &d_w.as_view(), &cutlass_w, kk, n, n_tokens, BenchTile::SmallM, reps)?;
        let swap = bench_tile(k, &d_w.as_view(), &cutlass_w, kk, n, n_tokens, BenchTile::SmallMSwap, reps)?;
        let default = bench_tile(k, &d_w.as_view(), &cutlass_w, kk, n, n_tokens, BenchTile::Default, reps)?;
        let stream_k = bench_tile(k, &d_w.as_view(), &cutlass_w, kk, n, n_tokens, BenchTile::StreamK, reps)?;
        let swap_streamk =
            bench_tile(k, &d_w.as_view(), &cutlass_w, kk, n, n_tokens, BenchTile::SmallMSwapStreamK, reps)?;
        println!(
            "{n_tokens:>8}  {small_m:>12.4}  {swap:>12.4}  {default:>12.4}  {stream_k:>12.4}  {swap_streamk:>14.4}  {:>10.3}  {:>10.3}",
            swap / small_m,
            swap / default
        );
    }
    Ok(())
}

fn main() -> Result<()> {
    let k = Kernels::new(Device::new(0)?);
    run_shape(&k, 5120, 17408, "real qwen38-27b-fp8 FFN gate/up projection")?;
    run_shape(&k, 17408, 5120, "real qwen38-27b-fp8 FFN down projection -- the transpose")?;
    run_shape(&k, 5120, 34816, "real qwen38-27b-fp8 fused gate_up (INFERO_FUSE_FFN=1) -- N doubled")?;
    run_shape(&k, 5120, 1024, "real qwen38-27b-fp8 attention K/V projection -- 8 swap-tiles, most idle-SM headroom")?;
    run_shape(&k, 5120, 6144, "real qwen38-27b-fp8 GDN in_proj_z -- 48 swap-tiles")?;
    Ok(())
}
