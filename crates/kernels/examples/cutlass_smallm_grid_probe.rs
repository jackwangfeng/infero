//! Checks whether the CUTLASS small-M path (`<64,128,128>` tile, no split-K)
//! has the same "grid too narrow for a small N" problem `mmv_f8_group_bench`
//! found and fixed for the plain scalar matvec -- at real n_tokens=2 (the
//! dual-stream decode shape), a GEMM's grid is just
//! `ceil(m/64) * ceil(n/128)` tiles with this schedule, no K-splitting, so a
//! small `n` starves the grid the same way a small matvec `n` did.
//!
//!   INFERO_CUTLASS_DIR=... INFERO_NVCC=... cargo run --release -p infero-kernels \
//!       --features cutlass --example cutlass_smallm_grid_probe

use std::time::Instant;

use anyhow::Result;
use infero_gpu::Device;
use infero_kernels::Kernels;
use infero_kernels::fp8::{ACT_QUANT_GROUP, FP8_BLOCK};

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

fn bench(k: &Kernels, k_dim: usize, n_dim: usize, n_tokens: usize, reps: usize) -> Result<()> {
    let stream = k.device().stream().clone();
    let quants = quant_bytes(n_dim * k_dim, 0xE4A3);
    let scale_n = n_dim / FP8_BLOCK;
    let scale_k = k_dim / FP8_BLOCK;
    let scales: Vec<f32> = (0..scale_n * scale_k).map(|i| 0.3 + 0.4 * (i % 5) as f32).collect();
    let w_buf = packed(&quants, &scales, k_dim, n_dim);
    let d_w = stream.clone_htod(&w_buf)?;
    let cutlass_w = k.prepare_cutlass_weight(&d_w.as_view(), k_dim, n_dim, false)?;

    let x: Vec<f32> = pseudo_random_f32(n_tokens * k_dim, 0xACE0);
    let d_x = stream.clone_htod(&x)?;
    let scale_cols = k_dim / ACT_QUANT_GROUP;
    let mut d_xq = stream.alloc_zeros::<u8>(n_tokens * k_dim)?;
    let mut d_sfa_t = stream.alloc_zeros::<f32>(scale_cols * n_tokens)?;
    k.quantize_act_e4m3_cutlass(
        &mut d_xq.as_view_mut(), &mut d_sfa_t.as_view_mut(), &d_x.as_view(), k_dim, n_tokens, n_tokens,
    )?;
    let mut d_out = stream.alloc_zeros::<f32>(n_tokens * n_dim)?;

    for _ in 0..3 {
        k.mma_e4m3_cutlass_sfa_f32out(
            &mut d_out.as_view_mut(), &d_w.as_view(), &cutlass_w, &d_xq.as_view(), &d_sfa_t.as_view(),
            k_dim, n_dim, n_tokens, false,
        )?;
    }
    k.device().synchronize()?;
    let t0 = Instant::now();
    for _ in 0..reps {
        k.mma_e4m3_cutlass_sfa_f32out(
            &mut d_out.as_view_mut(), &d_w.as_view(), &cutlass_w, &d_xq.as_view(), &d_sfa_t.as_view(),
            k_dim, n_dim, n_tokens, false,
        )?;
    }
    k.device().synchronize()?;
    let ms = t0.elapsed().as_secs_f64() * 1000.0 / reps as f64;

    let bytes = (n_dim * k_dim) as f64;
    let gbps = bytes / (ms / 1000.0) / 1e9;
    let n_tiles = n_dim.div_ceil(128);
    let m_tiles = n_tokens.div_ceil(64);
    println!(
        "K={k_dim:6} N={n_dim:6} tokens={n_tokens:3}  tiles={m_tiles}x{n_tiles:3}={:4}  {ms:8.4} ms  {gbps:7.1} GB/s",
        m_tiles * n_tiles,
    );
    Ok(())
}

fn bench_multi8(k: &Kernels, k_dim: usize, n_dim: usize, n_tokens: usize, reps: usize) -> Result<()> {
    let stream = k.device().stream().clone();
    let quants = quant_bytes(n_dim * k_dim, 0xE4A3);
    let scale_n = n_dim / FP8_BLOCK;
    let scale_k = k_dim / FP8_BLOCK;
    let scales: Vec<f32> = (0..scale_n * scale_k).map(|i| 0.3 + 0.4 * (i % 5) as f32).collect();
    let mut w_plain_buf = infero_kernels::fp8::pad_rows(&quants, k_dim, n_dim)?;
    for s in &scales {
        w_plain_buf.extend_from_slice(&s.to_le_bytes());
    }
    let d_w = stream.clone_htod(&w_plain_buf)?;
    let x: Vec<f32> = pseudo_random_f32(n_tokens * k_dim, 0xACE0);
    let d_x = stream.clone_htod(&x)?;
    let mut d_out = stream.alloc_zeros::<f32>(n_tokens * n_dim)?;

    for _ in 0..3 {
        k.mmv_f8_plain_multi8(&mut d_out.as_view_mut(), &d_w.as_view(), &d_x.as_view(), k_dim, n_dim, n_tokens, false)?;
    }
    k.device().synchronize()?;
    let t0 = Instant::now();
    for _ in 0..reps {
        k.mmv_f8_plain_multi8(&mut d_out.as_view_mut(), &d_w.as_view(), &d_x.as_view(), k_dim, n_dim, n_tokens, false)?;
    }
    k.device().synchronize()?;
    let ms = t0.elapsed().as_secs_f64() * 1000.0 / reps as f64;
    let bytes = (n_dim * k_dim) as f64;
    let gbps = bytes / (ms / 1000.0) / 1e9;
    println!("K={k_dim:6} N={n_dim:6} tokens={n_tokens:3}  mmv_f8_plain_multi8  {ms:8.4} ms  {gbps:7.1} GB/s");
    Ok(())
}

fn bench_multi2(k: &Kernels, k_dim: usize, n_dim: usize, reps: usize) -> Result<()> {
    let stream = k.device().stream().clone();
    let n_tokens = 2usize;
    let quants = quant_bytes(n_dim * k_dim, 0xE4A3);
    let scale_n = n_dim / FP8_BLOCK;
    let scale_k = k_dim / FP8_BLOCK;
    let scales: Vec<f32> = (0..scale_n * scale_k).map(|i| 0.3 + 0.4 * (i % 5) as f32).collect();
    let mut w_plain_buf = infero_kernels::fp8::pad_rows(&quants, k_dim, n_dim)?;
    for s in &scales {
        w_plain_buf.extend_from_slice(&s.to_le_bytes());
    }
    let d_w = stream.clone_htod(&w_plain_buf)?;
    let x: Vec<f32> = pseudo_random_f32(n_tokens * k_dim, 0xACE0);
    let d_x = stream.clone_htod(&x)?;
    let mut d_out = stream.alloc_zeros::<f32>(n_tokens * n_dim)?;

    for _ in 0..3 {
        k.mmv_f8_plain_multi2(&mut d_out.as_view_mut(), &d_w.as_view(), &d_x.as_view(), k_dim, n_dim, false)?;
    }
    k.device().synchronize()?;
    let t0 = Instant::now();
    for _ in 0..reps {
        k.mmv_f8_plain_multi2(&mut d_out.as_view_mut(), &d_w.as_view(), &d_x.as_view(), k_dim, n_dim, false)?;
    }
    k.device().synchronize()?;
    let ms = t0.elapsed().as_secs_f64() * 1000.0 / reps as f64;
    let bytes = (n_dim * k_dim) as f64;
    let gbps = bytes / (ms / 1000.0) / 1e9;
    println!("K={k_dim:6} N={n_dim:6} tokens=  2  mmv_f8_plain_multi2  {ms:8.4} ms  {gbps:7.1} GB/s");
    Ok(())
}

fn main() -> Result<()> {
    let k = Kernels::new(Device::new(0)?);
    println!("-- n_tokens=2 (the real dual-stream decode shape) --");
    bench(&k, 5120, 17408, 2, 200)?;
    bench(&k, 17408, 5120, 2, 200)?;
    bench(&k, 5120, 5120, 2, 200)?;
    bench(&k, 5120, 1024, 2, 200)?;
    println!("-- n_tokens=64 (small_m's own upper edge) for comparison --");
    bench(&k, 5120, 17408, 64, 100)?;
    bench(&k, 17408, 5120, 64, 100)?;
    println!("-- mmv_f8_plain_multi8 at n_tokens=2 (compare against the CUTLASS numbers above) --");
    bench_multi8(&k, 5120, 17408, 2, 200)?;
    bench_multi8(&k, 17408, 5120, 2, 200)?;
    bench_multi8(&k, 5120, 5120, 2, 200)?;
    bench_multi8(&k, 5120, 1024, 2, 200)?;
    println!("-- mmv_f8_plain_multi2 (TOKENS=2 exact instantiation) --");
    bench_multi2(&k, 5120, 17408, 200)?;
    bench_multi2(&k, 17408, 5120, 200)?;
    bench_multi2(&k, 5120, 5120, 200)?;
    bench_multi2(&k, 5120, 1024, 200)?;
    println!("-- crossover search: CUTLASS vs mmv_f8_plain_multi2 at N=1024,2048,3072,4096,6144 (K=5120) --");
    for n in [1024, 2048, 3072, 4096, 6144] {
        bench(&k, 5120, n, 2, 200)?;
        bench_multi2(&k, 5120, n, 200)?;
    }
    Ok(())
}
