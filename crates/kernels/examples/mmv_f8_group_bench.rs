//! `mmv_f8_plain` (four rows a block) against `mmv_f8_plain_g1` (one row a
//! block) at the 27B's real FFN shapes, reporting achieved DRAM bandwidth so
//! the two can be judged against this GPU's real peak (~1600-1800 GB/s on
//! the RTX PRO 6000 Blackwell this was measured on), not just against each
//! other.
//!
//!   INFERO_CUTLASS_DIR=... INFERO_NVCC=... cargo run --release -p infero-kernels \
//!       --features cutlass --example mmv_f8_group_bench

use std::time::Instant;

use anyhow::Result;
use infero_gpu::Device;
use infero_kernels::Kernels;
use infero_kernels::fp8::FP8_BLOCK;

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

fn bench_shape(k: &Kernels, k_dim: usize, n_dim: usize, reps: usize) -> Result<()> {
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
    let x: Vec<f32> = pseudo_random_f32(k_dim, 0xACE0);
    let d_x = stream.clone_htod(&x)?;

    let mut d_out = stream.alloc_zeros::<f32>(n_dim)?;
    for _ in 0..3 {
        k.mmv_f8_plain(&mut d_out.as_view_mut(), &d_w.as_view(), &d_x.as_view(), k_dim, n_dim, false)?;
    }
    k.device().synchronize()?;
    let t0 = Instant::now();
    for _ in 0..reps {
        k.mmv_f8_plain(&mut d_out.as_view_mut(), &d_w.as_view(), &d_x.as_view(), k_dim, n_dim, false)?;
    }
    k.device().synchronize()?;
    let ms_g4 = t0.elapsed().as_secs_f64() * 1000.0 / reps as f64;

    let mut d_out_g1 = stream.alloc_zeros::<f32>(n_dim)?;
    for _ in 0..3 {
        k.mmv_f8_plain_g1(&mut d_out_g1.as_view_mut(), &d_w.as_view(), &d_x.as_view(), k_dim, n_dim, false)?;
    }
    k.device().synchronize()?;
    let t0 = Instant::now();
    for _ in 0..reps {
        k.mmv_f8_plain_g1(&mut d_out_g1.as_view_mut(), &d_w.as_view(), &d_x.as_view(), k_dim, n_dim, false)?;
    }
    k.device().synchronize()?;
    let ms_g1 = t0.elapsed().as_secs_f64() * 1000.0 / reps as f64;

    let got_g4 = stream.clone_dtoh(&d_out)?;
    let got_g1 = stream.clone_dtoh(&d_out_g1)?;
    let max_diff = got_g4.iter().zip(&got_g1).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);

    let bytes = (n_dim * k_dim) as f64;
    let gbps_g4 = bytes / (ms_g4 / 1000.0) / 1e9;
    let gbps_g1 = bytes / (ms_g1 / 1000.0) / 1e9;
    let blocks_g4 = n_dim.div_ceil(4);
    println!(
        "K={k_dim:6} N={n_dim:6}  g4(blocks={blocks_g4:5}) {ms_g4:8.4} ms {gbps_g4:7.1} GB/s   \
         g1(blocks={n_dim:5}) {ms_g1:8.4} ms {gbps_g1:7.1} GB/s   speedup {:5.2}x   max_diff {max_diff:.4}",
        ms_g4 / ms_g1,
    );
    Ok(())
}

fn main() -> Result<()> {
    let k = Kernels::new(Device::new(0)?);
    println!("gate/up shape (K=5120 -> N=17408):");
    bench_shape(&k, 5120, 17408, 200)?;
    println!("down shape (K=17408 -> N=5120):");
    bench_shape(&k, 17408, 5120, 200)?;
    // Attention/GDN projection shapes are smaller on both dims -- worth
    // checking g1 doesn't regress where g4's grid was already adequate.
    println!("qkvo-ish shape (K=5120 -> N=5120):");
    bench_shape(&k, 5120, 5120, 200)?;
    println!("small shape (K=5120 -> N=1024):");
    bench_shape(&k, 5120, 1024, 200)?;
    println!("tiny shape (K=256 -> N=128):");
    bench_shape(&k, 256, 128, 200)?;
    println!("head-ish shape (K=128 -> N=5120):");
    bench_shape(&k, 128, 5120, 200)?;
    Ok(())
}
