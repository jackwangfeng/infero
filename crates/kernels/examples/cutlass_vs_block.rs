//! Wall-clock `mma_e4m3_cutlass` against `mma_e4m3_block` at the 27B's real
//! FFN shapes. `cutlass` feature only.
//!
//!   INFERO_CUTLASS_DIR=... INFERO_NVCC=... cargo run --release -p infero-kernels \
//!       --features cutlass --example cutlass_vs_block

use std::time::Instant;

use anyhow::Result;
use infero_gpu::Device;
use infero_kernels::Kernels;
use infero_kernels::fp8::{ACT_QUANT_GROUP, FP8_BLOCK, fp8_bytes};

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
    assert_eq!(v.len(), fp8_bytes(k, n));
    v
}

fn bench_shape(k: &Kernels, k_dim: usize, n_dim: usize, n_tokens: usize, reps: usize) -> Result<()> {
    let stream = k.device().stream().clone();
    let quants = quant_bytes(n_dim * k_dim, 0xE4A3);
    let scale_n = n_dim / FP8_BLOCK;
    let scale_k = k_dim / FP8_BLOCK;
    let scales: Vec<f32> = (0..scale_n * scale_k).map(|i| 0.3 + 0.4 * (i % 5) as f32).collect();
    let w_buf = packed(&quants, &scales, k_dim, n_dim);
    let d_w = stream.clone_htod(&w_buf)?;

    let x: Vec<f32> = pseudo_random_f32(n_tokens * k_dim, 0xACE0);
    let d_x = stream.clone_htod(&x)?;
    let scale_cols = k_dim / ACT_QUANT_GROUP;
    let mut d_xq = stream.alloc_zeros::<u8>(n_tokens * k_dim)?;
    let mut d_xs = stream.alloc_zeros::<f32>(n_tokens * scale_cols)?;
    k.quantize_act_e4m3(&mut d_xq.as_view_mut(), &mut d_xs.as_view_mut(), &d_x.as_view(), k_dim, n_tokens)?;

    let mut d_out = stream.alloc_zeros::<f32>(n_tokens * n_dim)?;
    let flops = 2.0 * n_tokens as f64 * k_dim as f64 * n_dim as f64;

    // warmup + correctness-agnostic timing
    let mut block_ran = true;
    for _ in 0..3 {
        block_ran = k.mma_e4m3_block(
            &mut d_out.as_view_mut(), &d_w.as_view(), &d_xq.as_view(), &d_xs.as_view(),
            k_dim, n_dim, n_tokens, false,
        )?;
    }
    k.device().synchronize()?;
    let block_ms = if block_ran {
        let t0 = Instant::now();
        for _ in 0..reps {
            k.mma_e4m3_block(
                &mut d_out.as_view_mut(), &d_w.as_view(), &d_xq.as_view(), &d_xs.as_view(),
                k_dim, n_dim, n_tokens, false,
            )?;
        }
        k.device().synchronize()?;
        Some(t0.elapsed().as_secs_f64() * 1000.0 / reps as f64)
    } else {
        None
    };

    let cutlass_w = k.prepare_cutlass_weight(&d_w.as_view(), k_dim, n_dim, false)?;
    for _ in 0..3 {
        k.mma_e4m3_cutlass(
            &mut d_out.as_view_mut(), &d_w.as_view(), &cutlass_w, &d_xq.as_view(), &d_xs.as_view(),
            k_dim, n_dim, n_tokens, false,
        )?;
    }
    k.device().synchronize()?;
    let t0 = Instant::now();
    for _ in 0..reps {
        k.mma_e4m3_cutlass(
            &mut d_out.as_view_mut(), &d_w.as_view(), &cutlass_w, &d_xq.as_view(), &d_xs.as_view(),
            k_dim, n_dim, n_tokens, false,
        )?;
    }
    k.device().synchronize()?;
    let cutlass_ms = t0.elapsed().as_secs_f64() * 1000.0 / reps as f64;

    match block_ms {
        Some(block_ms) => println!(
            "K={k_dim:6} N={n_dim:6} tokens={n_tokens:6}  block {block_ms:8.4} ms ({:6.1} TFLOPS)  \
             cutlass {cutlass_ms:8.4} ms ({:6.1} TFLOPS)  speedup {:5.2}x",
            flops / block_ms / 1e9,
            flops / cutlass_ms / 1e9,
            block_ms / cutlass_ms,
        ),
        None => println!(
            "K={k_dim:6} N={n_dim:6} tokens={n_tokens:6}  block declined  \
             cutlass {cutlass_ms:8.4} ms ({:6.1} TFLOPS)",
            flops / cutlass_ms / 1e9,
        ),
    }
    Ok(())
}

fn bench_mmv_floor(k: &Kernels, k_dim: usize, n_dim: usize, reps: usize) -> Result<()> {
    let stream = k.device().stream().clone();
    let quants = quant_bytes(n_dim * k_dim, 0xE4A3);
    let scale_n = n_dim / FP8_BLOCK;
    let scale_k = k_dim / FP8_BLOCK;
    let scales: Vec<f32> = (0..scale_n * scale_k).map(|i| 0.3 + 0.4 * (i % 5) as f32).collect();
    let w_buf = packed(&quants, &scales, k_dim, n_dim);
    let d_w = stream.clone_htod(&w_buf)?;
    let x: Vec<f32> = pseudo_random_f32(k_dim, 0xACE0);
    let d_x = stream.clone_htod(&x)?;
    let mut d_out = stream.alloc_zeros::<f32>(n_dim)?;

    for _ in 0..3 {
        k.mmv_f8_block(&mut d_out.as_view_mut(), &d_w.as_view(), &d_x.as_view(), k_dim, n_dim, false)?;
    }
    k.device().synchronize()?;
    let t0 = Instant::now();
    for _ in 0..reps {
        k.mmv_f8_block(&mut d_out.as_view_mut(), &d_w.as_view(), &d_x.as_view(), k_dim, n_dim, false)?;
    }
    k.device().synchronize()?;
    let ms = t0.elapsed().as_secs_f64() * 1000.0 / reps as f64;

    // Plain [n,k] row-major quants, no repack_rows -- CUTLASS's native
    // layout, to see if the interleave is still needed on this GPU.
    let mut w_plain_buf = infero_kernels::fp8::pad_rows(&quants, k_dim, n_dim)?;
    for s in &scales {
        w_plain_buf.extend_from_slice(&s.to_le_bytes());
    }
    let d_w_plain = stream.clone_htod(&w_plain_buf)?;
    let mut d_out2 = stream.alloc_zeros::<f32>(n_dim)?;
    for _ in 0..3 {
        k.mmv_f8_plain(&mut d_out2.as_view_mut(), &d_w_plain.as_view(), &d_x.as_view(), k_dim, n_dim, false)?;
    }
    k.device().synchronize()?;
    let t0 = Instant::now();
    for _ in 0..reps {
        k.mmv_f8_plain(&mut d_out2.as_view_mut(), &d_w_plain.as_view(), &d_x.as_view(), k_dim, n_dim, false)?;
    }
    k.device().synchronize()?;
    let ms_plain = t0.elapsed().as_secs_f64() * 1000.0 / reps as f64;

    let got = stream.clone_dtoh(&d_out)?;
    let got2 = stream.clone_dtoh(&d_out2)?;
    let max_diff = got.iter().zip(&got2).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);

    println!(
        "K={k_dim:6} N={n_dim:6} tokens=     1  mmv_f8_block(interleaved) {ms:8.4} ms   \
         mmv_f8_plain(no repack) {ms_plain:8.4} ms   ratio {:5.2}x   max_diff {max_diff:.4}",
        ms_plain / ms,
    );
    Ok(())
}

/// The bf16-scratch path (`quantize_act_e4m3_cutlass` + `mma_e4m3_cutlass_sfa`,
/// what the real server runs) against the f32-direct path
/// (`mma_e4m3_cutlass_sfa_f32out`, no separate store kernel) at the real
/// checkpoint's shapes and `batch_tokens`.
fn bench_f32out(k: &Kernels, k_dim: usize, n_dim: usize, n_tokens: usize, reps: usize) -> Result<()> {
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
        k.mma_e4m3_cutlass_sfa(
            &mut d_out.as_view_mut(), &d_w.as_view(), &cutlass_w, &d_xq.as_view(), &d_sfa_t.as_view(),
            k_dim, n_dim, n_tokens, n_tokens, false,
        )?;
    }
    k.device().synchronize()?;
    let t0 = Instant::now();
    for _ in 0..reps {
        k.mma_e4m3_cutlass_sfa(
            &mut d_out.as_view_mut(), &d_w.as_view(), &cutlass_w, &d_xq.as_view(), &d_sfa_t.as_view(),
            k_dim, n_dim, n_tokens, n_tokens, false,
        )?;
    }
    k.device().synchronize()?;
    let bf16_ms = t0.elapsed().as_secs_f64() * 1000.0 / reps as f64;

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
    let f32out_ms = t0.elapsed().as_secs_f64() * 1000.0 / reps as f64;

    println!(
        "K={k_dim:6} N={n_dim:6} tokens={n_tokens:6}  bf16-path {bf16_ms:8.4} ms  f32out {f32out_ms:8.4} ms  \
         speedup {:5.2}x",
        bf16_ms / f32out_ms,
    );
    Ok(())
}

fn main() -> Result<()> {
    let k = Kernels::new(Device::new(0)?);
    println!("gate/up shape (K=5120 -> N=17408):");
    bench_mmv_floor(&k, 5120, 17408, 50)?;
    for n_tokens in [1, 2, 4, 8, 16, 32, 64, 96, 128, 256, 1024, 4096] {
        bench_shape(&k, 5120, 17408, n_tokens, 20)?;
    }
    println!("down shape (K=17408 -> N=5120):");
    bench_mmv_floor(&k, 17408, 5120, 50)?;
    for n_tokens in [1, 2, 4, 8, 16, 32, 64, 96, 128, 256, 1024, 4096] {
        bench_shape(&k, 17408, 5120, n_tokens, 20)?;
    }
    println!("f32out vs bf16-scratch (real batch_tokens shapes):");
    for n_tokens in [128, 256, 1024] {
        bench_f32out(&k, 5120, 17408, n_tokens, 50)?;
        bench_f32out(&k, 17408, 5120, n_tokens, 50)?;
    }
    Ok(())
}
