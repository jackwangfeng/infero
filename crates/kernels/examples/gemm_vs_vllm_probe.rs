//! Standalone clean-timed CUTLASS FP8 GEMM probe at the real qwen38-27b-fp8
//! FFN gate-projection shape (M=8192, K=5120, N=17408), for a direct
//! side-by-side against vLLM's own registered FP8 GEMM op at the identical
//! shape. Throwaway probe, not part of the correctness test suite.
#![cfg(feature = "cutlass")]

use anyhow::Result;
use infero_cuda::Device;
use infero_kernels::Kernels;
use infero_kernels::fp8::{ACT_QUANT_GROUP, FP8_BLOCK, fp8_bytes, repack_rows};

const M: usize = 2048;
const K: usize = 5120;
const N: usize = 17408;

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
    let mut v = repack_rows(quants, k, n).expect("repack");
    for s in scales {
        v.extend_from_slice(&s.to_le_bytes());
    }
    assert_eq!(v.len(), fp8_bytes(k, n));
    v
}

fn main() -> Result<()> {
    let k = Kernels::new(Device::new(0)?);
    let stream = k.device().stream().clone();

    let quants = quant_bytes(N * K, 0xE4A3);
    let scale_grid_n = N / FP8_BLOCK;
    let scale_grid_k = K / FP8_BLOCK;
    let scales: Vec<f32> = (0..scale_grid_n * scale_grid_k)
        .map(|i| 0.3 + 0.4 * (i % 5) as f32)
        .collect();
    let w_buf = packed(&quants, &scales, K, N);
    let d_w = stream.clone_htod(&w_buf)?;
    let cutlass_w = k.prepare_cutlass_weight(&d_w.as_view(), K, N, false)?;

    let x: Vec<f32> = (0..M)
        .flat_map(|t| pseudo_random_f32(K, 0xACE0 + t as u64, 4.0))
        .collect();
    let d_x = stream.clone_htod(&x)?;
    let scale_cols = K / ACT_QUANT_GROUP;
    let mut d_xq = stream.alloc_zeros::<u8>(M * K)?;
    let mut d_xs = stream.alloc_zeros::<f32>(M * scale_cols)?;
    k.quantize_act_e4m3(
        &mut d_xq.as_view_mut(),
        &mut d_xs.as_view_mut(),
        &d_x.as_view(),
        K,
        M,
    )?;

    let mut d_out = stream.alloc_zeros::<f32>(M * N)?;

    // warmup
    for _ in 0..3 {
        k.mma_e4m3_cutlass_sfa_f32out(
            &mut d_out.as_view_mut(),
            &d_w.as_view(),
            &cutlass_w,
            &d_xq.as_view(),
            &d_xs.as_view(),
            K,
            N,
            M,
            false,
        )?;
    }
    k.device().synchronize()?;

    let reps = 20;
    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        k.mma_e4m3_cutlass_sfa_f32out(
            &mut d_out.as_view_mut(),
            &d_w.as_view(),
            &cutlass_w,
            &d_xq.as_view(),
            &d_xs.as_view(),
            K,
            N,
            M,
            false,
        )?;
    }
    k.device().synchronize()?;
    let elapsed = t0.elapsed();
    println!(
        "cutlass_fp8_gemm_f32out M={M} K={K} N={N}: {:.4} ms/call ({reps} reps, clean host timing, no profiler)",
        elapsed.as_secs_f64() * 1000.0 / reps as f64
    );
    Ok(())
}
