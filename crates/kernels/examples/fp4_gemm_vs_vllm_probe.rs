//! Standalone clean-timed CUTLASS NVFP4 GEMM probe at the real
//! Qwen3.8-27B-NVFP4 FFN gate-projection prefill shape (M=2048, K=5120,
//! N=17408), for a direct side-by-side against vLLM's own registered NVFP4
//! GEMM op (`cutlass_scaled_fp4_mm`) at the identical shape. Throwaway
//! probe, not part of the correctness test suite.
#![cfg(feature = "cutlass")]

use anyhow::Result;
use infero_cuda::Device;
use infero_kernels::Kernels;
use infero_kernels::fp4::F4E2M1_BLOCK;

const M: usize = 2048;
const K: usize = 5120;
const N: usize = 17408;

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

fn packed_weight(n: usize, k: usize, seed: u64) -> (Vec<u8>, Vec<u8>) {
    let mut s = seed | 1;
    let mut next_nibble = || -> u8 {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 24) & 0x0F) as u8
    };
    let bytes_per_row = k.div_ceil(2);
    let quants: Vec<u8> = (0..n * bytes_per_row)
        .map(|_| {
            let lo = next_nibble();
            let hi = next_nibble();
            lo | (hi << 4)
        })
        .collect();
    const SCALE_CODES: [u8; 5] = [0x38, 0x3C, 0x30, 0x34, 0x40];
    let blocks = k.div_ceil(F4E2M1_BLOCK);
    let scale_bytes: Vec<u8> = (0..n * blocks).map(|i| SCALE_CODES[i % SCALE_CODES.len()]).collect();
    (quants, scale_bytes)
}

fn main() -> Result<()> {
    let k = Kernels::new(Device::new(0)?);
    let stream = k.device().stream().clone();

    let (w_quants, w_scale_bytes) = packed_weight(N, K, 0xF4A1);
    let mut w_buf = w_quants.clone();
    w_buf.extend_from_slice(&w_scale_bytes);
    let d_w = stream.clone_htod(&w_buf)?;
    // Real, measured lm_head-scale magnitudes from this session's own bug
    // investigation (order-of-magnitude representative of a real checkpoint).
    let weight_scale_2 = 0.000_127_156_57_f32;
    let input_scale = 0.021_670_388_f32;
    let cw = k.prepare_cutlass_fp4_weight(&d_w.as_view(), K, N, weight_scale_2, input_scale)?;

    let x: Vec<f32> = (0..M).flat_map(|t| pseudo_random_f32(K, 0xACE0 + t as u64, 4.0)).collect();
    let d_x = stream.clone_htod(&x)?;

    let blocks = K.div_ceil(F4E2M1_BLOCK);
    let xq_len = M * K.div_ceil(2);
    let xs_len = M * blocks;
    let mut d_xq = stream.alloc_zeros::<u8>(xq_len)?;
    let mut d_xs = stream.alloc_zeros::<u8>(xs_len)?;
    for _ in 0..3 {
        k.quantize_act_e2m1_cutlass(
            &mut d_xq.as_view_mut(),
            &mut d_xs.as_view_mut(),
            &d_x.as_view(),
            1.0 / input_scale,
            K,
            M,
        )?;
    }
    k.device().synchronize()?;
    let qreps = 20;
    let qt0 = std::time::Instant::now();
    for _ in 0..qreps {
        k.quantize_act_e2m1_cutlass(
            &mut d_xq.as_view_mut(),
            &mut d_xs.as_view_mut(),
            &d_x.as_view(),
            1.0 / input_scale,
            K,
            M,
        )?;
    }
    k.device().synchronize()?;
    println!(
        "quantize_act_e2m1_cutlass M={M} K={K}: {:.4} ms/call ({qreps} reps, clean host timing)",
        qt0.elapsed().as_secs_f64() * 1000.0 / qreps as f64
    );

    let mut d_out = stream.alloc_zeros::<f32>(M * N)?;

    for _ in 0..3 {
        let ran = k.mma_e2m1_cutlass_sfa_f32out(
            &mut d_out.as_view_mut(),
            &d_w.as_view(),
            &cw,
            &d_xq.as_view(),
            &d_xs.as_view(),
            K,
            N,
            M,
            false,
        )?;
        anyhow::ensure!(ran, "mma_e2m1_cutlass_sfa_f32out declined M={M} K={K} N={N}");
    }
    k.device().synchronize()?;

    let reps = 20;
    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        k.mma_e2m1_cutlass_sfa_f32out(
            &mut d_out.as_view_mut(),
            &d_w.as_view(),
            &cw,
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
        "cutlass_fp4_gemm_f32out M={M} K={K} N={N}: {:.4} ms/call ({reps} reps, clean host timing, no profiler)",
        elapsed.as_secs_f64() * 1000.0 / reps as f64
    );
    Ok(())
}
