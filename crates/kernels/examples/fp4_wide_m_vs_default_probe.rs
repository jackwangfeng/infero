//! Standalone clean-timed head-to-head: the existing `default_tile`
//! CUTLASS NVFP4 GEMM (`fp4_bw_gemm.cu`, the one production dispatch uses)
//! against the new `wide_m` tile (mirrors vLLM's real SM120 NVFP4
//! `sm120_fp4_config_default`, see that namespace's own doc comment), at
//! the real Qwen3.8-27B-NVFP4 FFN gate-projection prefill shape
//! (M=2048, K=5120, N=17408) `fp4_gemm_vs_vllm_probe.rs` already measured
//! against vLLM's own kernel (~21.5% slower there). Same data-construction
//! and timing methodology as that file (copied, not reinvented) so the two
//! probes' numbers stay directly comparable. Throwaway probe, not part of
//! the correctness test suite -- `wide_m` is not wired into any production
//! dispatch yet.
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
    if !k.device().caps().fp4 {
        eprintln!("skipping: sm_{} has no NVFP4 tensor cores (need sm_120+)", k.device().arch());
        return Ok(());
    }
    let stream = k.device().stream().clone();

    let (w_quants, w_scale_bytes) = packed_weight(N, K, 0xF4A1);
    let mut w_buf = w_quants.clone();
    w_buf.extend_from_slice(&w_scale_bytes);
    let d_w = stream.clone_htod(&w_buf)?;
    // Same real, measured lm_head-scale magnitudes `fp4_gemm_vs_vllm_probe.rs`
    // uses -- kept identical for a direct, apples-to-apples comparison.
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
    k.quantize_act_e2m1_cutlass(&mut d_xq.as_view_mut(), &mut d_xs.as_view_mut(), &d_x.as_view(), 1.0 / input_scale, K, M)?;
    k.device().synchronize()?;

    let mut d_out_default = stream.alloc_zeros::<f32>(M * N)?;
    let mut d_out_wide = stream.alloc_zeros::<f32>(M * N)?;

    // Correctness check first: the two tiles must agree (within f32
    // GEMM-order-of-summation tolerance) before any timing number is worth
    // trusting -- a fast-but-wrong kernel is not a win.
    let ran_default = k.mma_e2m1_cutlass_sfa_f32out(&mut d_out_default.as_view_mut(), &d_w.as_view(), &cw, &d_xq.as_view(), &d_xs.as_view(), K, N, M, false)?;
    anyhow::ensure!(ran_default, "default_tile declined M={M} K={K} N={N}");
    let ran_wide = k.mma_e2m1_cutlass_sfa_f32out_wide_m_bench(&mut d_out_wide.as_view_mut(), &d_w.as_view(), &cw, &d_xq.as_view(), &d_xs.as_view(), K, N, M, false)?;
    anyhow::ensure!(ran_wide, "wide_m declined M={M} K={K} N={N}");
    k.device().synchronize()?;

    let out_default = stream.clone_dtoh(&d_out_default.as_view())?;
    let out_wide = stream.clone_dtoh(&d_out_wide.as_view())?;
    let mut max_abs_diff = 0.0f32;
    let mut max_rel_diff = 0.0f32;
    for (a, b) in out_default.iter().zip(out_wide.iter()) {
        let abs_diff = (a - b).abs();
        max_abs_diff = max_abs_diff.max(abs_diff);
        if a.abs() > 1e-6 {
            max_rel_diff = max_rel_diff.max(abs_diff / a.abs());
        }
    }
    println!("correctness: max_abs_diff={max_abs_diff:.6} max_rel_diff={max_rel_diff:.6} (sample out[0]: default={:.6} wide_m={:.6})", out_default[0], out_wide[0]);
    anyhow::ensure!(
        max_rel_diff < 0.01,
        "wide_m disagrees with default_tile by more than 1% relative -- NOT a valid comparison, this is a real bug"
    );

    for _ in 0..3 {
        k.mma_e2m1_cutlass_sfa_f32out(&mut d_out_default.as_view_mut(), &d_w.as_view(), &cw, &d_xq.as_view(), &d_xs.as_view(), K, N, M, false)?;
    }
    k.device().synchronize()?;
    let reps = 20;
    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        k.mma_e2m1_cutlass_sfa_f32out(&mut d_out_default.as_view_mut(), &d_w.as_view(), &cw, &d_xq.as_view(), &d_xs.as_view(), K, N, M, false)?;
    }
    k.device().synchronize()?;
    let default_ms = t0.elapsed().as_secs_f64() * 1000.0 / reps as f64;

    for _ in 0..3 {
        k.mma_e2m1_cutlass_sfa_f32out_wide_m_bench(&mut d_out_wide.as_view_mut(), &d_w.as_view(), &cw, &d_xq.as_view(), &d_xs.as_view(), K, N, M, false)?;
    }
    k.device().synchronize()?;
    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        k.mma_e2m1_cutlass_sfa_f32out_wide_m_bench(&mut d_out_wide.as_view_mut(), &d_w.as_view(), &cw, &d_xq.as_view(), &d_xs.as_view(), K, N, M, false)?;
    }
    k.device().synchronize()?;
    let wide_ms = t0.elapsed().as_secs_f64() * 1000.0 / reps as f64;

    println!("M={M} K={K} N={N} ({reps} reps each, clean host timing, no profiler)");
    println!("  default_tile (<128,128,256>, Pingpong):        {default_ms:.4} ms/call");
    println!("  wide_m       (<256,128,128>, Auto+Persistent): {wide_ms:.4} ms/call");
    println!("  ratio (default/wide_m): {:.4}x", default_ms / wide_ms);
    Ok(())
}
