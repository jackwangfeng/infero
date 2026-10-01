//! Isolated wall-clock comparison: `Kernels::narrow_gemm_f16` (new) against
//! the REAL unfused composition it is meant to replace (`to_f16` + the
//! cuBLAS-backed `gemm_f16`, the actual shipped kernels) -- at GDN's real
//! `in_proj_a`/`in_proj_b` shape, K=5120 N=48, across the real chunk sizes
//! `prefill_profile`'s own chunking produces for a 27295-token prefill
//! (8192, 8192, 8192, 2719), plus the real decode-scale M=4 (a speculative
//! verify pass at this checkpoint's real `INFERO_SPEC_K=3`).
//!
//! Not wired into the real forward pass -- a standalone go/no-go number for
//! whether this kernel is worth integrating at all.
//!
//! **Hardware caveat**: this was run on this development box's RTX A4000
//! (Ampere, sm_86), not the real bw/GPU3 Blackwell production card the rest
//! of this session's work targets -- absolute numbers (and this GPU's own
//! memory bandwidth/L2 size) do not transfer, only the real, measured
//! *result* that this kernel loses to the composition it was meant to
//! replace does (a structural reuse-ratio/traffic problem, not a
//! hardware-specific fluke -- see `cu/narrow_gemm.cu`'s own header for the
//! real numbers and the honest go/no-go).
//!
//!   cargo run --release -p infero-kernels --features cuda --example narrow_gemm_bench

use std::time::Instant;

use anyhow::Result;
use half::f16;
use infero_gpu::Device;
use infero_kernels::Kernels;

const K_DIM: usize = 5120;
const N_DIM: usize = 48;

fn pseudo_random_f16(n: usize, seed: u64) -> Vec<f16> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            f16::from_f32((((s >> 40) as f32 / (1u64 << 23) as f32) - 1.0) * 2.0)
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
            ((s >> 40) as f32 / (1u64 << 23) as f32) - 1.0
        })
        .collect()
}

fn bench_one(k: &Kernels, m: usize, reps: usize) -> Result<()> {
    let stream = k.device().stream().clone();

    let w: Vec<f16> = pseudo_random_f16(N_DIM * K_DIM, 0xF16F);
    let w_bytes: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();
    let d_w = stream.clone_htod(&w_bytes)?;
    let w16 = stream.clone_htod(&w)?;

    let x: Vec<f32> = pseudo_random_f32(m * K_DIM, 0xACE0);
    let d_x = stream.clone_htod(&x)?;

    let mut d_out_new = stream.alloc_zeros::<f32>(m * N_DIM)?;
    let mut x16 = stream.alloc_zeros::<f16>(m * K_DIM)?;
    let mut d_out_ref = stream.alloc_zeros::<f32>(m * N_DIM)?;

    // ---- new kernel ----
    for _ in 0..3 {
        k.narrow_gemm_f16(&mut d_out_new.as_view_mut(), &d_w.as_view(), &d_x.as_view(), K_DIM, N_DIM, m)?;
    }
    k.device().synchronize()?;
    let t0 = Instant::now();
    for _ in 0..reps {
        k.narrow_gemm_f16(&mut d_out_new.as_view_mut(), &d_w.as_view(), &d_x.as_view(), K_DIM, N_DIM, m)?;
    }
    k.device().synchronize()?;
    let new_ms = t0.elapsed().as_secs_f64() * 1000.0 / reps as f64;

    // ---- real reference composition: to_f16(x) then gemm_f16 ----
    for _ in 0..3 {
        k.to_f16(&mut x16.as_view_mut(), &d_x.as_view(), m * K_DIM)?;
        k.gemm_f16(&mut d_out_ref.as_view_mut(), &x16.as_view(), &w16.as_view(), m, K_DIM, N_DIM)?;
    }
    k.device().synchronize()?;
    let t0 = Instant::now();
    for _ in 0..reps {
        k.to_f16(&mut x16.as_view_mut(), &d_x.as_view(), m * K_DIM)?;
        k.gemm_f16(&mut d_out_ref.as_view_mut(), &x16.as_view(), &w16.as_view(), m, K_DIM, N_DIM)?;
    }
    k.device().synchronize()?;
    let ref_ms = t0.elapsed().as_secs_f64() * 1000.0 / reps as f64;

    println!(
        "M={m:6}  narrow_gemm_f16 {:8.4} ms   to_f16+gemm_f16 {:8.4} ms   speedup {:5.2}x",
        new_ms,
        ref_ms,
        ref_ms / new_ms
    );
    Ok(())
}

fn main() -> Result<()> {
    let k = Kernels::new(Device::new(0)?);
    println!("-- real GDN in_proj_a/in_proj_b shape: K={K_DIM} N={N_DIM} --");
    println!("-- real prefill chunk sizes (27295 tokens: 8192,8192,8192,2719) --");
    for m in [8192, 2719] {
        bench_one(&k, m, 50)?;
    }
    println!("-- real decode-scale M (speculative verify pass, k+1=4 at INFERO_SPEC_K=3) --");
    bench_one(&k, 4, 500)?;
    println!("-- plain decode M=1, for reference --");
    bench_one(&k, 1, 500)?;
    Ok(())
}
