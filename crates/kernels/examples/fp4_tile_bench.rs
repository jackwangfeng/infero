//! Task 10, Step 1: real timing sweep of the EXISTING default-wide-tile
//! CUTLASS NVFP4 GEMM (`fp4_bw_gemm.cu`, `Kernels::mma_e2m1_cutlass_sfa_f32out`,
//! Task 6 -- one tile shape only, no small-M/swap_ab variant yet) across the
//! real checkpoint shapes and decode-relevant `n_tokens` values named in this
//! task's own brief. This file does NOT build or benchmark a new tile variant
//! (that's Step 2, deferred pending these real numbers) -- it only measures
//! whether the existing wide tile is actually the fastest choice at small
//! `n_tokens`, mirroring the same real investigation
//! `swap_ab_vs_small_m_bench.rs` already did for the FP8 path.
//!
//! Structure mirrors `swap_ab_vs_small_m_bench.rs` line-for-line where
//! possible: same `#![cfg(feature = "cutlass")]` gating (no `[[example]]`
//! Cargo.toml entry -- confirmed this file's sibling uses the same bare
//! attribute, no `required-features` needed since `cargo check -p
//! infero-kernels` without `--examples` never compiles example targets at
//! all), same warmup-then-timed-loop methodology (3 warmup reps, then a timed
//! loop with one `synchronize()` before and after -- see
//! `swap_ab_vs_small_m_bench.rs:85-116`), same reps-scale-down-at-larger-
//! n_tokens convention (200 reps at `n_tokens <= 16`, 100 above --
//! `swap_ab_vs_small_m_bench.rs:135`), same per-shape table printout.
//!
//! Real NVFP4 test-data construction (packed e2m1 nibbles, per-block f8_e4m3
//! scale bytes, `weight_scale_2`, `input_scale`) mirrors
//! `crates/kernels/tests/cutlass_fp4_gemm.rs`'s own `packed_weight` helper and
//! its `WEIGHT_SCALE_2`/`INPUT_SCALE` constants exactly (both non-1.0, same
//! reasoning: a silently-dropped alpha would otherwise coincidentally still
//! produce a runnable, just numerically wrong, benchmark -- though this file
//! only measures latency, not correctness, so the exact scale values are
//! cosmetic here; kept identical anyway so this file's weight-construction
//! code is a straight copy of an already-reviewed, real pattern rather than a
//! new one).
//!
//! One real shape here (`lm_head`, K=5120 N=248320) is a full ~1.9GiB weight
//! buffer (`n * k.div_ceil(2)` packed nibbles + `n * k.div_ceil(16)` scale
//! bytes = 248320*2560 + 248320*320 ≈ 715MiB quant+scale, times roughly 2.6x
//! for the swizzled SFB copy `prepare_cutlass_fp4_weight` allocates
//! internally -- still comfortably under `bw`'s real VRAM) -- built ONCE per
//! shape and reused across every `n_tokens` in the sweep, exactly like
//! `swap_ab_vs_small_m_bench.rs`'s own `run_shape` builds `d_w`/`cutlass_w`
//! once outside the `n_tokens` loop (`swap_ab_vs_small_m_bench.rs:119-134`),
//! not reallocated per timing point.
#![cfg(feature = "cutlass")]

use std::time::Instant;

use anyhow::Result;
use infero_cuda::Device;
use infero_gpu::View;
use infero_kernels::fp4::F4E2M1_BLOCK;
use infero_kernels::{CutlassFp4Weight, Kernels};

// Same non-1.0 scale constants as `cutlass_fp4_gemm.rs`'s own real test, kept
// identical rather than reinvented (see this file's own doc comment).
const WEIGHT_SCALE_2: f32 = 1.75;
const INPUT_SCALE: f32 = 4.0;

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

/// Real, valid packed e2m1 weight bytes plus real per-16-block f8_e4m3 scale
/// bytes -- a straight copy of `cutlass_fp4_gemm.rs`'s own `packed_weight`
/// helper (that file's lines 50-79): e2m1 has no reserved/invalid nibble
/// pattern, so any pseudo-random nibble stream is already a valid quantized
/// weight, and no NaN-byte dodging (unlike FP8's e4m3 quant bytes) is needed.
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
    let scale_bytes: Vec<u8> = (0..n * blocks)
        .map(|i| SCALE_CODES[i % SCALE_CODES.len()])
        .collect();
    (quants, scale_bytes)
}

/// Times the existing (Task 6, single-tile) CUTLASS NVFP4 GEMM at one
/// `(k, n, n_tokens)` point: quantizes a fresh pseudo-random activation batch
/// via the real `quantize_act_e2m1_cutlass` device path (not a host-computed
/// stand-in), warms up 3 reps, then times `reps` more -- same structure as
/// `swap_ab_vs_small_m_bench.rs`'s own `bench_tile` (lines 58-117), minus the
/// `BenchTile` parameter since Task 6 built only one tile and this task has
/// not yet built a second.
fn bench_default_tile(
    k: &Kernels,
    d_w: &View<'_, u8>,
    cw: &CutlassFp4Weight,
    kk: usize,
    n: usize,
    n_tokens: usize,
    reps: usize,
) -> Result<f64> {
    let stream = k.device().stream().clone();
    let x: Vec<f32> = pseudo_random_f32(n_tokens * kk, 0xACE0, 3.0);
    let d_x = stream.clone_htod(&x)?;
    let blocks = kk.div_ceil(F4E2M1_BLOCK);
    let mut d_xq = stream.alloc_zeros::<u8>(n_tokens * kk.div_ceil(2))?;
    let mut d_xq_scale = stream.alloc_zeros::<u8>(n_tokens * blocks)?;
    k.quantize_act_e2m1_cutlass(
        &mut d_xq.as_view_mut(),
        &mut d_xq_scale.as_view_mut(),
        &d_x.as_view(),
        INPUT_SCALE,
        kk,
        n_tokens,
    )?;
    let mut d_out = stream.alloc_zeros::<f32>(n_tokens * n)?;

    for _ in 0..3 {
        let ran = k.mma_e2m1_cutlass_sfa_f32out(
            &mut d_out.as_view_mut(),
            d_w,
            cw,
            &d_xq.as_view(),
            &d_xq_scale.as_view(),
            kk,
            n,
            n_tokens,
            false,
        )?;
        anyhow::ensure!(
            ran,
            "mma_e2m1_cutlass_sfa_f32out declined n_tokens={n_tokens} at K={kk} N={n}"
        );
    }
    k.device().synchronize()?;
    let t0 = Instant::now();
    for _ in 0..reps {
        k.mma_e2m1_cutlass_sfa_f32out(
            &mut d_out.as_view_mut(),
            d_w,
            cw,
            &d_xq.as_view(),
            &d_xq_scale.as_view(),
            kk,
            n,
            n_tokens,
            false,
        )?;
    }
    k.device().synchronize()?;
    Ok(t0.elapsed().as_secs_f64() * 1000.0 / reps as f64)
}

fn run_shape(k: &Kernels, kk: usize, n: usize, label: &str) -> Result<()> {
    let stream = k.device().stream().clone();
    let (w_quants, w_scale_bytes) = packed_weight(n, kk, 0xF4A1);
    let mut w_buf = w_quants;
    w_buf.extend_from_slice(&w_scale_bytes);
    let d_w = stream.clone_htod(&w_buf)?;
    let cw = k.prepare_cutlass_fp4_weight(&d_w.as_view(), kk, n, WEIGHT_SCALE_2, INPUT_SCALE)?;

    println!("K={kk} N={n} ({label})");
    println!("{:>8}  {:>14}", "tokens", "default(ms)");
    for n_tokens in [1usize, 2, 4, 8, 16, 24, 32, 48, 64] {
        let reps = if n_tokens <= 16 { 200 } else { 100 };
        let default = bench_default_tile(k, &d_w.as_view(), &cw, kk, n, n_tokens, reps)?;
        println!("{n_tokens:>8}  {default:>14.4}");
    }
    Ok(())
}

fn main() -> Result<()> {
    let k = Kernels::new(Device::new(0)?);
    if !k.device().caps().fp4 {
        eprintln!(
            "skipping: sm_{} has no NVFP4 tensor cores (need sm_120+)",
            k.device().arch()
        );
        return Ok(());
    }
    run_shape(
        &k,
        5120,
        17408,
        "real qwen38-27b-nvfp4 FFN gate/up projection",
    )?;
    run_shape(
        &k,
        17408,
        5120,
        "real qwen38-27b-nvfp4 FFN down projection -- the transpose",
    )?;
    run_shape(
        &k,
        5120,
        248320,
        "real qwen38-27b-nvfp4 lm_head -- real vocab_size",
    )?;
    Ok(())
}
