//! `Kernels::narrow_gemm_f16` ([`infero_kernels::narrow_gemm`]) against the
//! REAL unfused composition it is meant to replace for GDN's narrow
//! `in_proj_a`/`in_proj_b` gate weights: today's real prefill path is a
//! separate `to_f16` cast (f32 activation -> f16) feeding the generic
//! cuBLAS-backed `gemm_f16`. Both kernels here are the actual shipped ones
//! (`Kernels::to_f16`, `Kernels::gemm_f16`), not a reimplementation of them.
//!
//! Not wired into the real forward pass — this is the isolated
//! correctness/edge-case/buffer-safety proof the task's own established
//! practice requires before anything in `model/src/lib.rs` is touched. See
//! `examples/narrow_gemm_bench.rs` for the isolated wall-clock comparison.
//!
//! ## Tolerance
//!
//! This kernel reads `x` directly as f32 and never rounds it to f16 — a
//! deliberate design choice (see `cu/narrow_gemm.cu`'s header), not a bug,
//! and it means the two sides are not computing the literal same arithmetic:
//! the reference composition rounds every activation element to half
//! precision *before* the dot product, this kernel does not. This project
//! already has a real precedent test for exactly this situation —
//! `crates/kernels/tests/quant.rs::prefill_gemm_agrees_with_decode_gemv`,
//! comparing a gemv path that reads f32 `x` directly against a gemm path
//! that rounds it to f16 first — and its own tolerance (cosine similarity
//! `> 0.9999`, peak-relative diff `< 0.01`) is what this file reuses rather
//! than inventing a new number.

mod common;

use anyhow::Result;
use common::*;
use half::f16;
use infero_kernels::narrow_gemm::NARROW_GEMM_MAX_N;

/// Sentinel written into every byte of a buffer, oversized well beyond what
/// the kernel should ever touch — the padding is checked for being *still*
/// this value afterward. Chosen as a bit pattern that is not a plausible
/// zero/small float (`f32::from_bits` of repeated `0xDEADBEEF` is a large
/// negative NaN-adjacent value, easy to tell apart from a real accumulated
/// dot product), matching the project's "an undersized test buffer silently
/// evaded memcheck once" lesson by making mine generously larger instead —
/// real reads/writes past the kernel's own region land in memory this test
/// can directly inspect.
const POISON_BITS: u32 = 0xDEAD_BEEF;
const POISON_F32: f32 = f32::from_bits(POISON_BITS);
/// Extra padding, in elements, appended past every buffer's real size.
const PAD: usize = 4096;

fn poison_f32(n: usize) -> Vec<f32> {
    vec![POISON_F32; n]
}

fn pseudo_random_f16(n: usize, seed: u64, scale: f32) -> Vec<f16> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            f16::from_f32((((s >> 40) as f32 / (1u64 << 23) as f32) - 1.0) * scale)
        })
        .collect()
}

/// Runs the REAL reference composition (`to_f16` + `gemm_f16`, the shipped
/// kernels, not a reimplementation) and the new kernel, both against
/// deliberately oversized, poison-filled device buffers, and checks:
/// * the real output region agrees within the cited tolerance;
/// * every byte of padding past the real output region is untouched.
fn check_shape(k_dim: usize, n_dim: usize, m: usize, seed: u64) -> Result<()> {
    let k = kernels()?;
    let stream = k.device().stream().clone();

    let w: Vec<f16> = pseudo_random_f16(n_dim * k_dim, seed, 2.0);
    let mut w_padded: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();
    w_padded.extend(std::iter::repeat(0xAAu8).take(PAD * 2)); // 2 bytes/half
    let d_w = stream.clone_htod(&w_padded)?;

    let x: Vec<f32> = pseudo_random(m * k_dim, seed ^ 0xACE0);
    let mut x_padded = x.clone();
    x_padded.extend(poison_f32(PAD));
    let d_x = stream.clone_htod(&x_padded)?;

    // ---- new kernel, oversized+poisoned output buffer ----
    let mut out_new = poison_f32(m * n_dim + PAD);
    let mut d_out_new = stream.clone_htod(&out_new)?;
    let ran = k.narrow_gemm_f16(
        &mut d_out_new.as_view_mut(),
        &d_w.as_view(),
        &d_x.as_view(),
        k_dim,
        n_dim,
        m,
    )?;
    assert!(ran, "narrow_gemm_f16 declined n={n_dim} <= NARROW_GEMM_MAX_N={NARROW_GEMM_MAX_N}");
    k.device().synchronize()?;
    out_new = stream.clone_dtoh(&d_out_new)?;
    for (i, &v) in out_new[m * n_dim..].iter().enumerate() {
        assert_eq!(
            v.to_bits(),
            POISON_BITS,
            "narrow_gemm_f16 wrote past its own output region at padding offset {i}"
        );
    }

    // ---- real reference: to_f16(x) then gemm_f16(x16, w) ----
    // `w` is already f16 on the host; upload directly rather than
    // round-tripping through `to_f16`, which exists to convert f32
    // activations, not weights.
    let w16 = stream.clone_htod(&w)?;
    let dx = stream.clone_htod(&x)?;
    let mut x16 = stream.alloc_zeros::<f16>(m * k_dim)?;
    k.to_f16(&mut x16.as_view_mut(), &dx.as_view(), m * k_dim)?;
    let mut d_out_ref = stream.alloc_zeros::<f32>(m * n_dim)?;
    k.gemm_f16(&mut d_out_ref.as_view_mut(), &x16.as_view(), &w16.as_view(), m, k_dim, n_dim)?;
    k.device().synchronize()?;
    let out_ref = stream.clone_dtoh(&d_out_ref)?;

    let got = &out_new[..m * n_dim];
    let cos = cosine(got, &out_ref);
    assert!(
        cos > 0.9999,
        "K={k_dim} N={n_dim} M={m}: cosine {cos} between narrow_gemm_f16 and the real to_f16+gemm_f16 composition"
    );
    let scale = out_ref.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-6);
    let (abs, at) = max_abs_diff(got, &out_ref);
    assert!(
        abs / scale < 0.01,
        "K={k_dim} N={n_dim} M={m}: peak-relative diff {} at {at} (got {} want {})",
        abs / scale,
        got[at],
        out_ref[at]
    );
    Ok(())
}

/// The real production shape at chunk-prefill scale: K=5120 (d_model),
/// N=48 (GDN value heads, the unfused `in_proj_a`/`in_proj_b`), M=8192 (one
/// real prefill chunk).
#[test]
fn matches_real_shape_at_chunk_scale() -> Result<()> {
    check_shape(5120, 48, 8192, 0xF16F)
}

/// The same real weight shape, but at plain decode scale (`n_tokens=1`) --
/// `matmul_pre`'s own dispatch does not distinguish prefill from decode, so
/// this weight/kernel type is reachable at M=1 too.
#[test]
fn matches_real_shape_at_decode_scale() -> Result<()> {
    check_shape(5120, 48, 1, 0xD1C0DE)
}

/// Real production decode batching with speculation on (`INFERO_SPEC_K=3`,
/// this checkpoint's own real running config, per `gemv_f16_ksplit`'s own
/// doc comment): a verify pass is `k+1=4` rows, never 1.
#[test]
fn matches_real_shape_at_speculative_verify_scale() -> Result<()> {
    check_shape(5120, 48, 4, 0x5_BEC)
}

/// `K`/`N`/`M` that divide none of `NARROW_GEMM_KTILE=128`,
/// `NARROW_GEMM_BLOCK=128`, or each other — exercises the partial last
/// K-tile and the partial last M-block, both real code paths the exact
/// production shape (K=5120=40*128 exactly, M=8192=64*128 exactly) never
/// takes.
#[test]
fn handles_nondivisible_k_n_and_m() -> Result<()> {
    check_shape(5003, 37, 131, 0x0DD5)
}

/// `N` at exactly the compile-time cap must still run.
#[test]
fn runs_at_exactly_the_max_n_cap() -> Result<()> {
    check_shape(256, NARROW_GEMM_MAX_N, 17, 0xCAFE)
}

/// `N` one past the compile-time cap must be declined, not silently
/// truncated or allowed to overrun its fixed-size accumulator.
#[test]
fn declines_n_past_the_max_n_cap() -> Result<()> {
    let k = kernels()?;
    let stream = k.device().stream().clone();
    let n_dim = NARROW_GEMM_MAX_N + 1;
    let k_dim = 64usize;
    let m = 8usize;
    let w: Vec<f16> = pseudo_random_f16(n_dim * k_dim, 0x1, 1.0);
    let w_bytes: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();
    let d_w = stream.clone_htod(&w_bytes)?;
    let x: Vec<f32> = pseudo_random(m * k_dim, 0x2);
    let d_x = stream.clone_htod(&x)?;
    let mut d_out = stream.alloc_zeros::<f32>(m * n_dim)?;
    let ran = k.narrow_gemm_f16(&mut d_out.as_view_mut(), &d_w.as_view(), &d_x.as_view(), k_dim, n_dim, m)?;
    assert!(!ran, "expected a decline at n={n_dim} > NARROW_GEMM_MAX_N={NARROW_GEMM_MAX_N}");
    Ok(())
}

/// `m == 0` is a legal no-op, not a launch of an empty/degenerate grid.
#[test]
fn m_zero_is_a_noop() -> Result<()> {
    let k = kernels()?;
    let stream = k.device().stream().clone();
    let (k_dim, n_dim) = (128usize, 48usize);
    let w: Vec<f16> = pseudo_random_f16(n_dim * k_dim, 0x3, 1.0);
    let w_bytes: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();
    let d_w = stream.clone_htod(&w_bytes)?;
    let d_x = stream.alloc_zeros::<f32>(1)?; // never read at m=0
    let mut d_out = stream.alloc_zeros::<f32>(1)?; // never written at m=0
    let ran = k.narrow_gemm_f16(&mut d_out.as_view_mut(), &d_w.as_view(), &d_x.as_view(), k_dim, n_dim, 0)?;
    assert!(ran);
    Ok(())
}
