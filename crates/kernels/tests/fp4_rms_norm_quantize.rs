//! Isolation proof for `Kernels::rms_norm_quantize_e2m1_cutlass`
//! (`cu/fp4.cu`'s `rms_norm_quantize_e2m1_f32`): a new, standalone kernel that
//! fuses the register-resident RMS norm (`Kernels::rms_norm`'s own fused
//! path, `cu/mmvq.cu`'s `rms_norm_f16_f32`) with NVFP4(E2M1) activation
//! quantization (`Kernels::quantize_act_e2m1_cutlass`), for one real caller:
//! `feed_forward`'s plain (unfused, F4E2M1 gate/up) norm immediately feeding
//! `matmul_pre_f4e2m1_dedup`'s quantize call.
//!
//! Per this project's own non-negotiable practice (see
//! `fp4_fused_silu_mul_quantize.rs`'s own doc comment for the precedent this
//! file follows), this proves the fused kernel correct in ISOLATION against
//! the REAL reference composition -- `Kernels::rms_norm` followed by
//! `Kernels::quantize_act_e2m1_cutlass`, run as two separate, unmodified
//! device kernel calls on the same input -- not a host re-derivation of the
//! math, and not a new, independent reading of the NVFP4/RMS-norm spec.
//!
//! Expected result: BIT-EXACT agreement, for the same reason
//! `fp4_fused_silu_mul_quantize.rs` expects it: the fused kernel's phase 1/2
//! use the IDENTICAL strided-register load, `block_reduce_sum`, and
//! `scale * weight[i]` formula `rms_norm_f16_f32` uses (down to the same
//! `rms_block(d)` block-size formula, so the reduction tree itself is
//! bit-identical, not just the math), and phase 3 uses the identical
//! per-block scale/threshold/pack math `quantize_act_e2m1_f32` uses on its
//! `x` argument -- an IEEE-754 f32 value written to global memory (by
//! `Kernels::rms_norm`) and read back (by `Kernels::quantize_act_e2m1_cutlass`)
//! is bit-identical to the same value kept in a register across the fused
//! kernel's own phase 2 -> phase 3 boundary, so there is no legitimate source
//! of a numeric difference.
//!
//! This sidesteps `fp4_quantize_act.rs`'s own sm_89+-only numeric gate
//! entirely, the same way `fp4_fused_silu_mul_quantize.rs` does: both the
//! fused kernel and the reference composition call the exact same
//! `f32_to_e4m3`/`e4m3_to_f32` device functions on the same hardware, so
//! whatever that conversion actually does here (real on sm_89+, a real
//! all-zero degenerate case below it), both paths see identical behavior and
//! must still match bit-for-bit.

mod common;

use anyhow::Result;
use common::*;
use infero_kernels::fp4::F4E2M1_BLOCK;
use infero_kernels::Kernels;

/// Generously oversized padding on every device output buffer -- real
/// allocations sized exactly to what the kernel should touch would already be
/// adequate (the kernel's own bounds checks are what's under test), but
/// padding further means a stray out-of-bounds write from a buggy kernel
/// lands in still-allocated, still-poisoned memory rather than possibly off
/// the end of a tightly-sized allocation, giving compute-sanitizer memcheck
/// the best chance of catching it if it exists. This project's own history:
/// an undersized TEST buffer (not a kernel bug) silently evaded memcheck
/// once already (`feedback_stack_overflow_evades_memcheck.md`).
const PAD: usize = 4096;

/// Runs the fused kernel and the reference composition (today's real,
/// unmodified `Kernels::rms_norm` + `Kernels::quantize_act_e2m1_cutlass`, as
/// two separate calls) on the same input, returning
/// `(fused_out, fused_xq, fused_xs, ref_out, ref_xq, ref_xs)`.
#[allow(clippy::too_many_arguments)]
fn run_both(
    k: &Kernels,
    x: &[f32],
    weight: &[f32],
    input_scale: f32,
    d: usize,
    n_tokens: usize,
    eps: f32,
) -> Result<(Vec<f32>, Vec<u8>, Vec<u8>, Vec<f32>, Vec<u8>, Vec<u8>)> {
    let stream = k.device().stream().clone();
    let blocks_per_row = d.div_ceil(F4E2M1_BLOCK);
    let bytes_per_row = d.div_ceil(2);

    let d_x = stream.clone_htod(x)?;
    let d_w = stream.clone_htod(weight)?;

    // ---- fused kernel ----
    let mut d_fused_out = stream.alloc_zeros::<f32>(n_tokens * d + PAD)?;
    let mut d_fused_xq = stream.alloc_zeros::<u8>(n_tokens * bytes_per_row + PAD)?;
    let mut d_fused_xs = stream.alloc_zeros::<u8>(n_tokens * blocks_per_row + PAD)?;
    k.rms_norm_quantize_e2m1_cutlass(
        &mut d_fused_out.slice_mut(0..n_tokens * d),
        &mut d_fused_xq.slice_mut(0..n_tokens * bytes_per_row),
        &mut d_fused_xs.slice_mut(0..n_tokens * blocks_per_row),
        &d_x.as_view(),
        &d_w.as_view(),
        input_scale,
        d,
        n_tokens,
        eps,
    )?;
    k.device().synchronize()?;
    let fused_out = stream.clone_dtoh(&d_fused_out)?[..n_tokens * d].to_vec();
    let fused_xq = stream.clone_dtoh(&d_fused_xq)?[..n_tokens * bytes_per_row].to_vec();
    let fused_xs = stream.clone_dtoh(&d_fused_xs)?[..n_tokens * blocks_per_row].to_vec();

    // ---- reference composition: today's real, unmodified two calls ----
    let mut d_ref_out = stream.alloc_zeros::<f32>(n_tokens * d + PAD)?;
    k.rms_norm(
        &mut d_ref_out.slice_mut(0..n_tokens * d),
        &d_x.as_view(),
        &d_w.as_view(),
        n_tokens,
        d,
        eps,
    )?;
    k.device().synchronize()?;

    let mut d_ref_xq = stream.alloc_zeros::<u8>(n_tokens * bytes_per_row + PAD)?;
    let mut d_ref_xs = stream.alloc_zeros::<u8>(n_tokens * blocks_per_row + PAD)?;
    k.quantize_act_e2m1_cutlass(
        &mut d_ref_xq.slice_mut(0..n_tokens * bytes_per_row),
        &mut d_ref_xs.slice_mut(0..n_tokens * blocks_per_row),
        &d_ref_out.slice(0..n_tokens * d),
        input_scale,
        d,
        n_tokens,
    )?;
    k.device().synchronize()?;
    let ref_out = stream.clone_dtoh(&d_ref_out)?[..n_tokens * d].to_vec();
    let ref_xq = stream.clone_dtoh(&d_ref_xq)?[..n_tokens * bytes_per_row].to_vec();
    let ref_xs = stream.clone_dtoh(&d_ref_xs)?[..n_tokens * blocks_per_row].to_vec();

    Ok((fused_out, fused_xq, fused_xs, ref_out, ref_xq, ref_xs))
}

/// Main correctness proof, at the REAL production width: `d = 5120`
/// (`d_model` for this plan's target Qwen3.8/Qwen3.5 checkpoint family), a
/// token count that is not a power of two on purpose, and a real
/// checkpoint-scale `input_scale` (same magnitude class as
/// `fp4_quantize_act.rs`'s own real-checkpoint test).
#[test]
fn fused_kernel_matches_reference_composition_at_real_d_model() -> Result<()> {
    let k = kernels()?;
    let (n_tokens, d) = (37usize, 5120usize);
    let x = pseudo_random(n_tokens * d, 0xD5120);
    let weight = pseudo_random(d, 0xD5121)
        .iter()
        .map(|v| v + 1.5) // norm gains are centred away from 0 in real checkpoints
        .collect::<Vec<_>>();
    let checkpoint_input_scale = 0.0014f32;
    let global_scale = 1.0 / checkpoint_input_scale;
    let eps = 1e-6;

    let (fused_out, fused_xq, fused_xs, ref_out, ref_xq, ref_xs) =
        run_both(&k, &x, &weight, global_scale, d, n_tokens, eps)?;

    assert_eq!(
        fused_out, ref_out,
        "fused kernel's normalized row diverged from Kernels::rms_norm"
    );
    assert_eq!(
        fused_xs, ref_xs,
        "fused kernel's scale bytes diverged from the reference composition"
    );
    assert_eq!(
        fused_xq, ref_xq,
        "fused kernel's packed bytes diverged from the reference composition"
    );
    Ok(())
}

/// Edge case: `n_tokens = 1` (the decode shape), at the same real `d_model`.
/// `feed_forward`'s own dispatch doesn't distinguish prefill from decode, so
/// a correctness bug at batch-one would be a real, shipped regression even
/// though this fusion's performance target is prefill.
#[test]
fn fused_kernel_handles_n_tokens_one_at_real_d_model() -> Result<()> {
    let k = kernels()?;
    let (n_tokens, d) = (1usize, 5120usize);
    let x = pseudo_random(n_tokens * d, 0x111D);
    let weight = pseudo_random(d, 0x222D)
        .iter()
        .map(|v| v + 1.5)
        .collect::<Vec<_>>();

    let (fused_out, fused_xq, fused_xs, ref_out, ref_xq, ref_xs) =
        run_both(&k, &x, &weight, 1.0, d, n_tokens, 1e-6)?;

    assert_eq!(fused_out, ref_out);
    assert_eq!(fused_xs, ref_xs);
    assert_eq!(fused_xq, ref_xq);
    Ok(())
}

/// Edge case: `d` NOT a multiple of 16 (the partial-last-E2M1-block path) --
/// mirrors `fp4_quantize_act.rs`'s own `..._handles_a_partial_last_block`
/// coverage for the unfused quantizer. `d_model` itself is always a multiple
/// of 16 for every real checkpoint this project targets, so this isn't a real
/// production shape, but the fused kernel must still not corrupt memory (or
/// silently diverge from the reference) at an odd `d` -- this is exactly the
/// "judgment-level edge case" the fusion's own investigation flagged: the
/// natural block/grid sizing here (driven by `rms_block(d)`, a multiple of
/// 32) is independent of whether `d` is a multiple of 16, so an unguarded
/// write here would be a real out-of-bounds write past `bytes_per_row`.
#[test]
fn fused_kernel_handles_a_partial_last_e2m1_block() -> Result<()> {
    let k = kernels()?;
    let (n_tokens, d) = (5usize, 37usize); // 37 = 2 full E2M1 blocks of 16 + 5
    let x = pseudo_random(n_tokens * d, 0x37A);
    let weight = pseudo_random(d, 0x37B)
        .iter()
        .map(|v| v + 1.5)
        .collect::<Vec<_>>();

    let (fused_out, fused_xq, fused_xs, ref_out, ref_xq, ref_xs) =
        run_both(&k, &x, &weight, 1.0, d, n_tokens, 1e-6)?;

    assert_eq!(fused_out, ref_out);
    assert_eq!(fused_xs, ref_xs);
    assert_eq!(fused_xq, ref_xq);
    Ok(())
}

/// A second, smaller non-multiple-of-16 width, with a token count that is
/// itself not a multiple of anything convenient either -- further coverage
/// of the same partial-block path at a different phase relative to the
/// block/warp boundaries (`d = 20` lands the partial block's `n_in_block`
/// at 4, not 5, exercising a different pair-boundary case: the partial
/// block's valid pairs are `(16,17)` only, `(18,19)` entirely out of range).
#[test]
fn fused_kernel_handles_a_different_partial_last_e2m1_block_phase() -> Result<()> {
    let k = kernels()?;
    let (n_tokens, d) = (3usize, 20usize); // 20 = 1 full block of 16 + 4
    let x = pseudo_random(n_tokens * d, 0x2020);
    let weight = pseudo_random(d, 0x2021)
        .iter()
        .map(|v| v + 1.5)
        .collect::<Vec<_>>();

    let (fused_out, fused_xq, fused_xs, ref_out, ref_xq, ref_xs) =
        run_both(&k, &x, &weight, 1.0, d, n_tokens, 1e-6)?;

    assert_eq!(fused_out, ref_out);
    assert_eq!(fused_xs, ref_xs);
    assert_eq!(fused_xq, ref_xq);
    Ok(())
}

/// Edge case: an all-zero row for one token (every `x[i] == 0`, so
/// `mean(x^2) == 0`, the norm's own `rsqrt(0 + eps)` is finite but large, and
/// every normalized value is `0 * weight[i] * scale == 0`) -- drives every
/// E2M1 block's `vec_max == 0` -> `scale_f32 == 0` -> `scale_q == 0` ->
/// `output_scale == 0` guard path, which `quantize_act_e2m1_f32` already
/// handles (see that kernel's own doc comment) and the fused kernel must
/// handle identically. Other tokens in the same buffer are real random data,
/// a non-degenerate control case.
#[test]
fn fused_kernel_handles_an_all_zero_row() -> Result<()> {
    let k = kernels()?;
    let (n_tokens, d) = (4usize, 256usize);
    let mut x = pseudo_random(n_tokens * d, 0x9001);
    for v in x[1 * d..2 * d].iter_mut() {
        *v = 0.0; // token 1: the all-zero row
    }
    let weight = pseudo_random(d, 0x9002)
        .iter()
        .map(|v| v + 1.5)
        .collect::<Vec<_>>();

    let (fused_out, fused_xq, fused_xs, ref_out, ref_xq, ref_xs) =
        run_both(&k, &x, &weight, 1.0, d, n_tokens, 1e-6)?;

    // Confirm the degenerate row really did produce an all-zero scale row in
    // the REFERENCE path (otherwise this test isn't exercising the guard it
    // claims to).
    let blocks_per_row = d / F4E2M1_BLOCK;
    assert!(
        ref_xs[1 * blocks_per_row..2 * blocks_per_row]
            .iter()
            .all(|&b| b == 0),
        "token 1's (all-zero input) scale bytes should all be 0 in the reference path"
    );
    assert!(
        ref_xq[1 * (d / 2)..2 * (d / 2)].iter().all(|&b| b == 0),
        "token 1's (all-zero input) packed bytes should all be 0 (code 0, positive) \
         in the reference path"
    );

    assert_eq!(fused_out, ref_out);
    assert_eq!(fused_xs, ref_xs);
    assert_eq!(fused_xq, ref_xq);
    Ok(())
}
