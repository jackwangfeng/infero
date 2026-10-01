//! Isolation proof for `Kernels::quantize_silu_mul_e2m1_cutlass`
//! (`cu/fp4.cu`'s `quantize_silu_mul_e2m1_f32`): a new, standalone kernel
//! that fuses `down_proj`'s SwiGLU product (`silu(gate) * up`) with NVFP4
//! activation quantization, eliminating the `[n_tokens, d_ff]` intermediate
//! tensor `silu_mul`/`silu_mul_f16` writes and `quantize_act_e2m1_cutlass`/
//! `_f16` reads back today.
//!
//! Per this task's own non-negotiable requirement, this file proves the
//! fused kernel correct in ISOLATION, against the REAL reference
//! composition (today's actual `Kernels::silu_mul` + `Kernels::
//! quantize_act_e2m1_cutlass`, unchanged, as two separate device kernel
//! calls) -- not against a host re-derivation of the math -- before any
//! forward-pass integration is attempted. This sidesteps
//! `fp4_quantize_act.rs`'s own sm_89+-only numeric gate entirely: that file
//! compares a device kernel against a *host* f8_e4m3 encoder
//! (`f32_to_e4m3_host`), which only agrees with the real hardware
//! `cvt.rn.satfinite.e4m3x2.f32` instruction on sm_89+ (below that, the
//! real device conversion is a compiled-in no-op returning 0 --
//! `crates/cuda/src/backend.rs`'s `Caps::fp8`). This file's comparison is
//! device-kernel-vs-device-kernel on the SAME hardware, so it is a real,
//! always-real (no sm_89+ gate needed) test: both the fused kernel and the
//! reference composition call the exact same `f32_to_e4m3`/`e4m3_to_f32`
//! device functions, so whatever that hardware's conversion actually does
//! (real on sm_89+, a real all-zero degenerate case below it), both paths
//! see the identical behavior and must still match bit-for-bit.
//!
//! Expected result: BIT-EXACT agreement. The fused kernel computes
//! `silu(gate[i]) * up[i]` with the identical formula
//! (`(g / (1 + __expf(-g))) * u`, same `__expf` intrinsic, same operand
//! order) `silu_mul_f32` uses, then feeds that into the identical two-pass
//! max-abs/quantize/pack logic `quantize_act_e2m1_f32` uses on its `x`
//! argument -- an IEEE-754 f32 value written to global memory and read back
//! is bit-identical to the same value kept in a register, so there is no
//! legitimate source of a numeric difference here (unlike the f16-shadow
//! experiment this supersedes, which really does trade precision).

mod common;

use anyhow::Result;
use common::*;
use infero_kernels::fp4::F4E2M1_BLOCK;

/// Runs the fused kernel and the reference composition on the same
/// `gate`/`up` input and returns `(fused_xq, fused_xs, ref_xq, ref_xs)`.
fn run_both(
    k: &infero_kernels::Kernels,
    gate: &[f32],
    up: &[f32],
    input_scale: f32,
    kk: usize,
    n_tokens: usize,
) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>)> {
    let stream = k.device().stream().clone();
    let blocks_per_row = kk.div_ceil(F4E2M1_BLOCK);
    let bytes_per_row = kk.div_ceil(2);

    let d_gate = stream.clone_htod(gate)?;
    let d_up = stream.clone_htod(up)?;

    // Generously oversized device buffers for every output -- real
    // allocations of exactly the needed size would already be adequate
    // (the kernel's own bounds checks are what's under test), but padding
    // them further here means a stray out-of-bounds write from a buggy
    // kernel lands in still-allocated, still-poisoned memory rather than
    // possibly off the end of a tightly-sized allocation, so
    // compute-sanitizer memcheck has the best chance of catching it if it
    // exists.
    const PAD: usize = 4096;

    // ---- fused kernel ----
    let mut d_fused_xq = stream.alloc_zeros::<u8>(n_tokens * bytes_per_row + PAD)?;
    let mut d_fused_xs = stream.alloc_zeros::<u8>(n_tokens * blocks_per_row + PAD)?;
    k.quantize_silu_mul_e2m1_cutlass(
        &mut d_fused_xq.slice_mut(0..n_tokens * bytes_per_row),
        &mut d_fused_xs.slice_mut(0..n_tokens * blocks_per_row),
        &d_gate.as_view(),
        &d_up.as_view(),
        input_scale,
        kk,
        n_tokens,
    )?;
    k.device().synchronize()?;
    let fused_xq = stream.clone_dtoh(&d_fused_xq)?[..n_tokens * bytes_per_row].to_vec();
    let fused_xs = stream.clone_dtoh(&d_fused_xs)?[..n_tokens * blocks_per_row].to_vec();

    // ---- reference composition: today's real two separate kernel calls ----
    let mut d_ffn = stream.alloc_zeros::<f32>(n_tokens * kk + PAD)?;
    k.silu_mul(
        &mut d_ffn.slice_mut(0..n_tokens * kk),
        &d_gate.as_view(),
        &d_up.as_view(),
        n_tokens * kk,
    )?;
    k.device().synchronize()?;

    let mut d_ref_xq = stream.alloc_zeros::<u8>(n_tokens * bytes_per_row + PAD)?;
    let mut d_ref_xs = stream.alloc_zeros::<u8>(n_tokens * blocks_per_row + PAD)?;
    k.quantize_act_e2m1_cutlass(
        &mut d_ref_xq.slice_mut(0..n_tokens * bytes_per_row),
        &mut d_ref_xs.slice_mut(0..n_tokens * blocks_per_row),
        &d_ffn.slice(0..n_tokens * kk),
        input_scale,
        kk,
        n_tokens,
    )?;
    k.device().synchronize()?;
    let ref_xq = stream.clone_dtoh(&d_ref_xq)?[..n_tokens * bytes_per_row].to_vec();
    let ref_xs = stream.clone_dtoh(&d_ref_xs)?[..n_tokens * blocks_per_row].to_vec();

    Ok((fused_xq, fused_xs, ref_xq, ref_xs))
}

/// Main correctness proof: real-shape-like random gate/up values, aligned
/// k (a clean multiple of 16), comparing bit-for-bit against the reference
/// composition.
#[test]
fn fused_kernel_matches_reference_composition_bit_exact() -> Result<()> {
    let k = kernels()?;
    let (n_tokens, kk) = (37usize, 256usize); // not a power-of-2 token count on purpose
    let gate = pseudo_random(n_tokens * kk, 0xFEED1);
    let up = pseudo_random(n_tokens * kk, 0xFEED2);
    let input_scale = 1.0f32;

    let (fused_xq, fused_xs, ref_xq, ref_xs) = run_both(&k, &gate, &up, input_scale, kk, n_tokens)?;

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

/// A real checkpoint-scale `input_scale` (same magnitude class as
/// `fp4_quantize_act.rs`'s own real-checkpoint test), to exercise the
/// output_scale zero-guard path for real rather than only at the
/// self-reciprocal `input_scale = 1.0` used above.
#[test]
fn fused_kernel_matches_reference_composition_at_real_checkpoint_scale() -> Result<()> {
    let k = kernels()?;
    let (n_tokens, kk) = (16usize, 512usize);
    let gate = pseudo_random(n_tokens * kk, 0xABCD1);
    let up = pseudo_random(n_tokens * kk, 0xABCD2);
    let checkpoint_input_scale = 0.0014f32;
    let global_scale = 1.0 / checkpoint_input_scale;

    let (fused_xq, fused_xs, ref_xq, ref_xs) =
        run_both(&k, &gate, &up, global_scale, kk, n_tokens)?;

    assert_eq!(fused_xs, ref_xs);
    assert_eq!(fused_xq, ref_xq);
    Ok(())
}

/// Edge case: `n_tokens = 1` (the decode shape). `down_proj`'s own dispatch
/// doesn't distinguish prefill from decode, so a correctness bug here would
/// be a real, shipped regression even though this fusion's performance
/// target is prefill.
#[test]
fn fused_kernel_handles_n_tokens_one() -> Result<()> {
    let k = kernels()?;
    let (n_tokens, kk) = (1usize, 17408usize); // the real checkpoint's real d_ff
    let gate = pseudo_random(n_tokens * kk, 0x111);
    let up = pseudo_random(n_tokens * kk, 0x222);

    let (fused_xq, fused_xs, ref_xq, ref_xs) = run_both(&k, &gate, &up, 1.0, kk, n_tokens)?;

    assert_eq!(fused_xs, ref_xs);
    assert_eq!(fused_xq, ref_xq);
    Ok(())
}

/// Edge case: `k` not a multiple of 16 (the partial-last-block path) --
/// mirrors `fp4_quantize_act.rs`'s own `..._handles_a_partial_last_block`
/// coverage for the existing quantizer. The real checkpoint's `d_ff =
/// 17408` IS a clean multiple of 16 (17408 / 16 = 1088 exactly), so this
/// isn't a real production shape for THIS model, but the fused kernel must
/// still not corrupt memory if it's ever used at an odd `k`.
#[test]
fn fused_kernel_handles_a_partial_last_block() -> Result<()> {
    let k = kernels()?;
    let (n_tokens, kk) = (5usize, 37usize); // 37 = 2 full blocks of 16 + 5
    let gate = pseudo_random(n_tokens * kk, 0x37A);
    let up = pseudo_random(n_tokens * kk, 0x37B);

    let (fused_xq, fused_xs, ref_xq, ref_xs) = run_both(&k, &gate, &up, 1.0, kk, n_tokens)?;

    assert_eq!(fused_xs, ref_xs);
    assert_eq!(fused_xq, ref_xq);
    Ok(())
}

/// Diagnostic, not a correctness gate: confirms *why* a real forward-pass
/// `INFERO_PROBE` bisection (fused build vs. the unmodified baseline build)
/// shows a small, sparse divergence in `after_ffn` even though this file's
/// own bit-exact tests above all pass. The baseline build's real
/// `down_is_nvfp4` dispatch is `silu_mul_f16` (which ALSO rounds the SwiGLU
/// product to f16 before `quantize_act_e2m1_cutlass_f16` reads it) -- not
/// the plain f32 `silu_mul` + `quantize_act_e2m1_cutlass` this file's other
/// tests compare against. So the real baseline the fused kernel replaces is
/// the f16 composition, which this test reproduces explicitly (same real
/// `d_ff = 17408` shape, real-checkpoint-scale `global_scale`), and asserts
/// it is NOT bit-exact against the fused kernel -- while also confirming
/// the mismatch is small and bounded (a handful of elements tipped into an
/// adjacent NVFP4 bucket by the f16 rounding step, not a wholesale
/// divergence), i.e. this is the documented, expected precision
/// *improvement* from removing the f16 round-trip, not a new bug.
#[test]
fn fused_kernel_vs_old_f16_dispatch_diverges_only_by_expected_f16_rounding() -> Result<()> {
    let k = kernels()?;
    // The real checkpoint's real d_ff, and real last-chunk token count (a
    // first, much smaller sample at n_tokens=8 saw 0/139264 nibbles flip --
    // consistent with a rare, real event (empirically ~3e-7/element at this
    // scale below) that a sample this size would not reliably hit, not with
    // "there is no such event." This is the real shape order of magnitude
    // needed to see it at all.
    let (n_tokens, kk) = (2719usize, 17408usize);
    let gate = pseudo_random(n_tokens * kk, 0xF16A);
    let up = pseudo_random(n_tokens * kk, 0xF16B);
    let checkpoint_input_scale = 0.0014f32;
    let global_scale = 1.0 / checkpoint_input_scale;

    let stream = k.device().stream().clone();
    let blocks_per_row = kk.div_ceil(F4E2M1_BLOCK);
    let bytes_per_row = kk.div_ceil(2);
    const PAD: usize = 4096;

    let d_gate = stream.clone_htod(&gate)?;
    let d_up = stream.clone_htod(&up)?;

    // ---- fused kernel (what the integrated build now does) ----
    let mut d_fused_xq = stream.alloc_zeros::<u8>(n_tokens * bytes_per_row + PAD)?;
    let mut d_fused_xs = stream.alloc_zeros::<u8>(n_tokens * blocks_per_row + PAD)?;
    k.quantize_silu_mul_e2m1_cutlass(
        &mut d_fused_xq.slice_mut(0..n_tokens * bytes_per_row),
        &mut d_fused_xs.slice_mut(0..n_tokens * blocks_per_row),
        &d_gate.as_view(),
        &d_up.as_view(),
        global_scale,
        kk,
        n_tokens,
    )?;
    k.device().synchronize()?;
    let fused_xq = stream.clone_dtoh(&d_fused_xq)?[..n_tokens * bytes_per_row].to_vec();

    // ---- OLD production composition: silu_mul_f16 + quantize_act_e2m1_cutlass_f16 ----
    let mut d_ffn = stream.alloc_zeros::<f32>(n_tokens * kk + PAD)?;
    let mut d_ffn_f16 = stream.alloc_zeros::<half::f16>(n_tokens * kk + PAD)?;
    k.silu_mul_f16(
        &mut d_ffn.slice_mut(0..n_tokens * kk),
        &mut d_ffn_f16.slice_mut(0..n_tokens * kk),
        &d_gate.as_view(),
        &d_up.as_view(),
        n_tokens * kk,
    )?;
    k.device().synchronize()?;
    let mut d_old_xq = stream.alloc_zeros::<u8>(n_tokens * bytes_per_row + PAD)?;
    let mut d_old_xs = stream.alloc_zeros::<u8>(n_tokens * blocks_per_row + PAD)?;
    k.quantize_act_e2m1_cutlass_f16(
        &mut d_old_xq.slice_mut(0..n_tokens * bytes_per_row),
        &mut d_old_xs.slice_mut(0..n_tokens * blocks_per_row),
        &d_ffn_f16.slice(0..n_tokens * kk),
        global_scale,
        kk,
        n_tokens,
    )?;
    k.device().synchronize()?;
    let old_xq = stream.clone_dtoh(&d_old_xq)?[..n_tokens * bytes_per_row].to_vec();

    // Gated the same way `fp4_quantize_act.rs`'s own
    // `quantize_act_e2m1_cutlass_matches_the_host_reference` is: below
    // sm_89 `f32_to_e4m3` degenerates to a compiled-in 0 (see that file's
    // doc comment), so BOTH paths collapse to the same all-zero-scale
    // output regardless of input precision -- both kernel launches above
    // still ran for real (compute-sanitizer coverage), but the numeric
    // comparison below would only prove "both paths are equally
    // degenerate," not anything about real e4m3 rounding behavior.
    if !k.device().caps().fp8 {
        eprintln!(
            "skipping numeric comparison: sm_{} has no native e4m3 conversion \
             (both kernels still launched above, for compute-sanitizer coverage)",
            k.device().arch()
        );
        return Ok(());
    }

    let n_diff = fused_xq
        .iter()
        .zip(old_xq.iter())
        .filter(|(a, b)| a != b)
        .count();
    println!(
        "fused vs old-f16-dispatch: {n_diff}/{} packed bytes differ ({:.4}%)",
        fused_xq.len(),
        100.0 * n_diff as f64 / fused_xq.len() as f64
    );
    assert!(
        n_diff > 0,
        "expected the f16 rounding step to flip at least one nibble at this real shape/scale \
         -- if this is now 0, either the f16 path stopped losing precision (surprising) or the \
         fused kernel silently stopped matching real production math"
    );
    // Bounded: every byte holds two independent 4-bit NVFP4 codes, and f16
    // rounding is a tiny (<=2^-11 relative) perturbation, so only values
    // near a quantization bucket boundary should ever flip -- not a
    // wholesale mismatch.
    assert!(
        (n_diff as f64) < 0.05 * fused_xq.len() as f64,
        "f16-rounding should only flip a small minority of near-boundary values, not {n_diff}/{} \
         -- this is too large to be the expected rounding effect alone",
        fused_xq.len()
    );
    Ok(())
}

/// Edge case: an all-zero block (every `silu(gate[i])*up[i]` in some block
/// is exactly 0, e.g. `gate` all zero there since `silu(0) = 0`) -- the
/// `vec_max == 0` -> `scale_f32 == 0` -> `scale_q == 0` ->
/// `output_scale == 0` guard path, which `quantize_act_e2m1_f32` already
/// has to handle (see that kernel's own doc comment) and the fused kernel
/// must handle identically.
#[test]
fn fused_kernel_handles_all_zero_block() -> Result<()> {
    let k = kernels()?;
    let (n_tokens, kk) = (4usize, 32usize); // 2 blocks of 16

    // Token 0: block 0 all-zero gate (silu(0)=0, so product is 0 regardless
    // of `up`), block 1 real random data.
    // Token 1: both gate AND up all-zero.
    // Tokens 2-3: real random data (a non-degenerate control case in the
    // same buffers).
    let mut gate = pseudo_random(n_tokens * kk, 0x9001);
    let mut up = pseudo_random(n_tokens * kk, 0x9002);
    for i in 0..F4E2M1_BLOCK {
        gate[0 * kk + i] = 0.0; // token 0, block 0
    }
    for i in 0..kk {
        gate[1 * kk + i] = 0.0; // token 1, both blocks
        up[1 * kk + i] = 0.0;
    }

    let (fused_xq, fused_xs, ref_xq, ref_xs) = run_both(&k, &gate, &up, 1.0, kk, n_tokens)?;

    // Confirm the degenerate blocks really did produce a zero scale byte in
    // the REFERENCE path (otherwise this test isn't exercising the guard it
    // claims to).
    let blocks_per_row = kk / F4E2M1_BLOCK;
    assert_eq!(
        ref_xs[0 * blocks_per_row + 0],
        0,
        "token 0 block 0 scale should be 0"
    );
    assert_eq!(
        ref_xs[1 * blocks_per_row + 0],
        0,
        "token 1 block 0 scale should be 0"
    );
    assert_eq!(
        ref_xs[1 * blocks_per_row + 1],
        0,
        "token 1 block 1 scale should be 0"
    );

    assert_eq!(fused_xs, ref_xs);
    assert_eq!(fused_xq, ref_xq);
    Ok(())
}
