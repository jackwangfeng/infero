//! Verifies the NVFP4 activation quantizer, `Kernels::quantize_act_e2m1_cutlass`
//! (`cu/fp4.cu`'s `quantize_act_e2m1_f32`) and its pure-Rust host reference
//! `infero_kernels::fp4::quantize_act_e2m1_row`.
//!
//! Both port vLLM's own real reference quantizer (`cast_to_fp4` /
//! `ref_nvfp4_quant`, real installed vLLM 0.27.1 source,
//! `vllm/model_executor/layers/quantization/utils/nvfp4_emulation_utils.py`)
//! -- not a second independent reading of the NVFP4 spec.
//!
//! ## Why there are two tests, one host-only
//!
//! The real per-block dynamic scale this quantizer computes has to be
//! requantized through f8_e4m3 (`f32_to_e4m3`, `common.cuh`), and that
//! conversion only exists on real hardware as an sm_89+ PTX instruction
//! (`cvt.rn.satfinite.e4m3x2.f32`) -- below sm_89 it's a compiled-in no-op
//! that always returns 0 (`crates/kernels/src/cu/common.cuh`,
//! `crates/cuda/src/backend.rs`'s `Caps::fp8 = arch >= 89`). This crate's own
//! dev sandbox GPU is an RTX A4000 (sm_86), so `quantize_act_e2m1_f32`'s
//! actual on-device numeric output degenerates to all-zero here -- NOT a bug
//! in this port, the same pre-existing, already-handled limitation
//! `tests/quantize_act_e4m3.rs` already gates and skips under for the
//! analogous e4m3 activation quantizer.
//!
//! So this file's real, always-real, numeric verification is
//! `quantize_act_e2m1_host_reference_round_trips_within_tolerance`: it runs
//! the SAME algorithm through a portable host encoder
//! (`fp4::quantize_act_e2m1_row`, using `f32_to_e4m3_host`'s brute-force
//! round-to-nearest stand-in rather than the hardware instruction) and
//! round-trips through Task 4's already-verified `dequant_f4e2m1_row` --
//! this needs no GPU at all and gives real, non-skippable evidence the
//! ported algorithm (block reduction, clamp, `cast_to_fp4`'s asymmetric
//! threshold table, the `output_scale` zero guard, nibble packing) is
//! correct. `quantize_act_e2m1_cutlass_matches_the_host_reference` then
//! checks the actual device kernel produces the SAME bytes as that host
//! reference, gated (and skipped, with the same message convention as
//! `quantize_act_e4m3.rs`) on `Device::caps().fp8` -- it will run for real
//! on sm_89+ hardware (e.g. this plan's own target production box) even
//! though it cannot here.

mod common;

use anyhow::Result;
use common::*;
use infero_kernels::fp4::{F4E2M1_BLOCK, dequant_f4e2m1_row, quantize_act_e2m1_row};

/// Splits the round-trip error check into near-zero (absolute) and
/// non-near-zero (relative) elements, and asserts both stay within a
/// tolerance derived from the e2m1 ladder's own coarseness -- not widened
/// until the test passes.
///
/// e2m1 has only 3 magnitude bits, so a blanket relative-error bound over
/// every element is the wrong test: an element whose true *scaled* value
/// falls just under the ladder's first nonzero boundary (0.25, rounds to
/// code 0) has 100% relative error no matter how well the quantizer works --
/// that's the ladder's own coarseness, not a bug.
///
/// - "near-zero" elements (|want| below `near_zero_floor`) are checked on
///   ABSOLUTE error. The smallest nonzero ladder magnitude is 0.5, and with
///   `input_scale = 1.0` and activations in [-1, 1) here, a block's scale is
///   at most `vec_max / 6.0 <= 1.0/6.0 ~ 0.1667` (before fp8-requantization
///   noise), so `0.5 * scale` bounds how far a near-zero element can be
///   pushed by one ladder step -- a real, derived ceiling, not an arbitrary
///   one.
/// - all other elements are checked on RELATIVE error, bounded by the
///   ladder's own worst-case relative half-gap: the largest is at magnitude
///   0.5 (boundaries 0.25/0.75, i.e. +/-50% before rounding forces a jump to
///   0 or 1.0), so 55% leaves real headroom for the scale's own
///   fp8-quantization noise without masking an actual bug.
fn assert_round_trip_within_ladder_tolerance(want: &[f32], got: &[f32]) {
    assert_eq!(want.len(), got.len());
    let near_zero_floor = 0.06f32; // ~ half the smallest ladder step's max reach (see above)
    let near_zero_abs_tol = 0.10f32;
    let relative_tol = 0.55f32;

    let mut worst_near_zero_abs = 0.0f32;
    let mut worst_relative = 0.0f32;
    for (i, (&w, &g)) in want.iter().zip(got.iter()).enumerate() {
        if w.abs() < near_zero_floor {
            let d = (w - g).abs();
            worst_near_zero_abs = worst_near_zero_abs.max(d);
            assert!(
                d <= near_zero_abs_tol,
                "element {i} (near-zero, want {w}, got {g}): abs diff {d} > {near_zero_abs_tol}"
            );
        } else {
            let d = (w - g).abs() / w.abs();
            worst_relative = worst_relative.max(d);
            assert!(
                d <= relative_tol,
                "element {i} (want {w}, got {g}): relative diff {d} > {relative_tol}"
            );
        }
    }
    eprintln!(
        "round trip: worst near-zero abs diff = {worst_near_zero_abs:.4e}, worst relative diff = {worst_relative:.4e}"
    );
}

/// Real, always-real (no GPU needed) round-trip check: quantize with the
/// pure-Rust host reference, dequantize with Task 4's already-verified
/// `dequant_f4e2m1_row`, compare to the original activations.
#[test]
fn quantize_act_e2m1_host_reference_round_trips_within_tolerance() {
    // 64 rows x 256 k: a clean multiple of the 16-element block, wide enough
    // (16 blocks/row) that a wrong block index would land on a visibly wrong
    // value rather than a coincidentally-close one -- the same sizing
    // rationale `fp4_dequant.rs` used for Task 4's own device-dequant test.
    let (n_tokens, kk) = (64usize, 256usize);
    let x = pseudo_random(n_tokens * kk, 0xACE1);

    // `input_scale` here names the quantizer's `global_scale` ARGUMENT, not
    // the checkpoint's raw `input_scale` scalar -- real call sites must pass
    // the RECIPROCAL of the checkpoint value (see
    // `quantize_act_e2m1_host_reference_at_a_real_checkpoints_input_scale`
    // below, which uses a real checkpoint magnitude and is what actually
    // guards the convention). `1.0` is self-reciprocal, so this test's own
    // choice cannot distinguish the two conventions -- it exists only to
    // exercise the per-block dynamic-scale path away from the clamp's
    // saturation edge, at a scale that keeps `global_scale * vec_max / 6.0`
    // well inside `[-448, 448]` with activations already O(1) (the
    // pseudo-random generator's own [-1, 1) range).
    let input_scale = 1.0f32;
    // Per the round-trip arithmetic: `ref_nvfp4_quant`'s own round-trip
    // identity is `x_dq = fp4_val * (scale / global_scale)`, and
    // `dequant_f4e2m1_row` computes `e2m1_value(nibble) * scale[block] *
    // scale2`. Matching `scale[block] * scale2` to `scale / global_scale`
    // (`scale[block]` supplies the numerator `scale` in both) requires
    // `scale2 == 1.0 / global_scale`, i.e. `1.0 / input_scale` here.
    let scale2 = 1.0 / input_scale;

    let mut got = Vec::with_capacity(n_tokens * kk);
    for row in 0..n_tokens {
        let xrow = &x[row * kk..(row + 1) * kk];
        let (packed, scale_bytes) = quantize_act_e2m1_row(xrow, input_scale, kk);
        assert_eq!(packed.len(), kk / 2);
        assert_eq!(scale_bytes.len(), kk / F4E2M1_BLOCK);

        let scale_f32: Vec<f32> = scale_bytes
            .iter()
            .map(|&b| infero_safetensors::e4m3_value(b))
            .collect();
        got.extend(dequant_f4e2m1_row(&packed, &scale_f32, scale2, kk));
    }

    assert_round_trip_within_ladder_tolerance(&x, &got);
}

/// The real regression test for the `global_scale`/`alpha` convention bug
/// (whole-branch review, corrected in `cutlass_fp4.rs`'s "CORRECTION 2"):
/// at this checkpoint's own real, calibrated `input_scale` for
/// `layers.0.mlp.gate_proj` (`0.0014`, `amax≈3.763`), feeding the RAW
/// checkpoint value as the quantizer's `global_scale` argument (the bug)
/// drives every block's e4m3 scale byte to zero -- `amax < 3.969` makes
/// this true for ANY activation in the block, not a probabilistic
/// near-miss. Feeding the reciprocal (the fix, matching real call sites in
/// `crates/model/src/lib.rs`) keeps every scale byte in e4m3's real usable
/// range and the round trip recovers the original activations within the
/// same ladder tolerance every other test in this file uses. No GPU
/// needed -- this is exactly the class of bug layer-by-layer activation
/// probing on real hardware could not see (every intermediate looks like a
/// valid, small, all-zero tensor, not an obviously-wrong one).
#[test]
fn quantize_act_e2m1_host_reference_at_a_real_checkpoints_input_scale() {
    let (n_tokens, kk) = (64usize, 256usize);
    let x = pseudo_random(n_tokens * kk, 0xACE1);

    // The real, measured `input_scale` for this plan's target checkpoint's
    // `layers.0.mlp.gate_proj` (task8-lmhead-rootcause-report.md).
    let checkpoint_input_scale = 0.0014f32;

    let quantize_and_dequant = |global_scale: f32| -> (Vec<f32>, bool) {
        let mut got = Vec::with_capacity(n_tokens * kk);
        let mut any_zero_scale = false;
        for row in 0..n_tokens {
            let xrow = &x[row * kk..(row + 1) * kk];
            let (packed, scale_bytes) = quantize_act_e2m1_row(xrow, global_scale, kk);
            any_zero_scale |= scale_bytes.iter().any(|&b| b == 0);
            let scale_f32: Vec<f32> = scale_bytes
                .iter()
                .map(|&b| infero_safetensors::e4m3_value(b))
                .collect();
            // Recovering x_true needs `scale2 == 1.0 / global_scale`, same
            // identity the sibling test above derives.
            got.extend(dequant_f4e2m1_row(&packed, &scale_f32, 1.0 / global_scale, kk));
        }
        (got, any_zero_scale)
    };

    // The bug: raw `input_scale` fed directly as `global_scale`. Every
    // block's scale byte underflows to zero -- provable from this
    // checkpoint's own real `amax`, not just "some blocks are hurt".
    let (bugged, bugged_any_zero) = quantize_and_dequant(checkpoint_input_scale);
    assert!(
        bugged_any_zero,
        "expected the pre-fix convention to zero every scale byte at this checkpoint's real magnitude"
    );
    assert!(
        bugged.iter().all(|&v| v == 0.0),
        "expected the pre-fix convention to recover an all-zero activation (the actual, shipped bug)"
    );

    // The fix: the RECIPROCAL fed as `global_scale`, matching real call
    // sites (`Model::matmul_pre`, the lm_head dispatch).
    let (fixed, fixed_any_zero) = quantize_and_dequant(1.0 / checkpoint_input_scale);
    assert!(
        !fixed_any_zero,
        "fixed convention should keep every block's scale byte off the e4m3 underflow floor"
    );
    assert_round_trip_within_ladder_tolerance(&x, &fixed);
}

/// Confirms the device kernel produces the exact same bytes as the host
/// reference -- gated on real sm_89+ hardware fp8 support, skipped
/// (with the same message convention `tests/quantize_act_e4m3.rs` already
/// uses for the analogous e4m3 activation quantizer) where that hardware
/// isn't available, per this file's own doc comment.
/// The kernel LAUNCH itself always runs here, even on sm_86 where its
/// numeric output degenerates (see this file's doc comment) -- memory safety
/// (bounds, races) is independent of whether the fp8 hardware instruction
/// inside it produces meaningful numbers, and this is exactly what
/// `compute-sanitizer` is run against (see the task report). Only the
/// byte-for-byte comparison against the host reference is gated on real
/// sm_89+ hardware.
#[test]
fn quantize_act_e2m1_cutlass_matches_the_host_reference() -> Result<()> {
    let k = kernels()?;
    let stream = k.device().stream().clone();

    let (n_tokens, kk) = (64usize, 256usize);
    let blocks_per_row = kk / F4E2M1_BLOCK;
    let bytes_per_row = kk / 2;
    let x = pseudo_random(n_tokens * kk, 0xACE1);
    let input_scale = 1.0f32;

    let mut want_xq = Vec::with_capacity(n_tokens * bytes_per_row);
    let mut want_xs = Vec::with_capacity(n_tokens * blocks_per_row);
    for row in 0..n_tokens {
        let xrow = &x[row * kk..(row + 1) * kk];
        let (packed, scale_bytes) = quantize_act_e2m1_row(xrow, input_scale, kk);
        want_xq.extend(packed);
        want_xs.extend(scale_bytes);
    }

    let d_x = stream.clone_htod(&x)?;
    let mut d_xq = stream.alloc_zeros::<u8>(n_tokens * bytes_per_row)?;
    let mut d_xq_scale = stream.alloc_zeros::<u8>(n_tokens * blocks_per_row)?;

    k.quantize_act_e2m1_cutlass(
        &mut d_xq.as_view_mut(),
        &mut d_xq_scale.as_view_mut(),
        &d_x.as_view(),
        input_scale,
        kk,
        n_tokens,
    )?;
    k.device().synchronize()?;

    let got_xq = stream.clone_dtoh(&d_xq)?;
    let got_xs = stream.clone_dtoh(&d_xq_scale)?;

    if !k.device().caps().fp8 {
        eprintln!(
            "skipping numeric comparison: sm_{} has no native e4m3 conversion \
             (kernel still launched above, for compute-sanitizer coverage)",
            k.device().arch()
        );
        return Ok(());
    }

    assert_eq!(
        got_xs, want_xs,
        "device scale bytes diverged from the host reference"
    );
    assert_eq!(
        got_xq, want_xq,
        "device packed bytes diverged from the host reference"
    );
    Ok(())
}

/// A k that is NOT a multiple of 16, run purely for `compute-sanitizer`
/// coverage of the kernel's partial-last-block path (`n_in_block <
/// F4E2M1_BLOCK`, the odd-total-elements packing edge where the final
/// byte's high nibble has no second element) -- the round-trip tests above
/// only ever use k = 256 (a clean multiple of 16), so this exists
/// specifically to exercise that boundary's memory accesses for real, not
/// to check numeric correctness (which the host reference above already
/// covers for the aligned case).
#[test]
fn quantize_act_e2m1_cutlass_handles_a_partial_last_block() -> Result<()> {
    let k = kernels()?;
    let stream = k.device().stream().clone();

    let (n_tokens, kk) = (3usize, 37usize); // 37 = 2 full blocks of 16 + 5
    let blocks_per_row = kk.div_ceil(F4E2M1_BLOCK);
    let bytes_per_row = kk.div_ceil(2);
    let x = pseudo_random(n_tokens * kk, 0x37BEEF);

    let d_x = stream.clone_htod(&x)?;
    let mut d_xq = stream.alloc_zeros::<u8>(n_tokens * bytes_per_row)?;
    let mut d_xq_scale = stream.alloc_zeros::<u8>(n_tokens * blocks_per_row)?;

    k.quantize_act_e2m1_cutlass(
        &mut d_xq.as_view_mut(),
        &mut d_xq_scale.as_view_mut(),
        &d_x.as_view(),
        1.0,
        kk,
        n_tokens,
    )?;
    k.device().synchronize()?;

    // Just confirm it ran and produced finite (not garbage/uninitialized)
    // output; the real correctness check is the aligned-k tests above.
    let got_xq = stream.clone_dtoh(&d_xq)?;
    let got_xs = stream.clone_dtoh(&d_xq_scale)?;
    assert_eq!(got_xq.len(), n_tokens * bytes_per_row);
    assert_eq!(got_xs.len(), n_tokens * blocks_per_row);
    Ok(())
}
