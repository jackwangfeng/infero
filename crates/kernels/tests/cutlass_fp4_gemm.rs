//! `mma_e2m1_cutlass_sfa_f32out` against Task 2's own host reference
//! (`dequant_f4e2m1_row` applied to the SAME already-quantized bytes both
//! sides consume, summed in f64 then cast to f32) -- the first, and so far
//! only, numeric ground-truth check for the CUTLASS NVFP4 GEMM this task
//! adds. Mirrors `cutlass_fp8_gemm.rs`'s own structure and tolerance
//! convention (see `max_rel_diff` below), substituting NVFP4's real
//! element/scale types for FP8's.
//!
//! This crate's own dev sandbox has neither `nvcc` (the `cutlass` feature's
//! AOT compile needs it) nor NVFP4 tensor cores (an RTX A4000, sm_86 --
//! `Device::caps().fp4` requires sm_120), so this test was written but never
//! run or even compiled by the agent that wrote it; see this plan's own
//! task-6 report for what that means for confidence in this file.

#![cfg(feature = "cutlass")]

mod common;

use anyhow::Result;
use common::*;
use infero_kernels::fp4::{F4E2M1_BLOCK, dequant_f4e2m1_row, quantize_act_e2m1_row};

// K must be a multiple of CUTLASS's own real NVFP4 operand alignment (32
// e2m1 elements, `128 bits / 4-bit elements` -- see `fp4_bw_gemm.cu`'s own
// `AlignmentA`/`AlignmentB`). 256 is also a full `MmaTileShape_MNK` K-tile
// (`Shape<_128,_128,_256>`, the real SM120 NVFP4 unit test's own tile shape
// this file's kernel uses verbatim), so this exercises one complete
// mainloop K-iteration rather than only a predicated partial one.
const K: usize = 256;
const N: usize = 128;

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
/// bytes. Unlike FP8's e4m3 quant bytes (`cutlass_fp8_gemm.rs`'s own
/// `quant_bytes`, which has to dodge the two NaN byte patterns), e2m1's 4-bit
/// codes have no reserved/invalid pattern at all (`fp4::E2M1_TABLE` has
/// exactly 16 valid entries) -- any pseudo-random nibble stream is already a
/// valid quantized weight, so no byte-rejection is needed here.
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

    // A handful of real, distinct f8_e4m3 byte codes (sign=0, moderate
    // exponent/mantissa combinations -- not NaN, not zero, not subnormal),
    // cycled across blocks/rows so the test exercises real scale-to-scale
    // variation. Their exact f32 meaning is read back via
    // `infero_safetensors::e4m3_value` below (the same real decode oracle
    // `fp4.rs`'s own `f32_to_e4m3_host` uses), not hand-derived here, so
    // there is no risk of this test's own reference math disagreeing with
    // the format on what these bytes mean.
    const SCALE_CODES: [u8; 5] = [0x38, 0x3C, 0x30, 0x34, 0x40]; // arbitrary distinct non-NaN e4m3 bytes
    let blocks = k.div_ceil(F4E2M1_BLOCK);
    let scale_bytes: Vec<u8> = (0..n * blocks).map(|i| SCALE_CODES[i % SCALE_CODES.len()]).collect();
    (quants, scale_bytes)
}

/// Dequantizes an `[n, k]` matrix (packed e2m1 + f8_e4m3 block scale, real
/// on-disk `WeightType::F4E2M1` layout minus the two trailing per-tensor
/// scalars) into f64, for a from-scratch reference matmul.
fn dequant_matrix_f64(packed: &[u8], scale_bytes: &[u8], scale2: f32, k: usize, n: usize) -> Vec<f64> {
    let blocks = k.div_ceil(F4E2M1_BLOCK);
    let bytes_per_row = k.div_ceil(2);
    let mut out = Vec::with_capacity(n * k);
    for row in 0..n {
        let row_packed = &packed[row * bytes_per_row..(row + 1) * bytes_per_row];
        let row_scale: Vec<f32> = scale_bytes[row * blocks..(row + 1) * blocks]
            .iter()
            .map(|&b| infero_safetensors::e4m3_value(b))
            .collect();
        let row_f32 = dequant_f4e2m1_row(row_packed, &row_scale, scale2, k);
        out.extend(row_f32.iter().map(|&v| v as f64));
    }
    out
}

#[test]
fn the_nvfp4_gemm_matches_the_host_reference() -> Result<()> {
    let k = kernels()?;
    if !k.device().caps().fp4 {
        eprintln!("skipping: sm_{} has no NVFP4 tensor cores", k.device().arch());
        return Ok(());
    }
    let stream = k.device().stream().clone();

    // Real per-tensor NVFP4 weight scale (`weight_scale_2`, applied as this
    // GEMM's own `alpha` -- see `cutlass_fp4.rs`'s doc comment for why) --
    // deliberately not 1.0, so a bug that silently drops or mis-scales
    // `alpha` shows up as a wrong answer rather than coincidentally passing.
    const WEIGHT_SCALE_2: f32 = 1.75;
    // Real per-tensor activation "global_scale" input to
    // `quantize_act_e2m1_row`'s own per-block scale computation (see that
    // function's doc comment) -- also not 1.0.
    const INPUT_SCALE: f32 = 4.0;

    let (w_quants, w_scale_bytes) = packed_weight(N, K, 0xF4A1);
    let mut w_buf = w_quants.clone();
    w_buf.extend_from_slice(&w_scale_bytes);
    let d_w = stream.clone_htod(&w_buf)?;
    let cw = k.prepare_cutlass_fp4_weight(&d_w.as_view(), K, N, WEIGHT_SCALE_2, INPUT_SCALE)?;

    let w_dequant_f64 = dequant_matrix_f64(&w_quants, &w_scale_bytes, WEIGHT_SCALE_2, K, N);

    for n_tokens in [1usize, 2, 8, 16, 32, 64, 65, 127, 128, 129] {
        let x: Vec<f32> =
            (0..n_tokens).flat_map(|t| pseudo_random_f32(K, 0xACE0 + t as u64, 3.0 + t as f32)).collect();
        let d_x = stream.clone_htod(&x)?;

        // Real per-token reference quantization (Task 2's own host oracle,
        // `quantize_act_e2m1_row` -- the same algorithm
        // `quantize_act_e2m1_cutlass`'s device kernel is checked against
        // elsewhere), then dequantized back to f64 for the reference matmul.
        // `scale2 = 1.0`: the activation's own per-block scale bytes already
        // fully absorb `INPUT_SCALE` (see `quantize_act_e2m1_row`'s own doc
        // comment), so no second post-multiply belongs here.
        let mut act_dequant_f64 = Vec::with_capacity(n_tokens * K);
        for t in 0..n_tokens {
            let row = &x[t * K..(t + 1) * K];
            let (xq_row, xs_row) = quantize_act_e2m1_row(row, INPUT_SCALE, K);
            let xs_row_f32: Vec<f32> = xs_row.iter().map(|&b| infero_safetensors::e4m3_value(b)).collect();
            let row_f32 = dequant_f4e2m1_row(&xq_row, &xs_row_f32, 1.0, K);
            act_dequant_f64.extend(row_f32.iter().map(|&v| v as f64));
        }

        for accum in [false, true] {
            let seed_out: Vec<f32> = if accum {
                pseudo_random_f32(n_tokens * N, 0xBEEF, 100.0)
            } else {
                vec![0.0f32; n_tokens * N]
            };

            // Reference: f64 matmul over the SAME already-quantized values
            // both sides consume (not against the original unquantized `x`),
            // isolating "does the CUTLASS kernel reproduce
            // dequant(A)@dequant(B)*alpha+beta*C" from "how lossy is e2m1
            // quantization" -- the latter is Task 2's own concern, already
            // covered there.
            let mut want = seed_out.clone();
            for t in 0..n_tokens {
                for j in 0..N {
                    let mut acc = 0.0f64;
                    for kk in 0..K {
                        acc += act_dequant_f64[t * K + kk] * w_dequant_f64[j * K + kk];
                    }
                    if accum {
                        want[t * N + j] += acc as f32;
                    } else {
                        want[t * N + j] = acc as f32;
                    }
                }
            }

            let mut d_xq = stream.alloc_zeros::<u8>(n_tokens * K.div_ceil(2))?;
            let mut d_xq_scale = stream.alloc_zeros::<u8>(n_tokens * K.div_ceil(F4E2M1_BLOCK))?;
            k.quantize_act_e2m1_cutlass(
                &mut d_xq.as_view_mut(),
                &mut d_xq_scale.as_view_mut(),
                &d_x.as_view(),
                INPUT_SCALE,
                K,
                n_tokens,
            )?;

            let mut d_got = stream.clone_htod(&seed_out)?;
            let ran = k.mma_e2m1_cutlass_sfa_f32out(
                &mut d_got.as_view_mut(),
                &d_w.as_view(),
                &cw,
                &d_xq.as_view(),
                &d_xq_scale.as_view(),
                K,
                N,
                n_tokens,
                accum,
            )?;
            assert!(ran, "mma_e2m1_cutlass_sfa_f32out declined {n_tokens} tokens at K={K}");

            k.device().synchronize()?;
            let got = stream.clone_dtoh(&d_got)?;
            k.device().synchronize()?;

            // e2m1's own ladder is coarse, but both sides here dequantize
            // the SAME quantized bytes -- the only real slack is tensor-core
            // MMA accumulation order vs. this test's own naive f64 sum, so
            // this uses the same tolerance shape (a relative-error floor
            // plus a relative term) `cutlass_fp8_gemm.rs`'s own comparisons
            // against a from-scratch reference use, not a looser one just
            // because the format is 4-bit.
            let (worst, at) = max_rel_diff(&got, &want, 3e-2);
            assert!(
                worst <= 1.0,
                "{n_tokens} tokens, accum={accum}: element {at} (token {}, row {}) is {worst:.1}x \
                 the tolerance: got {}, want {}",
                at / N,
                at % N,
                got[at],
                want[at]
            );
        }
    }
    Ok(())
}

fn max_rel_diff(got: &[f32], want: &[f32], rel: f32) -> (f32, usize) {
    let peak = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let floor = 3e-2 * peak.max(f32::MIN_POSITIVE);
    let mut worst = (0.0f32, 0usize);
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let r = (g - w).abs() / (floor + rel * w.abs());
        if r > worst.0 {
            worst = (r, i);
        }
    }
    worst
}
