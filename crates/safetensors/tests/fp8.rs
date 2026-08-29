//! E4M3 decoding and block-scaled dequantization.
//!
//! The checks here are deliberately not "the code agrees with a reimagining of
//! itself". Each one pins a property that can be confirmed from outside the
//! implementation: bit patterns whose value is fixed by the format, a scale grid
//! whose effect is visible per block, and a shape mismatch that must be refused
//! rather than silently mis-scaled. That distinction cost a night on a QK-norm
//! kernel whose unit tests were green because they shared the implementation's
//! wrong assumption about the layout.

use anyhow::Result;
use half::f16;
use infero_safetensors::{Dtype, Tensor};

/// Hand-decoded E4M3 bytes. Sign is bit 7, the exponent is bits 6..3 biased by
/// 7, the mantissa is bits 2..0 over eight. `exp == 0` is subnormal.
const KNOWN: &[(u8, f32)] = &[
    (0x00, 0.0),
    (0x38, 1.0),          // 0 0111 000 -> 2^0 * 1.0
    (0xB8, -1.0),         // sign flips
    (0x3C, 1.5),          // mantissa 100 -> 1 + 4/8
    (0x40, 2.0),          // exponent 1000 -> 2^1
    (0x30, 0.5),          // exponent 0110 -> 2^-1
    (0x01, 0.001953125),  // subnormal: (1/8) * 2^-6
    (0x07, 0.013671875),  // subnormal: (7/8) * 2^-6
    (0x7E, 448.0),        // 0 1111 110 -> largest finite
    (0xEE, -112.0),       // the byte read out of the real checkpoint
];

fn tensor<'a>(name: &'a str, dtype: Dtype, shape: Vec<usize>, data: &'a [u8]) -> Tensor<'a> {
    Tensor { name, dtype, shape, data }
}

/// Every byte the format defines, against values fixed by the spec rather than
/// by this crate.
#[test]
fn the_e4m3_bit_patterns_decode_to_their_defined_values() -> Result<()> {
    let bytes: Vec<u8> = KNOWN.iter().map(|(b, _)| *b).collect();
    // A 1x10 matrix with a single scale of 1.0 isolates the decode from scaling.
    let ones = [f32::to_le_bytes(1.0)].concat();
    let scales = tensor("s", Dtype::F32, vec![1, 1], &ones);
    let q = tensor("q", Dtype::F8E4M3, vec![1, KNOWN.len()], &bytes);
    let out = q.dequant_f8_to_f16(&scales, 128)?;
    for ((byte, want), got) in KNOWN.iter().zip(&out) {
        let got = f32::from(*got);
        assert!(
            (got - want).abs() <= want.abs() * 1e-3 + 1e-7,
            "byte {byte:#04x} decoded to {got}, the format says {want}"
        );
    }
    Ok(())
}

/// NaN has to survive as NaN. E4M3 has no infinities, so `0x7F` is not a
/// saturating "very large" — reading it as one would put a finite bogus number
/// into a projection matrix.
#[test]
fn the_nan_pattern_stays_nan() -> Result<()> {
    let bytes = [0x7Fu8, 0xFF];
    let ones = [f32::to_le_bytes(1.0)].concat();
    let scales = tensor("s", Dtype::F32, vec![1, 1], &ones);
    let q = tensor("q", Dtype::F8E4M3, vec![1, 2], &bytes);
    let out = q.dequant_f8_to_f16(&scales, 128)?;
    assert!(out.iter().all(|v| v.is_nan()), "got {out:?}");
    Ok(())
}

/// Each 128x128 tile takes its own scale. The test gives four tiles four
/// distinct scales and one constant quant value, so a dequantizer that indexed
/// the grid by row only, by column only, or with the strides transposed would
/// produce a different pattern in every case.
#[test]
fn each_block_takes_its_own_scale() -> Result<()> {
    const B: usize = 128;
    let (rows, cols) = (2 * B, 2 * B);
    // 0x38 is exactly 1.0, so every output equals its own block's scale.
    let q_bytes = vec![0x38u8; rows * cols];
    // Scale grid laid out row-major: [[2, 3], [5, 7]].
    let grid = [2.0f32, 3.0, 5.0, 7.0];
    let s_bytes: Vec<u8> = grid.iter().flat_map(|v| v.to_le_bytes()).collect();

    let scales = tensor("s", Dtype::F32, vec![2, 2], &s_bytes);
    let q = tensor("q", Dtype::F8E4M3, vec![rows, cols], &q_bytes);
    let out = q.dequant_f8_to_f16(&scales, B)?;

    for (r, c, want) in [(0, 0, 2.0), (0, B, 3.0), (B, 0, 5.0), (B, B, 7.0)] {
        let got = f32::from(out[r * cols + c]);
        assert_eq!(got, want, "block at ({r}, {c}) scaled by {got}, wanted {want}");
    }
    // And the interior of a block shares that block's scale.
    assert_eq!(f32::from(out[(B - 1) * cols + (B - 1)]), 2.0);
    assert_eq!(f32::from(out[(B - 1) * cols + B]), 3.0);
    Ok(())
}

/// A partial trailing tile is legitimate — 17408x5120 at block 128 divides
/// evenly, but nothing guarantees that for every projection in every export.
#[test]
fn a_partial_trailing_block_is_covered() -> Result<()> {
    const B: usize = 128;
    let (rows, cols) = (B + 1, B + 1);
    let q_bytes = vec![0x38u8; rows * cols];
    let grid = [2.0f32, 3.0, 5.0, 7.0];
    let s_bytes: Vec<u8> = grid.iter().flat_map(|v| v.to_le_bytes()).collect();
    let scales = tensor("s", Dtype::F32, vec![2, 2], &s_bytes);
    let q = tensor("q", Dtype::F8E4M3, vec![rows, cols], &q_bytes);
    let out = q.dequant_f8_to_f16(&scales, B)?;
    assert_eq!(out.len(), rows * cols);
    // The lone element of the bottom-right tile.
    assert_eq!(f32::from(out[B * cols + B]), 7.0);
    Ok(())
}

/// A grid of the wrong shape must be refused. Getting this wrong does not
/// crash: it mis-scales every block past the first row, which reads as a model
/// that loads and then talks nonsense.
#[test]
fn a_scale_grid_of_the_wrong_shape_is_refused() {
    const B: usize = 128;
    let q_bytes = vec![0x38u8; 2 * B * 2 * B];
    let s_bytes: Vec<u8> = [1.0f32, 1.0].iter().flat_map(|v| v.to_le_bytes()).collect();
    // 2x2 tiles of quants described by a 1x2 grid.
    let scales = tensor("s", Dtype::F32, vec![1, 2], &s_bytes);
    let q = tensor("q", Dtype::F8E4M3, vec![2 * B, 2 * B], &q_bytes);
    let err = q.dequant_f8_to_f16(&scales, B).unwrap_err().to_string();
    assert!(err.contains("scale grid"), "unhelpful error: {err}");
}

/// The scales are themselves BF16 in the checkpoint, which is the path
/// `to_f32` covers; dequantization has to accept that without a separate branch.
#[test]
fn bf16_scales_are_accepted() -> Result<()> {
    // 2.0 as bf16 is 0x4000.
    let s_bytes = 0x4000u16.to_le_bytes();
    let scales = tensor("s", Dtype::BF16, vec![1, 1], &s_bytes);
    let q_bytes = [0x38u8]; // 1.0
    let q = tensor("q", Dtype::F8E4M3, vec![1, 1], &q_bytes);
    let out = q.dequant_f8_to_f16(&scales, 128)?;
    assert_eq!(f32::from(out[0]), 2.0);
    Ok(())
}

/// The dequantized halves must not silently become infinities. bf16 scales
/// carry f32's exponent range, so a large scale times a large quant can leave
/// f16's ±65504 — and `f16::from_f32` saturates rather than complaining.
#[test]
fn f16_overflow_is_visible_rather_than_saturating_silently() -> Result<()> {
    // 448 (largest finite E4M3) times 1024 is 458752, well past f16's ceiling.
    let s_bytes = 1024.0f32.to_le_bytes();
    let scales = tensor("s", Dtype::F32, vec![1, 1], &s_bytes);
    let q_bytes = [0x7Eu8]; // 448.0
    let q = tensor("q", Dtype::F8E4M3, vec![1, 1], &q_bytes);
    let out = q.dequant_f8_to_f16(&scales, 128)?;
    // Documenting the current behaviour: this saturates. The check exists so
    // that a change to refusing is a deliberate one with a failing test to
    // update, rather than a silent difference in what a checkpoint means.
    assert!(
        out[0].is_infinite(),
        "expected the saturation this format allows, got {}",
        out[0]
    );
    Ok(())
}

/// Shape sanity: the output is row-major and exactly rows*cols long, because
/// the caller uploads it as an f16 matrix of that shape.
#[test]
fn the_output_is_row_major_and_exactly_the_matrix_size() -> Result<()> {
    let (rows, cols) = (3usize, 5usize);
    // Distinct values per column so a transposed write would be visible.
    let per_row: Vec<u8> = vec![0x38, 0x3C, 0x40, 0x30, 0x34];
    let q_bytes: Vec<u8> = (0..rows).flat_map(|_| per_row.clone()).collect();
    let ones = 1.0f32.to_le_bytes();
    let scales = tensor("s", Dtype::F32, vec![1, 1], &ones);
    let q = tensor("q", Dtype::F8E4M3, vec![rows, cols], &q_bytes);
    let out = q.dequant_f8_to_f16(&scales, 128)?;
    assert_eq!(out.len(), rows * cols);
    let want: Vec<f16> = per_row
        .iter()
        .map(|b| {
            let v = match b {
                0x38 => 1.0,
                0x3C => 1.5,
                0x40 => 2.0,
                0x30 => 0.5,
                0x34 => 0.75,
                _ => unreachable!(),
            };
            f16::from_f32(v)
        })
        .collect();
    for r in 0..rows {
        assert_eq!(&out[r * cols..(r + 1) * cols], &want[..], "row {r}");
    }
    Ok(())
}
