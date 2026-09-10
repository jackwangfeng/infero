//! Host-side reference for NVFP4 (e2m1) dequantization.
//!
//! This is the numeric ground truth every later NVFP4 task (the device-side
//! dequant kernel, the CUTLASS GEMM) gets checked against -- pure Rust, no
//! GPU/CUTLASS dependency. Getting the bit-pattern-to-value mapping and the
//! two-level scale math right here matters: a wrong encoding here would
//! silently validate wrong kernels later.
//!
//! See [`crate::WeightType::F4E2M1`] for the on-disk row layout this
//! dequantizes (packed quant bytes, then a trailing f8_e4m3 block-scale grid,
//! then the two f32 per-tensor scalars).

/// Number of e2m1 elements covered by one block scale, per NVFP4's two-level
/// (per-16-block f8_e4m3 scale, then a per-tensor f32 `weight_scale_2`) scheme.
pub const F4E2M1_BLOCK: usize = 16;

/// The 16-entry e2m1 value table. Bit 3 (the high bit of the 4-bit code) is
/// the sign (sign-magnitude, not two's complement); bits [2:0] index the
/// magnitude ladder `{0, 0.5, 1, 1.5, 2, 3, 4, 6}`.
///
/// This mapping is cross-checked against two independent real references
/// (not derived from the abstract "2 exponent bits, 1 mantissa bit"
/// description alone):
///
/// 1. CUTLASS's `cutlass::float_e2m1_t`. Its bit layout is defined generically
///    by `detail::FpBitRepresentation<uint8_t, /*NumBits=*/4, /*NumExpBits=*/2,
///    /*NumMantissaBits=*/1, NanInfEncoding::NONE>` (`cutlass/exmy_base.h`,
///    selected for `FpEncoding::E2M1` in `cutlass/float_subbyte.h`, both
///    fetched from `NVIDIA/cutlass` on GitHub this session). Evaluating that
///    template by hand: `EXP_BIAS = (1 << (2-1)) - 1 = 1`; `SIGN_SHIFT =
///    NumMantissaBits + NumExpBits = 3` (sign is the MSB); `HAS_DENORM = true`
///    (NumMantissaBits > 0), so exponent-bits `00` is the denormal case with
///    unbiased exponent `1 - EXP_BIAS = 0` and no hidden bit, while
///    exponent-bits `01..11` are normal with unbiased exponent
///    `exp_bits - EXP_BIAS` and an implicit leading 1. Working through all 8
///    non-negative 3-bit magnitude codes (`exp_bits:mantissa`) gives exactly
///    the ladder above, with the magnitude code equal to the plain 3-bit
///    integer value: `000`->0, `001`->0.5, `010`->1.0, `011`->1.5, `100`->2.0,
///    `101`->3.0, `110`->4.0, `111`->6.0.
/// 2. NVIDIA's own ModelOpt quantizer -- the tool that produces the real
///    checkpoints this format targets -- hard-codes the identical table in
///    `modelopt/torch/quantization/qtensor/nvfp4_tensor.py` (repo
///    `NVIDIA/Model-Optimizer`, fetched via `gh api` this session):
///    `e2m1_values = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1,
///    -1.5, -2, -3, -4, -6])`.
///
/// Both sources agree exactly, including that code `0b1000` is negative zero.
const E2M1_TABLE: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// Converts a 4-bit e2m1 code (only the low nibble of `nibble` is used) to
/// its `f32` value. See [`E2M1_TABLE`] for the real bit-pattern reference.
pub fn e2m1_value(nibble: u8) -> f32 {
    E2M1_TABLE[(nibble & 0x0F) as usize]
}

/// Dequantizes one row of `k` NVFP4 (e2m1) weight values, applying both the
/// per-16-block scale and the per-tensor `weight_scale_2`.
///
/// # Packing order
/// `packed` holds two e2m1 nibbles per byte. Per NVIDIA ModelOpt's own
/// packing/unpacking code (`modelopt/torch/quantization/qtensor/
/// nvfp4_tensor.py`: pack is `(q_weight[..., 1::2] << 4) | q_weight[...,
/// 0::2]`; unpack is `unpacked[..., 0::2] = input & 0x0F; unpacked[...,
/// 1::2] = input >> 4`), byte `b`'s LOW nibble (bits 3:0) holds element
/// `2*b` and its HIGH nibble (bits 7:4) holds element `2*b + 1`.
/// `packed` must have at least `k.div_ceil(2)` bytes.
///
/// # Scale format
/// `scale` is assumed ALREADY CONVERTED from its on-disk f8_e4m3
/// representation to `f32` by the caller. This function's job is only the
/// per-16-block index arithmetic and the two-level multiply (`value *
/// scale[block] * scale2`), not f8_e4m3 byte decoding -- that conversion is
/// the caller's (and, later, the device dequant kernel's) responsibility.
/// `scale` must have `k.div_ceil(F4E2M1_BLOCK)` entries; element `i`'s block
/// index is `i / F4E2M1_BLOCK`.
pub fn dequant_f4e2m1_row(packed: &[u8], scale: &[f32], scale2: f32, k: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(k);
    for i in 0..k {
        let byte = packed[i / 2];
        let nibble = if i % 2 == 0 { byte & 0x0F } else { byte >> 4 };
        let block = i / F4E2M1_BLOCK;
        out.push(e2m1_value(nibble) * scale[block] * scale2);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e2m1_matches_the_documented_ladder() {
        // Positive nibble codes 0..8 map to {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
        // per the real NVFP4 spec (verified against both CUTLASS's
        // `cutlass::float_e2m1_t` bit layout and NVIDIA ModelOpt's own
        // `e2m1_values` table -- see `E2M1_TABLE`'s doc comment).
        let expected_magnitudes = [0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
        for (code, &want) in expected_magnitudes.iter().enumerate() {
            let got = e2m1_value(code as u8);
            assert_eq!(got, want, "code {code}");
        }
    }

    #[test]
    fn e2m1_negative_codes_mirror_the_positive_ladder_with_sign_bit_3() {
        // Codes 8..16 are bit 3 (sign) set, ORed with the same magnitude
        // code used above (sign-magnitude encoding, not two's complement).
        let expected_magnitudes = [0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
        for (code, &mag) in expected_magnitudes.iter().enumerate() {
            let got = e2m1_value((code as u8) | 0x08);
            assert_eq!(got, -mag, "negative code {}", code + 8);
        }
    }

    #[test]
    fn dequant_f4e2m1_row_applies_both_scale_levels() {
        // 16 packed e2m1 values (8 bytes), one block-scale, one tensor-scale.
        // Every nibble encodes the value 1.0 (code 0b010 = 2, per the ladder
        // above). Block scale = 2.0 (a value f8_e4m3 represents exactly),
        // weight_scale_2 = 0.5. Expected dequantized value for every
        // element: 1.0 * 2.0 * 0.5 = 1.0.
        let one_code = 0b0010u8;
        let byte = (one_code << 4) | one_code;
        let packed: Vec<u8> = vec![byte; 8];
        let scale: Vec<f32> = vec![2.0]; // one block covering all 16 elements
        let scale2 = 0.5;
        let row = dequant_f4e2m1_row(&packed, &scale, scale2, 16);
        assert_eq!(row.len(), 16);
        for (i, &v) in row.iter().enumerate() {
            assert!((v - 1.0).abs() < 1e-6, "element {i}: got {v}");
        }
    }

    #[test]
    fn dequant_f4e2m1_row_unpacks_low_nibble_then_high_nibble() {
        // Byte's low nibble (element 0) = code for 0.5 (0b0001);
        // high nibble (element 1) = sign-bit-set code for -2.0 (0b1100).
        let byte = (0b1100u8 << 4) | 0b0001u8;
        let packed = vec![byte];
        let scale = vec![1.0f32];
        let row = dequant_f4e2m1_row(&packed, &scale, 1.0, 2);
        assert_eq!(row, vec![0.5, -2.0]);
    }

    #[test]
    fn dequant_f4e2m1_row_indexes_scale_by_block_of_16() {
        // 32 elements = 2 blocks of 16. First block all code-for-1.0,
        // second block all code-for-3.0 (0b0101 = 5). Distinct per-block
        // scales must select the right entry.
        let one_code = 0b0010u8;
        let three_code = 0b0101u8;
        let one_byte = (one_code << 4) | one_code;
        let three_byte = (three_code << 4) | three_code;
        let mut packed = vec![one_byte; 8];
        packed.extend(vec![three_byte; 8]);
        let scale = vec![10.0f32, 100.0f32];
        let row = dequant_f4e2m1_row(&packed, &scale, 1.0, 32);
        for (i, &v) in row.iter().enumerate().take(16) {
            assert!((v - 10.0).abs() < 1e-6, "element {i}: got {v}");
        }
        for (i, &v) in row.iter().enumerate().skip(16) {
            assert!((v - 300.0).abs() < 1e-4, "element {i}: got {v}");
        }
    }
}
