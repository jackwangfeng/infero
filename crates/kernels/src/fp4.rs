//! Host-side reference for NVFP4 (e2m1) dequantization, and the device-side
//! kernel checked against it.
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
//!
//! [`Kernels::dequant_f4e2m1`] below is the device counterpart: it mirrors
//! [`dequant_f4e2m1_row`]'s own table and index arithmetic exactly (see
//! `cu/fp4.cu`'s doc comment), so the two are checked against each other
//! rather than against a second independent reading of the NVFP4 spec.

use anyhow::{Context, Result};
use infero_gpu::{KernelArg, LaunchConfig, View, ViewMut};

use crate::{Kernels, fp4_src};

/// Threads a block for [`Kernels::dequant_f4e2m1`]'s k-dimension tiling.
const FP4_DEQUANT_BLOCK: u32 = 256;

/// Threads a block for [`Kernels::quantize_act_e2m1_cutlass`]'s
/// block-of-16 tiling (one thread per 16-element block, not per element).
const FP4_QUANT_BLOCK: u32 = 256;

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

/// Host-side round-to-nearest f32 -> f8_e4m3 encoder, for test/reference use
/// only (the real device path is `f32_to_e4m3` in `common.cuh`'s hardware
/// `cvt.rn.satfinite.e4m3x2.f32` instruction, sm_89+ only).
///
/// Brute-forces every non-NaN byte through the already-verified decode
/// oracle (`infero_safetensors::e4m3_value`, cross-checked in Task 1/2's own
/// work) and keeps whichever decodes closest to `f`, rather than hand-rolling
/// encode bit arithmetic a second time. This also reproduces the hardware
/// instruction's `satfinite` saturation for free: any `f` outside e4m3's
/// range is simply closest to one of the two extreme finite codes (+-448).
///
/// Exists because this crate's own dev sandbox GPU (an RTX A4000, sm_86)
/// cannot run the real hardware conversion at all -- `Device::caps().fp8`
/// is `false` below sm_89 (`crates/cuda/src/backend.rs`), the same gate
/// `quantize_act_e4m3_f32`'s own tests already skip under
/// (`tests/quantize_act_e4m3.rs`) -- so [`quantize_act_e2m1_row`] below
/// needs a portable stand-in to be checkable with real numbers on hardware
/// that can't run [`Kernels::quantize_act_e2m1_cutlass`]'s actual kernel.
fn f32_to_e4m3_host(f: f32) -> u8 {
    let mut best_byte = 0u8;
    let mut best_diff = f32::INFINITY;
    for b in 0u16..256 {
        if b == 0x7F || b == 0xFF {
            continue; // the two NaN patterns
        }
        let v = infero_safetensors::e4m3_value(b as u8);
        let d = (f - v).abs();
        if d < best_diff {
            best_diff = d;
            best_byte = b as u8;
        }
    }
    best_byte
}

/// Host-side reference for the real NVFP4 activation quantizer -- the
/// inverse direction of [`dequant_f4e2m1_row`]. Given one row of `k` f32
/// activations and the checkpoint's static per-tensor `global_scale`,
/// produces packed e2m1 bytes and per-16-block f8_e4m3 scale bytes, per
/// vLLM's own real reference quantizer (`cast_to_fp4` / `ref_nvfp4_quant`,
/// real installed vLLM 0.27.1 source, `vllm/model_executor/layers/
/// quantization/utils/nvfp4_emulation_utils.py`) -- NOT a plain "divide by
/// the checkpoint's static scale": a real per-16-block dynamic scale is
/// computed from the data itself, using `global_scale` only as an input to
/// that computation, mirroring [`Kernels::quantize_act_e2m1_cutlass`]'s
/// device kernel (`cu/fp4.cu`'s `quantize_act_e2m1_f32`) step for step, with
/// [`f32_to_e4m3_host`] standing in for that kernel's hardware
/// `f32_to_e4m3` call, since this crate's own dev sandbox cannot run it
/// (see that function's own doc comment).
///
/// Per block: `vec_max` = max(|x_i|) over the block; `scale_f32 =
/// clamp(global_scale * vec_max / 6.0, -448, 448)`, quantized to f8_e4m3
/// (the LOSSY requantized value is what both the returned scale byte and
/// the following `output_scale` computation use); `output_scale = 0` if
/// that requantized scale is `0`, else `global_scale / scale_q`. Each
/// element is then `scaled = x_i * output_scale`, clamped to `[-6, 6]`, and
/// rounded to the e2m1 ladder via `cast_to_fp4`'s own asymmetric `<=`/`<`
/// threshold table (copied verbatim, not renormalized).
///
/// Returns `(packed, scale_bytes)`: `packed` has `k.div_ceil(2)` bytes (the
/// same low-nibble-is-even-index convention [`dequant_f4e2m1_row`] expects),
/// `scale_bytes` has `k.div_ceil(F4E2M1_BLOCK)` raw f8_e4m3 bytes.
pub fn quantize_act_e2m1_row(x: &[f32], global_scale: f32, k: usize) -> (Vec<u8>, Vec<u8>) {
    let blocks = k.div_ceil(F4E2M1_BLOCK);
    let mut packed = vec![0u8; k.div_ceil(2)];
    let mut scale_bytes = vec![0u8; blocks];

    for block in 0..blocks {
        let base = block * F4E2M1_BLOCK;
        let n_in_block = F4E2M1_BLOCK.min(k - base);

        let vec_max = x[base..base + n_in_block]
            .iter()
            .fold(0.0f32, |m, &v| m.max(v.abs()));

        let scale_f32 = (global_scale * vec_max / 6.0).clamp(-448.0, 448.0);
        let scale_byte = f32_to_e4m3_host(scale_f32);
        scale_bytes[block] = scale_byte;
        let scale_q = infero_safetensors::e4m3_value(scale_byte);

        let output_scale = if scale_q == 0.0 {
            0.0
        } else {
            global_scale / scale_q
        };

        for i in 0..n_in_block {
            let idx = base + i;
            let scaled = x[idx] * output_scale;
            let clipped = scaled.clamp(-6.0, 6.0);
            let mag = clipped.abs();

            // `cast_to_fp4`'s own asymmetric threshold table, copied
            // verbatim -- see this function's own doc comment.
            let mag_code: u8 = if mag <= 0.25 {
                0
            } else if mag < 0.75 {
                1
            } else if mag <= 1.25 {
                2
            } else if mag < 1.75 {
                3
            } else if mag <= 2.5 {
                4
            } else if mag < 3.5 {
                5
            } else if mag <= 5.0 {
                6
            } else {
                7
            };
            let sign_bit: u8 = if clipped < 0.0 { 0x08 } else { 0x00 };
            let code = mag_code | sign_bit;

            let byte_idx = idx / 2;
            if idx % 2 == 0 {
                packed[byte_idx] = (packed[byte_idx] & 0xF0) | code;
            } else {
                packed[byte_idx] = (packed[byte_idx] & 0x0F) | (code << 4);
            }
        }
    }
    (packed, scale_bytes)
}

impl Kernels {
    /// Dequantizes an `n x k` NVFP4 (e2m1) matrix on-device, into `out`
    /// (`n * k` f32 elements, row-major -- matching [`dequant_f4e2m1_row`]'s
    /// per-row convention applied to every row of the matrix at once).
    ///
    /// `w` holds `n` rows of `k.div_ceil(2)` packed bytes each; `scale` holds
    /// `n` rows of `k.div_ceil(F4E2M1_BLOCK)` f8_e4m3 block-scale bytes each;
    /// `scale2` is the single per-tensor `weight_scale_2` scalar shared by
    /// the whole matrix. See `cu/fp4.cu`'s doc comment for the exact layout
    /// and [`crate::WeightType::F4E2M1`] for the on-disk version this mirrors.
    pub fn dequant_f4e2m1(
        &self,
        out: &mut ViewMut<'_, f32>,
        w: &View<'_, u8>,
        scale: &View<'_, u8>,
        scale2: f32,
        k: usize,
        n: usize,
    ) -> Result<()> {
        debug_assert!(
            out.len() >= n * k,
            "dequant output holds {} elements, need {}",
            out.len(),
            n * k
        );
        debug_assert!(
            w.len() >= n * k.div_ceil(2),
            "dequant input holds {} packed bytes, need {}",
            w.len(),
            n * k.div_ceil(2)
        );
        debug_assert!(
            scale.len() >= n * k.div_ceil(F4E2M1_BLOCK),
            "dequant scale holds {} bytes, need {}",
            scale.len(),
            n * k.div_ceil(F4E2M1_BLOCK)
        );
        let f = self
            .dev
            .kernels()
            .get("infero_fp4", fp4_src(), "dequant_f4e2m1_f32")?;
        let cfg = LaunchConfig {
            grid_dim: (n as u32, (k as u32).div_ceil(FP4_DEQUANT_BLOCK).max(1), 1),
            block_dim: (FP4_DEQUANT_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let (ki, ni) = (k as i32, n as i32);
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(out).arg(w).arg(scale).arg(&scale2).arg(&ki).arg(&ni);
        self.dev
            .profile()
            .time("dequant_f4e2m1", self.dev.stream(), || {
                unsafe { b.launch(cfg) }.context("dequant_f4e2m1")?;
                Ok(())
            })?;
        Ok(())
    }

    /// Quantizes an `n_tokens x k` f32 activation matrix to packed NVFP4
    /// (e2m1) bytes plus a real per-16-block dynamic f8_e4m3 scale, per
    /// vLLM's own NVFP4 reference quantizer (`cast_to_fp4` /
    /// `ref_nvfp4_quant`, real installed vLLM 0.27.1 source) -- NOT a plain
    /// "divide by the checkpoint's static scale" op. `input_scale` is the
    /// checkpoint's static per-tensor scale, used as `ref_nvfp4_quant`'s own
    /// `global_scale` input to the real per-block scale computation.
    ///
    /// `xq` gets `n_tokens` rows of `k.div_ceil(2)` packed bytes each (same
    /// low-nibble-is-even-index convention as [`Kernels::dequant_f4e2m1`]'s
    /// `w`). `xq_scale` gets `n_tokens` rows of `k.div_ceil(F4E2M1_BLOCK)`
    /// f8_e4m3 scale bytes each, row-major -- the SAME linear layout
    /// `dequant_f4e2m1`'s own `scale` argument consumes (deliberately not
    /// the transposed/padded layout `quantize_act_e4m3_cutlass_f32`'s
    /// `sfa_t` uses for its own, different, CUTLASS SFA convention; see
    /// `cu/fp4.cu`'s doc comment for why that transform is out of scope
    /// here). `x` holds `n_tokens` rows of `k` f32 elements each.
    pub fn quantize_act_e2m1_cutlass(
        &self,
        xq: &mut ViewMut<'_, u8>,
        xq_scale: &mut ViewMut<'_, u8>,
        x: &View<'_, f32>,
        input_scale: f32,
        k: usize,
        n_tokens: usize,
    ) -> Result<()> {
        let blocks_per_row = k.div_ceil(F4E2M1_BLOCK);
        debug_assert!(
            xq.len() >= n_tokens * k.div_ceil(2),
            "quantize output holds {} packed bytes, need {}",
            xq.len(),
            n_tokens * k.div_ceil(2)
        );
        debug_assert!(
            xq_scale.len() >= n_tokens * blocks_per_row,
            "quantize scale output holds {} bytes, need {}",
            xq_scale.len(),
            n_tokens * blocks_per_row
        );
        debug_assert!(
            x.len() >= n_tokens * k,
            "quantize input holds {} elements, need {}",
            x.len(),
            n_tokens * k
        );
        let f = self
            .dev
            .kernels()
            .get("infero_fp4", fp4_src(), "quantize_act_e2m1_f32")?;
        let cfg = LaunchConfig {
            grid_dim: (
                n_tokens as u32,
                (blocks_per_row as u32).div_ceil(FP4_QUANT_BLOCK).max(1),
                1,
            ),
            block_dim: (FP4_QUANT_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let (ki, ni) = (k as i32, n_tokens as i32);
        let mut b = self.dev.stream().launch_builder(&f);
        b.arg(xq)
            .arg(xq_scale)
            .arg(x)
            .arg(&input_scale)
            .arg(&ki)
            .arg(&ni);
        self.dev
            .profile()
            .time("quantize_act_e2m1_cutlass", self.dev.stream(), || {
                unsafe { b.launch(cfg) }.context("quantize_act_e2m1_cutlass")?;
                Ok(())
            })?;
        Ok(())
    }
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
