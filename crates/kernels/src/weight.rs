//! The subset of ggml types the kernels can decode.

use anyhow::{Result, bail};
use tuili_gguf::GgmlType;

/// A weight encoding with a matching CUDA decoder.
///
/// Adding one means writing its block layout into `common.cuh`, its element
/// decoder and its mat-vec into `quant.cu`, and a variant here — the three
/// stay in lockstep through [`WeightType::suffix`], which names the kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WeightType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q4K,
    Q6K,
    /// Four bits with an `f16` scale and zero point every 128 weights, laid out
    /// one output row at a time. This is what an AWQ checkpoint is repacked
    /// into at load; see [`crate::awq`].
    Q4G128,
    /// [`WeightType::Q8_0`] with the quants and the scales in separate regions:
    /// `k` contiguous int8 a row, then one `f16` scale per 32. Same bytes, and
    /// the same values in the same order — but a row's quants are one run, so the
    /// tile loader reads sixteen at a time where a 34-byte block forces two. Only
    /// the vocab projection uses it, because it is the one matrix tuili quantizes
    /// itself and so the only one whose layout is ours to choose.
    Q8_0S,
    /// [`WeightType::Q4G128`] with the 4-byte words inside each nibble run
    /// transposed and the scales moved to a trailing region, so a tensor-core
    /// lane's whole weight fragment is one aligned 16-byte read. Same total
    /// bytes; see [`crate::awq::transpose_words`].
    Q4G128T,
}

impl WeightType {
    pub const ALL: [WeightType; 12] = [
        WeightType::F32,
        WeightType::F16,
        WeightType::Q4_0,
        WeightType::Q4_1,
        WeightType::Q5_0,
        WeightType::Q5_1,
        WeightType::Q8_0,
        WeightType::Q4K,
        WeightType::Q6K,
        WeightType::Q4G128,
        WeightType::Q4G128T,
        WeightType::Q8_0S,
    ];

    pub fn from_ggml(t: GgmlType) -> Result<Self> {
        Ok(match t {
            GgmlType::F32 => WeightType::F32,
            GgmlType::F16 => WeightType::F16,
            GgmlType::Q4_0 => WeightType::Q4_0,
            GgmlType::Q4_1 => WeightType::Q4_1,
            GgmlType::Q5_0 => WeightType::Q5_0,
            GgmlType::Q5_1 => WeightType::Q5_1,
            GgmlType::Q8_0 => WeightType::Q8_0,
            GgmlType::Q4K => WeightType::Q4K,
            GgmlType::Q6K => WeightType::Q6K,
            other => bail!(
                "weight type {other} is not implemented; supported: \
                 F32, F16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q4_K, Q6_K"
            ),
        })
    }

    /// The kernel-name suffix, e.g. `gemv_q4_K`.
    pub const fn suffix(self) -> &'static str {
        match self {
            WeightType::F32 => "f32",
            WeightType::F16 => "f16",
            WeightType::Q4_0 => "q4_0",
            WeightType::Q4_1 => "q4_1",
            WeightType::Q5_0 => "q5_0",
            WeightType::Q5_1 => "q5_1",
            WeightType::Q8_0 => "q8_0",
            WeightType::Q8_0S => "q8_0s",
            WeightType::Q4K => "q4_K",
            WeightType::Q6K => "q6_K",
            WeightType::Q4G128 => "q4_g128",
            WeightType::Q4G128T => "q4_g128t",
        }
    }

    pub const fn block_size(self) -> usize {
        match self {
            WeightType::F32 | WeightType::F16 => 1,
            WeightType::Q4_0
            | WeightType::Q4_1
            | WeightType::Q5_0
            | WeightType::Q5_1
            | WeightType::Q8_0
            | WeightType::Q8_0S => 32,
            WeightType::Q4K | WeightType::Q6K => 256,
            WeightType::Q4G128 | WeightType::Q4G128T => 128,
        }
    }

    pub const fn type_size(self) -> usize {
        match self {
            WeightType::F32 => 4,
            WeightType::F16 => 2,
            WeightType::Q4_0 => 18,
            WeightType::Q4_1 => 20,
            WeightType::Q5_0 => 22,
            WeightType::Q5_1 => 24,
            // Counted as a block for accounting only; the bytes are split.
            WeightType::Q8_0 | WeightType::Q8_0S => 34,
            WeightType::Q4K => 144,
            WeightType::Q6K => 210,
            // `__half2` of {scale, scale * zero} then 128 nibbles.
            WeightType::Q4G128 | WeightType::Q4G128T => 68,
        }
    }

    pub const fn is_quantized(self) -> bool {
        self.block_size() != 1
    }

    /// How many independent chunks the mat-vec kernel walks for a row of `k`
    /// elements: one per block for the block-decoding kernels, one per element
    /// for the rest.
    ///
    /// This sizes the launch. A row of 896 Q8_0 values is only 28 blocks of
    /// work, so a 256-thread block would leave seven eighths of its threads
    /// idle and still pay for a full block-wide reduction.
    pub const fn gemv_work_items(self, k: usize) -> usize {
        match self {
            // Eight elements per thread.
            WeightType::Q8_0 => k / 8,
            // One 32-element group of a K-quant super-block per thread.
            WeightType::Q4K => k / 32,
            // Four elements per thread; see the kernel for why so fine.
            WeightType::Q6K => k / 4,
            // One 32-element quarter of a group per thread, as for Q4_K.
            WeightType::Q4G128 => k / 32,
            // The rest decode one element at a time.
            _ => k,
        }
    }

    /// Bits per weight, for reporting.
    pub fn bits_per_weight(self) -> f32 {
        self.type_size() as f32 * 8.0 / self.block_size() as f32
    }
}

impl std::fmt::Display for WeightType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            WeightType::F32 => "F32",
            WeightType::F16 => "F16",
            WeightType::Q4_0 => "Q4_0",
            WeightType::Q4_1 => "Q4_1",
            WeightType::Q5_0 => "Q5_0",
            WeightType::Q5_1 => "Q5_1",
            WeightType::Q8_0 => "Q8_0",
            WeightType::Q8_0S => "Q8_0S",
            WeightType::Q4K => "Q4_K",
            WeightType::Q6K => "Q6_K",
            WeightType::Q4G128 => "Q4_G128",
            WeightType::Q4G128T => "Q4_G128T",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layouts_agree_with_the_gguf_crate() {
        for w in WeightType::ALL {
            // Q4_G128 has no ggml counterpart: it is what an AWQ checkpoint is
            // repacked into, and nothing in a GGUF file ever carries it.
            // Neither Q4_G128 form has a ggml counterpart: they are what an
            // AWQ checkpoint is repacked into, and nothing in a GGUF file ever
            // carries one. Both hold 128 nibbles and a scale pair per block,
            // however they arrange them, so the byte count is the same.
            // Q8_0S is ours as well: same bytes as Q8_0, arranged differently.
            if w == WeightType::Q8_0S {
                assert_eq!(w.block_size(), 32);
                assert_eq!(w.type_size(), WeightType::Q8_0.type_size());
                continue;
            }
            if w == WeightType::Q4G128 || w == WeightType::Q4G128T {
                assert_eq!(w.block_size() * 4 + 4 * 8, w.type_size() * 8);
                continue;
            }
            let g = match w {
                WeightType::F32 => GgmlType::F32,
                WeightType::F16 => GgmlType::F16,
                WeightType::Q4_0 => GgmlType::Q4_0,
                WeightType::Q4_1 => GgmlType::Q4_1,
                WeightType::Q5_0 => GgmlType::Q5_0,
                WeightType::Q5_1 => GgmlType::Q5_1,
                WeightType::Q8_0 => GgmlType::Q8_0,
                WeightType::Q4K => GgmlType::Q4K,
                WeightType::Q6K => GgmlType::Q6K,
                WeightType::Q4G128 | WeightType::Q4G128T | WeightType::Q8_0S => {
                    unreachable!("handled above")
                }
            };
            assert_eq!(w.block_size(), g.block_size(), "{w}");
            assert_eq!(w.type_size(), g.type_size(), "{w}");
            assert_eq!(WeightType::from_ggml(g).unwrap(), w);
        }
    }

    #[test]
    fn unsupported_types_are_rejected_by_name() {
        let err = WeightType::from_ggml(GgmlType::Q5K)
            .unwrap_err()
            .to_string();
        assert!(err.contains("Q5_K"), "{err}");
    }

    #[test]
    fn bit_widths_are_what_the_names_claim() {
        assert_eq!(WeightType::Q8_0.bits_per_weight(), 8.5);
        assert_eq!(WeightType::Q4K.bits_per_weight(), 4.5);
    }
}
