//! ggml tensor element types and their block layouts.

use anyhow::{Result, bail};

/// Number of elements per K-quant super-block.
pub const QK_K: usize = 256;

/// A ggml tensor element type, as stored in the GGUF tensor info table.
///
/// The discriminants are the on-disk values and must not be renumbered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    Iq2Xxs = 16,
    Iq2Xs = 17,
    Iq3Xxs = 18,
    Iq1S = 19,
    Iq4Nl = 20,
    Iq3S = 21,
    Iq2S = 22,
    Iq4Xs = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    Iq1M = 29,
    BF16 = 30,
}

impl GgmlType {
    pub fn from_u32(v: u32) -> Result<Self> {
        use GgmlType::*;
        Ok(match v {
            0 => F32,
            1 => F16,
            2 => Q4_0,
            3 => Q4_1,
            6 => Q5_0,
            7 => Q5_1,
            8 => Q8_0,
            9 => Q8_1,
            10 => Q2K,
            11 => Q3K,
            12 => Q4K,
            13 => Q5K,
            14 => Q6K,
            15 => Q8K,
            16 => Iq2Xxs,
            17 => Iq2Xs,
            18 => Iq3Xxs,
            19 => Iq1S,
            20 => Iq4Nl,
            21 => Iq3S,
            22 => Iq2S,
            23 => Iq4Xs,
            24 => I8,
            25 => I16,
            26 => I32,
            27 => I64,
            28 => F64,
            29 => Iq1M,
            30 => BF16,
            other => bail!("unknown ggml type id {other}"),
        })
    }

    /// Elements packed into one block. Non-quantized types use 1.
    pub const fn block_size(self) -> usize {
        use GgmlType::*;
        match self {
            F32 | F16 | BF16 | I8 | I16 | I32 | I64 | F64 => 1,
            Q4_0 | Q4_1 | Q5_0 | Q5_1 | Q8_0 | Q8_1 | Iq4Nl => 32,
            Q2K | Q3K | Q4K | Q5K | Q6K | Q8K | Iq2Xxs | Iq2Xs | Iq3Xxs | Iq1S | Iq3S | Iq2S
            | Iq4Xs | Iq1M => QK_K,
        }
    }

    /// Bytes occupied by one block.
    pub const fn type_size(self) -> usize {
        use GgmlType::*;
        match self {
            F32 | I32 => 4,
            F16 | BF16 | I16 => 2,
            I8 => 1,
            I64 | F64 => 8,
            Q4_0 => 18, // d:f16 + 32 nibbles
            Q4_1 => 20, // d,m:f16 + 32 nibbles
            Q5_0 => 22,
            Q5_1 => 24,
            Q8_0 => 34, // d:f16 + 32 i8
            Q8_1 => 36,
            Q2K => 84,
            Q3K => 110,
            Q4K => 144, // d,dmin:f16 + 12 scale bytes + 128 nibbles
            Q5K => 176,
            Q6K => 210, // 128 low + 64 high + 16 scales + d:f16
            Q8K => 292,
            Iq2Xxs => 66,
            Iq2Xs => 74,
            Iq3Xxs => 98,
            Iq1S => 50,
            Iq1M => 56,
            Iq4Nl => 18,
            Iq3S => 110,
            Iq2S => 82,
            Iq4Xs => 136,
        }
    }

    pub const fn is_quantized(self) -> bool {
        self.block_size() != 1
    }

    /// Bytes needed to store `n_elements` of this type.
    pub fn size_for(self, n_elements: usize) -> Result<usize> {
        let blck = self.block_size();
        if !n_elements.is_multiple_of(blck) {
            bail!("{self:?} needs a multiple of {blck} elements, got {n_elements}");
        }
        Ok(n_elements / blck * self.type_size())
    }

    pub const fn name(self) -> &'static str {
        use GgmlType::*;
        match self {
            F32 => "F32",
            F16 => "F16",
            Q4_0 => "Q4_0",
            Q4_1 => "Q4_1",
            Q5_0 => "Q5_0",
            Q5_1 => "Q5_1",
            Q8_0 => "Q8_0",
            Q8_1 => "Q8_1",
            Q2K => "Q2_K",
            Q3K => "Q3_K",
            Q4K => "Q4_K",
            Q5K => "Q5_K",
            Q6K => "Q6_K",
            Q8K => "Q8_K",
            Iq2Xxs => "IQ2_XXS",
            Iq2Xs => "IQ2_XS",
            Iq3Xxs => "IQ3_XXS",
            Iq1S => "IQ1_S",
            Iq4Nl => "IQ4_NL",
            Iq3S => "IQ3_S",
            Iq2S => "IQ2_S",
            Iq4Xs => "IQ4_XS",
            I8 => "I8",
            I16 => "I16",
            I32 => "I32",
            I64 => "I64",
            F64 => "F64",
            Iq1M => "IQ1_M",
            BF16 => "BF16",
        }
    }
}

impl std::fmt::Display for GgmlType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_layouts_match_ggml() {
        // A row of 4096 elements, the numbers llama.cpp reports for these types.
        assert_eq!(GgmlType::F16.size_for(4096).unwrap(), 8192);
        assert_eq!(GgmlType::Q8_0.size_for(4096).unwrap(), 4096 / 32 * 34);
        assert_eq!(GgmlType::Q4K.size_for(4096).unwrap(), 4096 / 256 * 144);
        assert_eq!(GgmlType::Q6K.size_for(4096).unwrap(), 4096 / 256 * 210);
    }

    #[test]
    fn partial_block_is_rejected() {
        assert!(GgmlType::Q4K.size_for(100).is_err());
    }
}
