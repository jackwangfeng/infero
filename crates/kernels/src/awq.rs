//! AWQ checkpoints, repacked into [`WeightType::Q4G128`].
//!
//! AWQ stores a linear layer transposed and column-packed: `qweight` is
//! `[in_features, out_features / 8]` of `i32`, where one `i32` holds the 4-bit
//! codes of **eight neighbouring output channels** at a single input index.
//! Summing over `k` for one output therefore strides by `out_features / 8`
//! words, which is the opposite of what a mat-vec wants — it wants one output's
//! whole row contiguous so a block can stream it.
//!
//! Rather than write a second kernel shape for the sake of one file format,
//! the weights are transposed once at load into the layout the existing kernels
//! already read: output-major, 128 weights per block, an `f16` scale and zero
//! per block. This is what vLLM's `awq_marlin` does too, for the same reason.
//!
//! The nibble order inside an `i32` is not sequential. AutoAWQ packs output
//! offset `ORDER[i]` at bit `4 * i`, with `ORDER = [0, 2, 4, 6, 1, 3, 5, 7]`,
//! an artefact of the interleave its original CUDA kernel wanted. Getting this
//! wrong produces weights that look entirely plausible — the right values, in
//! the wrong columns — so `tests/awq.rs` checks the result against the same
//! model's GGUF quantization rather than against itself.

use anyhow::Result;

use crate::WeightType;

/// Weights per block, and AWQ's group size. They have to match: a block
/// carries exactly one scale, and so does a group.
pub const GROUP: usize = 128;

/// Bytes per packed block: `__half2` of {scale, scale * zero}, then 128 nibbles.
pub const BLOCK_BYTES: usize = 68;

/// Where output offset `i` within a packed `i32` actually lives: AutoAWQ writes
/// output `ORDER[i]` into bits `4 * i`.
const ORDER: [usize; 8] = [0, 2, 4, 6, 1, 3, 5, 7];

/// For output offset `m`, which nibble holds it. The inverse of [`ORDER`].
const NIBBLE_OF: [usize; 8] = {
    let mut inv = [0usize; 8];
    let mut i = 0;
    while i < 8 {
        inv[ORDER[i]] = i;
        i += 1;
    }
    inv
};

/// One AWQ linear layer, borrowed from the checkpoint.
pub struct AwqTensor<'a> {
    /// `[in_features, out_features / 8]`
    pub qweight: &'a [i32],
    /// `[in_features / GROUP, out_features / 8]`
    pub qzeros: &'a [i32],
    /// `[in_features / GROUP, out_features]`
    pub scales: &'a [half::f16],
    pub in_features: usize,
    pub out_features: usize,
}

impl AwqTensor<'_> {
    fn check(&self) -> Result<()> {
        let (k, n) = (self.in_features, self.out_features);
        anyhow::ensure!(
            k.is_multiple_of(GROUP),
            "in_features {k} is not a multiple of the {GROUP}-weight group"
        );
        anyhow::ensure!(n.is_multiple_of(8), "out_features {n} is not a multiple of 8");
        anyhow::ensure!(
            self.qweight.len() == k * n / 8,
            "qweight has {} words, expected {}",
            self.qweight.len(),
            k * n / 8
        );
        anyhow::ensure!(
            self.qzeros.len() == k / GROUP * n / 8,
            "qzeros has {} words, expected {}",
            self.qzeros.len(),
            k / GROUP * n / 8
        );
        anyhow::ensure!(
            self.scales.len() == k / GROUP * n,
            "scales has {} values, expected {}",
            self.scales.len(),
            k / GROUP * n
        );
        Ok(())
    }

    /// The 4-bit code for output `n` at input `k`.
    #[inline]
    fn code(&self, k: usize, n: usize) -> u32 {
        let word = self.qweight[k * (self.out_features / 8) + n / 8] as u32;
        (word >> (4 * NIBBLE_OF[n % 8])) & 0xF
    }

    /// The zero point for output `n` in the group covering input `k`.
    #[inline]
    fn zero(&self, k: usize, n: usize) -> u32 {
        let g = k / GROUP;
        let word = self.qzeros[g * (self.out_features / 8) + n / 8] as u32;
        (word >> (4 * NIBBLE_OF[n % 8])) & 0xF
    }

    /// The scale for output `n` in the group covering input `k`, which is also
    /// the quantization step and so the natural unit for any tolerance.
    #[inline]
    pub fn scale(&self, k: usize, n: usize) -> f32 {
        f32::from(self.scales[k / GROUP * self.out_features + n])
    }

    /// One dequantized weight, for tests and for the reference path.
    pub fn weight(&self, k: usize, n: usize) -> f32 {
        (self.code(k, n) as f32 - self.zero(k, n) as f32) * self.scale(k, n)
    }

    /// Bytes a [`WeightType::Q4G128`] repacking of this tensor occupies.
    pub fn packed_bytes(&self) -> usize {
        self.out_features * self.in_features / GROUP * BLOCK_BYTES
    }

    /// Repack into output-major [`WeightType::Q4G128`] blocks.
    ///
    /// Byte `b` of a block holds weight `b` in its low nibble and weight
    /// `b + 64` in its high nibble, so the four 32-weight quarters a Q8_1
    /// activation block covers come out as two `int` loads each — the same
    /// arrangement `Q4_0` uses, which is what lets the mat-vec reuse its dot
    /// product.
    pub fn repack(&self) -> Result<Vec<u8>> {
        self.check()?;
        let (k, n) = (self.in_features, self.out_features);
        let blocks_per_row = k / GROUP;
        let mut out = vec![0u8; self.packed_bytes()];

        // Rows are independent, and there are seven billion weights in an 8B
        // checkpoint: single-threaded this is a minute of load time.
        let threads = std::thread::available_parallelism()
            .map_or(1, |n| n.get())
            .min(n);
        let per_thread = n.div_ceil(threads);
        let row_bytes = blocks_per_row * BLOCK_BYTES;
        std::thread::scope(|scope| {
            for (t, chunk) in out.chunks_mut(per_thread * row_bytes).enumerate() {
                let first = t * per_thread;
                scope.spawn(move || self.fill_rows(chunk, first, blocks_per_row));
            }
        });
        Ok(out)
    }

    /// Pack `out.len() / (blocks * BLOCK_BYTES)` rows starting at `first`.
    fn fill_rows(&self, out: &mut [u8], first: usize, blocks_per_row: usize) {
        let rows = out.len() / (blocks_per_row * BLOCK_BYTES);
        for r in 0..rows {
            let row = first + r;
            for b in 0..blocks_per_row {
                let k0 = b * GROUP;
                let s = self.scale(k0, row);
                let z = self.zero(k0, row) as f32;
                let off = (r * blocks_per_row + b) * BLOCK_BYTES;
                // {scale, scale * zero}: the dot product wants the zero folded
                // into the scale, because it multiplies the activation block's
                // own sum rather than each weight.
                //
                // Storing the product rather than the zero costs an f16 ulp on
                // the offset, so a weight whose code equals its zero decodes to
                // something near zero rather than exactly zero. Q4_1 and Q4_K
                // carry their minimum the same way for the same reason: it
                // turns a per-weight subtraction into one per block.
                out[off..off + 2].copy_from_slice(&half::f16::from_f32(s).to_le_bytes());
                out[off + 2..off + 4]
                    .copy_from_slice(&half::f16::from_f32(s * z).to_le_bytes());
                for i in 0..64 {
                    let lo = self.code(k0 + i, row);
                    let hi = self.code(k0 + 64 + i, row);
                    out[off + 4 + i] = (lo | (hi << 4)) as u8;
                }
            }
        }
    }
}

/// Dequantize a repacked row, for tests.
pub fn unpack_row(packed: &[u8], k: usize, row: usize) -> Vec<f32> {
    let blocks = k / GROUP;
    let mut out = Vec::with_capacity(k);
    for b in 0..blocks {
        let off = (row * blocks + b) * BLOCK_BYTES;
        let s = f32::from(half::f16::from_le_bytes([packed[off], packed[off + 1]]));
        let sz = f32::from(half::f16::from_le_bytes([packed[off + 2], packed[off + 3]]));
        let qs = &packed[off + 4..off + 68];
        for i in 0..GROUP {
            let byte = qs[i % 64];
            let code = if i < 64 { byte & 0xF } else { byte >> 4 };
            out.push(code as f32 * s - sz);
        }
    }
    out
}

/// The weight type a repacked AWQ tensor has.
pub const fn packed_type() -> WeightType {
    WeightType::Q4G128
}

/// [`WeightType::Q4G128`] rearranged so a tensor-core lane's whole weight
/// fragment is one aligned 16-byte read.
///
/// Two things move. The 4x4 matrix of 4-byte words inside each 64-byte nibble
/// run is transposed, `new[c*16 + w*4 + i] = old[w*16 + c*4 + i]`, which puts
/// the four words a lane fetches — the low and high halves of both 32-byte
/// runs — side by side. And the scales leave the blocks for a region of their
/// own, because `qs` sits four bytes into a 68-byte block and no offset inside
/// it is ever 16-byte aligned.
///
/// The split is global: `n * nb` blocks of 64 quant bytes, then `n * nb`
/// scale pairs. A per-row split was tried first, because a kernel then needs
/// only the row base it already computes and the mat-vec macros take it with
/// no new plumbing — and it costs 12% at 32 tokens on `ffn_gate`, 224 GB/s
/// against 251. The scales interleaved into the row stream is what does it.
/// So the mat-vec learned to take the matrix width instead.
///
/// Same total bytes as the packed blocks. `k` still has to be a multiple of
/// 512 for a block's quants to land on 16 bytes.
/// Measured against the packed layout on the f16 GEMM, in GB/s of weights at
/// 32 tokens: `ffn_gate` 215 -> 252, `attn_q` 173 -> 196, `ffn_down` 227 ->
/// 229, `attn_k` 116 -> 123.
pub fn transpose_words(packed: &[u8], k: usize, n: usize) -> Vec<u8> {
    let nb = k / GROUP;
    let mut out = vec![0u8; n * nb * BLOCK_BYTES];
    let scales = n * nb * 64;
    for row in 0..n {
        for b in 0..nb {
            let src = (row * nb + b) * BLOCK_BYTES;
            let so = scales + (row * nb + b) * 4;
            out[so..so + 4].copy_from_slice(&packed[src..src + 4]);
            let qs = &packed[src + 4..src + BLOCK_BYTES];
            let dst = (row * nb + b) * 64;
            for c in 0..4 {
                for w in 0..4 {
                    for i in 0..4 {
                        // [0, 2, 1, 3]: see the note above.
                        const P: [usize; 4] = [0, 2, 1, 3];
                        out[dst + c * 16 + w * 4 + P[i]] = qs[w * 16 + c * 4 + i];
                    }
                }
            }
        }
    }
    out
}

/// Whether a row length admits [`transpose_words`]: the row stride has to put
/// every row's quants on a 16-byte boundary.
pub const fn transposable(k: usize) -> bool {
    k % (GROUP * 4) == 0
}

/// Two [`transpose_words`] tensors of the same `k`, stacked along `n`.
///
/// vLLM does this at load time and calls the results `qkv_proj` and
/// `gate_up_proj` (`model_executor/models/llama.py`, whose weight loader maps
/// `.q_proj` to the `"q"` shard of `.qkv_proj`). The reason is not tidiness: a
/// matmul's efficiency here rises steeply with its width, because a narrow one
/// cannot fill the device. Measured on a Blackwell RTX PRO 6000 at 32 tokens,
/// in GB/s of weights: `attn_k` (4096x1024) 261, `attn_q` (4096x4096) 608,
/// `qkv` (4096x6144) 799, `ffn_gate` (4096x14336) 1154, `gate_up`
/// (4096x28672) 1368. Fusing costs a layer 104.0 us where seven separate
/// matmuls cost 127.4.
///
/// The layout is two regions — all quants, then all scales — so this is not an
/// append. Both regions concatenate separately and the scale region moves.
pub fn concat_t(a: &[u8], n_a: usize, b: &[u8], n_b: usize, k: usize) -> Vec<u8> {
    let nb = k / GROUP;
    let (qa, qb) = (n_a * nb * 64, n_b * nb * 64);
    let mut out = vec![0u8; (n_a + n_b) * nb * BLOCK_BYTES];
    out[..qa].copy_from_slice(&a[..qa]);
    out[qa..qa + qb].copy_from_slice(&b[..qb]);
    let s = qa + qb;
    out[s..s + n_a * nb * 4].copy_from_slice(&a[qa..qa + n_a * nb * 4]);
    out[s + n_a * nb * 4..].copy_from_slice(&b[qb..qb + n_b * nb * 4]);
    out
}

/// Dequantize a row of [`transpose_words`] output, for tests.
pub fn unpack_row_t(packed: &[u8], k: usize, n: usize, row: usize) -> Vec<f32> {
    let nb = k / GROUP;
    let mut out = Vec::with_capacity(k);
    for b in 0..nb {
        let so = n * nb * 64 + (row * nb + b) * 4;
        let s = f32::from(half::f16::from_le_bytes([packed[so], packed[so + 1]]));
        let sz = f32::from(half::f16::from_le_bytes([packed[so + 2], packed[so + 3]]));
        let qs = &packed[(row * nb + b) * 64..(row * nb + b) * 64 + 64];
        for i in 0..GROUP {
            // Undo the word transpose to find element `i`'s byte.
            let j = i % 64;
            let (w, c, e) = (j / 16, (j % 16) / 4, j % 4);
            const P: [usize; 4] = [0, 2, 1, 3];
            let byte = qs[c * 16 + w * 4 + P[e]];
            let code = if i < 64 { byte & 0xF } else { byte >> 4 };
            out.push(code as f32 * s - sz);
        }
    }
    out
}

/// Quantize an `f16` matrix to Q8_0, for the vocabulary projection.
///
/// An AWQ checkpoint leaves `lm_head` in `f16`: 1.05 GB on an 8B model, a fifth
/// of everything a decode step reads, and the float mat-vec that has to consume
/// it manages 141 GB/s where the integer one reaches 366. Quantizing it at load
/// halves the bytes and moves it onto the fast path, which is worth 6 ms of a
/// 17 ms step — far more than eight bits costs a vocabulary projection, where
/// the only thing that matters is the argmax over 128k logits.
///
/// Q8_0 is 32 weights to one `f16` scale, `d = max|w| / 127`.
pub fn quantize_f16_to_q8_0(src: &[half::f16], k: usize) -> Result<Vec<u8>> {
    anyhow::ensure!(
        k.is_multiple_of(32),
        "row length {k} is not a multiple of the 32-weight block"
    );
    anyhow::ensure!(
        src.len().is_multiple_of(k),
        "{} values do not divide into rows of {k}",
        src.len()
    );
    const BLOCK: usize = 32;
    const BYTES: usize = 34; // f16 scale, then 32 int8
    let blocks = src.len() / BLOCK;
    let mut out = vec![0u8; blocks * BYTES];

    let threads = std::thread::available_parallelism()
        .map_or(1, |n| n.get())
        .min(blocks.max(1));
    let per = blocks.div_ceil(threads);
    std::thread::scope(|scope| {
        for (t, chunk) in out.chunks_mut(per * BYTES).enumerate() {
            let src = &src[t * per * BLOCK..];
            scope.spawn(move || {
                for (b, dst) in chunk.chunks_mut(BYTES).enumerate() {
                    let vals = &src[b * BLOCK..b * BLOCK + BLOCK];
                    let amax = vals.iter().fold(0.0f32, |m, v| m.max(f32::from(*v).abs()));
                    let d = amax / 127.0;
                    dst[..2].copy_from_slice(&half::f16::from_f32(d).to_le_bytes());
                    let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
                    for (i, v) in vals.iter().enumerate() {
                        dst[2 + i] = (f32::from(*v) * inv).round().clamp(-127.0, 127.0) as i8 as u8;
                    }
                }
            });
        }
    });
    Ok(out)
}
