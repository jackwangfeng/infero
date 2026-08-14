//! Stacking two transposed AWQ tensors along `n`.
//!
//! Fusing `gate`/`up` and `q`/`k`/`v` into one matmul is worth doing because a
//! narrow matmul cannot fill the device — but only if row `n_a + r` of the
//! result is exactly row `r` of the second tensor. The layout has two regions
//! rather than one, so a concatenation that treated the buffers as opaque and
//! appended them would put the first tensor's scales in the middle of the
//! second tensor's quants and still produce plausible-looking numbers.

use anyhow::Result;
use tuili_kernels::awq::{AwqTensor, concat_t, transpose_words, transposable, unpack_row_t};

/// A deterministic AWQ tensor of the given shape.
fn awq(k: usize, n: usize, seed: usize) -> Vec<u8> {
    let groups = k / 128;
    let qweight: Vec<i32> = (0..k * n / 8)
        .map(|i| ((i.wrapping_mul(2_654_435_761).wrapping_add(seed)) % 0xFFFF_FFFF) as i32)
        .collect();
    let qzeros: Vec<i32> = (0..groups * n / 8)
        .map(|i| ((i * 7919 + seed) % 0x7FFF_FFFF) as i32)
        .collect();
    let scales: Vec<half::f16> = (0..groups * n)
        .map(|i| half::f16::from_f32(((i * 37 + seed) % 97) as f32 / 400.0 + 0.01))
        .collect();
    let packed = AwqTensor {
        qweight: &qweight,
        qzeros: &qzeros,
        scales: &scales,
        in_features: k,
        out_features: n,
    }
    .repack()
    .expect("repack");
    transpose_words(&packed, k, n)
}

#[test]
fn a_stacked_tensor_holds_both_originals_row_for_row() -> Result<()> {
    let k = 1024usize;
    assert!(transposable(k));

    // Equal widths, as `gate`/`up` have, and unequal ones, as `q`/`k`/`v` do.
    for (n_a, n_b) in [(512usize, 512usize), (512, 128), (128, 512)] {
        let a = awq(k, n_a, 1);
        let b = awq(k, n_b, 2);
        let c = concat_t(&a, n_a, &b, n_b, k);
        assert_eq!(c.len(), a.len() + b.len(), "{n_a}+{n_b}: byte count");

        let n = n_a + n_b;
        for r in [0usize, 1, n_a / 2, n_a - 1] {
            let want = unpack_row_t(&a, k, n_a, r);
            let got = unpack_row_t(&c, k, n, r);
            assert_eq!(got, want, "{n_a}+{n_b}: first tensor row {r}");
        }
        for r in [0usize, 1, n_b / 2, n_b - 1] {
            let want = unpack_row_t(&b, k, n_b, r);
            let got = unpack_row_t(&c, k, n, n_a + r);
            assert_eq!(got, want, "{n_a}+{n_b}: second tensor row {r}");
        }
    }
    Ok(())
}
