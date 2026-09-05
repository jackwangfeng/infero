//! `Tensor::to_f16`'s BF16 narrowing path, parallelized across threads for a
//! vocab-sized matrix (embeddings/lm_head): this pins that the threaded
//! version produces byte-identical output to a plain serial reference, at a
//! size large enough to span many worker chunks regardless of how many cores
//! this machine actually has, plus the boundary/edge cases a chunking bug
//! would most likely get wrong.

use half::f16;
use infero_safetensors::{Dtype, Tensor};

fn tensor<'a>(name: &'a str, dtype: Dtype, shape: Vec<usize>, data: &'a [u8]) -> Tensor<'a> {
    Tensor { name, dtype, shape, data }
}

fn bf16_bytes(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 2);
    for v in vals {
        // BF16 is f32's top 16 bits: truncate, not round, matching how a real
        // BF16 checkpoint's bytes were produced upstream (this test is about
        // `to_f16`'s narrowing, not BF16 encoding, so any valid encoding of
        // each value will do).
        let bits = (v.to_bits() >> 16) as u16;
        out.extend_from_slice(&bits.to_le_bytes());
    }
    out
}

/// A plain, unthreaded reference for the same narrowing this crate's
/// production path now does across threads — the two must agree exactly, not
/// approximately, since it is the same bit-for-bit conversion either way.
fn reference_to_f16(vals: &[f32]) -> Vec<f16> {
    vals.iter().map(|v| f16::from_f32(*v)).collect()
}

#[test]
fn threaded_bf16_narrowing_matches_a_serial_reference_at_vocab_scale() {
    // Large enough to span many chunks on any real core count (16 threads
    // max per the implementation), and not a round multiple of any small
    // chunk size, so a boundary bug is likely to show up somewhere.
    const N: usize = 200_003;
    let vals: Vec<f32> = (0..N)
        .map(|i| {
            // A spread of magnitudes and signs, including values that are
            // exactly representable in bf16 (so the truncation round-trips
            // cleanly) and values near zero/one.
            let x = (i as f32 - N as f32 / 2.0) * 0.0078125; // multiple of 2^-7
            if i % 7 == 0 { -x } else { x }
        })
        .collect();
    let bytes = bf16_bytes(&vals);
    let t = tensor("embed_tokens.weight", Dtype::BF16, vec![1, N], &bytes);

    let got = t.to_f16().expect("narrowing in-range values must succeed");
    // What actually got encoded into bf16 (truncated, not `vals` itself) is
    // the real input to the narrowing step.
    let bf16_vals: Vec<f32> = bytes
        .chunks_exact(2)
        .map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16))
        .collect();
    let want = reference_to_f16(&bf16_vals);

    assert_eq!(got.len(), N);
    assert_eq!(&got[..], &want[..], "threaded narrowing diverged from the serial reference");
}

#[test]
fn threaded_bf16_narrowing_handles_small_and_singleton_inputs() {
    for n in [1usize, 2, 3, 15, 16, 17] {
        let vals: Vec<f32> = (0..n).map(|i| i as f32 * 0.5 - 1.0).collect();
        let bytes = bf16_bytes(&vals);
        let t = tensor("t", Dtype::BF16, vec![1, n], &bytes);
        let got = t.to_f16().unwrap_or_else(|e| panic!("n={n}: {e}"));
        let bf16_vals: Vec<f32> = bytes
            .chunks_exact(2)
            .map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16))
            .collect();
        let want = reference_to_f16(&bf16_vals);
        assert_eq!(&got[..], &want[..], "n={n}");
    }
}

#[test]
fn threaded_bf16_narrowing_still_rejects_a_value_outside_f16_range() {
    // f16's max finite magnitude is 65504; bf16 carries f32's exponent range,
    // so a bf16 value like 1e30 narrows to +inf silently unless this is
    // checked -- exactly the corruption this function's own doc comment
    // warns about, and the threaded rewrite must still catch it wherever in
    // the chunked sweep it happens to land.
    const N: usize = 5000;
    let mut vals: Vec<f32> = (0..N).map(|i| i as f32 * 0.01).collect();
    let bad_index = N - 7; // deliberately not at a chunk boundary
    vals[bad_index] = 1.0e30;
    let bytes = bf16_bytes(&vals);
    let t = tensor("t", Dtype::BF16, vec![1, N], &bytes);
    let err = t.to_f16().unwrap_err().to_string();
    assert!(err.contains("outside f16 range"), "got: {err}");
}
