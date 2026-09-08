//! `quantize_f16_to_q4g128`'s own round trip: a from-scratch encode into
//! [`infero_kernels::WeightType::Q4G128`]'s byte layout, not a repack of an
//! already-quantized AWQ tensor (that path is `awq.rs`'s own `AwqTensor::repack`,
//! covered by `tests/q4g128.rs`). Checked two ways: the CPU's own dequantized
//! reading of the bytes against the original F16 source (does the quantization
//! math itself round-trip within a 4-bit step), and the real `mmvq` kernel
//! against a dot product over the *original*, unquantized F16 weights (does the
//! kernel read these bytes the same way the encoder wrote them).

use anyhow::Result;
use infero_cuda::Device;
use infero_kernels::awq::{quantize_f16_to_q4g128, unpack_row};
use infero_kernels::{Kernels, WeightType};

fn synthetic_f16(n: usize, seed: u64) -> Vec<half::f16> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            half::f16::from_f32((((s >> 40) as f32 / (1u64 << 23) as f32) - 1.0) * 3.0)
        })
        .collect()
}

#[test]
fn the_encode_round_trips_within_a_4bit_step() -> Result<()> {
    for (k, n) in [(128usize, 4usize), (256, 8), (5120, 3)] {
        let w = synthetic_f16(k * n, 0xF00D + k as u64);
        let packed = quantize_f16_to_q4g128(&w, k)?;
        assert_eq!(packed.len(), n * (k / 128) * infero_kernels::awq::BLOCK_BYTES);

        for row in 0..n {
            let want: Vec<f32> = w[row * k..(row + 1) * k].iter().map(|v| f32::from(*v)).collect();
            let got = unpack_row(&packed, k, row);
            assert_eq!(got.len(), k);
            let range = want.iter().fold(1e-6f32, |m, v| m.max(v.abs()));
            for i in 0..k {
                // Per-128-block affine quantization at 4 bits: the worst a
                // value can be off is half a quantization step, and the step
                // itself is the block's own (max-min)/15 -- bounded here by
                // the row's own range as a simple, real, not-tuned-to-pass
                // tolerance rather than recomputing each block's own step.
                assert!(
                    (got[i] - want[i]).abs() <= (range / 15.0) * 1.5,
                    "k={k} n={n} row {row} elem {i}: {} vs {} (range {range})",
                    got[i],
                    want[i]
                );
            }
        }
    }
    Ok(())
}

#[test]
fn the_real_mmvq_kernel_matches_the_original_f16_weights() -> Result<()> {
    let dev = match Device::new(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skipping: no cuda device ({e})");
            return Ok(());
        }
    };
    let kern = Kernels::new(dev.clone());
    let stream = dev.stream();

    for (k, n) in [(128usize, 64usize), (5120, 96), (5120, 256)] {
        let w = synthetic_f16(k * n, 0xACE0 + k as u64 + n as u64);
        let packed = quantize_f16_to_q4g128(&w, k)?;
        let dw = stream.clone_htod(&packed)?;

        let x: Vec<f32> = (0..k)
            .map(|i| ((i * 2654435761usize) % 401) as f32 / 200.0 - 1.0)
            .collect();
        let want: Vec<f32> = (0..n)
            .map(|r| {
                (0..k)
                    .map(|i| f32::from(w[r * k + i]) as f64 * x[i] as f64)
                    .sum::<f64>() as f32
            })
            .collect();

        let dx = stream.clone_htod(&x)?;
        let mut q8 = stream.alloc_zeros::<u8>(Kernels::q8_1_bytes(k))?;
        kern.quantize_q8_1(&mut q8.as_view_mut(), &dx.as_view(), k)?;
        let mut out = stream.alloc_zeros::<f32>(n)?;
        kern.mmvq(
            &mut out.as_view_mut(),
            &dw.as_view(),
            WeightType::Q4G128,
            &q8.as_view(),
            k,
            n,
        )?;
        let got = stream.clone_dtoh(&out)?;
        dev.synchronize()?;

        let scale = want.iter().fold(1e-6f32, |m, v| m.max(v.abs()));
        for r in 0..n {
            // 4-bit weight quantization plus 8-bit activation quantization,
            // against the *original*, unquantized F16 weights -- real slack
            // needed here is the weight encode's own step (up to ~1/15 of a
            // block's own range) compounded with the activation's, not the
            // 2% `q4g128.rs`'s own tests use against an AWQ-repack reference
            // (which is unquantized-weight-vs-quantized-weight already
            // folded into `unpack_row`'s own answer, a smaller gap to check).
            assert!(
                (got[r] - want[r]).abs() <= 0.10 * scale,
                "k={k} n={n} row {r}: mmvq {} vs {} (want-scale {scale})",
                got[r],
                want[r]
            );
        }
    }
    Ok(())
}
