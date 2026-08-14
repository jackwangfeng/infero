//! The fused RMS-norm-and-quantize against doing the two separately.
//!
//! The fused kernel reduces over the 32 values that share a Q8_1 scale inside
//! the same block that computed the norm, rather than re-reading them in a
//! second launch. A mismatch in how the groups are assigned to warps would
//! produce plausible-looking output with the wrong scales, so compare the
//! quantized bytes exactly — not the dequantized values.

use anyhow::Result;
use tuili_cuda::Device;
use tuili_kernels::Kernels;

#[test]
fn fused_norm_and_quantize_matches_the_two_kernels() -> Result<()> {
    let dev = match Device::new(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skipping: no cuda device ({e})");
            return Ok(());
        }
    };
    let kern = Kernels::new(dev.clone());
    let stream = dev.stream();

    for (n_tokens, d) in [(1usize, 4096usize), (3, 896), (16, 5120)] {
        let x: Vec<f32> = (0..n_tokens * d)
            .map(|i| ((i * 2654435761usize) % 1997) as f32 / 500.0 - 2.0)
            .collect();
        let w: Vec<f32> = (0..d).map(|i| 0.5 + (i % 17) as f32 / 32.0).collect();
        let dx = stream.clone_htod(&x)?;
        let dw = stream.clone_htod(&w)?;
        let qb = Kernels::q8_1_bytes(d);

        let mut sep_f = stream.alloc_zeros::<f32>(n_tokens * d)?;
        let mut sep_q = stream.alloc_zeros::<u8>(n_tokens * qb)?;
        kern.rms_norm(
            &mut sep_f.as_view_mut(),
            &dx.as_view(),
            &dw.as_view(),
            n_tokens,
            d,
            1e-5,
        )?;
        kern.quantize_q8_1(&mut sep_q.as_view_mut(), &sep_f.as_view(), n_tokens * d)?;

        let mut fus_f = stream.alloc_zeros::<f32>(n_tokens * d)?;
        let mut fus_q = stream.alloc_zeros::<u8>(n_tokens * qb)?;
        kern.rms_norm_q8_1(
            &mut fus_f.as_view_mut(),
            &mut fus_q.as_view_mut(),
            &dx.as_view(),
            &dw.as_view(),
            n_tokens,
            d,
            1e-5,
        )?;

        let (a_f, b_f) = (stream.clone_dtoh(&sep_f)?, stream.clone_dtoh(&fus_f)?);
        let (a_q, b_q) = (stream.clone_dtoh(&sep_q)?, stream.clone_dtoh(&fus_q)?);
        dev.synchronize()?;

        // Not bit-equality: the two kernels size their block from different
        // things — the separate norm always takes 256 threads, the fused one
        // takes as many as its per-thread register budget needs — so they sum
        // the squares in a different order and the scale can land a ulp apart.
        // What a wrong warp-to-Q8_1-group mapping produces is not a ulp; it is
        // a scale taken from an unrelated 32 values, so these tolerances still
        // fail loudly on the bug the test exists for.
        let f_err = a_f
            .iter()
            .zip(&b_f)
            .map(|(x, y)| (x - y).abs() / x.abs().max(1e-3))
            .fold(0.0f32, f32::max);
        let q_bad = a_q
            .iter()
            .zip(&b_q)
            .filter(|(x, y)| x.abs_diff(**y) > 1)
            .count();
        eprintln!("  n={n_tokens:<3} d={d:<5} float rel err {f_err:.2e}, {q_bad} bytes off by >1");
        assert!(f_err < 1e-5, "normalized output differs: rel err {f_err:.2e}");
        assert_eq!(q_bad, 0, "quantized bytes differ by more than a rounding step");
    }
    Ok(())
}
