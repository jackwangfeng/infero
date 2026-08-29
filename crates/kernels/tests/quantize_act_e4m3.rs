//! Does `quantize_act_e4m3_f32` pick the scale `mma_e4m3`'s epilogue expects
//! and encode against it correctly?
//!
//! Forcing the group's scale to exactly 1.0 (by putting 448.0, e4m3's largest
//! finite magnitude, in the group) turns "does quantization round correctly"
//! into "does `f32_to_e4m3(v)` reproduce a known byte for a `v` this test
//! picked to already be e4m3-exact" — equality, not a tolerance that could
//! hide a scale computed one group over or an off-by-one in the reduction.

use anyhow::Result;
use infero_cuda::Device;
use infero_kernels::Kernels;

const GROUP: usize = 128;

/// Bit for bit `e4m3_to_f32` in `fp8.cu`, same second-copy reasoning as
/// `mma_e4m3.rs`.
fn e4m3_to_f32(b: u8) -> f32 {
    let sign = (b & 0x80) != 0;
    let exp = (b >> 3) & 0x0F;
    let man = (b & 0x07) as f32;
    let mag = if exp == 0 {
        man / 512.0
    } else {
        (1.0 + man / 8.0) * 2f32.powi(exp as i32 - 7)
    };
    if sign { -mag } else { mag }
}

fn run(x: &[f32]) -> Result<Option<(Vec<u8>, Vec<f32>)>> {
    let dev = match Device::new(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skipping: no cuda device ({e})");
            return Ok(None);
        }
    };
    if !dev.caps().fp8 {
        eprintln!("skipping: sm_{} has no native e4m3 conversion", dev.arch());
        return Ok(None);
    }
    let kern = Kernels::new(dev.clone());
    let stream = dev.stream();
    let n_tokens = x.len() / GROUP;
    let dx = stream.clone_htod(x)?;
    let mut d_xq = stream.alloc_zeros::<u8>(x.len())?;
    let mut d_xs = stream.alloc_zeros::<f32>(n_tokens)?;
    kern.quantize_act_e4m3(
        &mut d_xq.slice_mut(..),
        &mut d_xs.slice_mut(..),
        &dx.slice(..),
        GROUP,
        n_tokens,
    )?;
    let xq = stream.clone_dtoh(&d_xq)?;
    let xs = stream.clone_dtoh(&d_xs)?;
    dev.synchronize()?;
    Ok(Some((xq, xs)))
}

/// One group: element 0 is 448.0 (forces `scale == 1.0`), the rest are
/// e4m3-exact values covering a normal, a subnormal, zero, and a negative —
/// each with a hand-computed expected byte, independent of `e4m3_to_f32`.
#[test]
fn a_full_scale_group_quantizes_exactly() -> Result<()> {
    // (value, expected byte), value chosen e4m3-exact so the round-trip
    // through `f32_to_e4m3` at `scale = 1.0` is lossless.
    let known: &[(f32, u8)] = &[
        (448.0, 0x7E),  // the value that sets the scale
        (1.0, 0x38),    // normal: sign 0, exp 7, man 0
        (-1.0, 0xB8),   // same, negated
        (1.75, 0x3E),   // normal: exp 7, man 6
        (0.5, 0x30),    // normal: exp 6, man 0
        (0.0, 0x00),    // zero
        (7.0 / 512.0, 0x07), // subnormal: man 7
        (2.0, 0x40),    // normal: exp 8, man 0
    ];
    let mut x = vec![0.0f32; GROUP];
    for (i, &(v, _)) in known.iter().enumerate() {
        x[i] = v;
    }

    let Some((xq, xs)) = run(&x)? else {
        return Ok(());
    };
    assert_eq!(xs[0], 1.0, "448.0 in the group should force scale == 1.0");
    for (i, &(v, want)) in known.iter().enumerate() {
        assert_eq!(
            xq[i], want,
            "x[{i}] = {v}: got byte {:#04x}, want {want:#04x} (decodes to {})",
            xq[i],
            e4m3_to_f32(want)
        );
    }
    Ok(())
}

/// A group whose largest magnitude is small (well under e4m3's own dynamic
/// range) still has to scale so that magnitude reaches close to 448, not clip
/// to zero — the failure mode a scale floor set too high would produce.
#[test]
fn a_small_magnitude_group_still_uses_its_own_dynamic_range() -> Result<()> {
    let mut x = vec![0.0f32; GROUP];
    x[0] = 0.01;
    x[1] = -0.005;

    let Some((xq, xs)) = run(&x)? else {
        return Ok(());
    };
    let scale = xs[0];
    assert!(
        scale > 0.0 && scale < 1.0,
        "a 0.01-magnitude group scaled to {scale}, expected well under 1.0"
    );
    let got0 = e4m3_to_f32(xq[0]) * scale;
    let got1 = e4m3_to_f32(xq[1]) * scale;
    assert!(
        (got0 - 0.01).abs() < 0.01 * 0.15,
        "0.01 round-tripped to {got0}"
    );
    assert!(
        (got1 - (-0.005)).abs() < 0.005 * 0.15,
        "-0.005 round-tripped to {got1}"
    );
    Ok(())
}

/// An all-zero group must not produce a `NaN` scale.
#[test]
fn an_all_zero_group_has_a_finite_scale_and_encodes_zero() -> Result<()> {
    let x = vec![0.0f32; GROUP];
    let Some((xq, xs)) = run(&x)? else {
        return Ok(());
    };
    assert!(xs[0].is_finite(), "all-zero group produced scale {}", xs[0]);
    assert!(xq.iter().all(|&b| b == 0), "all-zero group did not encode to all-zero bytes");
    Ok(())
}
