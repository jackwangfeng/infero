//! Does `mma.m16n8k32.e4m3` put the numbers where `mma_e4m3` in `mma.cuh`
//! claims, the same question `mma.rs` asks of the `s8` and `f16` MMAs.
//!
//! e4m3 reuses the `s8` fragment layout byte for byte (see the comment on
//! `mma_e4m3`), so this is `mma.rs`'s int8 test with a different decode on
//! the reference side rather than a new layout to derive.

use anyhow::Result;
use infero_cuda::Device;
use infero_kernels::Kernels;

/// Bit for bit what `e4m3_to_f32` in `fp8.cu` does — the one definition lives
/// there, this is a from-scratch second copy so the test does not validate a
/// decoder against itself.
fn e4m3_to_f32(b: u8) -> f32 {
    let sign = (b & 0x80) != 0;
    let exp = (b >> 3) & 0x0F;
    let man = (b & 0x07) as f32;
    let mag = if exp == 0 {
        man / 512.0
    } else {
        // Rebias 7 to 127 by scaling: (1 + man/8) * 2^(exp - 7).
        (1.0 + man / 8.0) * 2f32.powi(exp as i32 - 7)
    };
    if sign { -mag } else { mag }
}

/// Row-major `a` (16x32) times the transpose of row-major `b` (8x32), decoded
/// from e4m3 bytes first.
fn reference(a: &[u8], b: &[u8]) -> Vec<f32> {
    let mut d = vec![0.0f32; 16 * 8];
    for m in 0..16 {
        for n in 0..8 {
            let mut acc = 0.0f32;
            for k in 0..32 {
                acc += e4m3_to_f32(a[m * 32 + k]) * e4m3_to_f32(b[n * 32 + k]);
            }
            d[m * 8 + n] = acc;
        }
    }
    d
}

fn run(a: &[u8], b: &[u8]) -> Result<Option<Vec<f32>>> {
    let dev = match Device::new(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skipping: no cuda device ({e})");
            return Ok(None);
        }
    };
    if !dev.caps().fp8 {
        eprintln!("skipping: sm_{} has no native e4m3 mma", dev.arch());
        return Ok(None);
    }
    let kern = Kernels::new(dev.clone());
    let stream = dev.stream();
    let da = stream.clone_htod(a)?;
    let db = stream.clone_htod(b)?;
    let mut dd = stream.alloc_zeros::<f32>(16 * 8)?;
    kern.mma_e4m3_probe(&mut dd.slice_mut(..), &da.slice(..), &db.slice(..))?;
    let got = stream.clone_dtoh(&dd)?;
    dev.synchronize()?;
    Ok(Some(got))
}

/// Bytes chosen to stay small — modulus `0x30` caps the exponent nibble at 6,
/// so the largest decoded magnitude is under 1 — and every code in range is a
/// dyadic rational with a small denominator. A first attempt at `0x4F` let the
/// exponent nibble reach 9, and the instruction's internal reduction order
/// (unlike this file's left-to-right one) disagreed with the reference in the
/// last bit of a handful of cells: still a correct sum, just not the same
/// rounding, and not a fragment-layout question at all — `mma.rs`'s equality
/// comparison works for `s8` and `f16` because their chosen values keep every
/// partial sum exactly representable, which is the property this modulus is
/// for. `a_single_nonzero_lands_in_a_single_cell` below is the one that
/// actually pins the layout down; this one is a broader sanity check on top.
#[test]
fn the_e4m3_mma_fragment_layout_is_what_mma_cuh_claims() -> Result<()> {
    let a: Vec<u8> = (0..16 * 32).map(|i| ((i * 7 + 3) % 0x30u32) as u8).collect();
    let b: Vec<u8> = (0..8 * 32).map(|i| ((i * 13 + 5) % 0x30u32) as u8).collect();

    let Some(got) = run(&a, &b)? else {
        return Ok(());
    };
    let want = reference(&a, &b);
    for m in 0..16 {
        for n in 0..8 {
            assert_eq!(
                got[m * 8 + n],
                want[m * 8 + n],
                "cell ({m}, {n}): fragment layout is wrong"
            );
        }
    }
    Ok(())
}

/// A one-hot pair localizes a mis-mapped index to a single cell, which is what
/// you want to read when the test above fails.
#[test]
fn a_single_nonzero_lands_in_a_single_cell() -> Result<()> {
    for &(row, col, kk) in &[(0usize, 0usize, 0usize), (3, 5, 17), (9, 7, 31), (15, 1, 16)] {
        let mut a = vec![0u8; 16 * 32];
        let mut b = vec![0u8; 8 * 32];
        a[row * 32 + kk] = 0x38; // 1.0
        b[col * 32 + kk] = 0x30; // 0.5
        let Some(got) = run(&a, &b)? else {
            return Ok(());
        };
        for (i, &v) in got.iter().enumerate() {
            let want = if i == row * 8 + col { 0.5 } else { 0.0 };
            assert_eq!(
                v,
                want,
                "a[{row}][{kk}]*b[{col}][{kk}] leaked into cell ({}, {})",
                i / 8,
                i % 8
            );
        }
    }
    Ok(())
}
