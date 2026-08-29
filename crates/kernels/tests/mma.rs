//! Does `mma.m16n8k32.s8` put the numbers where we think it does?
//!
//! The MMQ kernel builds its fragments by hand out of shared memory, so it is
//! entirely dependent on the register-to-(row, k) mapping in `mma.cuh` being
//! right. A wrong index there does not fail loudly — it produces a matrix
//! product that is close enough to look plausible in a cosine test and wrong
//! enough to ruin generation. So: pin it against an integer reference, where
//! "close" is not a thing that exists.

use anyhow::Result;
use infero_cuda::Device;
use infero_kernels::Kernels;

/// Row-major `a` (16x32) times the transpose of row-major `b` (8x32).
fn reference(a: &[i8], b: &[i8]) -> Vec<i32> {
    let mut d = vec![0i32; 16 * 8];
    for m in 0..16 {
        for n in 0..8 {
            let mut acc = 0i32;
            for k in 0..32 {
                acc += a[m * 32 + k] as i32 * b[n * 32 + k] as i32;
            }
            d[m * 8 + n] = acc;
        }
    }
    d
}

fn run(a: &[i8], b: &[i8]) -> Result<Option<Vec<i32>>> {
    let dev = match Device::new(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skipping: no cuda device ({e})");
            return Ok(None);
        }
    };
    if dev.arch() < 80 {
        eprintln!("skipping: sm_{} predates the int8 mma", dev.arch());
        return Ok(None);
    }
    let kern = Kernels::new(dev.clone());
    let stream = dev.stream();
    let da = stream.clone_htod(a)?;
    let db = stream.clone_htod(b)?;
    let mut dd = stream.alloc_zeros::<i32>(16 * 8)?;
    kern.mma_s8_probe(&mut dd.slice_mut(..), &da.slice(..), &db.slice(..))?;
    let got = stream.clone_dtoh(&dd)?;
    Ok(Some(got))
}

/// Distinct values in every cell, so any transposition or swapped index shows
/// up rather than cancelling out.
#[test]
fn the_int8_mma_fragment_layout_is_what_mma_cuh_claims() -> Result<()> {
    let a: Vec<i8> = (0..16 * 32)
        .map(|i| ((i * 7 + 3) % 127 - 63) as i8)
        .collect();
    let b: Vec<i8> = (0..8 * 32)
        .map(|i| ((i * 13 + 5) % 127 - 63) as i8)
        .collect();

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
    for &(row, col, kk) in &[
        (0usize, 0usize, 0usize),
        (3, 5, 17),
        (9, 7, 31),
        (15, 1, 16),
    ] {
        let mut a = vec![0i8; 16 * 32];
        let mut b = vec![0i8; 8 * 32];
        a[row * 32 + kk] = 3;
        b[col * 32 + kk] = 5;
        let Some(got) = run(&a, &b)? else {
            return Ok(());
        };
        for (i, &v) in got.iter().enumerate() {
            let want = if i == row * 8 + col { 15 } else { 0 };
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

// ---- the f16 tensor cores ------------------------------------------------
//
// Same question as above, for `mma.m16n8k16.f16`. The integer probe could
// assert exact equality; this one cannot, because the accumulator sums 16 f16
// products in f32 in an order the instruction chooses. So the values are picked
// small and exactly representable — halves and quarters — which makes every
// partial sum exact and puts equality back on the table.

/// Row-major `a` (16x16) times the transpose of row-major `b` (8x16).
fn reference_f16(a: &[half::f16], b: &[half::f16]) -> Vec<f32> {
    let mut d = vec![0.0f32; 16 * 8];
    for m in 0..16 {
        for n in 0..8 {
            let mut acc = 0.0f32;
            for k in 0..16 {
                acc += a[m * 16 + k].to_f32() * b[n * 16 + k].to_f32();
            }
            d[m * 8 + n] = acc;
        }
    }
    d
}

#[test]
fn the_f16_mma_fragment_layout_is_what_mma_cuh_claims() -> Result<()> {
    let dev = match Device::new(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skipping: no cuda device ({e})");
            return Ok(());
        }
    };
    if dev.arch() < 80 {
        eprintln!("skipping: sm_{} predates this mma", dev.arch());
        return Ok(());
    }
    let kern = Kernels::new(dev.clone());
    let stream = dev.stream();

    // Multiples of 1/4 in [-8, 8): exact in f16, and their products and sums
    // are exact in f32, so the comparison below is equality and not a
    // tolerance that could hide a swapped index.
    let a: Vec<half::f16> = (0..16 * 16)
        .map(|i| half::f16::from_f32(((i * 7 + 3) % 64) as f32 / 4.0 - 8.0))
        .collect();
    let b: Vec<half::f16> = (0..8 * 16)
        .map(|i| half::f16::from_f32(((i * 13 + 5) % 64) as f32 / 4.0 - 8.0))
        .collect();

    let da = stream.clone_htod(&a)?;
    let db = stream.clone_htod(&b)?;
    let mut dd = stream.alloc_zeros::<f32>(16 * 8)?;
    kern.mma_f16_probe(&mut dd.slice_mut(..), &da.slice(..), &db.slice(..))?;
    let got = stream.clone_dtoh(&dd)?;
    dev.synchronize()?;

    let want = reference_f16(&a, &b);
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

/// The claim that makes the f16 path cheap: an f16 A fragment occupies the same
/// 32 bytes per row as an s8 one, so `ldmatrix_a_s8` loads it unchanged. If
/// that is wrong, the kernel needs its own gather and the tile has to be
/// re-thought — so assert it rather than assume it.
#[test]
fn ldmatrix_loads_an_f16_a_fragment_too() -> Result<()> {
    let dev = match Device::new(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skipping: no cuda device ({e})");
            return Ok(());
        }
    };
    if dev.arch() < 80 {
        return Ok(());
    }
    let kern = Kernels::new(dev.clone());
    let stream = dev.stream();

    // The same bytes read both ways: `ldmatrix_a_probe` takes int8 and reports
    // what `ldmatrix` produced against what the scalar gather produced. An f16
    // tile is those bytes with a different interpretation, and neither path
    // interprets them — so agreement here is agreement for f16.
    let bytes: Vec<i8> = (0..16 * 32).map(|i| ((i * 31 + 7) % 251 - 125) as i8).collect();
    let dsrc = stream.clone_htod(&bytes)?;
    let mut out = stream.alloc_zeros::<i32>(32 * 8)?;
    kern.ldmatrix_probe(&mut out.slice_mut(..), &dsrc.slice(..))?;
    let got = stream.clone_dtoh(&out)?;
    dev.synchronize()?;
    for lane in 0..32 {
        for i in 0..4 {
            assert_eq!(
                got[lane * 8 + i],
                got[lane * 8 + 4 + i],
                "lane {lane} register {i}: ldmatrix and the scalar gather differ"
            );
        }
    }
    Ok(())
}
