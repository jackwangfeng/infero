//! The GEMM against the mat-vec, on the same weights.
//!
//! `gemm_f16` exists so that prefill reads the weights once instead of once per
//! four tokens. The question it has to answer is not "is a matmul correct" but
//! "does it agree with the kernel it replaces" -- and the mat-vec is already
//! checked against a host dequantisation in `infero-metal`'s `quant` tests, so
//! agreement here chains onto that.
//!
//! It also pins the two things a GEMM wiring gets wrong: which operand is
//! transposed, and whether a view's byte offset reaches the driver. A transpose
//! flag the wrong way round still produces plausible numbers of the right
//! magnitude, and an ignored offset reads a neighbouring row.

use anyhow::Result;
use half::f16;
use infero_gpu::Device;
use infero_kernels::{Kernels, WeightType};

fn kernels() -> Result<Kernels> {
    Ok(Kernels::new(Device::new(0)?))
}

/// Largest absolute difference, and where.
fn max_abs_diff(a: &[f32], b: &[f32]) -> (f32, usize) {
    assert_eq!(a.len(), b.len());
    let mut worst = (0.0f32, 0usize);
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        let d = (x - y).abs();
        if d > worst.0 {
            worst = (d, i);
        }
    }
    worst
}

/// Values with the spread of a real activation, deterministic.
fn noise(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 11) as f64 / (1u64 << 53) as f64) as f32 * 2.0 - 1.0
        })
        .collect()
}

/// `c[t][row] = sum_k a[t][k] * b[row][k]`, in f32 on the host.
fn reference(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for t in 0..m {
        for row in 0..n {
            let mut acc = 0.0f32;
            for i in 0..k {
                acc += a[t * k + i] * b[row * k + i];
            }
            out[t * n + row] = acc;
        }
    }
    out
}

#[test]
fn the_gemm_agrees_with_the_host() -> Result<()> {
    let kern = kernels()?;
    let s = kern.device().stream();

    // `k` is deliberately not tiny. MPS accumulates in f16 -- it requires all
    // three matrices to share a data type -- so the tolerance has to cover a
    // 2048-term sum of f16 products, and asserting a tight bound at k = 64
    // would pass while saying nothing about the shape prefill actually runs.
    for (m, k, n) in [(1usize, 512usize, 128usize), (8, 512, 128), (37, 2048, 320)] {
        let a = noise(m * k, 11);
        let b = noise(n * k, 29);
        let a16: Vec<f16> = a.iter().map(|&x| f16::from_f32(x)).collect();
        let b16: Vec<f16> = b.iter().map(|&x| f16::from_f32(x)).collect();

        let da = s.clone_htod(&a16)?;
        let db = s.clone_htod(&b16)?;
        let mut dc = s.alloc_zeros::<f32>(m * n)?;
        kern.gemm_f16(&mut dc.as_view_mut(), &da.as_view(), &db.as_view(), m, k, n)?;
        kern.device().synchronize()?;
        let got = s.clone_dtoh(&dc.as_view())?;

        let want = reference(&a, &b, m, k, n);
        let (worst, at) = max_abs_diff(&got, &want);
        // Scaled by the sum's own magnitude rather than absolute: a k-term sum
        // of unit-ish products has magnitude ~sqrt(k).
        let tol = 0.02 * (k as f32).sqrt();
        eprintln!("  {m}x{k}x{n}: worst {worst:.4} at {at}, tol {tol:.4}");
        assert!(worst <= tol, "{m}x{k}x{n}: worst {worst} at {at} exceeds {tol}");
    }
    Ok(())
}

#[test]
fn the_gemm_agrees_with_the_matvec_it_replaces() -> Result<()> {
    let kern = kernels()?;
    let s = kern.device().stream();

    // f16 weights, so no dequantisation stands between the two paths and a
    // disagreement is the GEMM's.
    let (m, k, n) = (16usize, 1024usize, 256usize);
    let a = noise(m * k, 5);
    let b = noise(n * k, 7);
    let a16: Vec<f16> = a.iter().map(|&x| f16::from_f32(x)).collect();
    let b16: Vec<f16> = b.iter().map(|&x| f16::from_f32(x)).collect();

    let da32 = s.clone_htod(&a)?;
    let da16 = s.clone_htod(&a16)?;
    let db = s.clone_htod(&b16)?;

    // The mat-vec reads the weights as bytes and is told their encoding.
    let db_bytes = unsafe { db.as_view().transmute::<u8>(n * k * 2) }
        .expect("an f16 buffer is byte-aligned");
    let mut via_gemv = s.alloc_zeros::<f32>(m * n)?;
    kern.gemv(
        &mut via_gemv.as_view_mut(),
        &db_bytes,
        WeightType::F16,
        &da32.as_view(),
        k,
        n,
        m,
    )?;

    let mut via_gemm = s.alloc_zeros::<f32>(m * n)?;
    kern.gemm_f16(
        &mut via_gemm.as_view_mut(),
        &da16.as_view(),
        &db.as_view(),
        m,
        k,
        n,
    )?;
    kern.device().synchronize()?;

    let (g, v) = (
        s.clone_dtoh(&via_gemm.as_view())?,
        s.clone_dtoh(&via_gemv.as_view())?,
    );
    let (worst, at) = max_abs_diff(&g, &v);
    // The mat-vec accumulates in f32 and the GEMM in f16, so they are allowed
    // to differ by the accumulation -- but not by a transpose, which would put
    // the error at the magnitude of the values themselves.
    let tol = 0.02 * (k as f32).sqrt();
    eprintln!("  gemm vs gemv {m}x{k}x{n}: worst {worst:.4} at {at} (row {}, tok {})", at % n, at / n);
    assert!(worst <= tol, "gemm and gemv disagree by {worst} at {at}");
    Ok(())
}

/// A view that does not start at element zero, which is the normal case: the
/// engine hands in windows of one scratch buffer.
#[test]
fn the_gemm_honours_a_view_offset() -> Result<()> {
    let kern = kernels()?;
    let s = kern.device().stream();
    let (m, k, n) = (4usize, 512usize, 64usize);

    let a = noise(m * k, 3);
    let b = noise(n * k, 13);
    let a16: Vec<f16> = a.iter().map(|&x| f16::from_f32(x)).collect();
    let b16: Vec<f16> = b.iter().map(|&x| f16::from_f32(x)).collect();

    // Both operands live past a pad, and the result lands past one too.
    let pad = 1024usize;
    let mut ha = vec![f16::ZERO; pad];
    ha.extend_from_slice(&a16);
    let mut hb = vec![f16::ZERO; pad];
    hb.extend_from_slice(&b16);
    let da = s.clone_htod(&ha)?;
    let db = s.clone_htod(&hb)?;
    let mut dc = s.alloc_zeros::<f32>(pad + m * n)?;

    kern.gemm_f16(
        &mut dc.slice_mut(pad..pad + m * n),
        &da.slice(pad..pad + m * k),
        &db.slice(pad..pad + n * k),
        m,
        k,
        n,
    )?;
    kern.device().synchronize()?;
    let got = s.clone_dtoh(&dc.as_view())?;

    // The pad must be untouched -- a widening pass that ignored the offset
    // would have written over it.
    assert!(
        got[..pad].iter().all(|&x| x == 0.0),
        "the widening pass wrote before the output window"
    );
    let want = reference(&a, &b, m, k, n);
    let (worst, at) = max_abs_diff(&got[pad..], &want);
    let tol = 0.02 * (k as f32).sqrt();
    eprintln!("  offset {m}x{k}x{n}: worst {worst:.4} at {at}");
    assert!(worst <= tol, "offset gemm: worst {worst} at {at}");
    Ok(())
}
