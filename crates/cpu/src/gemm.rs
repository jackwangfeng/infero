//! `c = a * b^T` with an f32 result -- the free function `Kernels::gemm_f16`
//! calls directly (not through the name dispatcher; see `infero_metal::gemm`
//! for the precedent this mirrors and `launch.rs`'s own doc comment for why
//! this crate cannot register it as a named kernel and let `infero-kernels`
//! own the math instead: there is no cycle to avoid here, since this
//! function needs nothing from that crate).
//!
//! f32-accumulating throughout -- no precision trade to make the way
//! Metal's MPS path has (it accumulates in f16 because MPS requires all
//! three matrices to share a type); this backend is not a performance
//! target, so there is no reason to take that loss too.
//!
//! The actual product is [`gemm_f32`], the `gemm` crate's SIMD- and
//! cache-blocked microkernel rather than a hand-rolled triple loop: a
//! straight `perf stat` on the hand-rolled version (rayon-parallel, one
//! `f32` multiply-add per element, nothing else in the loop) showed ~3.6
//! instructions per cycle -- good IPC, meaning the core wasn't stalling --
//! but tens of billions of instructions for a request that only needs low
//! single-digit GFLOP, i.e. the loop itself was the bottleneck, not memory
//! or scheduling. `-C target-feature=+avx2,+fma` made no measurable
//! difference in a controlled A/B, which is the signature of a loop the
//! auto-vectorizer isn't actually widening regardless of what the target
//! allows it to emit. `gemm` (the crate behind `faer`) is a maintained,
//! widely-used microkernel that dispatches to real AVX2/FMA/AVX-512 kernels
//! at runtime and handles cache blocking itself, instead of trusting one
//! more layer of LLVM auto-vectorization to do it for a hand-written loop.

use anyhow::Result;
use half::f16;
use rayon::prelude::*;

use crate::Device;
use crate::buffer::{View, ViewMut};

/// `dst(m×n) = lhs(m×k) * rhs_t(n×k)^T`, all row-major, via the `gemm`
/// crate's microkernel. `rhs_t` is stored as `n` rows of `k` (a weight
/// matrix's natural layout -- one row per output feature) rather than the
/// `k×n` a plain matmul would want; expressing that as a transpose through
/// strides costs nothing here, since the microkernel takes arbitrary
/// strides for every operand already.
pub fn gemm_f32(dst: &mut [f32], lhs: &[f32], rhs_t: &[f32], m: usize, k: usize, n: usize) {
    assert!(lhs.len() >= m * k, "gemm_f32 wants {m}x{k} of lhs, got {}", lhs.len());
    assert!(rhs_t.len() >= n * k, "gemm_f32 wants {n}x{k} of rhs_t, got {}", rhs_t.len());
    assert!(dst.len() >= m * n, "gemm_f32 writes {m}x{n}, got {}", dst.len());
    // SAFETY: the length asserts above cover every offset `gemm::gemm` can
    // reach given `m`, `k`, `n` and the strides passed below.
    unsafe {
        gemm::gemm(
            m,
            n,
            k,
            dst.as_mut_ptr(),
            1,
            n as isize,
            false,
            lhs.as_ptr(),
            1,
            k as isize,
            rhs_t.as_ptr(),
            k as isize,
            1,
            0.0f32,
            1.0f32,
            false,
            false,
            false,
            gemm::Parallelism::Rayon(0),
        );
    }
}

/// `out[t,row] = dot(w_f32[row,:], x[t,:])`, from an already-decoded,
/// persistently-cached f32 weight matrix -- no per-call decode, no
/// `WeightType` dispatch. The `infero-model`-side cache this serves is
/// `Matrix::cpu_f32_weight`; see that method's own doc comment for why a
/// persistent cache exists at all (a plain `gemv_f32`/`gemm_f32` call
/// re-decoding from raw bytes every request was paying that cost, and the
/// allocation it needs, on every single request regardless of token count).
pub fn gemv_f32_cached(
    out: &mut ViewMut<'_, f32>,
    w_f32: &[f32],
    x: &View<'_, f32>,
    k: usize,
    n: usize,
    n_tokens: usize,
) -> Result<()> {
    anyhow::ensure!(out.len() >= n_tokens * n, "gemv_f32_cached writes {n_tokens}x{n}, got {}", out.len());
    anyhow::ensure!(x.len() >= n_tokens * k, "gemv_f32_cached wants {n_tokens}x{k} of activations, got {}", x.len());
    anyhow::ensure!(w_f32.len() >= n * k, "gemv_f32_cached wants {n}x{k} of weights, got {}", w_f32.len());
    // SAFETY: both windows are live for the duration of this call, matching
    // every other reinterpret this file already does of its own `View`s.
    let (x_s, out_s) = unsafe {
        (
            std::slice::from_raw_parts(x.raw_ptr() as *const f32, x.len()),
            std::slice::from_raw_parts_mut(out.raw_ptr() as *mut f32, out.len()),
        )
    };
    gemm_f32(&mut out_s[..n_tokens * n], &x_s[..n_tokens * k], w_f32, n_tokens, k, n);
    Ok(())
}

pub fn gemm_f16_to_f32(
    _dev: &Device,
    c: &mut ViewMut<'_, f32>,
    a: &View<'_, f16>,
    b: &View<'_, f16>,
    m: usize,
    k: usize,
    n: usize,
) -> Result<()> {
    anyhow::ensure!(a.len() >= m * k, "gemm wants {m}x{k} of activations, got {}", a.len());
    anyhow::ensure!(b.len() >= n * k, "gemm wants {n}x{k} of weights, got {}", b.len());
    anyhow::ensure!(c.len() >= m * n, "gemm writes {m}x{n}, got {}", c.len());

    // SAFETY: both windows are live for the duration of this call, and both
    // were allocated with `f16`'s own alignment (typed `Buf<f16>` storage),
    // matching every other read this backend's `Stream::memcpy_dtoh` etc.
    // already perform through the same `raw_ptr()`.
    let (a_s, b_s, c_s) = unsafe {
        (
            std::slice::from_raw_parts(a.raw_ptr() as *const f16, a.len()),
            std::slice::from_raw_parts(b.raw_ptr() as *const f16, b.len()),
            std::slice::from_raw_parts_mut(c.raw_ptr() as *mut f32, c.len()),
        )
    };

    // Decode each operand once: `gemm_f32` wants plain `f32` operands, and
    // doing that decode once up front is (m+n)*k conversions rather than
    // the m*n*k an inline decode inside the product would cost.
    let a_f32: Vec<f32> = a_s[..m * k].par_iter().map(|v| v.to_f32()).collect();
    let b_f32: Vec<f32> = b_s[..n * k].par_iter().map(|v| v.to_f32()).collect();

    gemm_f32(&mut c_s[..m * n], &a_f32, &b_f32, m, k, n);
    Ok(())
}
