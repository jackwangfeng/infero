//! `c = a * b^T` with an f32 result -- the free function `Kernels::gemm_f16`
//! calls directly (not through the name dispatcher; see `infero_metal::gemm`
//! for the precedent this mirrors and `launch.rs`'s own doc comment for why
//! this crate cannot register it as a named kernel and let `infero-kernels`
//! own the math instead: there is no cycle to avoid here, since this
//! function needs nothing from that crate).
//!
//! Plain triple loop, f32-accumulating throughout -- no precision trade to
//! make the way Metal's MPS path has (it accumulates in f16 because MPS
//! requires all three matrices to share a type); this backend is not a
//! performance target, so there is no reason to take that loss too.

use anyhow::Result;
use half::f16;

use crate::Device;
use crate::buffer::{View, ViewMut};

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

    for row in 0..m {
        let ar = &a_s[row * k..row * k + k];
        for col in 0..n {
            let br = &b_s[col * k..col * k + k];
            let mut acc = 0.0f32;
            for i in 0..k {
                acc += ar[i].to_f32() * br[i].to_f32();
            }
            c_s[row * n + col] = acc;
        }
    }
    Ok(())
}
