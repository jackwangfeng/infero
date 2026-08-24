//! `MPSMatrixMultiplication`, which is this backend's GEMM.
//!
//! Prefill is what wants it. A mat-vec reads the whole weight matrix to produce
//! one column of output, so a 256-token chunk pushed through the mat-vec reads
//! 18 GiB of weights sixty-four times: measured at 48 ms a prompt token, flat,
//! which makes a 1000-token conversation cost 48 seconds before the first token
//! of the answer. A GEMM reads them once.
//!
//! Apple's rather than hand-written, for the same reason the CUDA side calls
//! cuBLAS rather than writing its own: a competitive tiled matmul on this
//! hardware is a tuning problem, not a correctness problem, and MPS has already
//! been tuned for it.
//!
//! Two things the API forces that are worth stating.
//!
//! **All three matrices share one data type.** cuBLAS `GemmEx` takes f16 inputs
//! and accumulates into an f32 result in one call; `MPSMatrixMultiplication`
//! does not, so the result comes back f16 and a conversion pass follows. That
//! pass is `n_tokens * n` elements -- 8.9 MB at the widest prefill shape here --
//! against a GEMM that just read hundreds of megabytes, so it is not where the
//! time goes. It does mean the accumulator is f16 where CUDA's is f32; see the
//! note on `gemm_f16` in `tuili-kernels` for what that costs.
//!
//! **It encodes onto a command buffer, not an encoder.** See
//! `Batch::encode_on_buffer`.

use std::collections::HashMap;
use std::sync::Mutex;

use anyhow::{Result, anyhow};
use objc2::AnyThread;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSUInteger;
use objc2_metal::MTLBuffer;
use objc2_metal_performance_shaders::{
    MPSDataType, MPSMatrix, MPSMatrixDescriptor, MPSMatrixMultiplication,
};

use crate::device::Device;

/// One multiplication kernel per shape.
///
/// `MPSMatrixMultiplication` bakes the shape in at construction, and a prefill
/// walks the same handful of shapes for every layer -- four or five per block,
/// the same ones in the next block -- so building them once and keeping them is
/// most of what this cache is for. The key is the shape triple; nothing else
/// about a call varies.
#[derive(Default)]
pub(crate) struct GemmCache {
    kernels: Mutex<HashMap<(usize, usize, usize), Retained<MPSMatrixMultiplication>>>,
    /// The f16 result, before it is widened.
    ///
    /// Grown, never shrunk, and shared by every shape: a prefill walks four or
    /// five matmuls a block and the widest of them sizes this once. 8.9 MB at
    /// `256 x 17408`.
    out: Mutex<Option<crate::buffer::Buf<half::f16>>>,
}

// SAFETY: MPS kernel objects are used behind the mutex and, like `MTLBuffer`,
// are documented safe to use from multiple threads once created.
unsafe impl Send for GemmCache {}
unsafe impl Sync for GemmCache {}

fn descriptor(rows: usize, columns: usize, row_bytes: usize) -> Result<Retained<MPSMatrixDescriptor>> {
    unsafe {
        Ok(MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(
            rows as NSUInteger,
            columns as NSUInteger,
            row_bytes as NSUInteger,
            MPSDataType::Float16,
        ))
    }
}

fn matrix(
    buf: &ProtocolObject<dyn MTLBuffer>,
    offset: usize,
    desc: &MPSMatrixDescriptor,
) -> Result<Retained<MPSMatrix>> {
    unsafe {
        MPSMatrix::initWithBuffer_offset_descriptor(MPSMatrix::alloc(), buf, offset as NSUInteger, desc)
    }
    .into_ok()
}

trait IntoOk {
    type T;
    fn into_ok(self) -> Result<Self::T>;
}

impl<T> IntoOk for Retained<T> {
    type T = Retained<T>;
    fn into_ok(self) -> Result<Retained<T>> {
        Ok(self)
    }
}

/// `c = a * b^T`, all f16.
///
/// `a` is `[m, k]` row-major, `b` is `[n, k]` row-major -- the layout every
/// weight in this engine already has, one output row contiguous -- and `c` is
/// `[m, n]`. So the right operand is transposed and the left is not, which is
/// the same pair of flags the cuBLAS call passes.
///
/// Offsets are in elements and become byte offsets here; a view into the middle
/// of a scratch buffer is the normal case.
pub(crate) fn gemm_f16(
    dev: &Device,
    c: (&ProtocolObject<dyn MTLBuffer>, usize),
    a: (&ProtocolObject<dyn MTLBuffer>, usize),
    b: (&ProtocolObject<dyn MTLBuffer>, usize),
    m: usize,
    k: usize,
    n: usize,
) -> Result<()> {
    anyhow::ensure!(m > 0 && k > 0 && n > 0, "gemm {m}x{k}x{n} has an empty side");

    let kernel = {
        let mut cache = dev.gemm_cache().kernels.lock().unwrap();
        cache
            .entry((m, k, n))
            .or_insert_with(|| unsafe {
                MPSMatrixMultiplication::initWithDevice_transposeLeft_transposeRight_resultRows_resultColumns_interiorColumns_alpha_beta(
                    MPSMatrixMultiplication::alloc(),
                    dev.raw(),
                    false,
                    true,
                    m as NSUInteger,
                    n as NSUInteger,
                    k as NSUInteger,
                    1.0,
                    0.0,
                )
            })
            .clone()
    };

    let two = std::mem::size_of::<half::f16>();
    let a_desc = descriptor(m, k, k * two)?;
    let b_desc = descriptor(n, k, k * two)?;
    let c_desc = descriptor(m, n, n * two)?;
    let a_mat = matrix(a.0, a.1 * two, &a_desc)?;
    let b_mat = matrix(b.0, b.1 * two, &b_desc)?;
    let c_mat = matrix(c.0, c.1 * two, &c_desc)?;

    dev.batch().encode_on_buffer(dev.raw_queue(), |cb| {
        unsafe {
            kernel.encodeToCommandBuffer_leftMatrix_rightMatrix_resultMatrix(
                cb, &a_mat, &b_mat, &c_mat,
            );
        }
        Ok(())
    })
    .map_err(|e| anyhow!("encoding a {m}x{k}x{n} MPS matmul: {e}"))
}


/// Widening the f16 result, which MPS will not do in the matmul.
///
/// Carried here rather than in `tuili-kernels` with the model's kernels because
/// that is where it belongs: it exists only to paper over one API's insistence
/// that its three matrices share a data type, and no forward pass should have to
/// know about it. The metal crate cannot depend on `tuili-kernels` -- the arrow
/// points the other way -- so it brings its own fifteen lines.
const WIDEN_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void gemm_widen_f16_f32(
    device float* out         [[buffer(0)]],
    device const half* in     [[buffer(1)]],
    constant int& n           [[buffer(2)]],
    uint3 tgid  [[threadgroup_position_in_grid]],
    uint3 tid   [[thread_position_in_threadgroup]],
    uint3 tgdim [[threads_per_threadgroup]]) {
    const int base = int(tgid.x * tgdim.x + tid.x) * 4;
    if (base >= n) return;
    for (int i = base; i < base + 4 && i < n; ++i) out[i] = float(in[i]);
}
"#;

/// `c = a * b^T` with an f32 result, which is the signature the engine wants.
///
/// The f16 intermediate and the widening pass are both internal: a caller hands
/// in the same `ViewMut<f32>` the cuBLAS path writes and cannot tell the
/// difference, except in the last bits. MPS accumulates in f16 here where cuBLAS
/// is asked for f32, so a long `k` loses precision the CUDA path keeps -- at
/// k = 17408 that is real, and it is why this is a prefill path and the decode
/// mat-vec, which accumulates in f32, is left alone.
pub fn gemm_f16_to_f32(
    dev: &Device,
    c: &mut crate::buffer::ViewMut<'_, f32>,
    a: &crate::buffer::View<'_, half::f16>,
    b: &crate::buffer::View<'_, half::f16>,
    m: usize,
    k: usize,
    n: usize,
) -> Result<()> {
    anyhow::ensure!(
        a.len() >= m * k,
        "gemm wants {m}x{k} of activations, got {}",
        a.len()
    );
    anyhow::ensure!(b.len() >= n * k, "gemm wants {n}x{k} of weights, got {}", b.len());
    anyhow::ensure!(c.len() >= m * n, "gemm writes {m}x{n}, got {}", c.len());

    // The f16 result, grown to fit and kept.
    let need = m * n;
    {
        let mut slot = dev.gemm_cache().out.lock().unwrap();
        let grow = match slot.as_ref() {
            Some(buf) => buf.len() < need,
            None => true,
        };
        if grow {
            *slot = Some(dev.stream().alloc_zeros::<half::f16>(need)?);
        }
    }
    let slot = dev.gemm_cache().out.lock().unwrap();
    let out16 = slot.as_ref().unwrap();

    gemm_f16(
        dev,
        (out16.as_view().raw_buf(), 0),
        (a.raw_buf(), a.byte_offset() / std::mem::size_of::<half::f16>()),
        (b.raw_buf(), b.byte_offset() / std::mem::size_of::<half::f16>()),
        m,
        k,
        n,
    )?;

    // Four elements a thread, which is the contract every elementwise kernel in
    // this backend uses and the one `add_assign` was silently breaking.
    let f = dev
        .kernels()
        .get("tuili_metal_gemm", WIDEN_MSL, "gemm_widen_f16_f32")?;
    let ni = need as i32;
    let src = out16.slice(..need);
    let mut b2 = dev.stream().launch_builder(&f);
    b2.arg(c).arg(&src).arg(&ni);
    let block = 256u32;
    unsafe {
        b2.launch(crate::launch::LaunchConfig {
            grid_dim: ((need as u32).div_ceil(block * 4).max(1), 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        })
    }
    .map_err(|e| anyhow!("widening a {m}x{n} gemm result: {e}"))
}
