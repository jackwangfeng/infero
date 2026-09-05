//! Tensor-parallel cross-GPU communication: a safe wrapper around NCCL.
//! See `crates/kernels/src/cu_vendor/nccl_shim.rs` for the raw FFI bindings
//! and `docs/superpowers/specs/2026-09-05-tensor-parallel-design.md` for the
//! design this implements.

use crate::cu_vendor::nccl_shim::*;
use anyhow::{ensure, Result};
use cudarc::driver::DevicePtrMut;
use infero_gpu::{Stream, ViewMut};

pub use crate::cu_vendor::nccl_shim::NCCL_UNIQUE_ID_BYTES;

pub struct NcclUniqueId(pub [u8; NCCL_UNIQUE_ID_BYTES]);

impl NcclUniqueId {
    pub fn generate() -> Result<Self> {
        let mut id = ncclUniqueId { internal: [0u8; NCCL_UNIQUE_ID_BYTES] };
        let rc = unsafe { ncclGetUniqueId(&mut id) };
        ensure!(rc == 0, "ncclGetUniqueId failed, code {rc}");
        Ok(Self(id.internal))
    }
}

/// Owns one rank's NCCL communicator handle for its tensor-parallel group.
/// `Send`/`Sync`: NCCL's own docs guarantee a `ncclComm_t` is safe to use
/// from any thread as long as calls into it are externally serialized (which
/// every call site here already is, via `&self` plus the single CUDA stream
/// they're issued on) -- the same safety argument this crate already makes
/// for its other opaque FFI handles.
pub struct NcclComm {
    handle: ncclComm_t,
}
unsafe impl Send for NcclComm {}
unsafe impl Sync for NcclComm {}

impl NcclComm {
    pub fn init_rank(unique_id: &NcclUniqueId, rank: i32, world_size: i32) -> Result<Self> {
        let mut handle: ncclComm_t = std::ptr::null_mut();
        let id = ncclUniqueId { internal: unique_id.0 };
        let rc = unsafe { ncclCommInitRank(&mut handle, world_size, id, rank) };
        ensure!(rc == 0, "ncclCommInitRank failed, code {rc}");
        Ok(Self { handle })
    }

    /// Sums `buf` (`count` f32 elements) across every rank in this
    /// communicator's group, in place, on `stream`. `count` is passed
    /// explicitly rather than read off `buf` -- matches this crate's existing
    /// FFI convention (see `flash_attn2.rs`'s `n_q_elems`/`n_out_elems`)
    /// rather than assuming a `View`/`ViewMut` exposes a length accessor.
    pub fn all_reduce_sum_f32(
        &self,
        buf: &mut ViewMut<'_, f32>,
        count: usize,
        stream: &Stream,
    ) -> Result<()> {
        let (ptr, guard) = buf.device_ptr_mut(stream);
        let rc = unsafe {
            ncclAllReduce(
                ptr as *const std::ffi::c_void,
                ptr as *mut std::ffi::c_void,
                count,
                ncclDataType_t::ncclFloat32,
                ncclRedOp_t::ncclSum,
                self.handle,
                stream.cu_stream(),
            )
        };
        drop(guard);
        ensure!(rc == 0, "ncclAllReduce failed, code {rc}");
        Ok(())
    }

    /// Broadcasts `count` bytes from `root`'s `buf` to every rank's `buf`, in
    /// place, on `stream`. `buf` must be DEVICE memory -- NCCL collectives
    /// operate on GPU buffers, not host memory, so a caller with host-side
    /// bytes (e.g. serialized scheduler metadata) must copy them to a device
    /// buffer before calling this and back after, or use a non-NCCL
    /// side-channel instead if that round-trip isn't worth it for small,
    /// infrequent control-plane data (Task 5 decides this, not this
    /// low-level wrapper). `ncclUint8` per-element, so `count` here is a
    /// byte count, matching `all_reduce_sum_f32`'s explicit-count convention.
    pub fn broadcast_bytes(
        &self,
        buf: &mut ViewMut<'_, u8>,
        count: usize,
        root: i32,
        stream: &Stream,
    ) -> Result<()> {
        let (ptr, guard) = buf.device_ptr_mut(stream);
        let rc = unsafe {
            ncclBroadcast(
                ptr as *const std::ffi::c_void,
                ptr as *mut std::ffi::c_void,
                count,
                ncclDataType_t::ncclUint8,
                root,
                self.handle,
                stream.cu_stream(),
            )
        };
        drop(guard);
        ensure!(rc == 0, "ncclBroadcast failed, code {rc}");
        Ok(())
    }
}

impl Drop for NcclComm {
    fn drop(&mut self) {
        unsafe {
            ncclCommDestroy(self.handle);
        }
    }
}
