//! The neutral names, pointed at cudarc.
//!
//! Aliases rather than wrappers: `Buf<T>` *is* `CudaSlice<T>`, so nothing above
//! this module pays for the rename and no method is hidden by accident. The
//! Metal backend implements the same surface with the same names.

pub use crate::Device;
pub use cudarc::driver::{
    CudaSlice as Buf, CudaStream as Stream, CudaView as View, CudaViewMut as ViewMut, LaunchConfig,
    PushKernelArg as KernelArg,
};
pub use crate::nvrtc::Kernel as Function;
pub use cudarc::driver::{CudaEvent as Event, CudaGraph as Graph, PinnedHostSlice};
pub use cudarc::driver::CudaStream as OwnedStream;
pub use cudarc::driver::sys::CUevent_flags as EventFlags;
pub use cudarc::driver::sys::CUstreamCaptureMode as CaptureMode;
pub use cudarc::driver::sys::CUgraphInstantiate_flags as GraphFlags;

/// Flags the engine asks for by name rather than by value, so the call sites
/// stay free of vendor enums.
pub const EVENT_DEFAULT: EventFlags = EventFlags::CU_EVENT_DEFAULT;
pub const CAPTURE_RELAXED: CaptureMode = CaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED;

/// Raise a kernel's dynamic shared-memory ceiling.
///
/// CUDA caps a launch at 48 KiB of dynamic shared memory unless the function
/// opts in; Metal has no such opt-in, so its implementation is a no-op. Kept in
/// the neutral layer rather than behind a cfg at each of the four call sites,
/// because the call reads the same either way: ask for the memory, and find out
/// whether the kernel can have it.
pub fn set_max_dynamic_shared(f: &Function, bytes: u32) -> anyhow::Result<()> {
    crate::nvrtc::set_max_dynamic_shared(f, bytes)
}

/// What the host is allowed to assume about this GPU.
///
/// Mirrors `tuili_metal::Caps`. On CUDA every field is derived from the compute
/// capability, which is where the four `arch() >= 80` gates used to read it
/// directly -- the capability name says *what* is being asked for, where the
/// number said only which cards happen to have it.
#[derive(Debug, Clone, Copy)]
pub struct Caps {
    /// `mmq.cu`: the integer tensor-core GEMM. sm_80 and later.
    pub int_tensor_gemm: bool,
    /// `fp8.cu`: block-scaled FP8 mat-vec. sm_89 and later.
    pub fp8: bool,
    /// `cp.async.bulk.tensor`. sm_90 and later.
    pub tma: bool,
    pub simd_width: u32,
    pub max_threads_per_group: u32,
}

impl Device {
    pub fn caps(&self) -> Caps {
        let a = self.arch();
        Caps {
            int_tensor_gemm: a >= 80,
            fp8: a >= 89,
            tma: a >= 90,
            simd_width: 32,
            max_threads_per_group: 1024,
        }
    }
}
