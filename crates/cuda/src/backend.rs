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
