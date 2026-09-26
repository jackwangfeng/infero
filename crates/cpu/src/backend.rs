//! The neutral names, pointed at this backend.
//!
//! The same surface `infero_cuda::backend` and `infero_metal::backend`
//! export, so `infero-kernels` and `infero-model` compile against any of the
//! three without naming a vendor.

pub use crate::buffer::{Buf, CopyDst, CopySrc, View, ViewMut};
pub use crate::device::{Device, Stream};
pub use crate::gemm::gemm_f16_to_f32;
pub use crate::launch::{Function, KernelArg, LaunchConfig, NullBuffer};

pub const NULL_BUFFER: NullBuffer = NullBuffer;
pub use crate::compat::{CaptureMode, Context, Event, EventFlags, Graph, GraphFlags, OwnedStream, PinnedHostSlice};

pub const EVENT_DEFAULT: EventFlags = EventFlags;
pub const CAPTURE_RELAXED: CaptureMode = CaptureMode;

/// Raise a kernel's dynamic shared-memory ceiling -- a no-op here, the same
/// way it is on Metal: there is no equivalent opt-in, and nothing on this
/// backend reads `shared_mem_bytes` as anything but an unused field.
pub fn set_max_dynamic_shared(_f: &Function, _bytes: u32) -> anyhow::Result<()> {
    Ok(())
}

/// Alias a file into "device" memory -- an ordinary `mmap`. See
/// `buffer::map_file`.
pub fn map_file(dev: &Device, path: &std::path::Path) -> anyhow::Result<Option<Buf<u8>>> {
    crate::buffer::map_file(dev, path).map(Some)
}
