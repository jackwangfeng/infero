//! The neutral names, pointed at Metal.
//!
//! The same surface `tuili_cuda::backend` exports, so `tuili-kernels` and
//! `tuili-model` compile against either without naming a vendor.

pub use crate::buffer::{Buf, CopyDst, CopySrc, View, ViewMut};
pub use crate::device::{Device, Stream};
pub use crate::launch::{Function, KernelArg, LaunchConfig};
pub use crate::compat::{
    CaptureMode, Context, Event, EventFlags, Graph, GraphFlags, OwnedStream, PinnedHostSlice,
};

/// Flags the engine asks for by name rather than by value, so the call sites
/// stay free of vendor enums.
pub const EVENT_DEFAULT: EventFlags = EventFlags;
pub const CAPTURE_RELAXED: CaptureMode = CaptureMode;

/// Raise a kernel's dynamic shared-memory ceiling -- a no-op here.
///
/// CUDA needs an explicit opt-in past 48 KiB; Metal's threadgroup memory limit
/// is a fixed device property (32 KiB on Apple GPUs) with nothing to opt into,
/// and a request past it fails at `dispatchThreadgroups` where it belongs. So
/// this accepts the request and lets the dispatch be the judge.
pub fn set_max_dynamic_shared(_f: &Function, _bytes: u32) -> anyhow::Result<()> {
    Ok(())
}
