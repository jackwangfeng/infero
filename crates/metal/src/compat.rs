//! The CUDA-shaped surface Metal does not have.
//!
//! Three subsystems in `infero-model` are CUDA-only and stay that way: layer
//! offload, CUDA graph capture, and the per-phase event timers. Between them
//! they touch 78 sites in `model/src/lib.rs`, so gating each one would bury a
//! hot file in `#[cfg]` -- which is the opposite of how this repository reads.
//!
//! Instead the types exist here so the code typechecks, and the *constructors*
//! fail. A feature that cannot work says so once, at the one place that asks
//! for it, rather than being silently faked or scattering conditionals through
//! the caller. Where the concept does survive translation -- pinned host memory
//! is just memory when the GPU shares the address space -- the implementation
//! is real.

use std::sync::Arc;

use anyhow::{Result, bail};

/// Host memory a DMA engine may read.
///
/// Real, and trivially so: on unified memory every host allocation is already
/// GPU-visible, so "pinned" is the default rather than a request. What has no
/// meaning here is what the engine *does* with it -- staging a layer from host
/// to device across a copy stream, when both names point at the same DRAM.
pub struct PinnedHostSlice<T> {
    v: Vec<T>,
}

impl<T: Copy + Default> PinnedHostSlice<T> {
    fn new(n: usize) -> Self {
        Self {
            v: vec![T::default(); n],
        }
    }
}

impl<T> PinnedHostSlice<T> {
    pub fn len(&self) -> usize {
        self.v.len()
    }
    pub fn is_empty(&self) -> bool {
        self.v.is_empty()
    }
    pub fn as_slice(&self) -> &[T] {
        &self.v
    }
    /// `Result` to match the CUDA signature, where the mapping can fail.
    pub fn as_mut_slice(&mut self) -> Result<&mut [T]> {
        Ok(&mut self.v)
    }
}

/// A second stream, for overlapping copies with compute.
///
/// This backend has one command queue. A second queue is easy to make; what is
/// not is the thing it would be for, since there is no host-to-device transfer
/// to hide behind arithmetic.
pub struct OwnedStream;

impl OwnedStream {
    pub fn wait(&self, _e: &Event) -> Result<()> {
        bail!("this backend has no second stream to wait on")
    }

    pub fn memcpy_htod<S: ?Sized, D>(&self, _src: &S, _dst: &mut D) -> Result<()> {
        bail!("offload copies need a second stream, which this backend has not")
    }
}

/// A cross-stream ordering point.
pub struct Event;

impl Event {
    pub fn record<S>(&self, _stream: S) -> Result<()> {
        bail!("events are a CUDA-only path on this backend")
    }
    pub fn synchronize(&self) -> Result<()> {
        bail!("events are a CUDA-only path on this backend")
    }
    pub fn elapsed_ms(&self, _other: &Event) -> Result<f32> {
        bail!("events are a CUDA-only path on this backend")
    }
}

/// Flags an event is created with. Opaque, and never read here.
#[derive(Debug, Clone, Copy, Default)]
pub struct EventFlags;

/// How a capture treats work from other threads. Opaque here.
#[derive(Debug, Clone, Copy, Default)]
pub struct CaptureMode;

/// Flags a captured graph is instantiated with.
///
/// `repr(transparent)` over a `u32` so the caller's `transmute::<u32, _>(0)`
/// for "no flags" stays sound on this backend too. Never read here.
#[derive(Debug, Clone, Copy, Default)]
#[repr(transparent)]
pub struct GraphFlags(pub u32);

impl GraphFlags {
    #[allow(non_upper_case_globals)]
    pub const CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH: Self = Self(1);
}

/// A recorded, replayable sequence of launches.
///
/// Metal has indirect command buffers, which are a different mechanism with a
/// different cost model: they encode a fixed argument set up front rather than
/// replaying a captured stream. Worth building, and not by pretending to be
/// this. Meanwhile the engine's `use_graph` reads false on this backend and the
/// capture path is never entered.
pub struct Graph;

impl Graph {
    pub fn launch(&self) -> Result<()> {
        bail!("graph replay is a CUDA-only path on this backend")
    }

    pub fn upload(&self) -> Result<()> {
        bail!("graph replay is a CUDA-only path on this backend")
    }
}

/// The device context, for the allocations and streams that hang off it.
pub struct Context;

impl Context {
    pub fn new_stream(&self) -> Result<Arc<OwnedStream>> {
        bail!("this backend has a single command queue; offload needs a second stream")
    }

    pub fn new_event(&self, _flags: Option<EventFlags>) -> Result<Event> {
        bail!("events are a CUDA-only path on this backend")
    }

    /// # Safety
    /// Matches the CUDA signature, which is unsafe because the allocation is
    /// uninitialised there. Here it is zeroed, so there is nothing to uphold.
    pub unsafe fn alloc_pinned<T: Copy + Default>(&self, n: usize) -> Result<PinnedHostSlice<T>> {
        Ok(PinnedHostSlice::new(n))
    }
}
