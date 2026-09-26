//! The CUDA-shaped surface this backend does not have.
//!
//! Mirrors `infero_metal::compat` almost exactly, and for the same reason:
//! layer offload, CUDA graph capture, and per-phase event timers are
//! CUDA-only concepts touching dozens of sites in `model/src/lib.rs`, so the
//! types exist here so the code typechecks and the *constructors* fail with
//! a clear message, rather than gating each call site or silently faking the
//! feature. Pinned host memory is the one exception: it is just memory here
//! too, so that implementation is real.

use std::sync::Arc;

use anyhow::{Result, bail};

/// Host memory a DMA engine may read -- real, and trivially so, since there
/// is no separate device memory for a DMA engine to read *into*.
pub struct PinnedHostSlice<T> {
    v: Vec<T>,
}

impl<T: Copy + Default> PinnedHostSlice<T> {
    fn new(n: usize) -> Self {
        Self { v: vec![T::default(); n] }
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
    pub fn as_mut_slice(&mut self) -> Result<&mut [T]> {
        Ok(&mut self.v)
    }
}

pub struct OwnedStream;

impl OwnedStream {
    pub fn wait(&self, _e: &Event) -> Result<()> {
        bail!("this backend has no second stream to wait on")
    }

    pub fn memcpy_htod<S: ?Sized, D>(&self, _src: &S, _dst: &mut D) -> Result<()> {
        bail!("offload copies need a second stream, which this backend has not")
    }
}

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

#[derive(Debug, Clone, Copy, Default)]
pub struct EventFlags;

#[derive(Debug, Clone, Copy, Default)]
pub struct CaptureMode;

#[derive(Debug, Clone, Copy, Default)]
#[repr(transparent)]
pub struct GraphFlags(pub u32);

impl GraphFlags {
    #[allow(non_upper_case_globals)]
    pub const CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH: Self = Self(1);
}

pub struct Graph;

impl Graph {
    pub fn launch(&self) -> Result<()> {
        bail!("graph replay is a CUDA-only path on this backend")
    }

    pub fn upload(&self) -> Result<()> {
        bail!("graph replay is a CUDA-only path on this backend")
    }
}

pub struct Context;

impl Context {
    pub fn new_stream(&self) -> Result<Arc<OwnedStream>> {
        bail!("this backend has a single stream; offload needs a second one")
    }

    pub fn new_event(&self, _flags: Option<EventFlags>) -> Result<Event> {
        bail!("events are a CUDA-only path on this backend")
    }

    /// # Safety
    /// Matches the CUDA signature, unsafe there because the allocation is
    /// uninitialised. Here it is zeroed, so there is nothing to uphold.
    pub unsafe fn alloc_pinned<T: Copy + Default>(&self, n: usize) -> Result<PinnedHostSlice<T>> {
        Ok(PinnedHostSlice::new(n))
    }
}
