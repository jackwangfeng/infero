//! The device layer the rest of the engine sees.
//!
//! Exactly one backend is compiled in, and everything above this crate spells
//! its types `Buf`, `View`, `ViewMut`, `LaunchConfig` rather than naming a
//! vendor. The names are cudarc's shapes because that is what the 160 launch
//! sites in `tuili-kernels` were written against; the Metal backend was built
//! to match them, which is why those sites need no change.
//!
//! What does *not* live here is the tuning and capture surface -- register
//! counts, occupancy probes, CUDA graphs, cuBLAS, pinned host memory for
//! offload. Those have no Metal counterpart, or a counterpart that answers a
//! different question, so they stay behind `#[cfg(feature = "cuda")]` at their
//! call sites rather than being faked here.

#[cfg(all(feature = "cuda", feature = "metal"))]
compile_error!(
    "tuili-gpu takes exactly one backend: build with --features cuda-13 or --features metal"
);

#[cfg(not(any(feature = "cuda", feature = "metal")))]
compile_error!("tuili-gpu needs a backend: --features cuda-13 or --features metal");

#[cfg(feature = "cuda")]
pub use tuili_cuda::backend::*;

#[cfg(feature = "metal")]
pub use tuili_metal::backend::*;

/// Which backend was compiled in, for the one or two places that log it.
pub const BACKEND: &str = if cfg!(feature = "metal") { "metal" } else { "cuda" };
