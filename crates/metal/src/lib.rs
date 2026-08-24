//! The Apple-GPU device layer.
//!
//! This crate is shaped like the subset of `cudarc` that `tuili-kernels` uses,
//! on purpose: the same method names in the same order, so that
//!
//! ```ignore
//! let f = dev.kernels().get("tuili_ops", ops_src(), "rms_norm_f32")?;
//! let mut b = dev.stream().launch_builder(&f);
//! b.arg(out).arg(x).arg(weight).arg(&d_i).arg(&eps_f);
//! unsafe { b.launch(cfg) }?;
//! ```
//!
//! compiles unchanged against either backend. The mapping is closer than it
//! looks: `.arg()` position becomes MSL's `[[buffer(n)]]` index, and a view's
//! element offset becomes the byte offset of `setBuffer:offset:atIndex:`.
//!
//! What does *not* map is the tuning surface -- register counts, occupancy,
//! `ldmatrix` probes. Those stay behind CUDA-only entry points; Metal reports
//! what it can (`threadExecutionWidth`, `maxTotalThreadsPerThreadgroup`) and
//! grids here are fixed constants rather than occupancy-derived.

pub mod backend;
mod compat;
mod buffer;
mod device;
mod launch;
mod msl;
mod profile;

pub use buffer::{Buf, CopyDst, CopySrc, View, ViewMut};
pub use device::{Caps, Device, Stream};
pub use launch::{Function, KernelArg, LaunchBuilder, LaunchConfig};
pub use profile::{Entry, Profile};
pub use msl::Modules;
