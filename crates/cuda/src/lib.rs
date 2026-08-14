//! Thin CUDA layer for tuili.
//!
//! Everything above this crate talks to the GPU through [`Device`]: one context,
//! one compute stream, one cuBLAS handle, and a runtime-compiled kernel cache.
//! There is no `nvcc` on the build box, so kernels are compiled by NVRTC on
//! first use and cached as PTX on disk.

pub mod device;
pub mod loader;
pub mod nvrtc;
pub mod profile;

pub use device::Device;
pub use nvrtc::{Kernel, KernelCache, set_max_dynamic_shared};
pub use profile::Profile;

/// Directory holding the CUDA headers NVRTC needs (`cuda_fp16.h` and friends).
pub const CUDA_INCLUDE_DIR: &str = env!("TUILI_CUDA_INCLUDE");

/// Directory holding `libcublas.so` / `libnvrtc.so`; baked into the rpath.
pub const CUDA_LIB_DIR: &str = env!("TUILI_CUDA_LIB");
