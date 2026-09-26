//! The CPU device layer -- for running a small, plain-dense-decoder model
//! (Qwen2/Qwen3-family, F32/F16/Q8_0 only) with no GPU at all, not a
//! performance target and not feature-complete with the CUDA/Metal
//! backends.
//!
//! **What this backend covers:** the same subset Metal's own doc comment
//! names -- a dense decoder's forward pass (embedding lookup, RMSNorm,
//! RoPE, causal GQA attention, SwiGLU, F32/F16/Q8_0 weights) -- implemented
//! in `crates/kernels/src/cpu/mod.rs` as plain, mostly `rayon`-parallel Rust,
//! not a port of the CUDA kernels' own tiled/warp-specialized algorithms.
//! **What it does not cover:** MoE, the vision tower, GatedDeltaNet/linear
//! attention, TurboQuant KV compression, tensor parallelism, speculative
//! decoding, CUDA graphs, or any quantization format beyond Q8_0. A model
//! that needs one of those fails at the specific dispatch that needed it,
//! with a message naming what was missing, rather than silently degrading.
//!
//! Shaped like the subset of `cudarc` `infero-kernels` uses, the same way
//! `infero-metal` is: same method names, same argument order, same
//! `LaunchConfig` field names, so the ~160 launch sites in that crate
//! compile against this backend unchanged. What *is* different from both
//! GPU backends: this crate does not itself know what any kernel computes.
//! There is no source to compile, so `Function::launch` calls back into a
//! dispatcher `infero-kernels` registers once (`set_dispatcher`) -- see
//! `launch.rs`'s own doc comment for why the numerical code cannot live
//! here.

pub mod backend;
mod buffer;
mod compat;
mod device;
mod gemm;
mod launch;
mod profile;

pub use buffer::{Buf, CopyDst, CopySrc, Elem, View, ViewMut};
pub use device::{Caps, Device, Stream};
pub use launch::{Arg, Dispatch, Function, KernelArg, LaunchBuilder, LaunchConfig, Modules, NullBuffer, set_dispatcher};
pub use profile::{Entry, Profile};
